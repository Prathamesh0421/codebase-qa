"""Retrieval evaluation harness: precision/recall for naive vs hybrid vs
hybrid+graph against a hand-labeled Flask Q&A set.

Run with:  python -m evals.runners.retrieval_eval

This is the project's central measurement, not a demo -- it exists to find
out whether call-graph expansion actually earns its complexity, and the
numbers go in the README whatever they turn out to say (see
docs/IMPLEMENTATION_PLAN.md's Phase 9 "done when" bar).

A metric subtlety worth stating up front: HybridGraphStrategy.retrieve()
returns primary + expanded, where primary is exactly HybridStrategy's own
top_k, unchanged. That means precision/recall computed over hybrid_graph's
first top_k results is numerically IDENTICAL to hybrid's -- graph-expanded
chunks live entirely past that cutoff by construction. Printed anyway,
because the identity itself is worth seeing rather than silently omitting a
row for the project's headline strategy. The metric that actually answers
"does graph expansion help" is recall over hybrid_graph's FULL returned list
(primary + expansion), compared against hybrid's own recall -- the lift.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from codeqa.config import get_settings
from codeqa.indexing.embeddings import EmbeddingProvider, build_embedder
from codeqa.indexing.pipeline import index_repo
from codeqa.indexing.store import register_repo
from codeqa.retrieval.strategy import RetrievedChunk, get_strategy
from evals.runners.metrics import GoldItem, PrecisionRecall, chunk_matches_gold, score

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "flask_qa.json"
FLASK_FIXTURE = Path(__file__).parent.parent.parent / "tests" / "fixtures" / "repos" / "flask"
FLASK_EVAL_SLUG = "flask-eval"


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    gold: tuple[GoldItem, ...]


def load_dataset(path: Path = DATASET_PATH) -> list[Question]:
    raw = json.loads(path.read_text())
    return [
        Question(
            id=item["id"],
            question=item["question"],
            gold=tuple(GoldItem(file=g["file"], symbol=g["symbol"]) for g in item["gold"]),
        )
        for item in raw
    ]


def ensure_flask_indexed(conn: psycopg.Connection, embedder: EmbeddingProvider) -> int:
    """Reuse an existing flask-eval index if one exists; index it fresh
    otherwise. Deliberately persistent across runs (unlike the test suite's
    per-test fixtures, which tear down) -- re-running the harness during
    interview prep shouldn't pay a ~5s re-index every time.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM repos WHERE slug = %s", (FLASK_EVAL_SLUG,))
        row = cur.fetchone()
    if row is not None:
        return row[0]

    settings = get_settings()
    repo_id = register_repo(
        conn, FLASK_EVAL_SLUG, "Flask (eval)", "local_path", str(FLASK_FIXTURE),
        settings.embedding_model, settings.embedding_dim,
    )
    index_repo(conn, repo_id, FLASK_FIXTURE, embedder)
    conn.commit()
    return repo_id


@dataclass(frozen=True)
class GraphRecoveryDetail:
    """A gold item that hybrid's own top_k missed but hybrid_graph's full
    list (primary + expansion) found -- and the source label attached to
    the chunk that recovered it. source == "graph" exactly means no other
    component's candidate pool had touched that chunk either; a "+"
    suffix means it was reachable another way too, just not highly enough
    ranked to survive fusion into hybrid's own top_k.
    """

    question_id: str
    gold: GoldItem
    source: str


@dataclass(frozen=True)
class QuestionResult:
    question: Question
    naive: PrecisionRecall
    hybrid: PrecisionRecall
    # First top_k of hybrid_graph's list -- identical to hybrid's, by construction.
    hybrid_graph_primary: PrecisionRecall
    hybrid_graph_full: PrecisionRecall  # primary + expansion
    graph_recoveries: tuple[GraphRecoveryDetail, ...]


def _graph_recoveries(
    question_id: str,
    gold: tuple[GoldItem, ...],
    hybrid_chunks: list[RetrievedChunk],
    hybrid_graph_chunks: list[RetrievedChunk],
) -> tuple[GraphRecoveryDetail, ...]:
    """Gold items satisfied somewhere in hybrid_graph's full list but not by
    hybrid's own top_k -- attributed to the specific chunk (and its source
    label) that satisfies each one.
    """
    recoveries = []
    for g in gold:
        if any(chunk_matches_gold(c, g) for c in hybrid_chunks):
            continue
        match = next((c for c in hybrid_graph_chunks if chunk_matches_gold(c, g)), None)
        if match is not None:
            recoveries.append(
                GraphRecoveryDetail(question_id=question_id, gold=g, source=match.source)
            )
    return tuple(recoveries)


def evaluate_question(
    conn: psycopg.Connection,
    repo_id: int,
    embedder: EmbeddingProvider,
    question: Question,
    top_k: int,
    graph_max_depth: int,
    graph_max_nodes: int,
) -> QuestionResult:
    q = question.question
    naive_chunks = get_strategy("naive").retrieve(conn, repo_id, q, embedder, top_k)
    hybrid_chunks = get_strategy("hybrid").retrieve(conn, repo_id, q, embedder, top_k)
    hg_strategy = get_strategy(
        "hybrid_graph", graph_max_depth=graph_max_depth, graph_max_nodes=graph_max_nodes
    )
    hybrid_graph_chunks = hg_strategy.retrieve(conn, repo_id, q, embedder, top_k)

    return QuestionResult(
        question=question,
        naive=score(naive_chunks, question.gold),
        hybrid=score(hybrid_chunks, question.gold),
        hybrid_graph_primary=score(hybrid_graph_chunks[:top_k], question.gold),
        hybrid_graph_full=score(hybrid_graph_chunks, question.gold),
        graph_recoveries=_graph_recoveries(
            question.id, question.gold, hybrid_chunks, hybrid_graph_chunks
        ),
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def print_per_question_table(results: list[QuestionResult]) -> None:
    # The aggregate mean can tie between strategies even when individual
    # questions differ in both directions -- printing every question's own
    # recall is what tells the honest story instead of one that could be
    # misread as "hybrid never changes anything."
    header = f"{'question':<24}{'naive':>8}{'hybrid':>8}{'+graph':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        n, h, g = r.naive.recall, r.hybrid.recall, r.hybrid_graph_full.recall
        print(f"{r.question.id:<24}{n:>8.2f}{h:>8.2f}{g:>8.2f}")


def print_report(results: list[QuestionResult]) -> None:
    print(f"\n{len(results)} questions evaluated\n")

    print_per_question_table(results)

    print()
    header = f"{'strategy':<24}{'precision':>10}{'recall':>10}{'avg results':>13}"
    print(header)
    print("-" * len(header))
    rows = [
        ("naive", [r.naive for r in results]),
        ("hybrid", [r.hybrid for r in results]),
        ("hybrid_graph (top_k only)", [r.hybrid_graph_primary for r in results]),
        ("hybrid_graph (full)", [r.hybrid_graph_full for r in results]),
    ]
    for name, scored in rows:
        precision = _mean([s.precision for s in scored])
        recall = _mean([s.recall for s in scored])
        avg_n = _mean([float(s.result_count) for s in scored])
        print(f"{name:<24}{precision:>10.2f}{recall:>10.2f}{avg_n:>13.1f}")

    print(
        "\nNote: hybrid_graph (top_k only) is identical to hybrid by "
        "construction -- HybridGraphStrategy's primary portion IS "
        "HybridStrategy's own fused top_k, unchanged. The full-list row is "
        "the number that reflects what graph expansion actually adds."
    )

    hybrid_recall = _mean([r.hybrid.recall for r in results])
    full_recall = _mean([r.hybrid_graph_full.recall for r in results])
    print(f"\nRecall lift from graph expansion: {full_recall - hybrid_recall:+.2f}")

    all_recoveries = [rec for r in results for rec in r.graph_recoveries]
    print(
        f"\nGold items recovered ONLY by graph expansion "
        f"(missed by hybrid's own top_k): {len(all_recoveries)}"
    )
    if all_recoveries:
        exact_graph = sum(1 for rec in all_recoveries if rec.source == "graph")
        print(
            f'  of which source == "graph" exactly (no other mechanism came '
            f"close): {exact_graph}/{len(all_recoveries)}"
        )
        for rec in all_recoveries:
            print(f"    [{rec.question_id}] {rec.gold}  source={rec.source}")


def main() -> None:
    settings = get_settings()
    conn = psycopg.connect(settings.dsn)
    register_vector(conn)
    try:
        embedder = build_embedder(
            settings.embedding_provider, settings.embedding_model, settings.embedding_dim,
            settings.embedding_batch_size, settings.embedding_api_key,
        )
        repo_id = ensure_flask_indexed(conn, embedder)
        questions = load_dataset()
        results = [
            evaluate_question(
                conn, repo_id, embedder, q, settings.retrieval_top_k,
                settings.graph_max_depth, settings.graph_max_nodes,
            )
            for q in questions
        ]
        print_report(results)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
