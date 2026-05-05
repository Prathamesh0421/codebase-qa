"""Pure precision/recall scoring against a hand-labeled gold set. No database
access here -- same separation as graph/resolve.py's _resolve_one and
chunker.py's Tags-to-Chunks core, so the scoring logic is testable without a
live Postgres connection or a real retrieval strategy.
"""

from dataclasses import dataclass

from codeqa.retrieval.strategy import RetrievedChunk


@dataclass(frozen=True)
class GoldItem:
    # file is a suffix match (endswith), not exact-path equality: the
    # labeled set is written against the repo's relative layout
    # ("app.py"), while a retrieved chunk's file_path may carry whatever
    # prefix the indexed repo root happened to have.
    file: str
    # Matched against a chunk's qualified_name when it has one (a method,
    # disambiguated from same-named methods on other classes), else its
    # bare symbol_name (a module-level function or a class).
    symbol: str

    def __str__(self) -> str:
        return f"{self.file}:{self.symbol}"


def chunk_matches_gold(chunk: RetrievedChunk, gold: GoldItem) -> bool:
    if not chunk.file_path.endswith(gold.file):
        return False
    name = chunk.qualified_name or chunk.symbol_name
    return name == gold.symbol


def matched_gold(chunks: list[RetrievedChunk], gold: tuple[GoldItem, ...]) -> set[GoldItem]:
    """Every gold item satisfied by at least one chunk in the retrieved list."""
    return {g for g in gold if any(chunk_matches_gold(c, g) for c in chunks)}


@dataclass(frozen=True)
class PrecisionRecall:
    precision: float
    recall: float
    result_count: int
    matched_count: int  # relevant chunks retrieved, i.e. |gold ∩ retrieved|
    gold_count: int


def score(chunks: list[RetrievedChunk], gold: tuple[GoldItem, ...]) -> PrecisionRecall:
    """precision = relevant retrieved / total retrieved.
    recall = relevant retrieved / total relevant (the gold set size).

    An empty gold set makes recall undefined (0/0) -- every labeled question
    in this project's dataset has at least one gold item, so this isn't
    special-cased; it would be a labeling bug, not a runtime case to handle
    quietly.
    """
    matched = matched_gold(chunks, gold)
    relevant_retrieved = sum(1 for c in chunks if any(chunk_matches_gold(c, g) for g in gold))
    precision = relevant_retrieved / len(chunks) if chunks else 0.0
    recall = len(matched) / len(gold)
    return PrecisionRecall(
        precision=precision,
        recall=recall,
        result_count=len(chunks),
        matched_count=relevant_retrieved,
        gold_count=len(gold),
    )
