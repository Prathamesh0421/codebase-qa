"""HybridStrategy and HybridGraphStrategy against real Postgres.

Small purpose-built repo, same reasoning as test_naive_retrieval.py: a
handful of semantically, lexically, and structurally distinct functions
makes it possible to assert exactly which mechanism found which chunk,
which the 446-chunk Flask fixture makes much harder to state precisely.

The fixture is deliberately shaped so each retrieval component has
something ONLY it can find:
  - vector:  "how is a user's password verified" has no shared vocabulary
             with check_password_hash's name, only its docstring meaning.
  - lexical: "rate limit" phrase-matches check_rate_limit's underscore-split
             tokens without being an exact identifier.
  - symbol:  "check_password_hash" and "SessionManager.issue_session" are
             exact identifier/qualified-name matches, bare and dotted.

graph is deliberately NOT tested via "a query only graph can answer" --
tried that first, and it doesn't hold up: a direct caller's source text
necessarily contains the callee's name as a literal call expression (e.g.
process_incoming_login_form's body contains the text "check_password_hash"),
so Postgres's full-text search over raw source code legitimately finds
one-hop callers too, via ordinary lexical match, not just via the graph.
What graph expansion actually adds, and what's tested below instead, is
context beyond a size-bounded primary retrieval's cutoff: at top_k=1, a
plain hybrid query returns only the one best-ranked chunk, while
hybrid_graph returns that same chunk plus its call-graph neighborhood,
regardless of whether those neighbors would also have ranked well enough
to survive top_k on their own.
"""

import os
from pathlib import Path

import psycopg
import pytest
from pgvector.psycopg import register_vector

from codeqa.indexing.embeddings import LocalEmbedder
from codeqa.indexing.pipeline import index_repo
from codeqa.indexing.store import register_repo
from codeqa.retrieval.hybrid import HybridGraphStrategy, HybridStrategy
from codeqa.retrieval.strategy import get_strategy

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).parent.parent / "fixtures" / "repos"
EMBEDDING_DIM = 384


@pytest.fixture(scope="module")
def embedder():
    return LocalEmbedder("BAAI/bge-small-en-v1.5", dimension=EMBEDDING_DIM, batch_size=16)


@pytest.fixture
def conn():
    dsn = os.environ.get("CODEQA_TEST_DSN", "postgresql://codeqa:codeqa@localhost:5432/codeqa")
    connection = psycopg.connect(dsn)
    register_vector(connection)
    yield connection
    connection.rollback()
    connection.close()


def drop_repo_by_slug(conn, slug: str) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM repos WHERE slug = %s", (slug,))
        row = cur.fetchone()
        if row is not None:
            cur.execute("SELECT drop_repo_partition(%s)", (row[0],))
            cur.execute("DELETE FROM repos WHERE id = %s", (row[0],))
    conn.commit()


@pytest.fixture
def indexed_repo(conn, embedder, tmp_path, request):
    slug = f"test-{request.node.name}".replace("[", "-").replace("]", "")[:60]
    drop_repo_by_slug(conn, slug)

    src = tmp_path / "src"
    src.mkdir()
    (src / "auth.py").write_text(
        "def hash_password(password):\n"
        "    '''Turn a plaintext password into a stored hash.'''\n"
        "    return _digest(password)\n"
        "\n"
        "\n"
        "def check_password_hash(pw_hash, password):\n"
        "    '''Confirm a plaintext password matches a previously stored hash.'''\n"
        "    return pw_hash == hash_password(password)\n"
    )
    (src / "login.py").write_text(
        "from auth import check_password_hash\n"
        "\n"
        "\n"
        "def process_incoming_login_form(username, password):\n"
        "    '''Handle a submitted HTML login form end to end.'''\n"
        "    record = lookup_account(username)\n"
        "    if not check_password_hash(record.pw_hash, password):\n"
        "        raise PermissionError('bad credentials')\n"
        "    return record\n"
    )
    (src / "rate_limit.py").write_text(
        "def check_rate_limit(key, redis_client, max_per_minute):\n"
        "    '''Enforce a per-key token bucket rate limit using Redis.'''\n"
        "    count = redis_client.incr(key)\n"
        "    return count <= max_per_minute\n"
    )
    (src / "session.py").write_text(
        "class SessionManager:\n"
        "    def issue_session(self, user):\n"
        "        '''Create and persist a new session for an authenticated user.'''\n"
        "        return _new_token(user)\n"
    )

    repo_id = register_repo(
        conn, slug, "Test Repo", "local_path", str(src), "BAAI/bge-small-en-v1.5", EMBEDDING_DIM
    )
    index_repo(conn, repo_id, src, embedder)
    yield repo_id
    drop_repo_by_slug(conn, slug)


class TestHybridStrategy:
    def test_semantically_relevant_chunk_ranks_first(self, conn, embedder, indexed_repo):
        results = HybridStrategy().retrieve(
            conn, indexed_repo, "how is a user's password verified", embedder, top_k=5
        )
        assert results[0].symbol_name == "check_password_hash"

    def test_exact_bare_symbol_match_is_found_and_labeled(self, conn, embedder, indexed_repo):
        results = HybridStrategy().retrieve(
            conn, indexed_repo, "check_password_hash", embedder, top_k=5
        )
        match = next(r for r in results if r.symbol_name == "check_password_hash")
        assert "symbol" in match.source.split("+")

    def test_dotted_qualified_name_match_is_found_and_labeled(self, conn, embedder, indexed_repo):
        # This is the concrete case that lexical search alone cannot find --
        # verified interactively in this phase that websearch_to_tsquery
        # treats "SessionManager.issue_session" as one compound token that
        # never matches the tsvector's separately-positioned tokens.
        results = HybridStrategy().retrieve(
            conn, indexed_repo, "SessionManager.issue_session", embedder, top_k=5
        )
        match = next(r for r in results if r.qualified_name == "SessionManager.issue_session")
        assert "symbol" in match.source.split("+")

    def test_lexical_phrase_match_finds_underscore_split_identifier(
        self, conn, embedder, indexed_repo
    ):
        results = HybridStrategy().retrieve(conn, indexed_repo, "rate limit", embedder, top_k=5)
        assert any(r.symbol_name == "check_rate_limit" for r in results)

    def test_respects_top_k(self, conn, embedder, indexed_repo):
        results = HybridStrategy().retrieve(conn, indexed_repo, "anything", embedder, top_k=2)
        assert len(results) == 2

    def test_scores_are_descending(self, conn, embedder, indexed_repo):
        results = HybridStrategy().retrieve(conn, indexed_repo, "password", embedder, top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_repo_id_scoping_is_not_bypassable(self, conn, embedder, indexed_repo, tmp_path):
        other_src = tmp_path / "other"
        other_src.mkdir()
        (other_src / "unique.py").write_text(
            "def zzz_completely_unrelated_marker_function():\n"
            "    '''This function must never appear in another repo's results.'''\n"
            "    pass\n"
        )
        other_repo_id = register_repo(
            conn, "test-other-repo-scoping-hybrid", "Other", "local_path", str(other_src),
            "BAAI/bge-small-en-v1.5", EMBEDDING_DIM,
        )
        try:
            index_repo(conn, other_repo_id, other_src, embedder)
            results = HybridStrategy().retrieve(
                conn, indexed_repo, "zzz_completely_unrelated_marker_function", embedder, top_k=10
            )
            assert all(r.symbol_name != "zzz_completely_unrelated_marker_function" for r in results)
        finally:
            drop_repo_by_slug(conn, "test-other-repo-scoping-hybrid")


class TestHybridGraphStrategy:
    def test_expands_beyond_a_size_bounded_primary_result(self, conn, embedder, indexed_repo):
        # At top_k=1, plain hybrid retrieval returns exactly the one
        # best-ranked chunk -- check_password_hash. hybrid_graph, given the
        # same top_k=1, still keeps that single primary result but adds its
        # call-graph neighbors (caller process_incoming_login_form, callee
        # hash_password) on top -- each labeled with "graph" as one of its
        # source components, though not necessarily the ONLY one: this
        # fixture repo is far smaller than a component's default 20-chunk
        # candidate pool, so a neighbor commonly sits in some component's
        # pool too (e.g. process_incoming_login_form's body literally
        # contains the text "check_password_hash", so lexical search finds
        # it as well) without that having made it into the fused top_k=1.
        # That's the exact scenario "graph" as a source label exists to
        # distinguish from "graph" alone -- see strategy.py's
        # RetrievedChunk.source docstring.
        query = "how is a user's password verified"

        primary_only = HybridStrategy().retrieve(conn, indexed_repo, query, embedder, top_k=1)
        assert [r.symbol_name for r in primary_only] == ["check_password_hash"]

        strategy = HybridGraphStrategy(graph_max_depth=2, graph_max_nodes=40)
        results = strategy.retrieve(conn, indexed_repo, query, embedder, top_k=1)
        by_symbol = {r.symbol_name: r for r in results}

        assert "graph" not in by_symbol["check_password_hash"].source.split("+")  # primary result
        assert "graph" in by_symbol["process_incoming_login_form"].source.split("+")  # caller
        assert "graph" in by_symbol["hash_password"].source.split("+")  # callee

    def test_graph_expanded_chunks_never_duplicate_a_primary_result(
        self, conn, embedder, indexed_repo
    ):
        strategy = HybridGraphStrategy(graph_max_depth=2, graph_max_nodes=40)
        results = strategy.retrieve(
            conn, indexed_repo, "how is a user's password verified", embedder, top_k=3
        )
        ids = [r.chunk_id for r in results]
        assert len(ids) == len(set(ids))

    def test_graph_max_nodes_zero_yields_no_expansion(self, conn, embedder, indexed_repo):
        strategy = HybridGraphStrategy(graph_max_depth=2, graph_max_nodes=0)
        query = "how is a user's password verified"
        primary_only = HybridStrategy().retrieve(conn, indexed_repo, query, embedder, top_k=3)
        results = strategy.retrieve(conn, indexed_repo, query, embedder, top_k=3)
        assert [r.chunk_id for r in results] == [r.chunk_id for r in primary_only]

    def test_repo_id_scoping_is_not_bypassable(self, conn, embedder, indexed_repo, tmp_path):
        other_src = tmp_path / "other"
        other_src.mkdir()
        (other_src / "unique.py").write_text(
            "def zzz_completely_unrelated_marker_function():\n"
            "    '''This function must never appear in another repo's results.'''\n"
            "    pass\n"
        )
        other_repo_id = register_repo(
            conn, "test-other-repo-scoping-hybrid-graph", "Other", "local_path", str(other_src),
            "BAAI/bge-small-en-v1.5", EMBEDDING_DIM,
        )
        try:
            index_repo(conn, other_repo_id, other_src, embedder)
            strategy = HybridGraphStrategy(graph_max_depth=2, graph_max_nodes=40)
            results = strategy.retrieve(
                conn, indexed_repo, "how is a user's password verified", embedder, top_k=10
            )
            assert all(r.symbol_name != "zzz_completely_unrelated_marker_function" for r in results)
        finally:
            drop_repo_by_slug(conn, "test-other-repo-scoping-hybrid-graph")


class TestAllThreeStrategiesViaConfig:
    def test_naive_hybrid_and_hybrid_graph_all_run_against_the_same_query(
        self, conn, embedder, indexed_repo
    ):
        # The project's central comparison (Phase 9) requires all three
        # strategies to be runnable, via the same get_strategy(name) config
        # path, against one query -- this is that smoke test, exercised
        # through the real factory rather than by constructing classes
        # directly like the tests above do.
        query = "how is a user's password verified"
        for name in ("naive", "hybrid", "hybrid_graph"):
            strategy = get_strategy(name, graph_max_depth=2, graph_max_nodes=40)
            results = strategy.retrieve(conn, indexed_repo, query, embedder, top_k=5)
            assert len(results) > 0


class TestFlaskFixtureHybridRetrieval:
    """The real anchor repo, not a hand-built one -- catches things a small
    fixture can't, which is exactly what happened while building this
    phase: hybrid ranked the ENTIRE Flask class ahead of the actual answer
    (Flask.dispatch_request) because "Flask" is both an ordinary-looking
    capitalized word and a real symbol_name, something no repo small enough
    to hand-write would ever surface. See fusion.py's
    filter_symbol_candidates for the fix.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def class_conn():
        dsn = os.environ.get(
            "CODEQA_TEST_DSN", "postgresql://codeqa:codeqa@localhost:5432/codeqa"
        )
        connection = psycopg.connect(dsn)
        register_vector(connection)
        yield connection
        connection.rollback()
        connection.close()

    @pytest.fixture(scope="class")
    @staticmethod
    def indexed(class_conn, embedder):
        slug = "flask-fixture-hybrid"
        drop_repo_by_slug(class_conn, slug)
        repo_id = register_repo(
            class_conn, slug, "Flask", "local_path", str(FIXTURES / "flask"),
            "BAAI/bge-small-en-v1.5", EMBEDDING_DIM,
        )
        index_repo(class_conn, repo_id, FIXTURES / "flask", embedder)
        yield repo_id
        drop_repo_by_slug(class_conn, slug)

    def test_the_giant_flask_class_chunk_no_longer_wins_via_bare_symbol_match(
        self, class_conn, embedder, indexed
    ):
        # Regression test for the exact bug found while building this
        # phase: "Flask" is a bare token in this query AND a real
        # symbol_name (the whole 1500+ line Flask class), so an unfiltered
        # symbol component let that giant, barely-relevant chunk win an
        # exact match and outrank the real answer in RRF fusion.
        results = HybridStrategy().retrieve(
            class_conn, indexed, "how does Flask dispatch a request to a view function?",
            embedder, top_k=5,
        )
        assert all("symbol" not in r.source.split("+") for r in results if r.symbol_name == "Flask")

    def test_graph_expansion_recovers_the_documented_multi_hop_chain(
        self, class_conn, embedder, indexed
    ):
        # The project's own reason for existing: full_dispatch_request ->
        # dispatch_request is the multi-hop chain naive RAG misses and
        # call-graph expansion should recover (see fixtures README and
        # Phase 6's "done when" bar). top_k=10 -- Flask.full_dispatch_request
        # (the caller) DOES survive into hybrid's own top_k at this size
        # (verified interactively), but Flask.dispatch_request (the callee,
        # and the actual answer) does not: three same-named
        # dispatch_request methods exist on Flask/View/MethodView, and
        # View's and MethodView's outrank Flask's on raw vector similarity.
        # Only graph expansion from full_dispatch_request recovers it.
        query = "how does Flask dispatch a request to a view function?"
        strategy = HybridGraphStrategy(graph_max_depth=2, graph_max_nodes=40)
        results = strategy.retrieve(class_conn, indexed, query, embedder, top_k=10)

        dispatch_request = [
            r for r in results
            if r.qualified_name == "Flask.dispatch_request" and "graph" in r.source.split("+")
        ]
        assert len(dispatch_request) == 1
