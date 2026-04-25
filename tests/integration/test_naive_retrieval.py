"""NaiveStrategy against real Postgres, with a small purpose-built repo.

This is deliberately not run against the full Flask fixture -- a handful of
distinct, semantically separated functions makes it possible to assert
"the right chunk actually ranked first," which a 446-chunk real repo makes
much harder to state precisely. The Flask fixture is exercised end to end
in test_ask_cli.py instead, where exact ranking doesn't need to be asserted.
"""

import os

import psycopg
import pytest
from pgvector.psycopg import register_vector

from codeqa.indexing.embeddings import LocalEmbedder
from codeqa.indexing.pipeline import index_repo
from codeqa.indexing.store import register_repo
from codeqa.retrieval.naive import NaiveStrategy
from codeqa.retrieval.strategy import get_strategy

pytestmark = pytest.mark.integration

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
        "def check_password_hash(pw_hash, password):\n"
        "    '''Verify a plaintext password against a stored hash.'''\n"
        "    return pw_hash == hash_password(password)\n"
    )
    (src / "math_utils.py").write_text(
        "def fibonacci(n):\n"
        "    '''Compute the nth Fibonacci number recursively.'''\n"
        "    if n < 2:\n"
        "        return n\n"
        "    return fibonacci(n - 1) + fibonacci(n - 2)\n"
    )
    (src / "rate_limit.py").write_text(
        "def check_rate_limit(key, redis_client, max_per_minute):\n"
        "    '''Enforce a per-key token bucket rate limit using Redis.'''\n"
        "    count = redis_client.incr(key)\n"
        "    return count <= max_per_minute\n"
    )

    repo_id = register_repo(
        conn, slug, "Test Repo", "local_path", str(src), "BAAI/bge-small-en-v1.5", EMBEDDING_DIM
    )
    index_repo(conn, repo_id, src, embedder)
    yield repo_id
    drop_repo_by_slug(conn, slug)


class TestNaiveStrategy:
    def test_semantically_relevant_chunk_ranks_first(self, conn, embedder, indexed_repo):
        results = NaiveStrategy().retrieve(
            conn, indexed_repo, "how is a user's password verified", embedder, top_k=3
        )
        assert results[0].symbol_name == "check_password_hash"

    def test_a_different_query_ranks_a_different_chunk_first(self, conn, embedder, indexed_repo):
        results = NaiveStrategy().retrieve(
            conn, indexed_repo, "limiting how many requests a client can make per minute",
            embedder, top_k=3,
        )
        assert results[0].symbol_name == "check_rate_limit"

    def test_respects_top_k(self, conn, embedder, indexed_repo):
        results = NaiveStrategy().retrieve(conn, indexed_repo, "anything", embedder, top_k=2)
        assert len(results) == 2

    def test_scores_are_descending(self, conn, embedder, indexed_repo):
        results = NaiveStrategy().retrieve(conn, indexed_repo, "recursion", embedder, top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_citation_and_qualified_name_are_populated(self, conn, embedder, indexed_repo):
        results = NaiveStrategy().retrieve(conn, indexed_repo, "fibonacci", embedder, top_k=1)
        r = results[0]
        assert r.citation.startswith("math_utils.py:")
        assert r.kind == "function"

    def test_repo_id_scoping_is_not_bypassable(self, conn, embedder, indexed_repo, tmp_path):
        # A second, unrelated repo with an unambiguous, distinctly-named
        # function. Querying the FIRST repo for something semantically
        # close to the second repo's content must never surface it --
        # this is the invariant every retrieval path in the project claims
        # to uphold, checked here at the one strategy that currently exists.
        other_src = tmp_path / "other"
        other_src.mkdir()
        (other_src / "unique.py").write_text(
            "def zzz_completely_unrelated_marker_function():\n"
            "    '''This function must never appear in another repo's results.'''\n"
            "    pass\n"
        )
        other_repo_id = register_repo(
            conn, "test-other-repo-scoping", "Other", "local_path", str(other_src),
            "BAAI/bge-small-en-v1.5", EMBEDDING_DIM,
        )
        try:
            index_repo(conn, other_repo_id, other_src, embedder)

            results = NaiveStrategy().retrieve(
                conn, indexed_repo, "completely unrelated marker function", embedder, top_k=10
            )
            assert all(r.symbol_name != "zzz_completely_unrelated_marker_function" for r in results)
        finally:
            drop_repo_by_slug(conn, "test-other-repo-scoping")


class TestGetStrategy:
    def test_naive_returns_naive_strategy(self):
        assert isinstance(get_strategy("naive"), NaiveStrategy)

    def test_hybrid_not_yet_implemented(self):
        with pytest.raises(NotImplementedError, match="Phase 8"):
            get_strategy("hybrid")

    def test_hybrid_graph_not_yet_implemented(self):
        with pytest.raises(NotImplementedError, match="Phase 8"):
            get_strategy("hybrid_graph")

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="unknown retrieval strategy"):
            get_strategy("made-up-strategy")
