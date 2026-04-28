"""Call-graph extraction and resolution through the real indexing pipeline.

Two anchors: a small synthetic repo reproducing Flask's exact ambiguity
shape (a method name defined on multiple classes) so the same-class
heuristic is verified through the whole pipeline, not just the pure
resolver logic already covered in test_resolve.py; and the real Flask
fixture, since Phase 6's actual "done when" bar is that
Flask.full_dispatch_request -> Flask.dispatch_request exists as a real,
correctly resolved edge in the database.
"""

import os
from pathlib import Path

import psycopg
import pytest
from pgvector.psycopg import register_vector

from codeqa.indexing.embeddings import LocalEmbedder
from codeqa.indexing.pipeline import index_repo
from codeqa.indexing.store import register_repo

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


def edges_for(conn, repo_id, caller_qualified_name=None):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT caller.qualified_name, caller.symbol_name, e.callee_name,
                   e.resolution, callee.qualified_name
              FROM call_edges e
              JOIN chunks caller
                ON caller.id = e.caller_chunk_id AND caller.repo_id = e.repo_id
              LEFT JOIN chunks callee
                ON callee.id = e.callee_chunk_id AND callee.repo_id = e.repo_id
             WHERE e.repo_id = %s
            """,
            (repo_id,),
        )
        rows = cur.fetchall()
    if caller_qualified_name is not None:
        rows = [r for r in rows if r[0] == caller_qualified_name]
    return rows


@pytest.fixture
def indexed_repo(conn, embedder, tmp_path, request):
    slug = f"test-{request.node.name}".replace("[", "-").replace("]", "")[:60]
    drop_repo_by_slug(conn, slug)

    src = tmp_path / "src"
    src.mkdir()
    # Reproduces Flask's exact shape: dispatch_request defined on three
    # classes, and a caller in one of them that must resolve to its own
    # class's definition, not an arbitrary one of the three.
    (src / "app.py").write_text(
        "class Flask:\n"
        "    def dispatch_request(self):\n"
        "        return 1\n"
        "\n"
        "    def full_dispatch_request(self):\n"
        "        return self.dispatch_request()\n"
        "\n"
        "class View:\n"
        "    def dispatch_request(self):\n"
        "        return 2\n"
        "\n"
        "class MethodView:\n"
        "    def dispatch_request(self):\n"
        "        return 3\n"
        "\n"
        "def helper():\n"
        "    return unknown_external_call()\n"
    )

    repo_id = register_repo(
        conn, slug, "Test Repo", "local_path", str(src), "BAAI/bge-small-en-v1.5", EMBEDDING_DIM
    )
    index_repo(conn, repo_id, src, embedder)
    yield repo_id
    drop_repo_by_slug(conn, slug)


class TestSameClassAmbiguityThroughFullPipeline:
    def test_resolves_to_the_callers_own_class(self, conn, indexed_repo):
        rows = edges_for(conn, indexed_repo, "Flask.full_dispatch_request")
        dispatch_edge = next(r for r in rows if r[2] == "dispatch_request")
        _, _, _, resolution, callee_qualified = dispatch_edge
        assert resolution == "approximate"
        assert callee_qualified == "Flask.dispatch_request"

    def test_external_call_is_unresolved_not_guessed(self, conn, indexed_repo):
        rows = edges_for(conn, indexed_repo, None)
        external = next(r for r in rows if r[2] == "unknown_external_call")
        assert external[3] == "unresolved"
        assert external[4] is None

    def test_stats_report_the_split(self, conn, embedder, tmp_path):
        # Distinct from indexed_repo -- checks IndexStats directly, which
        # the CLI's summary line depends on.
        slug = "test-stats-split"
        drop_repo_by_slug(conn, slug)
        src = tmp_path / "s"
        src.mkdir()
        (src / "a.py").write_text(
            "def one():\n    return two()\ndef two():\n    return unknown_thing()\n"
        )
        repo_id = register_repo(
            conn, slug, "T", "local_path", str(src), "BAAI/bge-small-en-v1.5", EMBEDDING_DIM
        )
        try:
            stats = index_repo(conn, repo_id, src, embedder)
            assert stats.call_edges_exact == 1  # one() -> two(), globally unique
            assert stats.call_edges_unresolved == 1  # unknown_thing()
        finally:
            drop_repo_by_slug(conn, slug)

    def test_reindexing_does_not_duplicate_edges(self, conn, embedder, tmp_path):
        slug = "test-reindex-edges"
        drop_repo_by_slug(conn, slug)
        src = tmp_path / "s"
        src.mkdir()
        (src / "a.py").write_text("def one():\n    return two()\ndef two():\n    return 1\n")
        repo_id = register_repo(
            conn, slug, "T", "local_path", str(src), "BAAI/bge-small-en-v1.5", EMBEDDING_DIM
        )
        try:
            index_repo(conn, repo_id, src, embedder)
            index_repo(conn, repo_id, src, embedder)
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM call_edges WHERE repo_id = %s", (repo_id,))
                assert cur.fetchone()[0] == 1
        finally:
            drop_repo_by_slug(conn, slug)


class TestFlaskFixtureCallGraph:
    """Phase 6's actual 'done when' bar."""

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
        slug = "flask-fixture-graph"
        drop_repo_by_slug(class_conn, slug)
        repo_id = register_repo(
            class_conn, slug, "Flask", "local_path", str(FIXTURES / "flask"),
            "BAAI/bge-small-en-v1.5", EMBEDDING_DIM,
        )
        stats = index_repo(class_conn, repo_id, FIXTURES / "flask", embedder)
        yield repo_id, stats
        drop_repo_by_slug(class_conn, slug)

    def test_full_dispatch_request_calls_dispatch_request_resolved(self, class_conn, indexed):
        repo_id, _stats = indexed
        rows = edges_for(class_conn, repo_id, "Flask.full_dispatch_request")
        dispatch_edge = next(r for r in rows if r[2] == "dispatch_request")
        _, _, _, resolution, callee_qualified = dispatch_edge

        # Real Flask defines dispatch_request three times (Flask, View,
        # MethodView) -- getting this right requires the same-class
        # heuristic, not luck. "exact" would mean the ambiguity wasn't
        # actually present; "approximate" is the honest, correct label.
        assert resolution == "approximate"
        assert callee_qualified == "Flask.dispatch_request"

    def test_the_dynamic_view_dispatch_is_not_captured(self, class_conn, indexed):
        # Empirically verified during design (see docs/deep-dive.html):
        # dispatch_request invokes the actual view via
        # self.ensure_sync(self.view_functions[rule.endpoint])(**view_args)
        # -- a call on the RETURN VALUE of another call, not a named symbol.
        # tree-sitter's reference.call query structurally cannot capture
        # this. Asserting its absence turns a known limitation into a
        # pinned, honest fact rather than a silent gap.
        repo_id, _stats = indexed
        rows = edges_for(class_conn, repo_id, "Flask.dispatch_request")
        callee_names = {r[2] for r in rows}
        assert "ensure_sync" in callee_names  # the inner call IS captured
        # No edge represents "the view function itself" -- there is no
        # symbolic name for it to be captured under.
        assert all(name != "view" for name in callee_names)

    def test_builtin_calls_are_unresolved_not_guessed(self, class_conn, indexed):
        repo_id, _stats = indexed
        rows = edges_for(class_conn, repo_id, None)
        getattr_calls = [r for r in rows if r[2] == "getattr"]
        assert getattr_calls
        assert all(r[3] == "unresolved" and r[4] is None for r in getattr_calls)

    def test_most_edges_resolve_given_a_real_codebase(self, class_conn, indexed):
        repo_id, stats = indexed
        # Not a tight bound -- just confirms resolution is doing real work,
        # not silently producing all-unresolved or (suspiciously) all-exact.
        assert stats.call_edges_exact > 0
        assert stats.call_edges_approximate > 0
        assert stats.call_edges_unresolved > 0
