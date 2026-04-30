"""Traversal against real Postgres: the differential test at the center of
Phase 7.

Chunk and call_edges rows are inserted directly rather than through the
indexing pipeline -- traversal doesn't care how an edge got there, only that
it exists, and hand-crafted topology gives exact control over cycles, fan-out
and depth without fighting tree-sitter to produce a specific shape indirectly.
Phase 6's tests already cover "does real code produce correct edges"; these
cover "given edges, does traversal behave correctly."
"""

import os
from pathlib import Path

import psycopg
import pytest
from pgvector.psycopg import register_vector

from codeqa.graph.traversal import traverse_networkx, traverse_sql
from codeqa.indexing.embeddings import LocalEmbedder
from codeqa.indexing.pipeline import index_repo
from codeqa.indexing.store import register_repo

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).parent.parent / "fixtures" / "repos"
EMBEDDING_DIM = 384


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


def make_chunks(conn, repo_id: int, names: list[str]) -> dict[str, int]:
    """One placeholder file + one trivial chunk per name. Returns name -> chunk_id."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO files (repo_id, path, language, tier, blob_sha, size_bytes) "
            "VALUES (%s, 'f.py', 'python', 'tier1', 'deadbeef', 1) RETURNING id",
            (repo_id,),
        )
        file_id = cur.fetchone()[0]

        ids = {}
        for i, name in enumerate(names):
            cur.execute(
                """
                INSERT INTO chunks
                    (repo_id, file_id, kind, symbol_name, start_line, end_line,
                     content, content_sha)
                VALUES (%s, %s, 'function', %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (repo_id, file_id, name, i + 1, i + 1, f"def {name}(): pass", f"sha-{name}"),
            )
            ids[name] = cur.fetchone()[0]
    conn.commit()
    return ids


def make_edges(conn, repo_id: int, ids: dict[str, int], pairs: list[tuple[str, str]]) -> None:
    with conn.cursor() as cur:
        for caller, callee in pairs:
            cur.execute(
                """
                INSERT INTO call_edges
                    (repo_id, caller_chunk_id, callee_name, callee_chunk_id, resolution)
                VALUES (%s, %s, %s, %s, 'exact')
                """,
                (repo_id, ids[caller], callee, ids[callee]),
            )
    conn.commit()


@pytest.fixture
def repo(conn, request):
    slug = f"test-{request.node.name}".replace("[", "-").replace("]", "")[:60]
    drop_repo_by_slug(conn, slug)
    repo_id = register_repo(
        conn, slug, "T", "local_path", "/tmp/x", "BAAI/bge-small-en-v1.5", EMBEDDING_DIM
    )
    yield repo_id
    drop_repo_by_slug(conn, slug)


class TestSqlTraversalCycleTermination:
    """The literal Phase 7 'done when' bar, against the real database."""

    def test_terminates_and_returns_correct_nodes(self, conn, repo):
        ids = make_chunks(conn, repo, ["a", "b", "c", "d"])
        make_edges(conn, repo, ids, [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")])

        result = traverse_sql(conn, repo, ids["a"], "callees", max_depth=10, max_nodes=100)
        by_id = {n.chunk_id: n.depth for n in result}

        assert by_id == {ids["b"]: 1, ids["c"]: 2, ids["d"]: 3}
        assert ids["a"] not in by_id  # start excluded, even though the cycle revisits it


class TestDifferentialAgreement:
    """SQL and networkx are independent implementations reading the same
    edges. Agreement here is real evidence, not two paths through one bug."""

    def _assert_agree(self, conn, repo, start_id, direction, max_depth=10, max_nodes=100):
        sql_result = traverse_sql(conn, repo, start_id, direction, max_depth, max_nodes)
        nx_result = traverse_networkx(conn, repo, start_id, direction, max_depth, max_nodes)
        assert sql_result == nx_result
        return sql_result

    def test_agree_on_a_cyclic_graph_callees(self, conn, repo):
        ids = make_chunks(conn, repo, ["a", "b", "c", "d"])
        make_edges(conn, repo, ids, [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")])
        result = self._assert_agree(conn, repo, ids["a"], "callees")
        assert len(result) == 3

    def test_agree_on_a_cyclic_graph_callers(self, conn, repo):
        ids = make_chunks(conn, repo, ["a", "b", "c", "d"])
        make_edges(conn, repo, ids, [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")])
        result = self._assert_agree(conn, repo, ids["d"], "callers")
        assert {n.chunk_id for n in result} == {ids["c"], ids["b"], ids["a"]}

    def test_agree_on_diamond_shaped_graph(self, conn, repo):
        # a -> b -> d and a -> c -> d: d is reachable by two paths, must
        # appear exactly once, at its shortest depth.
        ids = make_chunks(conn, repo, ["a", "b", "c", "d"])
        make_edges(conn, repo, ids, [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
        result = self._assert_agree(conn, repo, ids["a"], "callees")
        assert {n.chunk_id: n.depth for n in result} == {ids["b"]: 1, ids["c"]: 1, ids["d"]: 2}

    def test_agree_when_nothing_is_reachable(self, conn, repo):
        ids = make_chunks(conn, repo, ["lonely"])
        result = self._assert_agree(conn, repo, ids["lonely"], "callees")
        assert result == []

    def test_agree_respecting_depth_bound(self, conn, repo):
        ids = make_chunks(conn, repo, ["a", "b", "c", "d", "e"])
        make_edges(conn, repo, ids, [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")])
        result = self._assert_agree(conn, repo, ids["a"], "callees", max_depth=2)
        assert {n.chunk_id for n in result} == {ids["b"], ids["c"]}

    def test_agree_with_unresolved_edges_present(self, conn, repo):
        # An unresolved edge (callee_chunk_id NULL) must not appear as a
        # traversable node in either implementation.
        ids = make_chunks(conn, repo, ["a", "b"])
        make_edges(conn, repo, ids, [("a", "b")])
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO call_edges "
                "(repo_id, caller_chunk_id, callee_name, callee_chunk_id, resolution) "
                "VALUES (%s, %s, 'external_lib_fn', NULL, 'unresolved')",
                (repo, ids["a"]),
            )
        conn.commit()
        result = self._assert_agree(conn, repo, ids["a"], "callees")
        assert {n.chunk_id for n in result} == {ids["b"]}


class TestRepoScoping:
    def test_traversal_never_crosses_into_another_repos_edges(self, conn):
        slug_a, slug_b = "test-scope-a", "test-scope-b"
        drop_repo_by_slug(conn, slug_a)
        drop_repo_by_slug(conn, slug_b)
        repo_a = register_repo(
            conn, slug_a, "A", "local_path", "/tmp/a", "BAAI/bge-small-en-v1.5", EMBEDDING_DIM
        )
        repo_b = register_repo(
            conn, slug_b, "B", "local_path", "/tmp/b", "BAAI/bge-small-en-v1.5", EMBEDDING_DIM
        )
        try:
            ids_a = make_chunks(conn, repo_a, ["a1", "a2"])
            make_edges(conn, repo_a, ids_a, [("a1", "a2")])
            ids_b = make_chunks(conn, repo_b, ["b1", "b2"])
            make_edges(conn, repo_b, ids_b, [("b1", "b2")])

            result = traverse_sql(conn, repo_a, ids_a["a1"], "callees", 10, 100)
            assert {n.chunk_id for n in result} == {ids_a["a2"]}
            assert ids_b["b2"] not in {n.chunk_id for n in result}
        finally:
            drop_repo_by_slug(conn, slug_a)
            drop_repo_by_slug(conn, slug_b)


class TestMaxNodesBound:
    """Both implementations now compute the FULL correct reachable set (no
    SQL-level LIMIT -- see graph/traversal.py's comment on why DISTINCT ON
    made that necessary) and truncate identically in Python: sort by
    (depth, chunk_id), then slice to max_nodes. That means they're expected
    to agree exactly even when the true reachable set exceeds max_nodes,
    not just below it -- verified directly, not assumed from the shared
    truncation code being "probably" equivalent."""

    def test_sql_result_never_exceeds_max_nodes(self, conn, repo):
        names = ["hub"] + [f"leaf{i}" for i in range(20)]
        ids = make_chunks(conn, repo, names)
        make_edges(conn, repo, ids, [("hub", leaf) for leaf in names[1:]])

        result = traverse_sql(conn, repo, ids["hub"], "callees", max_depth=5, max_nodes=5)
        assert len(result) == 5

    def test_networkx_result_never_exceeds_max_nodes(self, conn, repo):
        names = ["hub"] + [f"leaf{i}" for i in range(20)]
        ids = make_chunks(conn, repo, names)
        make_edges(conn, repo, ids, [("hub", leaf) for leaf in names[1:]])

        result = traverse_networkx(conn, repo, ids["hub"], "callees", max_depth=5, max_nodes=5)
        assert len(result) == 5

    def test_below_the_bound_both_agree_exactly(self, conn, repo):
        names = ["hub"] + [f"leaf{i}" for i in range(3)]
        ids = make_chunks(conn, repo, names)
        make_edges(conn, repo, ids, [("hub", leaf) for leaf in names[1:]])

        sql_result = traverse_sql(conn, repo, ids["hub"], "callees", max_depth=5, max_nodes=100)
        nx_result = traverse_networkx(conn, repo, ids["hub"], "callees", max_depth=5, max_nodes=100)
        assert sql_result == nx_result
        assert len(sql_result) == 3

    def test_agree_exactly_even_when_truncation_actually_occurs(self, conn, repo):
        names = ["hub"] + [f"leaf{i}" for i in range(20)]
        ids = make_chunks(conn, repo, names)
        make_edges(conn, repo, ids, [("hub", leaf) for leaf in names[1:]])

        sql_result = traverse_sql(conn, repo, ids["hub"], "callees", max_depth=5, max_nodes=5)
        nx_result = traverse_networkx(conn, repo, ids["hub"], "callees", max_depth=5, max_nodes=5)
        assert sql_result == nx_result
        assert len(sql_result) == 5


class TestFlaskFixtureDifferential:
    """Agreement on real production data, not just hand-crafted fixtures --
    the synthetic tests above prove the algorithm is right; this proves it
    stays right against the shape of edges a real codebase actually
    produces (Phase 6's resolution heuristics, real fan-out, real depth)."""

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
    def indexed(class_conn):
        slug = "flask-fixture-traversal"
        drop_repo_by_slug(class_conn, slug)
        embedder = LocalEmbedder("BAAI/bge-small-en-v1.5", dimension=EMBEDDING_DIM, batch_size=16)
        repo_id = register_repo(
            class_conn, slug, "Flask", "local_path", str(FIXTURES / "flask"),
            "BAAI/bge-small-en-v1.5", EMBEDDING_DIM,
        )
        index_repo(class_conn, repo_id, FIXTURES / "flask", embedder)
        yield repo_id
        drop_repo_by_slug(class_conn, slug)

    @pytest.mark.parametrize("direction", ["callers", "callees"])
    def test_agree_from_a_real_ambiguous_method(self, class_conn, indexed, direction):
        # Flask.dispatch_request specifically: the chunk whose callers edge
        # (from full_dispatch_request) required the same-class resolution
        # heuristic in Phase 6. Real resolved/unresolved edges, real fan-out.
        repo_id = indexed
        with class_conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM chunks WHERE repo_id = %s AND qualified_name = %s",
                (repo_id, "Flask.dispatch_request"),
            )
            start_id = cur.fetchone()[0]

        sql_result = traverse_sql(
            class_conn, repo_id, start_id, direction, max_depth=3, max_nodes=200
        )
        nx_result = traverse_networkx(
            class_conn, repo_id, start_id, direction, max_depth=3, max_nodes=200
        )
        assert sql_result == nx_result
        assert len(sql_result) > 0
