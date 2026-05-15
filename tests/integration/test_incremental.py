"""incremental_index_repo against real Postgres. The single most important
assertion in this file is that an UNCHANGED chunk's id survives an
incremental run byte-for-byte -- that's the literal claim the whole module
exists to make true, not just "the counts look about right".
"""

import os

import psycopg
import pytest
from pgvector.psycopg import register_vector

from codeqa.indexing.embeddings import LocalEmbedder
from codeqa.indexing.incremental import incremental_index_repo
from codeqa.indexing.pipeline import index_repo
from codeqa.indexing.store import register_repo

pytestmark = pytest.mark.integration

EMBEDDING_DIM = 384


def _dsn() -> str:
    return os.environ.get("CODEQA_TEST_DSN", "postgresql://codeqa:codeqa@localhost:5432/codeqa")


@pytest.fixture(scope="module")
def embedder():
    return LocalEmbedder("BAAI/bge-small-en-v1.5", dimension=EMBEDDING_DIM, batch_size=16)


@pytest.fixture
def conn():
    connection = psycopg.connect(_dsn())
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


def _chunk_rows(conn, repo_id: int) -> dict[tuple[str, str], int]:
    """(file path, symbol_name) -> chunk_id, for asserting id stability."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.path, c.symbol_name, c.id
              FROM chunks c JOIN files f ON f.id = c.file_id AND f.repo_id = c.repo_id
             WHERE c.repo_id = %s
            """,
            (repo_id,),
        )
        return {(path, name): cid for path, name, cid in cur.fetchall()}


@pytest.fixture
def repo(conn, embedder, tmp_path, request):
    slug = f"test-{request.node.name}".replace("[", "-").replace("]", "")[:60]
    drop_repo_by_slug(conn, slug)

    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text(
        "def helper():\n    return 1\n\n\ndef unrelated():\n    return 2\n"
    )
    (src / "b.py").write_text("def caller():\n    return helper()\n")
    (src / "c.py").write_text("def standalone():\n    pass\n")

    repo_id = register_repo(
        conn, slug, "Test", "local_path", str(src), "BAAI/bge-small-en-v1.5", EMBEDDING_DIM
    )
    index_repo(conn, repo_id, src, embedder)
    yield repo_id, src
    drop_repo_by_slug(conn, slug)


class TestIncrementalIndexRepo:
    def test_an_unchanged_file_is_reported_unchanged_and_touches_nothing(
        self, conn, embedder, repo
    ):
        repo_id, src = repo
        before = _chunk_rows(conn, repo_id)

        stats = incremental_index_repo(conn, repo_id, src, embedder)

        assert stats.files_unchanged == 3
        assert stats.files_indexed == 0
        assert stats.chunks_created == 0
        assert _chunk_rows(conn, repo_id) == before

    def test_a_changed_functions_content_gets_a_new_chunk_but_siblings_keep_their_id(
        self, conn, embedder, repo
    ):
        repo_id, src = repo
        before = _chunk_rows(conn, repo_id)
        unrelated_id_before = before[("a.py", "unrelated")]
        caller_id_before = before[("b.py", "caller")]
        old_helper_id = before[("a.py", "helper")]

        (src / "a.py").write_text(
            "def helper():\n    return 100\n\n\ndef unrelated():\n    return 2\n"
        )
        stats = incremental_index_repo(conn, repo_id, src, embedder)

        after = _chunk_rows(conn, repo_id)
        # The literal claim this module exists to make true: a sibling
        # definition that didn't change keeps its exact chunk_id.
        assert after[("a.py", "unrelated")] == unrelated_id_before
        assert after[("b.py", "caller")] == caller_id_before
        # helper's content changed -- it must be a genuinely new row, not
        # the old one mutated in place.
        assert after[("a.py", "helper")] != old_helper_id

        assert stats.files_indexed == 1  # only a.py
        assert stats.files_unchanged == 2  # b.py, c.py
        assert stats.chunks_created == 1  # only the new helper body
        assert stats.chunks_preserved == 1  # unrelated, within the touched file

    def test_call_edge_into_a_changed_function_still_resolves_after_the_rebuild(
        self, conn, embedder, repo
    ):
        # caller (in b.py, untouched) calls helper (in a.py, changed). The
        # edge's old row is cascade-deleted when helper's old chunk goes --
        # this proves the full graph rebuild actually re-attaches it to
        # helper's NEW chunk_id, rather than leaving b.py's call dangling.
        repo_id, src = repo
        (src / "a.py").write_text(
            "def helper():\n    return 100\n\n\ndef unrelated():\n    return 2\n"
        )
        incremental_index_repo(conn, repo_id, src, embedder)

        after = _chunk_rows(conn, repo_id)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT callee_chunk_id, resolution FROM call_edges "
                "WHERE repo_id = %s AND caller_chunk_id = %s",
                (repo_id, after[("b.py", "caller")]),
            )
            rows = cur.fetchall()
        assert rows == [(after[("a.py", "helper")], "exact")]

    def test_an_added_file_is_indexed(self, conn, embedder, repo):
        repo_id, src = repo
        (src / "d.py").write_text("def brand_new():\n    return 42\n")

        stats = incremental_index_repo(conn, repo_id, src, embedder)

        assert stats.files_indexed == 1
        assert stats.chunks_created == 1
        after = _chunk_rows(conn, repo_id)
        assert ("d.py", "brand_new") in after

    def test_a_removed_file_deletes_its_chunks_and_their_edges(self, conn, embedder, repo):
        repo_id, src = repo
        before = _chunk_rows(conn, repo_id)
        standalone_id = before[("c.py", "standalone")]
        (src / "c.py").unlink()

        stats = incremental_index_repo(conn, repo_id, src, embedder)

        assert stats.files_removed == 1
        after = _chunk_rows(conn, repo_id)
        assert ("c.py", "standalone") not in after
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM chunks WHERE repo_id = %s AND id = %s",
                (repo_id, standalone_id),
            )
            assert cur.fetchone()[0] == 0

    def test_last_indexed_sha_is_recorded_when_given(self, conn, embedder, repo):
        repo_id, src = repo
        incremental_index_repo(conn, repo_id, src, embedder, commit_sha="deadbeef")
        with conn.cursor() as cur:
            cur.execute("SELECT last_indexed_sha FROM repos WHERE id = %s", (repo_id,))
            assert cur.fetchone()[0] == "deadbeef"

    def test_running_twice_with_no_changes_is_idempotent(self, conn, embedder, repo):
        repo_id, src = repo
        incremental_index_repo(conn, repo_id, src, embedder)
        before = _chunk_rows(conn, repo_id)
        stats = incremental_index_repo(conn, repo_id, src, embedder)
        assert _chunk_rows(conn, repo_id) == before
        assert stats.chunks_created == 0
