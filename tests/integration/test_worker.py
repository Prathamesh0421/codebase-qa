"""The worker against real Postgres. The one test that matters most here is
test_a_reclaimed_job_completes_on_a_fresh_worker_call -- it's the literal
mechanism behind Phase 12's "done when" bar: the job survives a worker
restart, not just "the code that would do that exists."
"""

import os

import psycopg
import pytest
from pgvector.psycopg import register_vector

from codeqa.config import Settings
from codeqa.indexing.jobs import claim_next_job, enqueue_job, reclaim_stale_jobs
from codeqa.indexing.store import register_repo
from codeqa.indexing.worker import process_one_job

pytestmark = pytest.mark.integration

EMBEDDING_DIM = 384


def _dsn() -> str:
    return os.environ.get("CODEQA_TEST_DSN", "postgresql://codeqa:codeqa@localhost:5432/codeqa")


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
            cur.execute("DELETE FROM index_jobs WHERE repo_id = %s", (row[0],))
            cur.execute("SELECT drop_repo_partition(%s)", (row[0],))
            cur.execute("DELETE FROM repos WHERE id = %s", (row[0],))
    conn.commit()


@pytest.fixture
def settings(tmp_path):
    return Settings(
        database_url=_dsn(),
        embedding_provider="local",
        embedding_batch_size=8,
        clone_workdir=str(tmp_path / "clones"),
    )


@pytest.fixture
def local_repo_id(conn, tmp_path, request):
    slug = f"test-{request.node.name}".replace("[", "-").replace("]", "")[:60]
    drop_repo_by_slug(conn, slug)

    src = tmp_path / "src"
    src.mkdir()
    (src / "greeter.py").write_text(
        "def greet(name):\n    '''Say hello.'''\n    return f'Hello, {name}!'\n"
    )
    repo_id = register_repo(
        conn, slug, "Test", "local_path", str(src), "BAAI/bge-small-en-v1.5", EMBEDDING_DIM
    )
    yield repo_id
    drop_repo_by_slug(conn, slug)


class TestProcessOneJob:
    def test_returns_false_when_queue_is_empty(self, conn, settings):
        assert process_one_job(conn, settings) is False

    def test_a_local_path_job_indexes_and_succeeds(self, conn, settings, local_repo_id):
        enqueue_job(conn, local_repo_id)
        did_work = process_one_job(conn, settings)
        assert did_work is True

        with conn.cursor() as cur:
            cur.execute("SELECT status FROM index_jobs WHERE repo_id = %s", (local_repo_id,))
            job_status = cur.fetchone()[0]
            cur.execute("SELECT status FROM repos WHERE id = %s", (local_repo_id,))
            repo_status = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM chunks WHERE repo_id = %s", (local_repo_id,))
            chunk_count = cur.fetchone()[0]
        assert job_status == "succeeded"
        assert repo_status == "ready"
        assert chunk_count > 0

    def test_an_unsafe_git_url_fails_the_job_cleanly(self, conn, settings, tmp_path):
        slug = "test-unsafe-git-url"
        drop_repo_by_slug(conn, slug)
        repo_id = register_repo(
            conn, slug, "Test", "git_url", "https://evil.example.com/repo.git",
            "BAAI/bge-small-en-v1.5", EMBEDDING_DIM,
        )
        try:
            enqueue_job(conn, repo_id)
            process_one_job(conn, settings)

            with conn.cursor() as cur:
                cur.execute("SELECT status, error FROM index_jobs WHERE repo_id = %s", (repo_id,))
                job_status, error = cur.fetchone()
                cur.execute("SELECT status FROM repos WHERE id = %s", (repo_id,))
                repo_status = cur.fetchone()[0]
            assert job_status == "failed"
            assert "not in the allowed" in error
            assert repo_status == "failed"
        finally:
            drop_repo_by_slug(conn, slug)


class TestIncrementalJobKind:
    def test_an_incremental_job_only_touches_what_actually_changed(
        self, conn, settings, tmp_path, request
    ):
        slug = f"test-{request.node.name}".replace("[", "-").replace("]", "")[:60]
        drop_repo_by_slug(conn, slug)
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("def helper():\n    return 1\n")
        repo_id = register_repo(
            conn, slug, "Test", "local_path", str(src), "BAAI/bge-small-en-v1.5", EMBEDDING_DIM
        )
        try:
            enqueue_job(conn, repo_id, kind="full")
            process_one_job(conn, settings)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM chunks WHERE repo_id = %s AND symbol_name = 'helper'",
                    (repo_id,),
                )
                original_chunk_id = cur.fetchone()[0]

            # Unchanged content -- an incremental job should report it as
            # such and never touch the existing chunk_id.
            enqueue_job(conn, repo_id, kind="incremental")
            did_work = process_one_job(conn, settings)
            assert did_work is True

            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM chunks WHERE repo_id = %s AND symbol_name = 'helper'",
                    (repo_id,),
                )
                assert cur.fetchone()[0] == original_chunk_id
                cur.execute(
                    """
                    SELECT status, stats FROM index_jobs
                     WHERE repo_id = %s ORDER BY id DESC LIMIT 1
                    """,
                    (repo_id,),
                )
                status, stats = cur.fetchone()
            assert status == "succeeded"
            assert stats["files_unchanged"] == 1
        finally:
            drop_repo_by_slug(conn, slug)


class TestSurvivesAWorkerRestart:
    def test_a_reclaimed_job_completes_on_a_fresh_worker_call(self, conn, settings, local_repo_id):
        enqueue_job(conn, local_repo_id)

        # Simulate a worker that claimed the job and then died before doing
        # any real work -- its heartbeat goes stale, nothing else changes.
        claimed = claim_next_job(conn)
        assert claimed is not None
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE index_jobs SET heartbeat_at = now() - interval '1 hour' WHERE id = %s",
                (claimed.id,),
            )
        conn.commit()

        requeued, failed = reclaim_stale_jobs(
            conn, settings.job_stale_after_seconds, settings.job_max_attempts
        )
        assert claimed.id in requeued
        assert claimed.id not in failed

        # A fresh worker call now picks the reclaimed job back up and
        # actually finishes it -- this is the behavior the phase's "done
        # when" bar names, not just the queue-state transition tested above.
        did_work = process_one_job(conn, settings)
        assert did_work is True

        with conn.cursor() as cur:
            cur.execute("SELECT status FROM index_jobs WHERE id = %s", (claimed.id,))
            assert cur.fetchone()[0] == "succeeded"
            cur.execute("SELECT status FROM repos WHERE id = %s", (local_repo_id,))
            assert cur.fetchone()[0] == "ready"
