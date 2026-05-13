"""The index_jobs queue against real Postgres: enqueue, claim, heartbeat,
reclaim, finish. Two connections are used where concurrent claiming is the
point -- FOR UPDATE SKIP LOCKED's guarantee is about separate sessions, and
a single connection can't demonstrate that two workers never double-claim.
"""

import os
import time

import psycopg
import pytest

from codeqa.indexing.jobs import (
    JobAlreadyLive,
    claim_next_job,
    complete_job,
    enqueue_job,
    fail_job,
    heartbeat,
    reclaim_stale_jobs,
)
from codeqa.indexing.store import register_repo

pytestmark = pytest.mark.integration


def _dsn() -> str:
    return os.environ.get("CODEQA_TEST_DSN", "postgresql://codeqa:codeqa@localhost:5432/codeqa")


@pytest.fixture
def conn():
    connection = psycopg.connect(_dsn())
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
def repo_id(conn, request):
    slug = f"test-{request.node.name}".replace("[", "-").replace("]", "")[:60]
    drop_repo_by_slug(conn, slug)
    rid = register_repo(conn, slug, "Test", "git_url", "https://github.com/x/y", "m", 384)
    yield rid
    drop_repo_by_slug(conn, slug)


class TestEnqueueAndClaim:
    def test_enqueue_creates_a_queued_job(self, conn, repo_id):
        job_id = enqueue_job(conn, repo_id)
        with conn.cursor() as cur:
            cur.execute("SELECT status, kind, attempts FROM index_jobs WHERE id = %s", (job_id,))
            status, kind, attempts = cur.fetchone()
        assert status == "queued"
        assert kind == "full"
        assert attempts == 0

    def test_a_second_enqueue_for_the_same_repo_fails_while_one_is_live(self, conn, repo_id):
        enqueue_job(conn, repo_id)
        with pytest.raises(JobAlreadyLive):
            enqueue_job(conn, repo_id)

    def test_claim_next_job_returns_none_when_queue_is_empty(self, conn):
        assert claim_next_job(conn) is None

    def test_claim_next_job_marks_it_running_and_increments_attempts(self, conn, repo_id):
        job_id = enqueue_job(conn, repo_id)
        job = claim_next_job(conn)
        assert job.id == job_id
        assert job.attempts == 1
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status, started_at IS NOT NULL FROM index_jobs WHERE id = %s", (job_id,)
            )
            status, has_started_at = cur.fetchone()
        assert status == "running"
        assert has_started_at

    def test_claim_next_job_claims_oldest_first(self, conn, repo_id, tmp_path):
        # A second repo so both jobs can be queued at once (one live job per
        # repo is enforced by the schema).
        other_id = register_repo(
            conn, "test-claim-order-other", "Other", "git_url", "https://github.com/a/b", "m", 384
        )
        try:
            first = enqueue_job(conn, repo_id)
            enqueue_job(conn, other_id)
            claimed = claim_next_job(conn)
            assert claimed.id == first
        finally:
            drop_repo_by_slug(conn, "test-claim-order-other")

    def test_two_connections_never_claim_the_same_job(self, conn, repo_id):
        # claim_next_job commits immediately, so this proves the functional
        # outcome (one job, claimed once) rather than stressing SKIP LOCKED's
        # actual lock-contention path -- that needs the first claim's
        # transaction still open while a second session queries concurrently,
        # which a sequential test on two connections can't reproduce.
        enqueue_job(conn, repo_id)
        other_conn = psycopg.connect(_dsn())
        try:
            job_a = claim_next_job(conn)
            job_b = claim_next_job(other_conn)
            assert job_a is not None
            assert job_b is None
        finally:
            other_conn.close()


class TestHeartbeatAndReclaim:
    def test_heartbeat_updates_the_timestamp(self, conn, repo_id):
        job_id = enqueue_job(conn, repo_id)
        claim_next_job(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT heartbeat_at FROM index_jobs WHERE id = %s", (job_id,))
            before = cur.fetchone()[0]
        time.sleep(0.05)
        heartbeat(conn, job_id)
        with conn.cursor() as cur:
            cur.execute("SELECT heartbeat_at FROM index_jobs WHERE id = %s", (job_id,))
            after = cur.fetchone()[0]
        assert after > before

    def test_a_fresh_heartbeat_is_left_alone_by_reclaim(self, conn, repo_id):
        job_id = enqueue_job(conn, repo_id)
        claim_next_job(conn)
        requeued, failed = reclaim_stale_jobs(conn, stale_after_seconds=60, max_attempts=3)
        assert job_id not in requeued
        assert job_id not in failed
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM index_jobs WHERE id = %s", (job_id,))
            assert cur.fetchone()[0] == "running"

    def test_a_stale_heartbeat_is_requeued_when_attempts_remain(self, conn, repo_id):
        job_id = enqueue_job(conn, repo_id)
        claim_next_job(conn)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE index_jobs SET heartbeat_at = now() - interval '1 hour' WHERE id = %s",
                (job_id,),
            )
        conn.commit()

        requeued, failed = reclaim_stale_jobs(conn, stale_after_seconds=60, max_attempts=3)
        assert job_id in requeued
        assert job_id not in failed
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM index_jobs WHERE id = %s", (job_id,))
            assert cur.fetchone()[0] == "queued"

    def test_a_stale_job_past_max_attempts_is_failed_not_requeued(self, conn, repo_id):
        job_id = enqueue_job(conn, repo_id)
        # Simulate three prior dead-worker cycles by claiming and going stale
        # three times before the final reclaim check.
        for _ in range(3):
            claim_next_job(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE index_jobs SET status='queued' WHERE id = %s", (job_id,)
                )
            conn.commit()
        claim_next_job(conn)  # attempts is now 4
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE index_jobs SET heartbeat_at = now() - interval '1 hour' WHERE id = %s",
                (job_id,),
            )
        conn.commit()

        requeued, failed = reclaim_stale_jobs(conn, stale_after_seconds=60, max_attempts=3)
        assert job_id in failed
        assert job_id not in requeued
        with conn.cursor() as cur:
            cur.execute("SELECT status, error FROM index_jobs WHERE id = %s", (job_id,))
            status, error = cur.fetchone()
        assert status == "failed"
        assert "max attempts" in error


class TestCompleteAndFail:
    def test_complete_job_marks_succeeded_and_updates_the_repo(self, conn, repo_id):
        job_id = enqueue_job(conn, repo_id)
        claim_next_job(conn)
        complete_job(conn, job_id, repo_id, commit_sha="abc123", stats={"files": 4})

        with conn.cursor() as cur:
            cur.execute("SELECT status, stats FROM index_jobs WHERE id = %s", (job_id,))
            job_status, stats = cur.fetchone()
            cur.execute(
                "SELECT status, last_indexed_sha FROM repos WHERE id = %s", (repo_id,)
            )
            repo_status, sha = cur.fetchone()
        assert job_status == "succeeded"
        assert stats == {"files": 4}
        assert repo_status == "ready"
        assert sha == "abc123"

    def test_fail_job_marks_failed_and_updates_the_repo(self, conn, repo_id):
        job_id = enqueue_job(conn, repo_id)
        claim_next_job(conn)
        fail_job(conn, job_id, repo_id, "clone timed out")

        with conn.cursor() as cur:
            cur.execute("SELECT status, error FROM index_jobs WHERE id = %s", (job_id,))
            job_status, error = cur.fetchone()
            cur.execute("SELECT status FROM repos WHERE id = %s", (repo_id,))
            repo_status = cur.fetchone()[0]
        assert job_status == "failed"
        assert error == "clone timed out"
        assert repo_status == "failed"
