"""The index_jobs queue: enqueue, claim, heartbeat, reclaim, finish.

Postgres is the queue -- no broker, same reasoning as the "job boundary the
design doc lacked" decision from Phase 0. claim_next_job uses
`FOR UPDATE SKIP LOCKED` so multiple worker processes can claim distinct
jobs from the same queued set without blocking on each other's row locks;
each function commits its own change immediately, so a claimed row's lock
is released the instant the claim succeeds.

reclaim_stale_jobs is the literal mechanism behind "the job survives a
worker restart": a running job whose heartbeat has gone stale is presumed
to belong to a dead worker and is reset to queued so a different worker
process can pick it back up, unless it has already exhausted its retry
budget, in which case it's marked failed instead of retried forever.
"""

import json
from dataclasses import dataclass

import psycopg

from codeqa.indexing.store import mark_failed, mark_indexed


class JobAlreadyLive(RuntimeError):
    pass


@dataclass(frozen=True)
class Job:
    id: int
    repo_id: int
    kind: str
    attempts: int


def enqueue_job(conn: psycopg.Connection, repo_id: int, kind: str = "full") -> int:
    """Insert a queued job. index_jobs_one_live_per_repo (0001_init.sql)
    enforces at the database that a repo can't have two live jobs -- caught
    here and rolled back rather than left to leak a raw psycopg error and an
    aborted transaction to the caller, the same pattern register_repo
    (store.py) uses for a duplicate slug.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO index_jobs (repo_id, kind) VALUES (%s, %s) RETURNING id",
                (repo_id, kind),
            )
            job_id = cur.fetchone()[0]
        conn.commit()
        return job_id
    except psycopg.errors.UniqueViolation as exc:
        conn.rollback()
        raise JobAlreadyLive(f"repo_id {repo_id} already has a live index job") from exc


def claim_next_job(conn: psycopg.Connection) -> Job | None:
    """Atomically claim the oldest queued job, or None if the queue is empty.

    SKIP LOCKED is what makes this safe under concurrent workers: a second
    worker's identical query simply skips a row the first worker's
    transaction already has locked, rather than blocking until it commits
    (which would serialize workers into claiming jobs one at a time) or
    reading stale data (a plain SELECT without FOR UPDATE could return the
    same row to two workers at once).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE index_jobs
               SET status = 'running', started_at = now(), heartbeat_at = now(),
                   attempts = attempts + 1
             WHERE id = (
                 SELECT id FROM index_jobs
                  WHERE status = 'queued'
                  ORDER BY created_at
                  FOR UPDATE SKIP LOCKED
                  LIMIT 1
             )
            RETURNING id, repo_id, kind, attempts
            """
        )
        row = cur.fetchone()
    conn.commit()
    if row is None:
        return None
    return Job(id=row[0], repo_id=row[1], kind=row[2], attempts=row[3])


def heartbeat(conn: psycopg.Connection, job_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute("UPDATE index_jobs SET heartbeat_at = now() WHERE id = %s", (job_id,))
    conn.commit()


def reclaim_stale_jobs(
    conn: psycopg.Connection, stale_after_seconds: int, max_attempts: int
) -> tuple[list[int], list[int]]:
    """Running jobs whose heartbeat is older than stale_after_seconds are
    presumed abandoned by a dead worker. Returns (requeued_ids, failed_ids).

    A job that has already reached max_attempts is marked failed instead of
    requeued -- otherwise a permanently broken job (a URL that will never
    clone, say) would cycle running -> stale -> queued forever, with no
    outcome ever reported.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE index_jobs
               SET status = 'failed', error = 'exceeded max attempts after being reclaimed',
                   finished_at = now()
             WHERE status = 'running'
               AND heartbeat_at < now() - make_interval(secs => %s)
               AND attempts >= %s
            RETURNING id
            """,
            (stale_after_seconds, max_attempts),
        )
        failed_ids = [row[0] for row in cur.fetchall()]

        cur.execute(
            """
            UPDATE index_jobs
               SET status = 'queued'
             WHERE status = 'running'
               AND heartbeat_at < now() - make_interval(secs => %s)
               AND attempts < %s
            RETURNING id
            """,
            (stale_after_seconds, max_attempts),
        )
        requeued_ids = [row[0] for row in cur.fetchall()]
    conn.commit()
    return requeued_ids, failed_ids


def complete_job(
    conn: psycopg.Connection, job_id: int, repo_id: int, commit_sha: str | None, stats: dict
) -> None:
    """Mark succeeded and update the repo's ready state in one transaction --
    delegates the repos update to mark_indexed (store.py) rather than
    duplicating it, and relies on mark_indexed's own commit to also flush
    this function's job-status update, made on the same connection just
    before it. A job that succeeded but left the repo not reflecting it (or
    vice versa) would be a worse inconsistency than either failing outright.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE index_jobs
               SET status = 'succeeded', finished_at = now(), stats = %s
             WHERE id = %s
            """,
            (json.dumps(stats), job_id),
        )
    mark_indexed(conn, repo_id, commit_sha)


def fail_job(conn: psycopg.Connection, job_id: int, repo_id: int, error: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE index_jobs SET status = 'failed', error = %s, finished_at = now()
             WHERE id = %s
            """,
            (error, job_id),
        )
    mark_failed(conn, repo_id)
