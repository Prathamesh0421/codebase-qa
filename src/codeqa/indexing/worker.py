"""The durable indexing worker: claim a job, run it, report the outcome.

Indexing takes minutes, so it never runs inside an HTTP request (the "job
boundary the design doc lacked" decision) -- this module is what actually
does the work POST /v1/repos merely schedules. A heartbeat thread runs
alongside the synchronous indexing call so a supervisor elsewhere can tell
this worker is still alive without index_repo itself needing to know
anything about jobs.
"""

import shutil
import threading
import time
from pathlib import Path

import psycopg

from codeqa.config import Settings
from codeqa.indexing.clone import CloneFailed, UnsafeCloneURL, check_disk_quota, safe_clone
from codeqa.indexing.embeddings import build_embedder
from codeqa.indexing.incremental import incremental_index_repo
from codeqa.indexing.jobs import Job, claim_next_job, complete_job, fail_job, heartbeat
from codeqa.indexing.jobs import reclaim_stale_jobs as _reclaim_stale_jobs
from codeqa.indexing.pipeline import index_repo


def _heartbeat_loop(dsn: str, job_id: int, interval: float, stop: threading.Event) -> None:
    # Its own connection, not the worker's main one -- a psycopg Connection
    # is not safe to use concurrently from two threads, and index_repo is
    # running a long sequence of queries on the main connection at the same
    # time this loop is ticking.
    hb_conn = psycopg.connect(dsn)
    try:
        while not stop.wait(interval):
            heartbeat(hb_conn, job_id)
    finally:
        hb_conn.close()


def _run_job(conn: psycopg.Connection, job: Job, settings: Settings) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT source_kind, source_ref, embedding_model, embedding_dim
              FROM repos WHERE id = %s
            """,
            (job.repo_id,),
        )
        row = cur.fetchone()
        if row is None:
            # repo_id comes from an already-claimed job's own FK to repos,
            # so a missing row here means real data corruption, not a race
            # -- raised explicitly (not asserted) so fail_job's error column
            # gets something an operator can act on, not a stripped-assert
            # TypeError from unpacking None two lines down.
            raise RuntimeError(f"repo {job.repo_id} referenced by job {job.id} does not exist")
        source_kind, source_ref, model, dim = row

    base_workdir = Path(settings.clone_workdir)
    workdir = base_workdir / str(job.repo_id)
    commit_sha: str | None = None

    try:
        if source_kind == "git_url":
            # disk_usage needs an existing path -- base_workdir may not
            # exist yet on a fresh install, before any repo has been cloned.
            base_workdir.mkdir(parents=True, exist_ok=True)
            check_disk_quota(base_workdir, settings.disk_min_free_mb)
            # Still a fresh --depth 1 clone every time, even for an
            # incremental job -- Phase 13 made re-EMBEDDING incremental,
            # not re-CLONING. incremental_index_repo diffs blob_sha against
            # what's stored in Postgres, which needs only the current
            # checkout's content, not any git history -- so a shallow
            # re-clone is sufficient and clone-bandwidth savings were never
            # this phase's target. See indexing/incremental.py's docstring.
            shutil.rmtree(workdir, ignore_errors=True)
            commit_sha = safe_clone(
                source_ref, workdir, settings.allowed_clone_hosts,
                settings.clone_timeout_seconds, settings.clone_max_mb,
            )
            root = workdir
        else:
            root = Path(source_ref)

        embedder = build_embedder(
            settings.embedding_provider, model, dim,
            settings.embedding_batch_size, settings.embedding_api_key,
            settings.llm_max_retries,
        )
        if job.kind == "incremental":
            stats = incremental_index_repo(conn, job.repo_id, root, embedder, commit_sha)
        else:
            stats = index_repo(conn, job.repo_id, root, embedder)
        complete_job(
            conn, job.id, job.repo_id, commit_sha,
            {
                "files_indexed": stats.files_indexed,
                "chunks_created": stats.chunks_created,
                "files_unchanged": stats.files_unchanged,
                "duration_seconds": stats.duration_seconds,
            },
        )
    except (UnsafeCloneURL, CloneFailed) as exc:
        conn.rollback()
        fail_job(conn, job.id, job.repo_id, str(exc))
    except Exception as exc:  # noqa: BLE001 -- one bad job must not kill the worker loop
        conn.rollback()
        fail_job(conn, job.id, job.repo_id, str(exc))


def process_one_job(conn: psycopg.Connection, settings: Settings) -> bool:
    """Claim and fully run one job. Returns False if the queue was empty.

    Split from run_worker's loop so tests can drive exactly one job
    deterministically instead of racing an infinite polling loop.
    """
    job = claim_next_job(conn)
    if job is None:
        return False

    stop = threading.Event()
    hb_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(settings.dsn, job.id, settings.job_heartbeat_interval_seconds, stop),
        daemon=True,
    )
    hb_thread.start()
    try:
        _run_job(conn, job, settings)
    finally:
        stop.set()
        hb_thread.join()
    return True


def run_worker(conn: psycopg.Connection, settings: Settings, once: bool = False) -> None:
    """Poll forever (or process exactly one job, with once=True, for tests
    and for `codeqa worker --once`).
    """
    while True:
        _reclaim_stale_jobs(conn, settings.job_stale_after_seconds, settings.job_max_attempts)
        did_work = process_one_job(conn, settings)
        if once:
            return
        if not did_work:
            time.sleep(settings.worker_poll_interval_seconds)
