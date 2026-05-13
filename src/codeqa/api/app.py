"""POST /v1/repos and its job-status counterpart -- the first two routes of
the api/ layer the layout has always planned for. No auth, no rate
limiting, no connection pooling: those are Phase 14's "hardening" job, not
this one's. This phase's job is narrower -- "any repo, safely" -- and is
scoped to exactly that.

A connection is opened and closed per request rather than pooled
(psycopg_pool is an installed extra but unused here) -- the simplest thing
that works, deliberately not optimized ahead of a load number that doesn't
exist yet. Revisit once Phase 14 actually measures request latency under
concurrency.
"""

from typing import Literal

import psycopg
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from codeqa.config import Settings, get_settings
from codeqa.indexing.clone import UnsafeCloneURL, validate_clone_url
from codeqa.indexing.jobs import enqueue_job
from codeqa.indexing.store import RepoAlreadyExists, register_repo

app = FastAPI(title="CodeQA API")


def get_conn(settings: Settings = Depends(get_settings)):  # noqa: B008
    conn = psycopg.connect(settings.dsn)
    try:
        yield conn
    finally:
        conn.close()


class CreateRepoRequest(BaseModel):
    slug: str
    display_name: str
    source_kind: Literal["git_url", "local_path"]
    source_ref: str
    default_branch: str | None = None


class CreateRepoResponse(BaseModel):
    repo_id: int
    job_id: int


class JobStatusResponse(BaseModel):
    id: int
    repo_id: int
    kind: str
    status: str
    attempts: int
    error: str | None
    stats: dict


@app.post("/v1/repos", response_model=CreateRepoResponse, status_code=201)
def create_repo(
    body: CreateRepoRequest,
    conn: psycopg.Connection = Depends(get_conn),  # noqa: B008
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> CreateRepoResponse:
    # Rejected here, not just later inside the worker: failing an obviously
    # unsafe URL at request time is a clearer signal to the caller than a
    # job that's queued only to fail a few seconds later for a reason it
    # could have been told immediately.
    if body.source_kind == "git_url":
        try:
            validate_clone_url(body.source_ref, settings.allowed_clone_hosts)
        except UnsafeCloneURL as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        repo_id = register_repo(
            conn, body.slug, body.display_name, body.source_kind, body.source_ref,
            settings.embedding_model, settings.embedding_dim, body.default_branch,
        )
    except RepoAlreadyExists as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    # JobAlreadyLive can't fire here -- repo_id was just created, so no job
    # could already exist for it. That guard belongs to a future reindex
    # endpoint that calls enqueue_job against an EXISTING repo_id instead.
    job_id = enqueue_job(conn, repo_id)

    return CreateRepoResponse(repo_id=repo_id, job_id=job_id)


@app.get("/v1/repos/{slug}/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(
    slug: str, job_id: int, conn: psycopg.Connection = Depends(get_conn)  # noqa: B008
) -> JobStatusResponse:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT j.id, j.repo_id, j.kind, j.status, j.attempts, j.error, j.stats
              FROM index_jobs j
              JOIN repos r ON r.id = j.repo_id
             WHERE r.slug = %s AND j.id = %s
            """,
            (slug, job_id),
        )
        row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"no job {job_id} for repo {slug!r}")
    return JobStatusResponse(
        id=row[0], repo_id=row[1], kind=row[2], status=row[3],
        attempts=row[4], error=row[5], stats=row[6],
    )
