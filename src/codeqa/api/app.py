"""POST /v1/repos, POST /v1/query, and GET /health -- the api/ layer the
layout has always planned for.

Auth, rate limiting, and a query cache are deliberately still absent (split
out to a follow-up pass): this phase's "done when" bar is narrower -- a
query streams end-to-end over HTTP and shows as one trace in Jaeger -- and
adding unauthenticated production surface area beyond that bar isn't free,
so it's scoped to exactly what proves the bar.

A connection is opened and closed per request rather than pooled
(psycopg_pool is an installed extra but unused here) -- the simplest thing
that works, deliberately not optimized ahead of a load number that doesn't
exist yet. Revisit once concurrency is actually measured.
"""

import json
from typing import Literal

import psycopg
import redis
import structlog
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from codeqa.agents.graph import build_agent_graph
from codeqa.agents.state import AgentState
from codeqa.config import Settings, get_settings
from codeqa.grounding import ground_answer
from codeqa.indexing.clone import UnsafeCloneURL, validate_clone_url
from codeqa.indexing.embeddings import build_embedder
from codeqa.indexing.jobs import enqueue_job
from codeqa.indexing.store import RepoAlreadyExists, register_repo
from codeqa.obs.logging import configure_logging
from codeqa.obs.tracing import configure_tracing, get_tracer
from codeqa.retrieval.strategy import get_strategy

_settings = get_settings()
configure_tracing("codeqa-api", _settings.otel_endpoint)
configure_logging(_settings.log_level)

app = FastAPI(title="CodeQA API")
FastAPIInstrumentor.instrument_app(app)

_tracer = get_tracer(__name__)
_log = structlog.get_logger()


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


@app.get("/health")
def health(settings: Settings = Depends(get_settings)) -> JSONResponse:  # noqa: B008
    # Postgres is load-bearing: nothing works without it, so it gates the
    # status code. Redis isn't wired into anything yet in this phase, but is
    # reported anyway since it's a preview of the degrade-don't-fail stance
    # a future cache/rate-limit pass will take: Redis down means degraded
    # service, not down service, so it never gates the code by itself.
    db_ok = True
    try:
        with psycopg.connect(settings.dsn, connect_timeout=2) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
    except Exception:
        db_ok = False

    redis_ok = True
    try:
        redis.Redis.from_url(str(settings.redis_url), socket_connect_timeout=2).ping()
    except Exception:
        redis_ok = False

    return JSONResponse(
        status_code=200 if db_ok else 503,
        content={"status": "ok" if db_ok else "down", "db": db_ok, "redis": redis_ok},
    )


class QueryRequest(BaseModel):
    repo_slug: str
    question: str
    top_k: int | None = None


def _stream_query(
    conn: psycopg.Connection,
    repo_id: int,
    embedding_model: str,
    embedding_dim: int,
    question: str,
    top_k: int,
    settings: Settings,
):
    """Runs the same locate -> trace -> synthesize graph as `codeqa ask
    --agent` (agents/graph.py), over HTTP instead of the terminal. A plain
    (sync) generator, not async def -- sse-starlette wraps it with
    starlette's iterate_in_threadpool, which is what actually keeps the
    blocking psycopg/litellm calls inside off the event loop; making this
    function async wouldn't change that, since nothing in the retrieval or
    LLM path is awaitable.

    Root-span parenting is the one real subtlety here, and it's why
    locate/trace/synthesize (agents/nodes.py) take an explicit
    parent_context instead of relying on OTel's ambient "current span".
    iterate_in_threadpool (see starlette.concurrency) drives this generator
    by calling next() once per yielded event, each dispatched independently
    via anyio.to_thread.run_sync -- nothing guarantees two consecutive
    next() calls land on the same worker thread, or that a later call
    doesn't land on a thread a DIFFERENT earlier call already used.
    Plain sync generators (unlike coroutines) get no automatic per-call
    context isolation, so a span opened with `start_as_current_span`
    wrapped around this whole generator -- or even just an attach() left
    open across a yield -- ends up trying to reset a contextvars Token on a
    thread that didn't create it, and OTel logs "Token ... was created in a
    different Context" (reproduced empirically under real uvicorn
    concurrency; a TestClient run does not reproduce it, which is what
    makes this easy to miss in dev). Passing parent_context explicitly into
    each node's own start_as_current_span call sidesteps ambient state
    entirely: each node's span still opens and closes within one
    synchronous call on one thread, so it's self-contained no matter which
    thread happens to run it.
    """
    root_span = _tracer.start_span("query")
    root_span.set_attribute("codeqa.repo_id", repo_id)
    parent_ctx = trace.set_span_in_context(root_span)
    log = _log.bind(repo_id=repo_id)
    log.info("query.start", question=question)

    embedder = build_embedder(
        settings.embedding_provider, embedding_model, embedding_dim,
        settings.embedding_batch_size, settings.embedding_api_key,
    )
    strategy = get_strategy(
        settings.retrieval_strategy,
        graph_max_depth=settings.graph_max_depth,
        graph_max_nodes=settings.graph_max_nodes,
    )
    graph = build_agent_graph(
        conn, embedder, strategy, top_k, settings.llm_model, settings.llm_api_key, parent_ctx
    )
    state = AgentState(
        repo_id=repo_id, question=question, current_query=question,
        max_attempts=settings.agent_max_attempts,
    )

    final_chunks: list = []
    answer_tokens: list[str] = []
    try:
        for mode, payload in graph.stream(state, stream_mode=["updates", "custom"]):
            if mode == "custom":
                answer_tokens.append(payload)
                yield {"event": "token", "data": payload}
                continue
            for node_name, update in payload.items():
                if node_name == "locate":
                    final_chunks = update["chunks"]
                    yield {
                        "event": "progress",
                        "data": json.dumps(
                            {
                                "stage": "locate",
                                "attempt": update["attempt"],
                                "chunk_count": len(update["chunks"]),
                            }
                        ),
                    }
                elif node_name == "trace":
                    yield {
                        "event": "progress",
                        "data": json.dumps(
                            {"stage": "trace", "sufficient": update["sufficient"]}
                        ),
                    }
    except Exception as exc:  # noqa: BLE001 -- must degrade to an SSE event, not a crash
        log.error("query.failed", error=str(exc))
        root_span.record_exception(exc)
        root_span.set_status(Status(StatusCode.ERROR, str(exc)))
        yield {"event": "error", "data": json.dumps({"message": str(exc)})}
        return
    finally:
        root_span.end()

    final_answer = "".join(answer_tokens)
    result = ground_answer(final_answer, final_chunks)
    log.info("query.done", chunk_count=len(final_chunks), citations_dropped=len(result.dropped))
    yield {
        "event": "done",
        "data": json.dumps(
            {
                "citations_dropped": [c.raw for c in result.dropped],
                "chunks": [
                    {
                        "citation": c.citation,
                        "score": c.score,
                        "symbol": c.qualified_name or c.symbol_name,
                    }
                    for c in final_chunks
                ],
            }
        ),
    }


@app.post("/v1/query")
def query(
    body: QueryRequest,
    conn: psycopg.Connection = Depends(get_conn),  # noqa: B008
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> EventSourceResponse:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, embedding_model, embedding_dim FROM repos WHERE slug = %s",
            (body.repo_slug,),
        )
        row = cur.fetchone()
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"no repo registered with slug {body.repo_slug!r}"
        )
    repo_id, embedding_model, embedding_dim = row
    top_k = body.top_k or settings.retrieval_top_k

    return EventSourceResponse(
        _stream_query(conn, repo_id, embedding_model, embedding_dim, body.question, top_k, settings)
    )
