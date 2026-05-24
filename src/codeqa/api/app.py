"""POST /v1/repos, POST /v1/query, and GET /health -- the api/ layer the
layout has always planned for. Every route except /health requires a
bearer API key (Phase 14b); /v1/query additionally sits behind a per-key
Redis rate limiter and a query result cache.

A connection is opened and closed per request rather than pooled
(psycopg_pool is an installed extra but unused here) -- the simplest thing
that works, deliberately not optimized ahead of a load number that doesn't
exist yet. Revisit once concurrency is actually measured.
"""

import json
from collections.abc import Iterator
from typing import Any, Literal

import psycopg
import redis
import structlog
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from codeqa.agents.graph import build_agent_graph
from codeqa.agents.state import AgentState
from codeqa.api.auth import ApiKeyRecord, InvalidApiKey, verify_api_key
from codeqa.api.cache import CachedAnswer, QueryCache
from codeqa.api.rate_limit import RateLimiter
from codeqa.config import Settings, get_settings
from codeqa.grounding import ground_answer
from codeqa.indexing.clone import UnsafeCloneURL, validate_clone_url
from codeqa.indexing.embeddings import build_embedder
from codeqa.indexing.jobs import enqueue_job
from codeqa.indexing.store import RepoAlreadyExists, register_repo
from codeqa.obs.logging import configure_logging
from codeqa.obs.tracing import configure_tracing, get_tracer
from codeqa.retrieval.strategy import RetrievedChunk, get_strategy

_settings = get_settings()
configure_tracing("codeqa-api", _settings.otel_endpoint)
configure_logging(_settings.log_level)

app = FastAPI(title="CodeQA API")
FastAPIInstrumentor.instrument_app(app)

_tracer = get_tracer(__name__)
_log = structlog.get_logger()

# Built once at process startup from whatever Settings were active at
# import time, the same convention configure_tracing/configure_logging
# above already use in this file -- redis.Redis manages its own connection
# pool internally, so this isn't "one shared connection", it's one client
# object reused across requests.
_redis_client = redis.Redis.from_url(str(_settings.redis_url))
_rate_limiter = RateLimiter(_redis_client)
_query_cache = QueryCache(_redis_client, _settings.cache_ttl_seconds)


def get_conn(settings: Settings = Depends(get_settings)) -> Iterator[psycopg.Connection]:  # noqa: B008
    conn = psycopg.connect(settings.dsn)
    try:
        yield conn
    finally:
        conn.close()


def require_api_key(
    authorization: str | None = Header(None),
    conn: psycopg.Connection = Depends(get_conn),  # noqa: B008
) -> ApiKeyRecord:
    # Doesn't distinguish "missing key" from "invalid key" from "revoked
    # key" in the response -- all three collapse to one 401, since telling
    # a caller precisely which way their key is wrong is a minor
    # information leak for callers who don't already have a good key, and
    # no benefit to callers who do.
    if authorization is None or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing or malformed Authorization header")
    plaintext = authorization.split(" ", 1)[1].strip()
    try:
        return verify_api_key(conn, plaintext)
    except InvalidApiKey as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


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
    stats: dict[str, Any]


@app.post("/v1/repos", response_model=CreateRepoResponse, status_code=201)
def create_repo(
    body: CreateRepoRequest,
    conn: psycopg.Connection = Depends(get_conn),  # noqa: B008
    settings: Settings = Depends(get_settings),  # noqa: B008
    api_key: ApiKeyRecord = Depends(require_api_key),  # noqa: B008
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
    slug: str,
    job_id: int,
    conn: psycopg.Connection = Depends(get_conn),  # noqa: B008
    api_key: ApiKeyRecord = Depends(require_api_key),  # noqa: B008
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


def _replay_cached(cached: CachedAnswer) -> Iterator[dict[str, Any]]:
    """A cache hit skips retrieval and synthesis entirely -- no locate/trace
    progress events, because neither node ran. The answer is replayed as a
    single token event rather than split back into per-token chunks: the
    granularity of the ORIGINAL stream isn't recoverable from the cached
    string, and re-chunking it arbitrarily would misrepresent the second
    request as having streamed live when it didn't.
    """
    yield {"event": "token", "data": cached.answer}
    yield {
        "event": "done",
        "data": json.dumps(
            {
                "citations_dropped": cached.citations_dropped,
                "chunks": cached.chunks,
                "cached": True,
            }
        ),
    }


def _stream_query(
    conn: psycopg.Connection,
    repo_id: int,
    embedding_model: str,
    embedding_dim: int,
    question: str,
    top_k: int,
    settings: Settings,
    cache: QueryCache,
    last_indexed_sha: str | None,
) -> Iterator[dict[str, Any]]:
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
        settings.llm_max_retries,
    )
    strategy = get_strategy(
        settings.retrieval_strategy,
        graph_max_depth=settings.graph_max_depth,
        graph_max_nodes=settings.graph_max_nodes,
    )
    graph = build_agent_graph(
        conn,
        embedder,
        strategy,
        top_k,
        settings.llm_model,
        settings.llm_api_key,
        parent_ctx,
        settings.llm_max_retries,
    )
    state = AgentState(
        repo_id=repo_id, question=question, current_query=question,
        max_attempts=settings.agent_max_attempts,
    )

    final_chunks: list[RetrievedChunk] = []
    answer_tokens: list[str] = []
    try:
        for mode, payload in graph.stream(state, stream_mode=["updates", "custom"]):
            if mode == "custom":
                answer_tokens.append(payload)
                yield {"event": "token", "data": payload}
                continue
            # stream_mode as a list collapses payload's stubbed type to
            # "dict[str, Any] | Any" (see cli.py's identical narrowing) --
            # every non-"custom" mode payload is a {node_name: update} dict
            # at runtime.
            assert isinstance(payload, dict)
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
    citations_dropped = [c.raw for c in result.dropped]
    chunk_payload = [
        {
            "citation": c.citation,
            "score": c.score,
            "symbol": c.qualified_name or c.symbol_name,
        }
        for c in final_chunks
    ]
    # Cached only on a clean completion -- the except branch above returns
    # before reaching here, so a failed run is never cached as if it
    # succeeded.
    cache.set(
        repo_id,
        question,
        last_indexed_sha,
        CachedAnswer(
            answer=final_answer, chunks=chunk_payload, citations_dropped=citations_dropped
        ),
    )
    yield {
        "event": "done",
        "data": json.dumps({"citations_dropped": citations_dropped, "chunks": chunk_payload}),
    }


@app.post("/v1/query")
def query(
    body: QueryRequest,
    conn: psycopg.Connection = Depends(get_conn),  # noqa: B008
    settings: Settings = Depends(get_settings),  # noqa: B008
    api_key: ApiKeyRecord = Depends(require_api_key),  # noqa: B008
) -> EventSourceResponse:
    # Rate limit is checked here, in the route function, before
    # EventSourceResponse is ever constructed -- not inside _stream_query.
    # Once SSE headers go out the response is committed to 200; a 429 has
    # to be a real HTTP error response, raised before that point, or a
    # rate-limited client would see a 200 stream carrying an "error" event
    # instead of the 429 it actually needs to back off correctly.
    if not _rate_limiter.allow(api_key.id, api_key.rate_limit_rpm):
        raise HTTPException(status_code=429, detail="rate limit exceeded")

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, embedding_model, embedding_dim, last_indexed_sha
              FROM repos WHERE slug = %s
            """,
            (body.repo_slug,),
        )
        row = cur.fetchone()
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"no repo registered with slug {body.repo_slug!r}"
        )
    repo_id, embedding_model, embedding_dim, last_indexed_sha = row
    top_k = body.top_k or settings.retrieval_top_k

    cached = _query_cache.get(repo_id, body.question, last_indexed_sha)
    if cached is not None:
        log = _log.bind(repo_id=repo_id)
        log.info("query.cache_hit", question=body.question)
        return EventSourceResponse(_replay_cached(cached))

    return EventSourceResponse(
        _stream_query(
            conn,
            repo_id,
            embedding_model,
            embedding_dim,
            body.question,
            top_k,
            settings,
            _query_cache,
            last_indexed_sha,
        )
    )
