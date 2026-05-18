"""Node factories for the locate -> trace -> synthesize graph.

Each is a factory: a function that closes over its infrastructure
dependencies (a connection, an embedder, a retrieval strategy, model
name/key) and returns the actual `(state) -> dict` callable LangGraph calls.
Infrastructure is wired at construction time rather than threaded through
LangGraph's context_schema machinery -- the same reasoning as get_strategy
and build_embedder elsewhere in this project -- so every node is directly
constructible and testable with a real or fake dependency, no graph required.
"""

import litellm
import psycopg
from langgraph.config import get_stream_writer
from opentelemetry.context import Context

from codeqa.agents.logic import build_trace_messages, merge_chunks, parse_trace_response
from codeqa.agents.state import AgentState
from codeqa.indexing.embeddings import EmbeddingProvider
from codeqa.obs.tracing import get_tracer
from codeqa.retrieval.strategy import RetrievalStrategy
from codeqa.synthesis import synthesize

_tracer = get_tracer(__name__)


def make_locate_node(
    conn: psycopg.Connection,
    embedder: EmbeddingProvider,
    strategy: RetrievalStrategy,
    top_k: int,
    parent_context: Context | None = None,
):
    def locate(state: AgentState) -> dict:
        # context=parent_context, not the ambient/"current" context: the API
        # path drives this node through a sync generator dispatched by
        # starlette's iterate_in_threadpool, which reuses OS threads across
        # calls with no guarantee a later call lands on the same thread as
        # an earlier one. OTel's attach()/detach() hand out a Token that is
        # only valid to reset on the thread that created it, so relying on
        # an ambient "current span" set by an outer wrapper crashes
        # ("Token ... was created in a different Context") the moment the
        # thread pool reuses a worker -- reproduced empirically under real
        # uvicorn concurrency (a TestClient run did not reproduce it, which
        # is what makes this easy to miss). Passing the parent explicitly
        # sidesteps ambient state entirely: this span's own attach/detach
        # pair still happens within one synchronous call, on one thread, so
        # it's self-contained regardless of who called it or from where.
        with _tracer.start_as_current_span("locate", context=parent_context) as span:
            new_chunks = strategy.retrieve(
                conn, state.repo_id, state.current_query, embedder, top_k
            )
            span.set_attribute("codeqa.chunks_found", len(new_chunks))
            return {
                "chunks": merge_chunks(state.chunks, new_chunks),
                "attempt": state.attempt + 1,
            }

    return locate


def make_trace_node(model: str, api_key: str | None, parent_context: Context | None = None):
    def trace(state: AgentState) -> dict:
        with _tracer.start_as_current_span("trace", context=parent_context) as span:
            messages = build_trace_messages(state.question, state.chunks)
            # Non-streaming: this is a control-flow decision consumed by the
            # graph, not prose shown to a user -- nothing benefits from
            # streaming it token by token the way the final answer does.
            response = litellm.completion(
                model=model, messages=messages, api_key=api_key, stream=False
            )
            sufficient, next_query, reasoning = parse_trace_response(
                response.choices[0].message.content, state.question
            )
            span.set_attribute("codeqa.sufficient", sufficient)
            update = {"sufficient": sufficient, "trace_reasoning": reasoning}
            if not sufficient:
                update["current_query"] = next_query
            return update

    return trace


def make_synthesize_node(model: str, api_key: str | None, parent_context: Context | None = None):
    def synthesize_node(state: AgentState) -> dict:
        # get_stream_writer(), not a return-value generator: LangGraph nodes
        # return a single state update, so streaming has to go out-of-band
        # via a custom stream event per token -- verified interactively that
        # a caller using app.stream(..., stream_mode="custom") receives
        # these as they're written, not batched after the node returns.
        # This is what keeps synthesize()'s own streaming contract (see its
        # docstring) intact all the way through the graph instead of being
        # flattened into one blocking call.
        with _tracer.start_as_current_span("synthesize", context=parent_context):
            writer = get_stream_writer()
            tokens = []
            for token in synthesize(state.question, state.chunks, model, api_key):
                writer(token)
                tokens.append(token)
            return {"answer": "".join(tokens)}

    return synthesize_node
