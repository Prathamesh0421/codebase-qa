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

from codeqa.agents.logic import build_trace_messages, merge_chunks, parse_trace_response
from codeqa.agents.state import AgentState
from codeqa.indexing.embeddings import EmbeddingProvider
from codeqa.retrieval.strategy import RetrievalStrategy
from codeqa.synthesis import synthesize


def make_locate_node(
    conn: psycopg.Connection, embedder: EmbeddingProvider, strategy: RetrievalStrategy, top_k: int
):
    def locate(state: AgentState) -> dict:
        new_chunks = strategy.retrieve(conn, state.repo_id, state.current_query, embedder, top_k)
        return {
            "chunks": merge_chunks(state.chunks, new_chunks),
            "attempt": state.attempt + 1,
        }

    return locate


def make_trace_node(model: str, api_key: str | None):
    def trace(state: AgentState) -> dict:
        messages = build_trace_messages(state.question, state.chunks)
        # Non-streaming: this is a control-flow decision consumed by the
        # graph, not prose shown to a user -- nothing benefits from
        # streaming it token by token the way the final answer does.
        response = litellm.completion(model=model, messages=messages, api_key=api_key, stream=False)
        sufficient, next_query, reasoning = parse_trace_response(
            response.choices[0].message.content, state.question
        )
        update = {"sufficient": sufficient, "trace_reasoning": reasoning}
        if not sufficient:
            update["current_query"] = next_query
        return update

    return trace


def make_synthesize_node(model: str, api_key: str | None):
    def synthesize_node(state: AgentState) -> dict:
        # get_stream_writer(), not a return-value generator: LangGraph nodes
        # return a single state update, so streaming has to go out-of-band
        # via a custom stream event per token -- verified interactively that
        # a caller using app.stream(..., stream_mode="custom") receives
        # these as they're written, not batched after the node returns.
        # This is what keeps synthesize()'s own streaming contract (see its
        # docstring) intact all the way through the graph instead of being
        # flattened into one blocking call.
        writer = get_stream_writer()
        tokens = []
        for token in synthesize(state.question, state.chunks, model, api_key):
            writer(token)
            tokens.append(token)
        return {"answer": "".join(tokens)}

    return synthesize_node
