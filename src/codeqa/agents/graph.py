"""Build the locate -> trace -> synthesize state graph.

The one real cycle in this project's AGENT graph -- not to be confused with
the call GRAPH's cycles, which graph/traversal.py handles as a completely
separate concern. If trace decides the located context is insufficient, this
routes back to locate with a refined query, bounded by
state.max_attempts (config.agent_max_attempts) so a model that keeps saying
INSUFFICIENT can never spin the pipeline forever.

Without this edge, locate -> trace -> synthesize is a straight line, and a
straight line doesn't need a graph library -- three function calls would do
exactly as well. This cycle is what actually earns LangGraph's place here.
"""

import psycopg
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from opentelemetry.context import Context

from codeqa.agents.logic import route_after_trace
from codeqa.agents.nodes import make_locate_node, make_synthesize_node, make_trace_node
from codeqa.agents.state import AgentState
from codeqa.indexing.embeddings import EmbeddingProvider
from codeqa.retrieval.strategy import RetrievalStrategy


def build_agent_graph(
    conn: psycopg.Connection,
    embedder: EmbeddingProvider,
    strategy: RetrievalStrategy,
    top_k: int,
    llm_model: str,
    llm_api_key: str | None,
    parent_context: Context | None = None,
    llm_max_retries: int = 0,
) -> CompiledStateGraph[AgentState]:
    g: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)
    # The three ignores below: langgraph's add_node overloads don't resolve
    # for a closure typed as Callable[[AgentState], dict[str, Any]], only
    # for a literal def or a class __call__. Verified in isolation against
    # langgraph 1.2.11 with a three-way minimal repro (top-level function:
    # OK; class __call__: OK; Callable-typed closure: fails). The runtime
    # contract is identical; this is a stub limitation, not a type error in
    # the node factories.
    g.add_node("locate", make_locate_node(conn, embedder, strategy, top_k, parent_context))  # type: ignore[call-overload]
    g.add_node(
        "trace", make_trace_node(llm_model, llm_api_key, parent_context, llm_max_retries)  # type: ignore[call-overload]
    )
    g.add_node(
        "synthesize",
        make_synthesize_node(llm_model, llm_api_key, parent_context, llm_max_retries),  # type: ignore[call-overload]
    )

    g.add_edge(START, "locate")
    g.add_edge("locate", "trace")
    g.add_conditional_edges(
        "trace", route_after_trace, {"locate": "locate", "synthesize": "synthesize"}
    )
    g.add_edge("synthesize", END)

    return g.compile()
