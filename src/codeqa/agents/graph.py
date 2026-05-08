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
) -> CompiledStateGraph:
    g: StateGraph = StateGraph(AgentState)
    g.add_node("locate", make_locate_node(conn, embedder, strategy, top_k))
    g.add_node("trace", make_trace_node(llm_model, llm_api_key))
    g.add_node("synthesize", make_synthesize_node(llm_model, llm_api_key))

    g.add_edge(START, "locate")
    g.add_edge("locate", "trace")
    g.add_conditional_edges(
        "trace", route_after_trace, {"locate": "locate", "synthesize": "synthesize"}
    )
    g.add_edge("synthesize", END)

    return g.compile()
