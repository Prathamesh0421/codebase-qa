"""The retrieval strategy interface: one contract, three implementations.

Built now, with only `naive` implemented, deliberately -- not because Phase 5
needs three strategies, but because retrofitting this interface after callers
already assume a bare function signature is expensive, and the whole point of
this project is measuring naive vs hybrid vs hybrid+graph against each other.
That comparison requires all three to satisfy the same contract from day one,
selected by config, so none of them can quietly drift out of a runnable state
while another one is being built.

`naive` must never be deleted, refactored into something incompatible, or
left to bitrot once hybrid exists. It is the eval baseline (Phase 9); if it
stops running, the entire "does call-graph expansion help" measurement
becomes unreproducible at that commit.
"""

from dataclasses import dataclass
from typing import Protocol

import psycopg

from codeqa.indexing.embeddings import EmbeddingProvider


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    file_path: str
    kind: str
    symbol_name: str
    qualified_name: str | None
    start_line: int
    end_line: int
    content: str
    # Meaning varies by strategy: cosine similarity for naive, an RRF-fused
    # score for hybrid/hybrid_graph. Always higher-is-better within one
    # strategy's own results, never comparable across strategies.
    score: float
    # Which retrieval mechanism(s) produced this chunk: any combination of
    # "vector", "lexical", "symbol" joined with "+", or "graph" (optionally
    # ALSO joined with one of those three, when a chunk that graph
    # expansion surfaced also sat in another component's candidate pool
    # without surviving that component's own fusion into top_k -- see
    # HybridGraphStrategy.retrieve). No default -- every strategy must say
    # explicitly what it means, not silently inherit "vector" when it might
    # not be true. Not cosmetic: Phase 9's eval harness needs this to answer
    # the project's central question -- does call-graph expansion find
    # chunks nothing else comes close to -- by checking whether a correct
    # chunk's source is EXACTLY "graph", with no other component appended.
    source: str

    @property
    def citation(self) -> str:
        """path:start-end -- the one citation format used everywhere in this
        project, from CLI output through to the VS Code extension's
        clickable links (Phase 16)."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


class RetrievalStrategy(Protocol):
    def retrieve(
        self,
        conn: psycopg.Connection,
        repo_id: int,
        query_text: str,
        embedder: EmbeddingProvider,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Find the top_k chunks most relevant to query_text, within repo_id.

        Takes the raw query text and an embedder rather than a pre-computed
        vector: naive only needs the vector, but hybrid (Phase 8) also needs
        the text for full-text and symbol lookup, and hybrid+graph needs the
        located chunks before it can expand along call edges. One stable
        signature across all three avoids changing every caller when the
        later strategies land.
        """
        ...


def get_strategy(
    name: str, graph_max_depth: int = 2, graph_max_nodes: int = 40
) -> RetrievalStrategy:
    """Select a strategy by name -- config.retrieval_strategy is the only
    place this decision gets made. An unimplemented strategy fails loudly
    with the phase that adds it, not silently falling back to naive.

    graph_max_depth/graph_max_nodes are only meaningful for hybrid_graph and
    ignored otherwise -- passed as explicit parameters rather than a Settings
    object, matching build_embedder's convention, so this factory (and
    HybridGraphStrategy itself) can be constructed and tested without a
    config file. Defaults mirror config.py's so callers that don't care about
    graph tuning can omit them.
    """
    if name == "naive":
        from codeqa.retrieval.naive import NaiveStrategy

        return NaiveStrategy()
    if name == "hybrid":
        from codeqa.retrieval.hybrid import HybridStrategy

        return HybridStrategy()
    if name == "hybrid_graph":
        from codeqa.retrieval.hybrid import HybridGraphStrategy

        return HybridGraphStrategy(graph_max_depth=graph_max_depth, graph_max_nodes=graph_max_nodes)
    raise ValueError(f"unknown retrieval strategy: {name!r}")
