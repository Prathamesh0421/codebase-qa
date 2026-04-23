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
    # Meaning varies by strategy: cosine similarity for naive/hybrid's dense
    # component, an RRF-fused score once Phase 8 lands. Always higher-is-better.
    score: float

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


def get_strategy(name: str) -> RetrievalStrategy:
    """Select a strategy by name -- config.retrieval_strategy is the only
    place this decision gets made. An unimplemented strategy fails loudly
    with the phase that adds it, not silently falling back to naive."""
    if name == "naive":
        from codeqa.retrieval.naive import NaiveStrategy

        return NaiveStrategy()
    if name == "hybrid":
        raise NotImplementedError("hybrid retrieval arrives in Phase 8")
    if name == "hybrid_graph":
        raise NotImplementedError("call-graph expansion arrives in Phase 8")
    raise ValueError(f"unknown retrieval strategy: {name!r}")
