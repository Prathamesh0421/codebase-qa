"""HybridStrategy: fuse dense, lexical and exact-symbol retrieval via RRF.
HybridGraphStrategy: HybridStrategy, then expand along the call graph.

Three independently-ranked candidate lists per query -- vector cosine
similarity, ts_rank full-text, and exact symbol/qualified-name match -- fused
by codeqa.retrieval.fusion.reciprocal_rank_fusion rather than by combining
their raw scores, which live on unrelated, untunable scales (see fusion.py's
docstring). The symbol component exists specifically because of something
verified empirically while building this phase: websearch_to_tsquery treats
a dotted qualified name like "Flask.dispatch_request" as a single compound
token that never matches the tsvector's separately-positioned tokens, so
lexical search alone cannot find a chunk by its qualified name -- only an
exact-match WHERE clause can.
"""

import psycopg

from codeqa.graph.traversal import traverse_sql
from codeqa.indexing.embeddings import EmbeddingProvider
from codeqa.retrieval.fusion import (
    extract_identifier_candidates,
    filter_symbol_candidates,
    reciprocal_rank_fusion,
)
from codeqa.retrieval.strategy import RetrievedChunk

# Each component fetches a pool of candidates LARGER than top_k before
# fusion -- standard RRF practice. A chunk ranked #8 by vector search and #2
# by lexical search should survive into the fused top_k even though #8 alone
# wouldn't make a bare top-10 cut from vector search alone. Bounded (not
# unbounded) so query cost stays predictable regardless of top_k.
_CANDIDATE_POOL_MULTIPLIER = 3
_MIN_CANDIDATE_POOL = 20

_VECTOR_RANKING_QUERY = """
    SELECT c.id
      FROM chunks c
     WHERE c.repo_id = %(repo_id)s
     ORDER BY c.embedding <=> %(qvec)s::vector
     LIMIT %(pool_size)s
"""

_LEXICAL_RANKING_QUERY = """
    SELECT c.id
      FROM chunks c
     WHERE c.repo_id = %(repo_id)s
       AND c.tsv @@ websearch_to_tsquery('english', %(query_text)s)
     ORDER BY ts_rank(c.tsv, websearch_to_tsquery('english', %(query_text)s)) DESC
     LIMIT %(pool_size)s
"""

# Qualified-name matches ordered ahead of bare-name matches: "Flask.
# dispatch_request" in the query is a stronger, more specific signal than
# the bare token "dispatch_request" alone, which could belong to any class.
_SYMBOL_RANKING_QUERY = """
    SELECT c.id
      FROM chunks c
     WHERE c.repo_id = %(repo_id)s
       AND (c.qualified_name = ANY(%(dotted)s) OR c.symbol_name = ANY(%(bare)s))
     ORDER BY (CASE WHEN c.qualified_name = ANY(%(dotted)s) THEN 0 ELSE 1 END), c.id
     LIMIT %(pool_size)s
"""

_HYDRATE_QUERY = """
    SELECT c.id, f.path, c.kind, c.symbol_name, c.qualified_name,
           c.start_line, c.end_line, c.content
      FROM chunks c
      JOIN files f ON f.id = c.file_id AND f.repo_id = c.repo_id
     WHERE c.repo_id = %(repo_id)s AND c.id = ANY(%(chunk_ids)s)
"""


def _fetch_ranking(conn: psycopg.Connection, query: str, params: dict) -> list[int]:
    with conn.cursor() as cur:
        cur.execute(query, params)
        return [row[0] for row in cur.fetchall()]


def _hydrate(conn: psycopg.Connection, repo_id: int, chunk_ids: list[int]) -> dict[int, tuple]:
    # ANY('{}') matches nothing in Postgres (verified interactively), but an
    # empty list is also the common case when there's simply nothing to
    # hydrate -- skip the round trip rather than relying on that behavior.
    if not chunk_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(_HYDRATE_QUERY, {"repo_id": repo_id, "chunk_ids": chunk_ids})
        return {row[0]: row for row in cur.fetchall()}


def _to_chunk(row: tuple, score: float, source: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=row[0],
        file_path=row[1],
        kind=row[2],
        symbol_name=row[3],
        qualified_name=row[4],
        start_line=row[5],
        end_line=row[6],
        content=row[7],
        score=score,
        source=source,
    )


def _matching_sources(
    cid: int, vector_set: set[int], lexical_set: set[int], symbol_set: set[int]
) -> list[str]:
    pools = (("vector", vector_set), ("lexical", lexical_set), ("symbol", symbol_set))
    return [name for name, member in pools if cid in member]


def _rank_and_fuse(
    conn: psycopg.Connection,
    repo_id: int,
    query_text: str,
    embedder: EmbeddingProvider,
    top_k: int,
) -> tuple[list[RetrievedChunk], set[int], set[int], set[int]]:
    """The shared core of HybridStrategy.retrieve: runs all three component
    rankings once and fuses them. Returns the fused top_k AND each
    component's full candidate-POOL id set (not just what survived into
    top_k) -- HybridGraphStrategy needs the pools, not just the final
    results, to know what vector/lexical/symbol search already considered
    before deciding what actually required graph expansion to find (see
    HybridGraphStrategy.retrieve). Split out rather than having
    HybridGraphStrategy call HybridStrategy.retrieve() and then re-run
    these same three queries again for that purpose -- same pool_size,
    same query, so re-running would just be duplicate database round trips
    for an answer this function already has.
    """
    pool_size = max(top_k * _CANDIDATE_POOL_MULTIPLIER, _MIN_CANDIDATE_POOL)

    query_vector = embedder.embed([query_text])[0]
    vector_params = {"repo_id": repo_id, "qvec": query_vector, "pool_size": pool_size}
    vector_ranking = _fetch_ranking(conn, _VECTOR_RANKING_QUERY, vector_params)
    lexical_params = {"repo_id": repo_id, "query_text": query_text, "pool_size": pool_size}
    lexical_ranking = _fetch_ranking(conn, _LEXICAL_RANKING_QUERY, lexical_params)

    bare, dotted = filter_symbol_candidates(*extract_identifier_candidates(query_text))
    # Skip the round trip entirely when the query has no identifier-shaped
    # tokens at all -- an empty bare/dotted list matches nothing anyway
    # (verified: ANY('{}') is always false), so this is purely an
    # optimization, not a correctness guard.
    symbol_ranking = (
        _fetch_ranking(
            conn,
            _SYMBOL_RANKING_QUERY,
            {"repo_id": repo_id, "bare": bare, "dotted": dotted, "pool_size": pool_size},
        )
        if bare or dotted
        else []
    )

    fused = reciprocal_rank_fusion([vector_ranking, lexical_ranking, symbol_ranking])
    top_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:top_k]
    rows = _hydrate(conn, repo_id, top_ids)

    vector_set = set(vector_ranking)
    lexical_set = set(lexical_ranking)
    symbol_set = set(symbol_ranking)
    results = []
    for cid in top_ids:
        row = rows.get(cid)
        if row is None:
            continue
        sources = _matching_sources(cid, vector_set, lexical_set, symbol_set)
        results.append(_to_chunk(row, score=fused[cid], source="+".join(sources)))
    return results, vector_set, lexical_set, symbol_set


class HybridStrategy:
    def retrieve(
        self,
        conn: psycopg.Connection,
        repo_id: int,
        query_text: str,
        embedder: EmbeddingProvider,
        top_k: int,
    ) -> list[RetrievedChunk]:
        results, _vector_set, _lexical_set, _symbol_set = _rank_and_fuse(
            conn, repo_id, query_text, embedder, top_k
        )
        return results


class HybridGraphStrategy:
    """HybridStrategy's results, plus their call-graph neighborhood.

    graph_max_depth/graph_max_nodes are constructor parameters, not a
    Settings object, matching build_embedder's convention -- this class
    shouldn't need to know how config is loaded to be tested or reused.
    """

    def __init__(self, graph_max_depth: int, graph_max_nodes: int):
        self._max_depth = graph_max_depth
        self._max_nodes = graph_max_nodes

    def retrieve(
        self,
        conn: psycopg.Connection,
        repo_id: int,
        query_text: str,
        embedder: EmbeddingProvider,
        top_k: int,
    ) -> list[RetrievedChunk]:
        primary, vector_set, lexical_set, symbol_set = _rank_and_fuse(
            conn, repo_id, query_text, embedder, top_k
        )
        # Only true duplicates of what's already being returned are excluded
        # here -- NOT everything any component's candidate pool considered.
        # First attempt at this used the full pool union for exclusion and
        # it was wrong in a worse way than the bug it fixed: verified
        # against the real Flask repo that it silently dropped
        # Flask.dispatch_request from hybrid_graph's results entirely --
        # ranked #10 by raw vector similarity (inside the pool) but pushed
        # out of the fused top_k by chunks that matched multiple
        # components, so it was neither a primary result NOR re-added by
        # expansion. A pool-sourced chunk still deserves to be surfaced;
        # what it doesn't deserve is being mislabeled PURELY "graph" when
        # another mechanism also found it -- that's handled below by
        # appending any pool memberships to its source label instead.
        already_present = {c.chunk_id for c in primary}

        # Expand outward from every primary chunk, both directions. A chunk
        # reachable from more than one primary chunk, or via more than one
        # direction, keeps its SHORTEST depth across all of them -- depth is
        # a proximity signal, and the closest path is the honest one.
        best_depth: dict[int, int] = {}
        for chunk in primary:
            for direction in ("callers", "callees"):
                for node in traverse_sql(
                    conn, repo_id, chunk.chunk_id, direction, self._max_depth, self._max_nodes
                ):
                    if node.chunk_id in already_present:
                        continue
                    if node.chunk_id not in best_depth or node.depth < best_depth[node.chunk_id]:
                        best_depth[node.chunk_id] = node.depth

        # graph_max_nodes bounds each individual traverse_sql call already,
        # but fanning out from several primary chunks can still union past
        # it -- re-apply it here as an overall cap on total expansion,
        # keeping the closest nodes (sorted by depth, then id for
        # determinism) rather than an arbitrary subset.
        by_proximity = sorted(best_depth, key=lambda cid: (best_depth[cid], cid))
        expansion_ids = by_proximity[: self._max_nodes]
        rows = _hydrate(conn, repo_id, expansion_ids)

        # score=0.0 is a sentinel, not a ranking signal. Graph-expanded
        # chunks are unranked structural context reached by a fundamentally
        # different mechanism than RRF-fused primary results -- inventing a
        # depth-decayed score for them would reintroduce exactly the
        # cross-mechanism score comparison RRF exists to avoid (a made-up
        # score here has no principled relationship to a fused RRF score).
        #
        # source is always "graph", plus "+vector"/"+lexical"/"+symbol" for
        # any component whose candidate pool also contained this chunk --
        # it just didn't survive that component's own fusion into top_k.
        # source == "graph" EXACTLY (no plus-suffix) is the precise signal
        # Phase 9's eval needs: found ONLY by walking the call graph, not
        # by any other mechanism at any rank.
        expanded = []
        for cid in expansion_ids:
            row = rows.get(cid)
            if row is None:
                continue
            extra_sources = _matching_sources(cid, vector_set, lexical_set, symbol_set)
            expanded.append(_to_chunk(row, score=0.0, source="+".join(["graph", *extra_sources])))
        return primary + expanded
