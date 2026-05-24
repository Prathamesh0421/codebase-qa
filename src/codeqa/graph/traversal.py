"""Walk the call graph outward from a chunk, safely, in one of two ways.

Bounded depth and cycle-safety both matter here: the call graph is real
code, and real code recurses. "Bounded" means two independent limits --
graph_max_depth (how many hops) and graph_max_nodes (how many results) --
because a shallow traversal can still explode in width at a popular utility
function.

Two implementations, deliberately sharing no traversal logic between them:

  traverse_sql       -- production. A single WITH RECURSIVE ... CYCLE query.
                        Cycle detection is Postgres's own (the CYCLE clause),
                        not a visited-set we maintain by hand.
  traverse_networkx  -- test oracle. Loads the same edges into an in-memory
                        graph and calls networkx's own
                        single_source_shortest_path_length -- its BFS, not
                        a hand-rolled loop dressed up as a second
                        implementation. Used only in tests -- if it agrees
                        with traverse_sql, that's real evidence the SQL is
                        correct, not two paths through one bug.

Both return TraversalNode(chunk_id, depth), excluding the start chunk
itself -- callers already know what they started from; traversal's job is
what's reachable from it, not restating the input.

networkx is imported lazily, inside the two functions that need it, not at
module level -- it's a dev-only dependency (pyproject.toml), and this module
also holds traverse_sql, which production code (Phase 10's trace agent)
imports. A module-level import here would make networkx a hard production
requirement for a function it never calls, the same reasoning behind
LocalEmbedder's lazy sentence-transformers import in indexing/embeddings.py.
`from __future__ import annotations` keeps the nx.DiGraph type hint below
from being evaluated at import time, so it's safe even when networkx isn't
installed at all (the deployed image).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import psycopg

if TYPE_CHECKING:
    import networkx as nx

Direction = Literal["callers", "callees"]


@dataclass(frozen=True)
class TraversalNode:
    chunk_id: int
    depth: int


# Two things were wrong here on the first pass, both caught by tests before
# this was trusted, not by inspection:
#
# 1. CYCLE only detects a node repeating WITHIN one path -- it does not
#    deduplicate a node reached via two INDEPENDENT paths at the same or
#    different depths (a diamond: a->b->d and a->c->d). That's inherent to
#    how SQL recursive CTEs work (they enumerate simple paths, not a single
#    BFS tree), not a bug in this query specifically. DISTINCT ON (chunk_id)
#    ... ORDER BY chunk_id, depth picks each node's minimum-depth occurrence.
#
# 2. The callees recursive term selects e.callee_chunk_id, which is NULL for
#    an unresolved edge -- an unfiltered recursion tries to recurse into
#    NULL as if it were a real chunk id. Needs an explicit
#    callee_chunk_id IS NOT NULL (the callers direction doesn't need the
#    mirror check: its join condition is `e.callee_chunk_id = r.chunk_id`,
#    and NULL = x is never true in SQL, so unresolved edges are naturally
#    excluded there without an explicit filter).
#
# Consequence of fix 1: DISTINCT ON requires ORDER BY to start with
# chunk_id, not depth -- verified via EXPLAIN ANALYZE that this defeats
# Postgres's early-termination on a wide fan-out (materializes fully before
# sorting; measured 5.4ms vs 0.03ms on a 5000-edge synthetic single-node
# fan-out). Accepted rather than worked around: correctness (never
# double-counting a reachable node) matters more than an optimization that
# only pays off on a pathological width unlikely at this project's default
# max_depth of 2, and graph_max_depth remains the primary bound on database
# work regardless. No LIMIT in the SQL at all now -- truncation to
# max_nodes happens in Python, after sorting by (depth, chunk_id), so a
# truncated result keeps the CLOSEST nodes rather than an arbitrary
# chunk_id-ordered subset (which is what LIMIT after this ORDER BY would
# have given instead).
_TRAVERSE_CALLEES = """
    WITH RECURSIVE reachable(chunk_id, depth) AS (
        SELECT %(start_chunk_id)s::bigint, 0
      UNION ALL
        SELECT e.callee_chunk_id, r.depth + 1
          FROM reachable r
          JOIN call_edges e
            ON e.caller_chunk_id = r.chunk_id AND e.repo_id = %(repo_id)s
         WHERE r.depth < %(max_depth)s AND e.callee_chunk_id IS NOT NULL
    ) CYCLE chunk_id SET is_cycle USING path
    SELECT DISTINCT ON (chunk_id) chunk_id, depth
      FROM reachable
     WHERE NOT is_cycle AND depth > 0
     ORDER BY chunk_id, depth
"""

_TRAVERSE_CALLERS = """
    WITH RECURSIVE reachable(chunk_id, depth) AS (
        SELECT %(start_chunk_id)s::bigint, 0
      UNION ALL
        SELECT e.caller_chunk_id, r.depth + 1
          FROM reachable r
          JOIN call_edges e
            ON e.callee_chunk_id = r.chunk_id AND e.repo_id = %(repo_id)s
         WHERE r.depth < %(max_depth)s
    ) CYCLE chunk_id SET is_cycle USING path
    SELECT DISTINCT ON (chunk_id) chunk_id, depth
      FROM reachable
     WHERE NOT is_cycle AND depth > 0
     ORDER BY chunk_id, depth
"""


def traverse_sql(
    conn: psycopg.Connection,
    repo_id: int,
    start_chunk_id: int,
    direction: Direction,
    max_depth: int,
    max_nodes: int,
) -> list[TraversalNode]:
    query = _TRAVERSE_CALLEES if direction == "callees" else _TRAVERSE_CALLERS
    with conn.cursor() as cur:
        cur.execute(
            query,
            {"start_chunk_id": start_chunk_id, "repo_id": repo_id, "max_depth": max_depth},
        )
        rows = cur.fetchall()
    nodes = [TraversalNode(chunk_id=r[0], depth=r[1]) for r in rows]
    nodes.sort(key=lambda n: (n.depth, n.chunk_id))
    return nodes[:max_nodes]


def bfs_over_graph(
    graph: nx.DiGraph[int],
    start_chunk_id: int,
    direction: Direction,
    max_depth: int,
) -> list[TraversalNode]:
    """Pure: bounded reachability over an already-built networkx graph, no
    database access. Split out from traverse_networkx below so it's testable
    against a directly-constructed nx.DiGraph fixture, the same separation
    Phase 6 used for _resolve_one.

    single_source_shortest_path_length is networkx's own BFS, not a
    hand-rolled loop -- the point of a test oracle is an independently
    implemented traversal, not a second copy of the same algorithm with
    different variable names. G.reverse(copy=False) is a view, not a copy,
    so "callers" costs no extra memory over "callees".
    """
    import networkx as nx  # noqa: PLC0415 -- lazy: see module docstring

    g = graph if direction == "callees" else graph.reverse(copy=False)
    try:
        distances = nx.single_source_shortest_path_length(g, start_chunk_id, cutoff=max_depth)
    except nx.NodeNotFound:
        # start_chunk_id has no edges at all in either direction -- nothing
        # reachable, same as a chunk with zero rows in call_edges.
        return []
    return [TraversalNode(chunk_id=cid, depth=d) for cid, d in distances.items() if d > 0]


def traverse_networkx(
    conn: psycopg.Connection,
    repo_id: int,
    start_chunk_id: int,
    direction: Direction,
    max_depth: int,
    max_nodes: int,
) -> list[TraversalNode]:
    """Test oracle only -- never called from production code. Builds the
    same directed graph traverse_sql reads (caller_chunk_id -> callee_chunk_id
    edges, unresolved edges excluded since callee_chunk_id is NULL) and
    delegates to the pure BFS above.
    """
    import networkx as nx  # noqa: PLC0415 -- lazy: see module docstring

    graph: nx.DiGraph[int] = nx.DiGraph()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT caller_chunk_id, callee_chunk_id FROM call_edges "
            "WHERE repo_id = %s AND callee_chunk_id IS NOT NULL",
            (repo_id,),
        )
        graph.add_edges_from(cur.fetchall())

    nodes = bfs_over_graph(graph, start_chunk_id, direction, max_depth)
    nodes.sort(key=lambda n: (n.depth, n.chunk_id))
    return nodes[:max_nodes]
