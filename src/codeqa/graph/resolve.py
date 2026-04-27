"""Resolve raw callee names to real chunks, repo-wide, and persist edges.

Call resolution is genuinely hard without type inference: `self.foo()`
could mean any class's `foo` method, and this project deliberately does not
attempt type inference or dynamic-dispatch resolution (a stated non-goal,
not an oversight -- see docs/deep-dive.html). What's achievable without it is
a small, honest set of heuristics, tried in order of confidence, that never
guesses when the evidence genuinely doesn't disambiguate:

  1. Zero candidates for the callee name in this repo -> unresolved.
     The common case: builtins, stdlib, third-party library calls.
  2. Exactly one candidate repo-wide -> exact. Nothing to disambiguate.
  3. Multiple candidates, exactly one defined on the caller's own class
     -> approximate. The `self.x()` / `this.x()` pattern.
  4. Multiple candidates, exactly one in the caller's own file
     -> approximate. A local helper shadowing a same-named function
     elsewhere in the repo.
  5. Otherwise -> unresolved. Never pick an arbitrary candidate and label
     it approximate; an unjustified guess is worse than admitting we don't
     know, because it would be indistinguishable from a real finding once
     it's sitting in the database.

Loads the whole repo's symbol index into memory once rather than one query
per call site -- a repo-scoped dict lookup instead of N round trips.
"""

from collections import defaultdict
from dataclasses import dataclass

import psycopg

from codeqa.graph.extraction import CallSite


@dataclass(frozen=True)
class PendingEdge:
    caller_chunk_id: int
    caller_qualified_name: str | None
    caller_file_id: int
    callee_name: str
    call_line: int


@dataclass(frozen=True)
class Candidate:
    chunk_id: int
    qualified_name: str | None
    file_id: int


@dataclass
class ResolveStats:
    exact: int = 0
    approximate: int = 0
    unresolved: int = 0

    @property
    def total(self) -> int:
        return self.exact + self.approximate + self.unresolved


def to_pending_edge(site: CallSite, caller_chunk_id: int, caller_file_id: int) -> PendingEdge:
    """Bridge from extraction.py's file-local CallSite (which knows a Chunk
    object, not a database id) to the repo-wide PendingEdge resolve_and_persist
    consumes. The pipeline calls this once it knows what id replace_chunks
    assigned to the caller chunk.
    """
    return PendingEdge(
        caller_chunk_id=caller_chunk_id,
        caller_qualified_name=site.caller.qualified_name,
        caller_file_id=caller_file_id,
        callee_name=site.callee_name,
        call_line=site.call_line,
    )


def _resolve_one(
    edge: PendingEdge, index: dict[str, list[Candidate]]
) -> tuple[int | None, str]:
    """Pure resolution logic, independent of the database -- see module
    docstring for the ordered heuristic. Returns (callee_chunk_id, resolution)."""
    candidates = index.get(edge.callee_name, [])

    if not candidates:
        return None, "unresolved"
    if len(candidates) == 1:
        return candidates[0].chunk_id, "exact"

    if edge.caller_qualified_name and "." in edge.caller_qualified_name:
        caller_class = edge.caller_qualified_name.rsplit(".", 1)[0]
        same_class = [
            c for c in candidates
            if c.qualified_name and c.qualified_name.rsplit(".", 1)[0] == caller_class
        ]
        if len(same_class) == 1:
            return same_class[0].chunk_id, "approximate"

    same_file = [c for c in candidates if c.file_id == edge.caller_file_id]
    if len(same_file) == 1:
        return same_file[0].chunk_id, "approximate"

    return None, "unresolved"


def _load_symbol_index(conn: psycopg.Connection, repo_id: int) -> dict[str, list[Candidate]]:
    index: dict[str, list[Candidate]] = defaultdict(list)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, symbol_name, qualified_name, file_id
              FROM chunks
             WHERE repo_id = %s AND kind IN ('function', 'method')
            """,
            (repo_id,),
        )
        for chunk_id, symbol_name, qualified_name, file_id in cur.fetchall():
            index[symbol_name].append(Candidate(chunk_id, qualified_name, file_id))
    return index


def resolve_and_persist(
    conn: psycopg.Connection, repo_id: int, edges: list[PendingEdge]
) -> ResolveStats:
    """Resolve every pending edge and replace the repo's call_edges.

    Delete-then-insert, matching store.replace_chunks's reasoning: this is
    the full-index path, and it keeps re-indexing an unchanged repo from
    duplicating edges instead of producing an identical set.
    """
    index = _load_symbol_index(conn, repo_id)
    stats = ResolveStats()

    with conn.cursor() as cur:
        cur.execute("DELETE FROM call_edges WHERE repo_id = %s", (repo_id,))
        for edge in edges:
            callee_chunk_id, resolution = _resolve_one(edge, index)
            cur.execute(
                """
                INSERT INTO call_edges
                    (repo_id, caller_chunk_id, callee_name, callee_chunk_id,
                     resolution, call_line)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    repo_id, edge.caller_chunk_id, edge.callee_name,
                    callee_chunk_id, resolution, edge.call_line,
                ),
            )
            setattr(stats, resolution, getattr(stats, resolution) + 1)
    conn.commit()
    return stats
