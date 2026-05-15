"""Incremental re-index: only genuinely changed content gets re-embedded.

The full pipeline (pipeline.py) re-chunks and re-embeds every file on every
run -- correct, and the only thing Phase 4 needed, but wasteful for a repo
that's already indexed and had one file touched. This module makes that
case cheap.

Two diffs, at two different granularities, both keyed on content hashes
rather than trusting a changed-path list from anywhere:

  file-level:  current blob_sha (git's own hash) vs files.blob_sha already
               stored -- decides which files are unchanged (skip entirely),
               changed, added, or removed.
  chunk-level: within a changed file, content_sha (chunker.py's own hash,
               per-definition) vs what that file's existing chunks already
               have -- decides which of the file's own chunks are actually
               new content (need embedding) versus byte-identical to
               before (keep the existing chunk_id untouched, and therefore
               its call_edges untouched too).

Deliberately NOT diffing against a specific old git commit via `git diff
old_sha new_sha`: Phase 12's clones are --depth 1, so the old commit may no
longer be reachable to diff against by the time a repo is re-indexed. Blob
SHA comparison against what's stored in Postgres needs no git history at
all -- only "what does this file look like now", which any current checkout
already has, shallow or not.

The call graph is NOT diffed incrementally -- resolve_and_persist (Phase 6,
untouched here) already deletes and rebuilds a repo's whole call_edges set
every time it runs, so every currently-present file's call sites are
re-extracted and re-resolved on every incremental run too. That's the
honest cost of this approach: only embedding is truly incremental here;
graph rebuild cost still scales with repo size, not with how much changed.
Accepted because tree-sitter parsing is cheap relative to embedding (Phase
4's own measurement) -- and it's what makes "what happens to call edges
pointing at deleted chunks" a non-question: there's never a dangling edge,
because the edge table is always rebuilt fresh from the current, correct
set of chunks, looked up fresh from the database rather than tracked
through this function's own bookkeeping.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import psycopg

from codeqa.graph.extraction import extract_call_sites
from codeqa.graph.resolve import PendingEdge, resolve_and_persist, to_pending_edge
from codeqa.indexing.chunker import Chunk, chunk_file
from codeqa.indexing.embeddings import EmbeddingProvider
from codeqa.indexing.pipeline import blob_sha
from codeqa.indexing.store import (
    IndexStats,
    check_embedder_matches_repo,
    delete_chunks_by_id,
    existing_chunks_by_content_sha,
    insert_chunks,
    mark_indexed,
    upsert_file,
)
from codeqa.indexing.walker import walk_repo
from codeqa.languages import LanguageSpec, detect_language


@dataclass
class _FileRef:
    file_id: int
    spec: LanguageSpec


def _existing_files(conn: psycopg.Connection, repo_id: int) -> dict[str, tuple[int, str]]:
    with conn.cursor() as cur:
        cur.execute("SELECT path, id, blob_sha FROM files WHERE repo_id = %s", (repo_id,))
        return {path: (file_id, sha) for path, file_id, sha in cur.fetchall()}


def _diff_file_chunks(
    old_by_sha: dict[str, list[int]], new_chunks: list[Chunk]
) -> tuple[list[int], list[Chunk]]:
    """Pair a file's new chunk list against its existing (content_sha ->
    chunk_id list) map, positionally rather than by a plain dict lookup --
    two definitions can be byte-identical (content_sha has no idea about
    symbol names), so consuming each hash bucket front-to-back is what
    keeps duplicates from all mapping onto the same existing id.

    Returns (removed_chunk_ids, added_chunks) -- what's genuinely new
    content (needs embedding) and what no longer exists in the new set
    (needs deleting). Chunks that matched an existing id aren't returned at
    all: nothing needs to happen to them.
    """
    remaining = {sha: list(ids) for sha, ids in old_by_sha.items()}
    added: list[Chunk] = []

    for chunk in new_chunks:
        bucket = remaining.get(chunk.content_sha)
        if bucket:
            bucket.pop(0)
        else:
            added.append(chunk)

    removed = [chunk_id for bucket in remaining.values() for chunk_id in bucket]
    return removed, added


def _map_chunks_to_ids(chunks: list[Chunk], by_sha: dict[str, list[int]]) -> dict[int, int]:
    """id(chunk) -> chunk_id for a file's CURRENT chunk list against its
    CURRENT database state (queried fresh, after any mutations) -- the
    single source of truth the call-graph rebuild attributes every call
    site's caller against, for both touched and untouched files alike.
    """
    remaining = {sha: list(ids) for sha, ids in by_sha.items()}
    result: dict[int, int] = {}
    for chunk in chunks:
        bucket = remaining.get(chunk.content_sha)
        if bucket:
            result[id(chunk)] = bucket.pop(0)
    return result


def incremental_index_repo(
    conn: psycopg.Connection,
    repo_id: int,
    root: Path,
    embedder: EmbeddingProvider,
    commit_sha: str | None = None,
) -> IndexStats:
    """Re-index repo_id against root, embedding only content that's
    genuinely new since the last run. Assumes repo_id was already indexed
    at least once (register_repo + a prior index_repo call) -- files.blob_sha
    rows are what this diffs against, and there are none on a repo's first
    index, which is exactly why the first index always goes through the
    full pipeline instead.
    """
    started = time.perf_counter()
    stats = IndexStats()

    check_embedder_matches_repo(conn, repo_id, embedder.model_name, embedder.dimension)

    existing = _existing_files(conn, repo_id)
    all_files: dict[str, _FileRef] = {}  # every currently-present file, touched or not
    pending_new_chunks: list[tuple[int, list[Chunk]]] = []  # (file_id, added chunks)
    seen_paths: set[str] = set()

    for rel_path in walk_repo(root):
        spec = detect_language(rel_path)
        if spec is None:
            stats.files_skipped_no_language += 1
            continue

        path_str = str(rel_path)
        seen_paths.add(path_str)

        try:
            content = (root / rel_path).read_bytes()
            current_sha = blob_sha(content)
            prior = existing.get(path_str)

            if prior is not None and prior[1] == current_sha:
                stats.files_unchanged += 1
                all_files[path_str] = _FileRef(file_id=prior[0], spec=spec)
                continue

            chunks = chunk_file(spec, path_str, content)
            if prior is None:
                file_id = upsert_file(
                    conn, repo_id, path_str, spec.name, spec.tier, current_sha, len(content)
                )
                added_chunks = chunks
            else:
                file_id, _old_sha = prior
                old_by_sha = existing_chunks_by_content_sha(conn, repo_id, file_id)
                removed_ids, added_chunks = _diff_file_chunks(old_by_sha, chunks)
                delete_chunks_by_id(conn, repo_id, removed_ids)
                stats.chunks_removed += len(removed_ids)
                upsert_file(
                    conn, repo_id, path_str, spec.name, spec.tier, current_sha, len(content)
                )

            all_files[path_str] = _FileRef(file_id=file_id, spec=spec)
            stats.chunks_preserved += len(chunks) - len(added_chunks)
            if added_chunks:
                pending_new_chunks.append((file_id, added_chunks))
            stats.files_indexed += 1
        except Exception as exc:  # noqa: BLE001 -- one bad file must not abort the run
            stats.files_failed += 1
            stats.errors.append((path_str, str(exc)))

    for path, (file_id, _sha) in existing.items():
        if path not in seen_paths:
            # Cascades to chunks, and from there to call_edges -- one
            # delete, the schema does the rest.
            with conn.cursor() as cur:
                cur.execute("DELETE FROM files WHERE repo_id = %s AND id = %s", (repo_id, file_id))
            stats.files_removed += 1

    # One batched embedding call across every genuinely new chunk across
    # every touched file -- same batching win pipeline.py's full index
    # gets, just over a much smaller set.
    all_new_texts = [c.content for _fid, chunks in pending_new_chunks for c in chunks]
    all_new_vectors = embedder.embed(all_new_texts) if all_new_texts else []

    offset = 0
    for file_id, added_chunks in pending_new_chunks:
        n = len(added_chunks)
        vectors = all_new_vectors[offset : offset + n]
        offset += n
        insert_chunks(conn, repo_id, file_id, added_chunks, vectors)
        stats.chunks_created += n
    conn.commit()

    # Call graph: re-extract from EVERY currently-present file, touched or
    # not, so resolve_and_persist's full rebuild has a complete, correct
    # picture -- see the module docstring for why this isn't itself
    # incremental. Chunk ids are looked up fresh from the database (now
    # reflecting every insert/delete above), not carried through this
    # function's own bookkeeping, so this is correct regardless of which
    # branch above a given file went through.
    pending_edges: list[PendingEdge] = []
    for path, ref in all_files.items():
        content = (root / path).read_bytes()
        chunks = chunk_file(ref.spec, path, content)
        by_sha = existing_chunks_by_content_sha(conn, repo_id, ref.file_id)
        chunk_id_by_identity = _map_chunks_to_ids(chunks, by_sha)

        sites = extract_call_sites(ref.spec, content, chunks)
        for site in sites:
            caller_id = chunk_id_by_identity.get(id(site.caller))
            if caller_id is None:
                continue
            pending_edges.append(to_pending_edge(site, caller_id, ref.file_id))

    graph_stats = resolve_and_persist(conn, repo_id, pending_edges)
    stats.call_edges_exact = graph_stats.exact
    stats.call_edges_approximate = graph_stats.approximate
    stats.call_edges_unresolved = graph_stats.unresolved

    mark_indexed(conn, repo_id, commit_sha)
    stats.duration_seconds = time.perf_counter() - started
    return stats
