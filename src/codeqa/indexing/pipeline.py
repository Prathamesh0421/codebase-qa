"""Full index: walk -> chunk -> embed -> persist.

Embedding is batched across the WHOLE repo, not per file -- measured 3.5x
throughput from batching (see test_embeddings.py), and most files don't have
enough chunks on their own to fill even a modest batch. That means chunking
happens for every file first, and embedding happens once, over every chunk in
the repo, before any of it is written to Postgres.

Per-file failures do not abort the run. A repo can contain one file with an
encoding quirk or a parser edge case tree-sitter chokes on; losing the other
999 files' worth of indexing to that one file would be a worse failure mode
than skipping it and recording why.
"""

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import psycopg

from codeqa.indexing.chunker import Chunk, chunk_file
from codeqa.indexing.embeddings import EmbeddingProvider
from codeqa.indexing.store import (
    IndexStats,
    check_embedder_matches_repo,
    mark_indexed,
    replace_chunks,
    upsert_file,
)
from codeqa.indexing.walker import walk_repo
from codeqa.languages import detect_language


def _blob_sha(content: bytes) -> str:
    """git's own blob hashing: sha1("blob {len}\\0" + content).

    Verified against a real `git hash-object` call before writing this
    (see docs/deep-dive.html) -- computing it independent of an actual git
    repository being present means it works identically for a git-cloned
    checkout and a client-pushed local path (Phase 12's two ingestion modes),
    and it's what Phase 13's incremental re-index compares against to skip
    files git reports as touched but whose content didn't actually change.
    """
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


@dataclass
class _PendingFile:
    path: str
    language: str
    tier: str
    blob_sha: str
    size_bytes: int
    chunks: list[Chunk]


def index_repo(
    conn: psycopg.Connection,
    repo_id: int,
    root: Path,
    embedder: EmbeddingProvider,
) -> IndexStats:
    """Index every recognized source file under root into repo_id.

    Assumes repo_id is already registered (register_repo) with a matching
    embedding_model/embedding_dim -- checked before any work happens, not
    discovered after chunks have already been embedded and thrown away.
    """
    started = time.perf_counter()
    stats = IndexStats()

    check_embedder_matches_repo(conn, repo_id, embedder.model_name, embedder.dimension)

    pending: list[_PendingFile] = []
    for rel_path in walk_repo(root):
        spec = detect_language(rel_path)
        if spec is None:
            stats.files_skipped_no_language += 1
            continue

        try:
            content = (root / rel_path).read_bytes()
            chunks = chunk_file(spec, str(rel_path), content)
            pending.append(
                _PendingFile(
                    path=str(rel_path),
                    language=spec.name,
                    tier=spec.tier,
                    blob_sha=_blob_sha(content),
                    size_bytes=len(content),
                    chunks=chunks,
                )
            )
        except Exception as exc:  # noqa: BLE001 -- one bad file must not abort the run
            stats.files_failed += 1
            stats.errors.append((str(rel_path), str(exc)))

    # One batched embedding call across every chunk in the repo, then sliced
    # back out per file below -- this is the operation the 3.5x measurement
    # in test_embeddings.py exists to justify.
    all_texts = [c.content for pf in pending for c in pf.chunks]
    all_vectors = embedder.embed(all_texts) if all_texts else []

    offset = 0
    for pf in pending:
        n = len(pf.chunks)
        vectors = all_vectors[offset : offset + n]
        offset += n

        try:
            file_id = upsert_file(
                conn, repo_id, pf.path, pf.language, pf.tier, pf.blob_sha, pf.size_bytes
            )
            replace_chunks(conn, repo_id, file_id, pf.chunks, vectors)
            conn.commit()
            stats.files_indexed += 1
            stats.chunks_created += n
        except Exception as exc:  # noqa: BLE001 -- see module docstring
            conn.rollback()
            stats.files_failed += 1
            stats.errors.append((pf.path, str(exc)))

    mark_indexed(conn, repo_id)
    stats.duration_seconds = time.perf_counter() - started
    return stats
