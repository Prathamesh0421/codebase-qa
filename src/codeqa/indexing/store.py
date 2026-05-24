"""Persistence: repos/files/chunks rows, against the partitioned schema from
0001_init.sql.

Two things verified empirically before writing this module, both real
psycopg + pgvector gotchas rather than assumptions:

  * pgvector stores vectors as float4 (single precision). A Python float64
    list round-trips through Postgres with precision loss (0.1 becomes
    0.10000000149011612) -- expected, not a bug, but it means comparing
    embeddings for exact equality anywhere in this codebase is wrong. Change
    detection compares content_sha, never embedding values.

  * register_vector() lets a Python list adapt automatically when INSERTing
    into a known vector column, but a vector used as a query PARAMETER (in a
    WHERE or ORDER BY clause) needs an explicit ::vector cast, or Postgres
    infers double precision[] and every vector operator fails to resolve.
"""

from dataclasses import dataclass, field

import psycopg

from codeqa.indexing.chunker import Chunk


class EmbeddingConfigMismatch(RuntimeError):
    """The embedder passed to index_repo disagrees with what the repo was registered with."""


class RepoAlreadyExists(RuntimeError):
    pass


@dataclass
class IndexStats:
    files_indexed: int = 0
    files_skipped_no_language: int = 0
    files_failed: int = 0
    chunks_created: int = 0
    # Populated by call-graph resolution (Phase 6). See graph/resolve.py for
    # what each resolution level actually means.
    call_edges_exact: int = 0
    call_edges_approximate: int = 0
    call_edges_unresolved: int = 0
    duration_seconds: float = 0.0
    errors: list[tuple[str, str]] = field(default_factory=list)
    # Populated only by incremental_index_repo (Phase 13) -- always 0 from
    # the full pipeline, which doesn't have the concept of "unchanged".
    files_unchanged: int = 0
    files_removed: int = 0
    chunks_preserved: int = 0
    chunks_removed: int = 0


def register_repo(
    conn: psycopg.Connection,
    slug: str,
    display_name: str,
    source_kind: str,
    source_ref: str,
    embedding_model: str,
    embedding_dim: int,
    default_branch: str | None = None,
) -> int:
    """Insert the repos row and create its chunks partition together.

    Both happen in one transaction: a repo row without a partition can't
    accept chunks, so the two must commit or fail as a unit (see
    create_repo_partition's own docstring in 0001_init.sql for why there's
    no DEFAULT partition to fall back on).
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO repos
                    (slug, display_name, source_kind, source_ref,
                     default_branch, embedding_model, embedding_dim)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    slug, display_name, source_kind, source_ref,
                    default_branch, embedding_model, embedding_dim,
                ),
            )
            row = cur.fetchone()
            assert row is not None  # INSERT ... RETURNING always yields exactly one row
            repo_id = int(row[0])
            cur.execute("SELECT create_repo_partition(%s)", (repo_id,))
        conn.commit()
        return repo_id
    except psycopg.errors.UniqueViolation as exc:
        conn.rollback()
        raise RepoAlreadyExists(f"repo slug {slug!r} is already registered") from exc


def check_embedder_matches_repo(
    conn: psycopg.Connection, repo_id: int, model: str, dim: int
) -> None:
    """Refuse to index if the embedder disagrees with what the repo was
    registered with.

    Dimension mismatch is caught by Postgres regardless (the column is a
    fixed-width vector(N)), but model-name mismatch at the SAME dimension
    is not: two different 384-dim models produce vectors that silently
    compare as if they meant the same thing. That corruption is invisible
    until search quality degrades, so it's checked here rather than left to
    be discovered later.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT embedding_model, embedding_dim FROM repos WHERE id = %s", (repo_id,)
        )
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"repo_id {repo_id} is not registered")
    recorded_model, recorded_dim = row
    if recorded_model != model or recorded_dim != dim:
        raise EmbeddingConfigMismatch(
            f"repo was registered with model={recorded_model!r} dim={recorded_dim}, "
            f"but the embedder given to index_repo is model={model!r} dim={dim}. "
            f"Mixing embeddings from different models in one column corrupts "
            f"similarity search without raising an error -- refusing to proceed."
        )


def upsert_file(
    conn: psycopg.Connection, repo_id: int, path: str, language: str, tier: str,
    blob_sha: str, size_bytes: int,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO files (repo_id, path, language, tier, blob_sha, size_bytes)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (repo_id, path) DO UPDATE SET
                language = EXCLUDED.language,
                tier = EXCLUDED.tier,
                blob_sha = EXCLUDED.blob_sha,
                size_bytes = EXCLUDED.size_bytes,
                indexed_at = now()
            RETURNING id
            """,
            (repo_id, path, language, tier, blob_sha, size_bytes),
        )
        row = cur.fetchone()
        assert row is not None  # INSERT ... RETURNING always yields exactly one row
        return int(row[0])


def insert_chunks(
    conn: psycopg.Connection,
    repo_id: int,
    file_id: int,
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> list[int]:
    """Insert chunks, touching nothing already there -- the primitive both
    replace_chunks (delete-then-insert-everything) and Phase 13's
    incremental path (insert only the genuinely new/changed ones, leaving
    unrelated existing rows and their chunk_ids, and therefore their
    call_edges, completely alone) build on.

    Returns the inserted ids, positionally matching `chunks` -- callers that
    need to attribute call sites to a caller chunk_id (Phase 6's extraction,
    here and in pipeline.py) need a real id back, and this is the one place
    that assigns them.
    """
    with conn.cursor() as cur:
        ids: list[int] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            cur.execute(
                """
                INSERT INTO chunks
                    (repo_id, file_id, kind, symbol_name, qualified_name,
                     start_line, end_line, content, content_sha, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    repo_id, file_id, chunk.kind, chunk.symbol_name, chunk.qualified_name,
                    chunk.start_line, chunk.end_line, chunk.content, chunk.content_sha, vector,
                ),
            )
            row = cur.fetchone()
            assert row is not None  # INSERT ... RETURNING always yields exactly one row
            ids.append(int(row[0]))
    return ids


def replace_chunks(
    conn: psycopg.Connection,
    repo_id: int,
    file_id: int,
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> list[int]:
    """Delete a file's existing chunks and insert the freshly computed set.

    Delete-then-insert rather than a smarter diff: this is the full-index
    path (Phase 4), not incremental re-indexing (Phase 13, which reuses
    insert_chunks directly instead so a file's untouched chunks never get
    new ids). It makes index_repo safely re-runnable -- re-indexing an
    unchanged repo replaces each file's chunks with an identical set rather
    than duplicating them -- which matters for tests and for recovering from
    a partial prior run.
    """
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE repo_id = %s AND file_id = %s", (repo_id, file_id))
    return insert_chunks(conn, repo_id, file_id, chunks, vectors)


def delete_chunks_by_id(conn: psycopg.Connection, repo_id: int, chunk_ids: list[int]) -> None:
    """Delete specific chunks by id -- cascades to any call_edges referencing
    them as caller or callee (0001_init.sql's ON DELETE CASCADE), which is
    exactly the intended cleanup: an edge pointing at a chunk that no longer
    exists should not exist either.
    """
    if not chunk_ids:
        return
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM chunks WHERE repo_id = %s AND id = ANY(%s)", (repo_id, chunk_ids)
        )


def existing_chunks_by_content_sha(
    conn: psycopg.Connection, repo_id: int, file_id: int
) -> dict[str, list[int]]:
    """content_sha -> chunk_id(s) for a file's currently-stored chunks --
    the lookup Phase 13's incremental diff uses to tell "this exact chunk
    already exists, keep its id" from "this is genuinely new content".
    A list, not a single id: two distinct definitions can be byte-identical
    (e.g. two trivially identical stub methods), and content_sha alone can't
    tell them apart -- the caller pairs them up by consuming this list in
    order rather than assuming uniqueness.
    """
    result: dict[str, list[int]] = {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT content_sha, id FROM chunks WHERE repo_id = %s AND file_id = %s",
            (repo_id, file_id),
        )
        for content_sha, chunk_id in cur.fetchall():
            result.setdefault(content_sha, []).append(chunk_id)
    return result


def mark_indexed(conn: psycopg.Connection, repo_id: int, commit_sha: str | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE repos
               SET status = 'ready',
                   last_indexed_at = now(),
                   last_indexed_sha = COALESCE(%s, last_indexed_sha)
             WHERE id = %s
            """,
            (commit_sha, repo_id),
        )
    conn.commit()


def mark_failed(conn: psycopg.Connection, repo_id: int) -> None:
    """The job-worker counterpart to mark_indexed (Phase 12) -- a repo whose
    only indexing attempt raised should not sit at 'registered' forever,
    silently indistinguishable from "not yet tried".
    """
    with conn.cursor() as cur:
        cur.execute("UPDATE repos SET status = 'failed' WHERE id = %s", (repo_id,))
    conn.commit()
