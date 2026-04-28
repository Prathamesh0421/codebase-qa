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
            repo_id = cur.fetchone()[0]
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
        return cur.fetchone()[0]


def replace_chunks(
    conn: psycopg.Connection,
    repo_id: int,
    file_id: int,
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> list[int]:
    """Delete a file's existing chunks and insert the freshly computed set.

    Delete-then-insert rather than a smarter diff: this is the full-index
    path (Phase 4), not incremental re-indexing (Phase 13). It makes
    index_repo safely re-runnable -- re-indexing an unchanged repo replaces
    each file's chunks with an identical set rather than duplicating them --
    which matters for tests and for recovering from a partial prior run.

    Returns the inserted ids, positionally matching `chunks` -- Phase 6's
    call-graph extraction needs a real chunk_id to attribute each call site
    to its caller, and this is the one place that assigns them. Returned as a
    plain list rather than a dict keyed by Chunk: Chunk equality is
    field-based, and nothing here needs to assume two distinct definitions
    can never produce field-identical Chunk values.
    """
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE repo_id = %s AND file_id = %s", (repo_id, file_id))
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
            ids.append(cur.fetchone()[0])
    return ids


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
