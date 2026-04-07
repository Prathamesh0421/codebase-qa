-- 0001_init.sql
--
-- Core schema for the codebase Q&A system.
--
-- Two conventions that the rest of the system depends on:
--
--   1. repo_id is on every retrievable row (chunks, symbols, call_edges).
--      Every retrieval path filters on it. Cross-repo contamination shows up
--      as a retrieval-quality regression, not as an obvious bug, so the
--      scoping lives in the schema rather than in caller discipline.
--
--   2. ${EMBEDDING_DIM} is substituted by the migration runner from config.
--      pgvector requires a fixed dimension at DDL time to build an index, but
--      the embedding model is configurable. repos.embedding_model and
--      repos.embedding_dim record what a repo was actually indexed with, so a
--      model swap is detected at query time and forces a re-index instead of
--      silently comparing vectors from different models.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ---------------------------------------------------------------- repos

CREATE TYPE repo_source_kind AS ENUM ('git_url', 'local_path');
CREATE TYPE repo_status AS ENUM ('registered', 'indexing', 'ready', 'failed');

CREATE TABLE repos (
    id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    slug             TEXT             NOT NULL UNIQUE,
    display_name     TEXT             NOT NULL,
    source_kind      repo_source_kind NOT NULL,
    -- git remote URL, or an opaque client-supplied identifier for local pushes
    source_ref       TEXT             NOT NULL,
    default_branch   TEXT,
    -- Drives incremental re-indexing: on the next run we diff against this SHA
    -- and re-embed only the files git reports as changed. NULL => never indexed.
    last_indexed_sha TEXT,
    last_indexed_at  TIMESTAMPTZ,
    embedding_model  TEXT             NOT NULL,
    embedding_dim    INTEGER          NOT NULL,
    status           repo_status      NOT NULL DEFAULT 'registered',
    created_at       TIMESTAMPTZ      NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ      NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------- files

-- Language support is tiered by what the grammar's tags.scm can express:
--   tier1 - definitions + @reference.call  => real call edges
--   tier2 - definitions only               => chunks + symbol-name call approximation
--   tier3 - no tag queries                 => chunks + symbol index only
CREATE TYPE language_tier AS ENUM ('tier1', 'tier2', 'tier3');

CREATE TABLE files (
    id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    repo_id    BIGINT        NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    path       TEXT          NOT NULL,
    language   TEXT          NOT NULL,
    tier       language_tier NOT NULL,
    -- git blob SHA; lets an incremental run skip files whose content is
    -- unchanged even when git reports the path as touched.
    blob_sha   TEXT          NOT NULL,
    size_bytes INTEGER       NOT NULL,
    indexed_at TIMESTAMPTZ   NOT NULL DEFAULT now(),
    UNIQUE (repo_id, path)
);

CREATE INDEX files_repo_lang_idx ON files (repo_id, language);

-- ---------------------------------------------------------------- chunks

CREATE TYPE chunk_kind AS ENUM ('function', 'method', 'class', 'module');

-- PARTITIONED BY repo_id. This is not premature optimization -- it is a
-- correctness fix, measured before it was written:
--
--   An HNSW index is approximate. Postgres searches the index first and applies
--   WHERE repo_id = ... afterwards, so when the nearest neighbours all belong to
--   other repos they are found, then filtered away, and the query returns fewer
--   rows than requested -- silently. Measured on 30.5k vectors across 3 repos,
--   asking for the 10 nearest chunks in the smallest repo:
--
--     unpartitioned, ef_search=40 (default)          ->  0 rows   (truth: 10)
--     unpartitioned, iterative_scan=relaxed_order    ->  0 rows   (budget exhausted)
--     unpartitioned, + scan_mem_multiplier=4         -> 10 rows   (tuned, fragile)
--     unpartitioned, ef_search=1000                  -> 10 rows   (brute force, slow)
--     PARTITIONED, ef_search=40 (default)            -> 10 rows   (correct by construction)
--
--   With one partition per repo, `WHERE repo_id = N` prunes to a single
--   partition and every row in it already satisfies the filter, so there is
--   nothing left to filter away and no recall to lose. The other approaches
--   make the failure less likely by spending more work; partitioning removes it.
--
-- Cost: a partition must be created when a repo is registered (see
-- create_repo_partition below), and the primary key must include the partition
-- key -- which is why chunk references elsewhere are composite (repo_id, id).
-- That is a feature: it makes a cross-repo edge a foreign-key violation rather
-- than a silent retrieval-quality bug.
CREATE TABLE chunks (
    id          BIGINT GENERATED ALWAYS AS IDENTITY,
    repo_id     BIGINT     NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    file_id     BIGINT     NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    kind        chunk_kind NOT NULL,
    -- Bare symbol name ("dispatch_request") plus the dotted path when the
    -- grammar gives us enough context ("Flask.dispatch_request"). Callee
    -- resolution matches on both, preferring the qualified form.
    symbol_name TEXT       NOT NULL,
    qualified_name TEXT,
    -- 1-indexed and inclusive, matching what editors and citations display.
    start_line  INTEGER    NOT NULL,
    end_line    INTEGER    NOT NULL,
    content     TEXT       NOT NULL,
    content_sha TEXT       NOT NULL,
    embedding   vector(${EMBEDDING_DIM}),
    -- Lexical half of hybrid retrieval, fused with vector results via RRF.
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (repo_id, id),
    CONSTRAINT chunks_line_span_valid CHECK (end_line >= start_line AND start_line >= 1)
) PARTITION BY LIST (repo_id);

-- Declared on the parent. Postgres propagates it to every existing partition
-- and automatically creates a matching index on partitions added later, so
-- repo registration does not have to know this index exists.
CREATE INDEX chunks_embedding_idx ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Creates the partition backing one repo. Called in the same transaction that
-- registers the repo: a repo without its partition cannot accept chunks, so
-- the two must commit together or not at all.
--
-- There is deliberately no DEFAULT partition. A default would let chunks for an
-- unregistered repo land somewhere queryable but unprunable -- exactly the
-- silent degradation this design exists to prevent -- and adopting those rows
-- into a real partition later requires a full scan under lock. Failing loudly
-- at insert time is the better trade.
CREATE FUNCTION create_repo_partition(p_repo_id BIGINT) RETURNS void AS $$
BEGIN
    EXECUTE format(
        'CREATE TABLE IF NOT EXISTS chunks_repo_%s PARTITION OF chunks FOR VALUES IN (%s)',
        p_repo_id, p_repo_id
    );
END;
$$ LANGUAGE plpgsql;

-- Mirrors create_repo_partition. Dropping the partition is O(1) metadata work,
-- where DELETE FROM chunks WHERE repo_id = N would rewrite and re-index the
-- heap. That is the point of doing it this way.
--
-- The order below is not arbitrary, and it is the one thing about this schema
-- that is easy to get wrong:
--
--   symbols and call_edges hold composite FKs into chunks. Postgres validates
--   those on DETACH -- it refuses to strand referencing rows -- so the
--   dependents must go first. DROP ... CASCADE would "work" by dropping the
--   FK constraints themselves, which are defined on the parent tables and
--   shared by every repo. That would silently disarm the cross-repo guarantee
--   for the whole database in order to delete one repo. Never use CASCADE here.
--
-- The dependent DELETEs are indexed on repo_id and touch far less data than
-- the chunk rows they protect, so the cheap-teardown property survives.
CREATE FUNCTION drop_repo_partition(p_repo_id BIGINT) RETURNS void AS $$
BEGIN
    DELETE FROM call_edges WHERE repo_id = p_repo_id;
    DELETE FROM symbols    WHERE repo_id = p_repo_id;

    IF to_regclass(format('chunks_repo_%s', p_repo_id)) IS NOT NULL THEN
        EXECUTE format('ALTER TABLE chunks DETACH PARTITION chunks_repo_%s', p_repo_id);
        EXECUTE format('DROP TABLE chunks_repo_%s', p_repo_id);
    END IF;
END;
$$ LANGUAGE plpgsql;

CREATE INDEX chunks_tsv_idx     ON chunks USING gin (tsv);
CREATE INDEX chunks_repo_idx    ON chunks (repo_id);
CREATE INDEX chunks_file_idx    ON chunks (file_id);
CREATE INDEX chunks_symbol_idx  ON chunks (repo_id, symbol_name);

-- ---------------------------------------------------------------- symbols

-- Exact-match lookup table: "where is dispatch_request defined?" resolves
-- without touching the vector index at all.
-- The FK is composite because chunks' primary key is (repo_id, id). The useful
-- side effect: a symbol row physically cannot point at a chunk in a different
-- repo. The scoping invariant is enforced by the database, not by remembering
-- to write the right WHERE clause.
CREATE TABLE symbols (
    id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    repo_id        BIGINT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    chunk_id       BIGINT NOT NULL,
    name           TEXT   NOT NULL,
    qualified_name TEXT,
    kind           chunk_kind NOT NULL,
    FOREIGN KEY (repo_id, chunk_id) REFERENCES chunks (repo_id, id) ON DELETE CASCADE
);

CREATE INDEX symbols_lookup_idx ON symbols (repo_id, name);
CREATE INDEX symbols_qualified_idx ON symbols (repo_id, qualified_name);
-- Fuzzy symbol lookup for near-miss questions ("dispatchRequest").
CREATE INDEX symbols_trgm_idx ON symbols USING gin (name gin_trgm_ops);

-- ---------------------------------------------------------------- call graph

-- How confident we are that caller -> callee is a real edge. Surfaced in
-- answers: an approximate edge is never presented as a certain call path.
CREATE TYPE edge_resolution AS ENUM ('exact', 'approximate', 'unresolved');

-- Both endpoints use composite FKs into chunks, so an edge cannot span two
-- repos. Graph traversal therefore cannot wander out of the repo it started in
-- even if a query forgets to scope -- the edges to leave simply do not exist.
CREATE TABLE call_edges (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    repo_id         BIGINT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    caller_chunk_id BIGINT NOT NULL,
    -- The raw callee name as written at the call site. Retained even when
    -- resolution succeeds, so approximations stay auditable.
    callee_name     TEXT   NOT NULL,
    -- NULL when the callee is external (stdlib, third-party) or unresolvable.
    callee_chunk_id BIGINT,
    resolution      edge_resolution NOT NULL,
    call_line       INTEGER,
    FOREIGN KEY (repo_id, caller_chunk_id) REFERENCES chunks (repo_id, id) ON DELETE CASCADE,
    FOREIGN KEY (repo_id, callee_chunk_id) REFERENCES chunks (repo_id, id) ON DELETE CASCADE,
    -- An edge is resolved iff it names a chunk. Keeps the two columns from
    -- drifting into states the traversal query would have to defend against.
    CONSTRAINT call_edges_resolution_consistent CHECK (
        (resolution = 'unresolved' AND callee_chunk_id IS NULL) OR
        (resolution IN ('exact', 'approximate') AND callee_chunk_id IS NOT NULL)
    )
);

-- Traversal is a recursive CTE in both directions, so index both endpoints.
CREATE INDEX call_edges_caller_idx ON call_edges (repo_id, caller_chunk_id)
    WHERE callee_chunk_id IS NOT NULL;
CREATE INDEX call_edges_callee_idx ON call_edges (repo_id, callee_chunk_id)
    WHERE callee_chunk_id IS NOT NULL;
CREATE INDEX call_edges_name_idx   ON call_edges (repo_id, callee_name);

-- ---------------------------------------------------------------- jobs

CREATE TYPE job_kind   AS ENUM ('full', 'incremental');
CREATE TYPE job_status AS ENUM ('queued', 'running', 'succeeded', 'failed', 'cancelled');

-- Indexing takes minutes, so it cannot run inside a request. A worker polls
-- this table; heartbeat_at lets a supervisor reclaim jobs whose worker died
-- mid-index rather than leaving them 'running' forever.
CREATE TABLE index_jobs (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    repo_id      BIGINT     NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    kind         job_kind   NOT NULL,
    status       job_status NOT NULL DEFAULT 'queued',
    attempts     INTEGER    NOT NULL DEFAULT 0,
    error        TEXT,
    stats        JSONB      NOT NULL DEFAULT '{}'::jsonb,
    heartbeat_at TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at   TIMESTAMPTZ,
    finished_at  TIMESTAMPTZ
);

CREATE INDEX index_jobs_queue_idx ON index_jobs (status, created_at)
    WHERE status IN ('queued', 'running');
CREATE INDEX index_jobs_repo_idx  ON index_jobs (repo_id, created_at DESC);

-- At most one live job per repo. Two concurrent indexers on one repo would
-- interleave upserts and delete each other's chunks, and the loser would leave
-- the repo half-indexed with a last_indexed_sha that claims otherwise.
-- Enforced as a partial unique index so a second enqueue fails fast at the
-- database rather than depending on an advisory lock the caller might skip.
CREATE UNIQUE INDEX index_jobs_one_live_per_repo ON index_jobs (repo_id)
    WHERE status IN ('queued', 'running');

-- ---------------------------------------------------------------- auth

-- Keys are stored hashed; the plaintext is shown once at creation and never
-- persisted. prefix is a non-secret display/lookup aid ("cq_live_a1b2...").
CREATE TABLE api_keys (
    id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name           TEXT   NOT NULL,
    key_prefix     TEXT   NOT NULL,
    key_hash       TEXT   NOT NULL UNIQUE,
    rate_limit_rpm INTEGER NOT NULL DEFAULT 60,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at   TIMESTAMPTZ,
    revoked_at     TIMESTAMPTZ
);

CREATE INDEX api_keys_prefix_idx ON api_keys (key_prefix);
