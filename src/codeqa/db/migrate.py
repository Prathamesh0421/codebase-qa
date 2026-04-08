"""Forward-only SQL migration runner.

Plain numbered .sql files rather than Alembic: there is no ORM here to
autogenerate from, and the interesting DDL (partitioned tables, HNSW indexes,
plpgsql helpers) is hand-written anyway. Alembic would add a dependency and a
layer of indirection over SQL we would still be writing by hand.

Four things this does beyond executing files in order:

  * Records what has been applied, so runs are idempotent.
  * Detects drift -- a migration edited after it was applied. Silent divergence
    between the file and the deployed schema is the classic way a migration
    system stops being trustworthy.
  * Takes an advisory lock, so two app instances booting at once cannot race.
  * Substitutes ${EMBEDDING_DIM} and then verifies the dimension actually
    deployed matches configuration. See check_embedding_dim for why that
    verification is not optional.
"""

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

# Arbitrary but fixed. Any process running migrations takes this lock, so
# concurrent boots serialize instead of racing on CREATE TABLE.
ADVISORY_LOCK_KEY = 0x0C0DE0A1

_FILENAME_RE = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")

BOOTSTRAP_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version      INTEGER     PRIMARY KEY,
    name         TEXT        NOT NULL,
    -- sha256 of the raw file, before ${...} substitution. Substituted values
    -- come from config and legitimately differ between environments; the file
    -- itself must not.
    checksum     TEXT        NOT NULL,
    applied_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    duration_ms  INTEGER     NOT NULL
);
"""


class MigrationError(RuntimeError):
    """Base class for migration failures."""


class MigrationDriftError(MigrationError):
    """An already-applied migration file has been modified."""


class EmbeddingDimMismatch(MigrationError):
    """Deployed vector dimension disagrees with configuration."""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    path: Path
    template: str
    checksum: str

    @property
    def label(self) -> str:
        return f"{self.version:04d}_{self.name}"


def discover(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    """Load migrations from disk, ordered by version.

    Rejects duplicate version numbers outright. Two files claiming 0003 apply
    in an order that depends on the filesystem, which is exactly the kind of
    thing that works locally and diverges in CI.
    """
    migrations: dict[int, Migration] = {}

    for path in sorted(directory.glob("*.sql")):
        match = _FILENAME_RE.match(path.name)
        if not match:
            raise MigrationError(
                f"{path.name!r} does not match the NNNN_name.sql convention"
            )
        version = int(match.group(1))
        if version in migrations:
            raise MigrationError(
                f"duplicate migration version {version:04d}: "
                f"{migrations[version].path.name} and {path.name}"
            )
        raw = path.read_bytes()
        migrations[version] = Migration(
            version=version,
            name=match.group(2),
            path=path,
            template=raw.decode("utf-8"),
            checksum=hashlib.sha256(raw).hexdigest(),
        )

    return [migrations[v] for v in sorted(migrations)]


def render(migration: Migration, params: dict[str, object]) -> str:
    """Substitute ${NAME} placeholders, failing loudly on anything unresolved.

    A missing substitution must never reach the server: ${EMBEDDING_DIM} left
    verbatim in a CREATE TABLE is a syntax error at best, and at worst -- in a
    context where it parses -- silently wrong DDL.
    """
    sql = migration.template
    for key, value in params.items():
        sql = sql.replace(f"${{{key}}}", str(value))

    if leftover := set(re.findall(r"\$\{([A-Z_]+)\}", sql)):
        raise MigrationError(
            f"{migration.label}: unsubstituted placeholders {sorted(leftover)}"
        )
    return sql


def _applied(conn: psycopg.Connection) -> dict[int, tuple[str, str]]:
    with conn.cursor() as cur:
        cur.execute("SELECT version, name, checksum FROM schema_migrations")
        return {row[0]: (row[1], row[2]) for row in cur.fetchall()}


def _check_drift(migrations: list[Migration], applied: dict[int, tuple[str, str]]) -> None:
    for migration in migrations:
        record = applied.get(migration.version)
        if record is None:
            continue
        _, checksum = record
        if checksum != migration.checksum:
            raise MigrationDriftError(
                f"{migration.label} was modified after being applied "
                f"(recorded {checksum[:12]}, file {migration.checksum[:12]}). "
                f"Migrations are immutable once applied -- add a new one instead."
            )

    # A version recorded in the database with no corresponding file usually
    # means someone checked out an older revision. Applying nothing would look
    # like success while the code expects a schema it cannot see.
    known = {m.version for m in migrations}
    if orphans := sorted(set(applied) - known):
        raise MigrationDriftError(
            f"database has migrations with no file on disk: {orphans}. "
            f"The checkout is probably older than the database."
        )


def deployed_embedding_dim(conn: psycopg.Connection) -> int | None:
    """Read the vector dimension actually deployed on chunks.embedding.

    format_type renders the typmod as 'vector(384)', which is stable across
    pgvector versions -- more so than decoding atttypmod arithmetic ourselves.
    Returns None if the column does not exist yet.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
              FROM pg_attribute a
             WHERE a.attrelid = to_regclass('chunks')
               AND a.attname  = 'embedding'
               AND a.attnum > 0 AND NOT a.attisdropped
            """
        )
        row = cur.fetchone()

    if row is None or row[0] is None:
        return None
    match = re.search(r"\((\d+)\)", row[0])
    return int(match.group(1)) if match else None


def check_embedding_dim(conn: psycopg.Connection, expected: int) -> None:
    """Fail if the deployed vector dimension disagrees with configuration.

    This guards a failure that is otherwise invisible. Vectors from a 384-dim
    model compared against a 768-dim column do not raise -- pgvector rejects
    the insert, but the subtler case is a *changed* model with the same
    dimension, or a schema built under a previous EMBEDDING_DIM that nobody
    re-migrated. Similarity scores stay well-formed and become meaningless.
    Better to refuse to boot.
    """
    actual = deployed_embedding_dim(conn)
    if actual is None or actual == expected:
        return
    raise EmbeddingDimMismatch(
        f"chunks.embedding is vector({actual}) but EMBEDDING_DIM={expected}. "
        f"Changing the embedding model changes the schema: create a new "
        f"migration for the new dimension and re-index affected repos. "
        f"Existing vectors cannot be reinterpreted at a different dimension."
    )


def migrate(
    dsn: str,
    embedding_dim: int,
    *,
    directory: Path = MIGRATIONS_DIR,
    dry_run: bool = False,
) -> list[Migration]:
    """Apply pending migrations. Returns those applied (or pending, if dry_run)."""
    migrations = discover(directory)
    params: dict[str, object] = {"EMBEDDING_DIM": embedding_dim}

    with psycopg.connect(dsn) as conn:
        conn.execute(BOOTSTRAP_SQL)
        conn.commit()

        # Held until the connection closes. A second migrator blocks here
        # rather than executing the same DDL concurrently.
        conn.execute("SELECT pg_advisory_lock(%s)", (ADVISORY_LOCK_KEY,))

        try:
            applied = _applied(conn)
            _check_drift(migrations, applied)

            pending = [m for m in migrations if m.version not in applied]
            if dry_run:
                return pending

            for migration in pending:
                sql = render(migration, params)
                started = time.perf_counter()
                # Postgres DDL is transactional: a migration that fails
                # halfway leaves no partial schema behind.
                with conn.transaction():
                    conn.execute(sql)
                    conn.execute(
                        """
                        INSERT INTO schema_migrations
                            (version, name, checksum, duration_ms)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (
                            migration.version,
                            migration.name,
                            migration.checksum,
                            int((time.perf_counter() - started) * 1000),
                        ),
                    )

            check_embedding_dim(conn, embedding_dim)
            return pending
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (ADVISORY_LOCK_KEY,))
