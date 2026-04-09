"""Migration runner behaviour against a real Postgres.

These are integration tests on purpose. The properties worth testing --
idempotence, drift detection, dimension enforcement -- are properties of the
runner's interaction with the database's own state, and a mock of psycopg would
be asserting that our mock behaves like our code.
"""

import os

import psycopg
import pytest

from codeqa.db.migrate import (
    EmbeddingDimMismatch,
    MigrationDriftError,
    deployed_embedding_dim,
    discover,
    migrate,
    render,
)

pytestmark = pytest.mark.integration

EMBEDDING_DIM = 384


@pytest.fixture
def dsn() -> str:
    base = os.environ.get(
        "CODEQA_TEST_DSN", "postgresql://postgres:test@localhost:55432/postgres"
    )
    db = f"codeqa_test_{os.getpid()}"

    admin = psycopg.connect(base, autocommit=True)
    with admin:
        admin.execute(f'DROP DATABASE IF EXISTS "{db}" WITH (FORCE)')
        admin.execute(f'CREATE DATABASE "{db}"')

    yield base.rsplit("/", 1)[0] + "/" + db

    admin = psycopg.connect(base, autocommit=True)
    with admin:
        admin.execute(f'DROP DATABASE IF EXISTS "{db}" WITH (FORCE)')


def test_applies_and_is_idempotent(dsn: str) -> None:
    first = migrate(dsn, EMBEDDING_DIM)
    assert [m.label for m in first] == ["0001_init"]

    # Second run is a no-op rather than an error.
    assert migrate(dsn, EMBEDDING_DIM) == []

    with psycopg.connect(dsn) as conn:
        assert deployed_embedding_dim(conn) == EMBEDDING_DIM
        count = conn.execute("SELECT count(*) FROM schema_migrations").fetchone()[0]
        assert count == 1


def test_dry_run_reports_without_applying(dsn: str) -> None:
    pending = migrate(dsn, EMBEDDING_DIM, dry_run=True)
    assert [m.label for m in pending] == ["0001_init"]

    with psycopg.connect(dsn) as conn:
        # Bootstrap table exists, but no user tables were created.
        assert conn.execute("SELECT to_regclass('chunks')").fetchone()[0] is None


def test_detects_drift_in_applied_migration(dsn: str, tmp_path) -> None:
    migrate(dsn, EMBEDDING_DIM)

    # Simulate someone editing 0001 after it shipped.
    tampered = tmp_path / "0001_init.sql"
    original = discover()[0].template
    tampered.write_text(original + "\n-- an innocent-looking comment\n")

    with pytest.raises(MigrationDriftError, match="modified after being applied"):
        migrate(dsn, EMBEDDING_DIM, directory=tmp_path)


def test_detects_database_ahead_of_checkout(dsn: str, tmp_path) -> None:
    migrate(dsn, EMBEDDING_DIM)

    # An empty directory stands in for a checkout that predates 0001.
    with pytest.raises(MigrationDriftError, match="no file on disk"):
        migrate(dsn, EMBEDDING_DIM, directory=tmp_path)


def test_rejects_embedding_dim_change_without_migration(dsn: str) -> None:
    migrate(dsn, EMBEDDING_DIM)

    # Schema is vector(384); config now claims 768. Booting against this would
    # compare vectors that cannot be compared.
    with pytest.raises(EmbeddingDimMismatch, match=r"vector\(384\).*768"):
        migrate(dsn, 768)


def test_migration_is_atomic(dsn: str, tmp_path) -> None:
    """A migration that fails partway leaves no partial schema behind."""
    bad = tmp_path / "0001_init.sql"
    bad.write_text("CREATE TABLE ok_so_far (id int); SELECT 1/0;")

    with pytest.raises(psycopg.errors.DivisionByZero):
        migrate(dsn, EMBEDDING_DIM, directory=tmp_path)

    with psycopg.connect(dsn) as conn:
        assert conn.execute("SELECT to_regclass('ok_so_far')").fetchone()[0] is None
        count = conn.execute("SELECT count(*) FROM schema_migrations").fetchone()[0]
        assert count == 0


def test_render_rejects_unsubstituted_placeholders(tmp_path) -> None:
    (tmp_path / "0001_x.sql").write_text("CREATE TABLE t (e vector(${EMBEDDING_DIM}), z ${NOPE});")
    migration = discover(tmp_path)[0]

    with pytest.raises(Exception, match=r"unsubstituted placeholders \['NOPE'\]"):
        render(migration, {"EMBEDDING_DIM": 384})


def test_rejects_duplicate_versions(tmp_path) -> None:
    (tmp_path / "0001_a.sql").write_text("SELECT 1;")
    (tmp_path / "0001_b.sql").write_text("SELECT 1;")

    with pytest.raises(Exception, match="duplicate migration version"):
        discover(tmp_path)


def test_rejects_misnamed_files(tmp_path) -> None:
    (tmp_path / "init.sql").write_text("SELECT 1;")

    with pytest.raises(Exception, match="NNNN_name.sql"):
        discover(tmp_path)
