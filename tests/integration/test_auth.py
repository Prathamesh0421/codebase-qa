"""api/auth.py against real Postgres: a key round-trips from creation
through verification to revocation, and a wrong or revoked key is
rejected -- the property require_api_key (api/app.py) actually depends on.
"""

import os

import psycopg
import pytest

from codeqa.api.auth import InvalidApiKey, create_api_key, revoke_api_key, verify_api_key

pytestmark = pytest.mark.integration


def _dsn() -> str:
    return os.environ.get("CODEQA_TEST_DSN", "postgresql://codeqa:codeqa@localhost:5432/codeqa")


@pytest.fixture
def conn():
    connection = psycopg.connect(_dsn())
    yield connection
    connection.rollback()
    connection.close()


def _cleanup(conn, key_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM api_keys WHERE id = %s", (key_id,))
    conn.commit()


class TestApiKeyLifecycle:
    def test_a_created_key_verifies_and_carries_its_own_rate_limit(self, conn):
        created = create_api_key(conn, "test-key", rate_limit_rpm=42)
        try:
            record = verify_api_key(conn, created.plaintext)
            assert record.id == created.id
            assert record.rate_limit_rpm == 42
        finally:
            _cleanup(conn, created.id)

    def test_verifying_updates_last_used_at(self, conn):
        created = create_api_key(conn, "test-key", rate_limit_rpm=60)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT last_used_at FROM api_keys WHERE id = %s", (created.id,))
                assert cur.fetchone()[0] is None

            verify_api_key(conn, created.plaintext)

            with conn.cursor() as cur:
                cur.execute("SELECT last_used_at FROM api_keys WHERE id = %s", (created.id,))
                assert cur.fetchone()[0] is not None
        finally:
            _cleanup(conn, created.id)

    def test_an_unknown_key_is_rejected(self, conn):
        with pytest.raises(InvalidApiKey):
            verify_api_key(conn, "cq_not-a-real-key")

    def test_a_revoked_key_is_rejected(self, conn):
        created = create_api_key(conn, "test-key", rate_limit_rpm=60)
        try:
            revoke_api_key(conn, created.id)
            with pytest.raises(InvalidApiKey):
                verify_api_key(conn, created.plaintext)
        finally:
            _cleanup(conn, created.id)

    def test_the_plaintext_is_never_recoverable_from_storage(self, conn):
        created = create_api_key(conn, "test-key", rate_limit_rpm=60)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT key_hash FROM api_keys WHERE id = %s", (created.id,))
                stored_hash = cur.fetchone()[0]
            assert created.plaintext not in stored_hash
            assert stored_hash != created.plaintext
        finally:
            _cleanup(conn, created.id)
