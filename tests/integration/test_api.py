"""POST /v1/repos and GET .../jobs/{id} against a real Postgres, via
FastAPI's TestClient. get_settings is overridden through FastAPI's own
dependency_overrides mechanism rather than monkeypatching env vars --
app.py resolves settings per-request through Depends, so this is the
supported seam for tests, not a workaround.
"""

import os

import psycopg
import pytest
from fastapi.testclient import TestClient

from codeqa.api.app import app, get_settings
from codeqa.api.auth import create_api_key
from codeqa.config import Settings

pytestmark = pytest.mark.integration


def _dsn() -> str:
    return os.environ.get("CODEQA_TEST_DSN", "postgresql://codeqa:codeqa@localhost:5432/codeqa")


def _test_settings() -> Settings:
    return Settings(database_url=_dsn())


@pytest.fixture(autouse=True)
def override_settings():
    app.dependency_overrides[get_settings] = _test_settings
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def conn():
    connection = psycopg.connect(_dsn())
    yield connection
    connection.rollback()
    connection.close()


def drop_repo_by_slug(conn, slug: str) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM repos WHERE slug = %s", (slug,))
        row = cur.fetchone()
        if row is not None:
            cur.execute("DELETE FROM index_jobs WHERE repo_id = %s", (row[0],))
            cur.execute("SELECT drop_repo_partition(%s)", (row[0],))
            cur.execute("DELETE FROM repos WHERE id = %s", (row[0],))
    conn.commit()


@pytest.fixture
def api_key(conn):
    # A high per-test rate limit -- this file exercises routes, not the
    # rate limiter itself (see test_rate_limit.py for that), so the limit
    # is set generously to never accidentally trip during a test.
    created = create_api_key(conn, "test-key", rate_limit_rpm=10_000)
    yield created
    with conn.cursor() as cur:
        cur.execute("DELETE FROM api_keys WHERE id = %s", (created.id,))
    conn.commit()


@pytest.fixture
def client(api_key):
    return TestClient(app, headers={"Authorization": f"Bearer {api_key.plaintext}"})


@pytest.fixture
def anonymous_client():
    return TestClient(app)


class TestCreateRepo:
    def test_local_path_repo_is_registered_and_a_job_is_queued(self, client, conn, tmp_path):
        slug = "test-api-local-path"
        drop_repo_by_slug(conn, slug)
        try:
            response = client.post(
                "/v1/repos",
                json={
                    "slug": slug,
                    "display_name": "Test",
                    "source_kind": "local_path",
                    "source_ref": str(tmp_path),
                },
            )
            assert response.status_code == 201, response.text
            body = response.json()
            assert body["repo_id"] > 0
            assert body["job_id"] > 0

            with conn.cursor() as cur:
                cur.execute("SELECT status FROM index_jobs WHERE id = %s", (body["job_id"],))
                assert cur.fetchone()[0] == "queued"
        finally:
            drop_repo_by_slug(conn, slug)

    def test_an_unsafe_git_url_is_rejected_at_request_time(self, client, conn):
        slug = "test-api-unsafe-url"
        drop_repo_by_slug(conn, slug)
        response = client.post(
            "/v1/repos",
            json={
                "slug": slug,
                "display_name": "Test",
                "source_kind": "git_url",
                "source_ref": "https://evil.example.com/repo.git",
            },
        )
        assert response.status_code == 400
        assert "not in the allowed" in response.json()["detail"]

        with conn.cursor() as cur:
            cur.execute("SELECT id FROM repos WHERE slug = %s", (slug,))
            # Rejected before register_repo ever ran -- no row, no job.
            assert cur.fetchone() is None

    def test_an_allowlisted_git_url_is_accepted(self, client, conn):
        slug = "test-api-safe-url"
        drop_repo_by_slug(conn, slug)
        try:
            response = client.post(
                "/v1/repos",
                json={
                    "slug": slug,
                    "display_name": "Test",
                    "source_kind": "git_url",
                    "source_ref": "https://github.com/psf/requests.git",
                },
            )
            assert response.status_code == 201, response.text
        finally:
            drop_repo_by_slug(conn, slug)

    def test_a_duplicate_slug_returns_409(self, client, conn, tmp_path):
        slug = "test-api-duplicate-slug"
        drop_repo_by_slug(conn, slug)
        try:
            body = {
                "slug": slug, "display_name": "Test",
                "source_kind": "local_path", "source_ref": str(tmp_path),
            }
            first = client.post("/v1/repos", json=body)
            assert first.status_code == 201

            second = client.post("/v1/repos", json=body)
            assert second.status_code == 409
        finally:
            drop_repo_by_slug(conn, slug)

    def test_an_invalid_source_kind_is_rejected_by_request_validation(self, client):
        response = client.post(
            "/v1/repos",
            json={
                "slug": "whatever", "display_name": "Test",
                "source_kind": "ftp_url", "source_ref": "ftp://example.com/repo",
            },
        )
        assert response.status_code == 422


class TestJobStatus:
    def test_returns_the_queued_job(self, client, conn, tmp_path):
        slug = "test-api-job-status"
        drop_repo_by_slug(conn, slug)
        try:
            created = client.post(
                "/v1/repos",
                json={
                    "slug": slug, "display_name": "Test",
                    "source_kind": "local_path", "source_ref": str(tmp_path),
                },
            ).json()

            response = client.get(f"/v1/repos/{slug}/jobs/{created['job_id']}")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "queued"
            assert body["attempts"] == 0
            assert body["repo_id"] == created["repo_id"]
        finally:
            drop_repo_by_slug(conn, slug)

    def test_unknown_job_id_returns_404(self, client, conn, tmp_path):
        slug = "test-api-unknown-job"
        drop_repo_by_slug(conn, slug)
        try:
            client.post(
                "/v1/repos",
                json={
                    "slug": slug, "display_name": "Test",
                    "source_kind": "local_path", "source_ref": str(tmp_path),
                },
            )
            response = client.get(f"/v1/repos/{slug}/jobs/999999")
            assert response.status_code == 404
        finally:
            drop_repo_by_slug(conn, slug)

    def test_unknown_repo_slug_returns_404(self, client):
        response = client.get("/v1/repos/does-not-exist/jobs/1")
        assert response.status_code == 404


class TestAuth:
    def test_no_authorization_header_is_rejected(self, anonymous_client):
        response = anonymous_client.post(
            "/v1/repos",
            json={
                "slug": "whatever", "display_name": "Test",
                "source_kind": "local_path", "source_ref": "/tmp",
            },
        )
        assert response.status_code == 401

    def test_a_malformed_authorization_header_is_rejected(self, anonymous_client):
        response = anonymous_client.get(
            "/v1/repos/does-not-exist/jobs/1", headers={"Authorization": "not-a-bearer-token"}
        )
        assert response.status_code == 401

    def test_an_unknown_key_is_rejected(self, anonymous_client):
        response = anonymous_client.get(
            "/v1/repos/does-not-exist/jobs/1",
            headers={"Authorization": "Bearer cq_not-a-real-key"},
        )
        assert response.status_code == 401

    def test_health_needs_no_authorization(self, anonymous_client):
        # Infra probes (load balancers, orchestrators) can't be expected to
        # carry an API key -- /health is deliberately the one open route.
        response = anonymous_client.get("/health")
        assert response.status_code == 200
