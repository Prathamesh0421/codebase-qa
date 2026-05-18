"""POST /v1/query and GET /health, via FastAPI's TestClient against real
Postgres, with litellm mocked -- same mock_response delegation pattern as
test_agent_graph.py, so response object shapes stay litellm's real ones.

The one thing a mocked-LLM test cannot prove is the OTel span-nesting fix
in agents/nodes.py / api/app.py (that bug only reproduces under real
uvicorn thread-pool concurrency -- TestClient's ASGI transport doesn't
exhibit it, which is exactly why it was found by hand against a live
server, not by this suite). This suite instead pins the contract these
tests CAN check deterministically: the response is a well-formed SSE
event stream carrying locate/trace progress, streamed tokens, and a final
grounding-aware done event.
"""

import os
from unittest.mock import patch

import litellm as real_litellm
import psycopg
import pytest
from fastapi.testclient import TestClient
from pgvector.psycopg import register_vector

from codeqa.api.app import app, get_settings
from codeqa.config import Settings
from codeqa.indexing.embeddings import LocalEmbedder
from codeqa.indexing.pipeline import index_repo
from codeqa.indexing.store import register_repo

pytestmark = pytest.mark.integration

EMBEDDING_DIM = 384
_real_completion = real_litellm.completion


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
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def embedder():
    return LocalEmbedder("BAAI/bge-small-en-v1.5", dimension=EMBEDDING_DIM, batch_size=16)


@pytest.fixture
def conn():
    connection = psycopg.connect(_dsn())
    register_vector(connection)
    yield connection
    connection.rollback()
    connection.close()


def drop_repo_by_slug(conn, slug: str) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM repos WHERE slug = %s", (slug,))
        row = cur.fetchone()
        if row is not None:
            cur.execute("SELECT drop_repo_partition(%s)", (row[0],))
            cur.execute("DELETE FROM repos WHERE id = %s", (row[0],))
    conn.commit()


@pytest.fixture
def indexed_repo(conn, embedder, tmp_path, request):
    slug = f"test-{request.node.name}".replace("[", "-").replace("]", "")[:60]
    drop_repo_by_slug(conn, slug)

    src = tmp_path / "src"
    src.mkdir()
    (src / "auth.py").write_text(
        "def check_password_hash(pw_hash, password):\n"
        "    '''Verify a plaintext password against a stored hash.'''\n"
        "    return pw_hash == hash_password(password)\n"
    )
    repo_id = register_repo(
        conn, slug, "Test Repo", "local_path", str(src), "BAAI/bge-small-en-v1.5", EMBEDDING_DIM
    )
    index_repo(conn, repo_id, src, embedder)
    yield slug
    drop_repo_by_slug(conn, slug)


def _mock_completion(trace_response: str, answer_text: str):
    def _delegate(**kwargs):
        kwargs.pop("api_key", None)
        if kwargs.get("stream"):
            return _real_completion(mock_response=answer_text, **kwargs)
        return _real_completion(mock_response=trace_response, **kwargs)

    return patch("litellm.completion", side_effect=_delegate)


def _parse_sse(body: str) -> list[tuple[str, str]]:
    # sse-starlette writes \r\n line endings, so a blank *line* (not "\n\n"
    # literally) is what separates events -- splitlines() handles \r\n
    # correctly, a naive split("\n\n") silently finds zero boundaries and
    # collapses the whole stream into a single "event".
    events = []
    event, data = None, None
    for line in body.splitlines():
        if line.startswith("event:"):
            event = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            # Per the SSE spec, only a single leading space after "data:" is
            # stripped -- a token's own trailing space is real content
            # (e.g. "hi " before "there"), and .strip() here would eat it,
            # corrupting reconstruction of the streamed answer text.
            data = line.removeprefix("data:").removeprefix(" ")
        elif line == "" and event is not None:
            events.append((event, data))
            event, data = None, None
    if event is not None:
        events.append((event, data))
    return events


class TestHealth:
    def test_reports_db_and_redis_reachability(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["db"] is True


class TestQueryEndpoint:
    def test_unknown_repo_slug_is_a_404_not_an_sse_stream(self, client):
        response = client.post(
            "/v1/query", json={"repo_slug": "does-not-exist", "question": "anything"}
        )
        assert response.status_code == 404

    def test_a_real_query_streams_progress_tokens_and_a_grounded_done_event(
        self, client, indexed_repo
    ):
        with _mock_completion("SUFFICIENT", "It hashes the password and compares."):
            response = client.post(
                "/v1/query",
                json={
                    "repo_slug": indexed_repo,
                    "question": "how is a password verified",
                    "top_k": 5,
                },
            )
        assert response.status_code == 200
        events = _parse_sse(response.text)
        event_types = [e for e, _ in events]

        assert "progress" in event_types
        assert "token" in event_types
        assert event_types[-1] == "done"

        streamed = "".join(
            data for event, data in events if event == "token"
        )
        assert streamed == "It hashes the password and compares."

        import json as jsonlib

        done_payload = jsonlib.loads(events[-1][1])
        assert done_payload["citations_dropped"] == []
        assert len(done_payload["chunks"]) > 0
