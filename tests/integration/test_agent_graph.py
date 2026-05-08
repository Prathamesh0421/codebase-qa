"""agents/graph.py against real Postgres, with litellm mocked -- same
mock_response delegation pattern as test_synthesis.py, so response object
shapes (ModelResponse for the non-streaming trace call, CustomStreamWrapper
for the streaming synthesize call) are litellm's real ones, not hand-rolled.

The point of this file is proving the retry edge ACTUALLY fires, not just
that it's wired into the graph -- a mock that returns "INSUFFICIENT" then
"SUFFICIENT" and asserting locate really ran twice is the only way to tell
"the cycle exists in the graph" from "the cycle executes correctly."
"""

import os
from unittest.mock import patch

import litellm as real_litellm
import psycopg
import pytest
from pgvector.psycopg import register_vector

from codeqa.agents.graph import build_agent_graph
from codeqa.agents.state import AgentState
from codeqa.indexing.embeddings import LocalEmbedder
from codeqa.indexing.pipeline import index_repo
from codeqa.indexing.store import register_repo
from codeqa.retrieval.naive import NaiveStrategy

pytestmark = pytest.mark.integration

EMBEDDING_DIM = 384
_LLM_MODEL = "gemini/gemini-2.0-flash"
_real_completion = real_litellm.completion


def _mock_completion(trace_responses: list[str], answer_text: str):
    """trace_responses are consumed in order by successive non-streaming
    (trace) calls. The streaming (synthesize) call always gets answer_text.
    Distinguishing the two by the stream kwarg, not by call count, since a
    retry means MORE trace calls than synthesize calls (exactly one
    synthesize call regardless of how many locate/trace rounds happened).
    """
    state = {"trace_call": 0}

    def _delegate(**kwargs):
        kwargs.pop("api_key", None)
        if kwargs.get("stream"):
            return _real_completion(mock_response=answer_text, **kwargs)
        text = trace_responses[state["trace_call"]]
        state["trace_call"] += 1
        return _real_completion(mock_response=text, **kwargs)

    return patch("litellm.completion", side_effect=_delegate)


@pytest.fixture(scope="module")
def embedder():
    return LocalEmbedder("BAAI/bge-small-en-v1.5", dimension=EMBEDDING_DIM, batch_size=16)


@pytest.fixture
def conn():
    dsn = os.environ.get("CODEQA_TEST_DSN", "postgresql://codeqa:codeqa@localhost:5432/codeqa")
    connection = psycopg.connect(dsn)
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
        "\n\n"
        "def hash_password(password):\n"
        "    '''Turn a plaintext password into a stored hash.'''\n"
        "    return _digest(password)\n"
    )
    repo_id = register_repo(
        conn, slug, "Test Repo", "local_path", str(src), "BAAI/bge-small-en-v1.5", EMBEDDING_DIM
    )
    index_repo(conn, repo_id, src, embedder)
    yield repo_id
    drop_repo_by_slug(conn, slug)


def _initial_state(repo_id: int, question: str, max_attempts: int = 2) -> AgentState:
    return AgentState(
        repo_id=repo_id, question=question, current_query=question, max_attempts=max_attempts
    )


class TestAgentGraph:
    def test_sufficient_on_first_attempt_never_retries(self, conn, embedder, indexed_repo):
        with _mock_completion(["SUFFICIENT"], "It hashes the password and compares."):
            g = build_agent_graph(conn, embedder, NaiveStrategy(), 5, _LLM_MODEL, None)
            result = g.invoke(_initial_state(indexed_repo, "how is a password verified"))

        assert result["attempt"] == 1
        assert result["sufficient"] is True
        assert result["answer"] == "It hashes the password and compares."

    def test_insufficient_then_sufficient_actually_retries_locate(
        self, conn, embedder, indexed_repo
    ):
        # This is the load-bearing assertion for this phase: attempt == 2
        # is only possible if locate genuinely ran a second time after
        # trace routed back to it -- not evidence the edge merely exists.
        with _mock_completion(
            ["INSUFFICIENT: find hash_password", "SUFFICIENT"], "Full grounded answer."
        ):
            g = build_agent_graph(conn, embedder, NaiveStrategy(), 5, _LLM_MODEL, None)
            result = g.invoke(_initial_state(indexed_repo, "how is a password verified"))

        assert result["attempt"] == 2
        assert result["sufficient"] is True
        assert result["answer"] == "Full grounded answer."

    def test_retry_uses_the_refined_query_and_accumulates_chunks(
        self, conn, embedder, indexed_repo
    ):
        with _mock_completion(
            ["INSUFFICIENT: find hash_password", "SUFFICIENT"], "answer"
        ):
            g = build_agent_graph(conn, embedder, NaiveStrategy(), 1, _LLM_MODEL, None)
            result = g.invoke(_initial_state(indexed_repo, "how is a password verified"))

        # top_k=1 per locate call, but two locate calls ran (initial +
        # retry with the refined query "find hash_password", which -- being
        # a near-exact repeat of hash_password's own name -- reliably ranks
        # it #1, a DIFFERENT chunk than attempt 1's check_password_hash).
        # Asserting the specific accumulated symbols, not just a count: a
        # weaker len() >= 1 assertion would pass even if the retry returned
        # a duplicate of attempt 1's only chunk and accumulated nothing new.
        symbols = {c.symbol_name for c in result["chunks"]}
        assert symbols == {"check_password_hash", "hash_password"}
        assert result["current_query"] == "find hash_password"

    def test_persistent_insufficiency_is_bounded_by_max_attempts(
        self, conn, embedder, indexed_repo
    ):
        with _mock_completion(
            ["INSUFFICIENT: a", "INSUFFICIENT: b", "INSUFFICIENT: c"], "best effort answer"
        ):
            g = build_agent_graph(conn, embedder, NaiveStrategy(), 5, _LLM_MODEL, None)
            result = g.invoke(_initial_state(indexed_repo, "q", max_attempts=2))

        # Never a third locate call, even though trace kept saying
        # INSUFFICIENT -- the attempt counter, not the model, is what
        # stops the loop.
        assert result["attempt"] == 2
        assert result["sufficient"] is False
        assert result["answer"] == "best effort answer"

    def test_answer_tokens_stream_through_custom_stream_mode(self, conn, embedder, indexed_repo):
        with _mock_completion(["SUFFICIENT"], "streamed tokens here"):
            g = build_agent_graph(conn, embedder, NaiveStrategy(), 5, _LLM_MODEL, None)
            tokens = list(
                g.stream(
                    _initial_state(indexed_repo, "how is a password verified"),
                    stream_mode="custom",
                )
            )

        assert "".join(tokens) == "streamed tokens here"
        assert len(tokens) > 1  # genuinely arrived as multiple pieces, not one blob
