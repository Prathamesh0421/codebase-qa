"""`codeqa ask` end to end: real Postgres, real embeddings, real retrieval,
mocked LLM call (litellm.completion patched via mock_response -- see
tests/unit/test_synthesis.py for why that boundary and not a hand-rolled
mock). This is the closest thing to Phase 5's literal "done when" bar that
can run without a live API key.
"""

import os
from unittest.mock import patch

import litellm as real_litellm
import psycopg
import pytest
from pgvector.psycopg import register_vector
from typer.testing import CliRunner

from codeqa.cli import app
from codeqa.config import get_settings
from codeqa.indexing.embeddings import LocalEmbedder
from codeqa.indexing.pipeline import index_repo
from codeqa.indexing.store import register_repo

pytestmark = pytest.mark.integration

EMBEDDING_DIM = 384
_real_completion = real_litellm.completion


@pytest.fixture(autouse=True)
def clear_settings_cache():
    # get_settings() is @cache'd (config.py) so a single process reuses the
    # first Settings() built -- deliberate in production (config is read
    # once), but it means monkeypatch.setenv has no effect on a later
    # get_settings() call within the same test session unless the cache is
    # cleared first. Cleared before AND after so no test leaks env-derived
    # settings into the next one.
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _mocked_llm(mock_text: str):
    def _delegate(**kwargs):
        kwargs.pop("api_key", None)
        return _real_completion(mock_response=mock_text, **kwargs)

    return patch("codeqa.synthesis.litellm.completion", side_effect=_delegate)


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
def indexed_repo(conn, tmp_path):
    slug = "greeter"
    drop_repo_by_slug(conn, slug)

    src = tmp_path / "src"
    src.mkdir()
    (src / "greeter.py").write_text(
        "def greet(name):\n"
        "    '''Return a friendly greeting for the given name.'''\n"
        "    return f'Hello, {name}!'\n"
    )

    embedder = LocalEmbedder("BAAI/bge-small-en-v1.5", dimension=EMBEDDING_DIM, batch_size=8)
    repo_id = register_repo(
        conn, slug, "Greeter", "local_path", str(src), "BAAI/bge-small-en-v1.5", EMBEDDING_DIM
    )
    index_repo(conn, repo_id, src, embedder)
    yield slug
    drop_repo_by_slug(conn, slug)


class TestAskCommand:
    def test_streams_answer_and_lists_retrieved_chunks(self, indexed_repo, monkeypatch):
        monkeypatch.setenv("CODEQA_RETRIEVAL_STRATEGY", "naive")
        runner = CliRunner()

        with _mocked_llm("Greets a user by name, citing greeter.py:1-3."):
            result = runner.invoke(
                app, ["ask", "how does greeting work?", "--repo", "greeter"]
            )

        assert result.exit_code == 0, result.output
        assert "Greets a user by name" in result.output
        assert "Retrieved:" in result.output
        assert "greeter.py:1-3" in result.output
        assert "greet" in result.output

    @pytest.mark.parametrize("strategy_name", ["naive", "hybrid", "hybrid_graph"])
    def test_all_three_strategies_are_selectable_purely_via_config(
        self, indexed_repo, monkeypatch, strategy_name
    ):
        # config.retrieval_strategy is the only place this decision is meant
        # to be made (see strategy.py's docstring) -- this drives that exact
        # path, CODEQA_RETRIEVAL_STRATEGY env var through get_settings()
        # through get_strategy(), for all three strategies against the same
        # repo and question, rather than constructing strategy classes
        # directly like test_hybrid_retrieval.py does.
        monkeypatch.setenv("CODEQA_RETRIEVAL_STRATEGY", strategy_name)
        runner = CliRunner()

        with _mocked_llm("Greets a user by name, citing greeter.py:1-3."):
            result = runner.invoke(
                app, ["ask", "how does greeting work?", "--repo", "greeter"]
            )

        assert result.exit_code == 0, result.output
        assert "Greets a user by name" in result.output
        assert "greeter.py:1-3" in result.output

    def test_unknown_repo_slug_fails_clearly(self, monkeypatch):
        monkeypatch.setenv("CODEQA_RETRIEVAL_STRATEGY", "naive")
        runner = CliRunner()

        result = runner.invoke(app, ["ask", "anything", "--repo", "does-not-exist"])

        assert result.exit_code == 1
        assert "No repo registered" in result.output

    def test_bracket_syntax_in_the_answer_is_not_swallowed_as_markup(
        self, indexed_repo, monkeypatch
    ):
        # Real, reproduced bug: rich's Console treats "[...]" as a markup tag
        # by default, and an LLM answering questions about code will
        # routinely emit real bracket syntax. "List[str]" printed as "List"
        # before this was fixed with markup=False on the streamed tokens.
        monkeypatch.setenv("CODEQA_RETRIEVAL_STRATEGY", "naive")
        runner = CliRunner()

        answer = "The return type is list[str] and chunks[0] holds the first result."
        with _mocked_llm(answer):
            result = runner.invoke(app, ["ask", "what does it return?", "--repo", "greeter"])

        assert result.exit_code == 0, result.output
        assert answer in result.output
