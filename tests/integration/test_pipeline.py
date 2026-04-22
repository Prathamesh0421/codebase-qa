"""End-to-end indexing against a real Postgres.

This is where Phases 1-4 actually connect: schema, language detection,
AST-aware chunking, and embedding all have to agree for a single
index_repo() call to work. The Flask fixture test is this project's
literal "Done when" bar for Phase 4.
"""

import os
from pathlib import Path

import psycopg
import pytest
from pgvector.psycopg import register_vector

from codeqa.indexing.embeddings import LocalEmbedder
from codeqa.indexing.pipeline import index_repo
from codeqa.indexing.store import (
    EmbeddingConfigMismatch,
    RepoAlreadyExists,
    check_embedder_matches_repo,
    register_repo,
)

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).parent.parent / "fixtures" / "repos"
EMBEDDING_DIM = 384


@pytest.fixture(scope="module")
def embedder():
    # One model load shared across every test in this module -- construction
    # is the expensive part, embed() calls are fast once loaded.
    return LocalEmbedder("BAAI/bge-small-en-v1.5", dimension=EMBEDDING_DIM, batch_size=32)


@pytest.fixture
def conn():
    dsn = os.environ.get("CODEQA_TEST_DSN", "postgresql://codeqa:codeqa@localhost:5432/codeqa")
    connection = psycopg.connect(dsn)
    register_vector(connection)
    yield connection
    connection.rollback()
    connection.close()


def drop_repo_by_slug(conn, slug: str) -> None:
    """Remove a repo and its partition if it exists. Safe to call when absent.

    Called on setup as well as teardown, deliberately: a test run that dies
    mid-fixture (a crash, a Ctrl-C) leaves the repos row behind, and every
    later run then fails on the unique slug constraint with an error that
    points at the wrong thing entirely. Tests should not inherit state from a
    previous run's failure.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM repos WHERE slug = %s", (slug,))
        row = cur.fetchone()
        if row is not None:
            cur.execute("SELECT drop_repo_partition(%s)", (row[0],))
            cur.execute("DELETE FROM repos WHERE id = %s", (row[0],))
    conn.commit()


@pytest.fixture
def repo(conn, request):
    """Registers a repo under a slug unique to the test, cleans it up after.

    drop_repo_partition (Phase 1) makes teardown correctness-critical, not
    just tidy: leaving a stale partition around risks a slug collision or a
    partition-count creep across a long test run.
    """
    slug = f"test-{request.node.name}".replace("[", "-").replace("]", "")[:60]
    drop_repo_by_slug(conn, slug)
    repo_id = register_repo(
        conn, slug, "Test Repo", "local_path", "/tmp/x", "BAAI/bge-small-en-v1.5", EMBEDDING_DIM
    )
    yield repo_id
    drop_repo_by_slug(conn, slug)


def make_repo_dir(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, content in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return tmp_path


class TestRegisterRepo:
    def test_creates_a_queryable_partition(self, conn, repo):
        with conn.cursor() as cur:
            cur.execute(
                "SELECT to_regclass(%s) IS NOT NULL", (f"chunks_repo_{repo}",)
            )
            assert cur.fetchone()[0] is True

    def test_duplicate_slug_raises_and_leaves_no_partial_row(self, conn, repo):
        with conn.cursor() as cur:
            cur.execute("SELECT slug FROM repos WHERE id = %s", (repo,))
            slug = cur.fetchone()[0]

        with pytest.raises(RepoAlreadyExists):
            register_repo(
                conn, slug, "Dup", "local_path", "/tmp/y", "BAAI/bge-small-en-v1.5", EMBEDDING_DIM
            )

        # The failed attempt's rollback must not have taken the original
        # registration down with it.
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM repos WHERE slug = %s", (slug,))
            assert cur.fetchone()[0] == 1


class TestEmbedderConsistencyGuard:
    def test_matching_model_and_dim_passes(self, conn, repo):
        check_embedder_matches_repo(conn, repo, "BAAI/bge-small-en-v1.5", EMBEDDING_DIM)

    def test_different_model_same_dim_raises(self, conn, repo):
        # The dangerous case: same dimension, different model -- Postgres's
        # vector(N) column accepts it silently, so this has to be caught here.
        with pytest.raises(EmbeddingConfigMismatch, match="different models"):
            check_embedder_matches_repo(conn, repo, "some-other-384d-model", EMBEDDING_DIM)

    def test_different_dim_raises(self, conn, repo):
        with pytest.raises(EmbeddingConfigMismatch):
            check_embedder_matches_repo(conn, repo, "BAAI/bge-small-en-v1.5", 768)

    def test_unregistered_repo_raises(self, conn):
        with pytest.raises(ValueError, match="not registered"):
            check_embedder_matches_repo(conn, 999_999_999, "any-model", EMBEDDING_DIM)


class TestIndexRepo:
    def test_indexes_files_and_chunks_with_correct_dimension(self, conn, repo, embedder, tmp_path):
        root = make_repo_dir(
            tmp_path,
            {
                "app.py": (
                    "def greet(name):\n"
                    "    return say_hello(name)\n"
                    "\n"
                    "class Greeter:\n"
                    "    def hello(self):\n"
                    "        return greet('world')\n"
                ),
            },
        )

        stats = index_repo(conn, repo, root, embedder)

        assert stats.files_indexed == 1
        assert stats.files_failed == 0
        assert stats.chunks_created == 3  # greet, Greeter, Greeter.hello

        with conn.cursor() as cur:
            cur.execute(
                "SELECT symbol_name, kind, qualified_name, embedding "
                "FROM chunks WHERE repo_id = %s ORDER BY symbol_name",
                (repo,),
            )
            rows = {r[0]: r for r in cur.fetchall()}

        assert set(rows) == {"greet", "Greeter", "hello"}
        assert rows["hello"][1] == "method"
        assert rows["hello"][2] == "Greeter.hello"
        assert len(rows["greet"][3].to_list()) == EMBEDDING_DIM

    def test_skips_files_with_no_registered_language(self, conn, repo, embedder, tmp_path):
        root = make_repo_dir(
            tmp_path, {"app.py": "def f(): pass", "README.md": "# hello", "data.json": "{}"}
        )

        stats = index_repo(conn, repo, root, embedder)

        assert stats.files_indexed == 1
        assert stats.files_skipped_no_language == 2

    def test_rerunning_is_idempotent_not_duplicating(self, conn, repo, embedder, tmp_path):
        root = make_repo_dir(tmp_path, {"app.py": "def f():\n    return 1\n"})

        first = index_repo(conn, repo, root, embedder)
        second = index_repo(conn, repo, root, embedder)

        assert first.chunks_created == second.chunks_created == 1
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks WHERE repo_id = %s", (repo,))
            assert cur.fetchone()[0] == 1

    def test_editing_a_file_replaces_its_chunks_not_appends(self, conn, repo, embedder, tmp_path):
        root = make_repo_dir(tmp_path, {"app.py": "def one(): pass\n"})
        index_repo(conn, repo, root, embedder)

        (root / "app.py").write_text("def one(): pass\ndef two(): pass\n")
        index_repo(conn, repo, root, embedder)

        with conn.cursor() as cur:
            cur.execute(
                "SELECT symbol_name FROM chunks WHERE repo_id = %s ORDER BY symbol_name", (repo,)
            )
            names = [r[0] for r in cur.fetchall()]
        assert names == ["one", "two"]

    def test_embedder_mismatch_aborts_before_any_writes(self, conn, repo, tmp_path):
        # Caught by check_embedder_matches_repo before any file is even
        # walked -- stronger than "before any writes": before any work.
        root = make_repo_dir(tmp_path, {"app.py": "def f(): pass"})
        wrong = LocalEmbedder("BAAI/bge-small-en-v1.5", dimension=768, batch_size=8)

        with pytest.raises(EmbeddingConfigMismatch):
            index_repo(conn, repo, root, wrong)

        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM files WHERE repo_id = %s", (repo,))
            assert cur.fetchone()[0] == 0

    def test_one_bad_file_does_not_abort_the_whole_run(self, conn, repo, embedder, tmp_path):
        root = make_repo_dir(
            tmp_path, {"good.py": "def works(): pass", "bad.py": "def broken(:\n"}
        )
        # bad.py has invalid Python syntax. tree-sitter is error-tolerant by
        # design (it always returns a tree), so this exercises the
        # try/except path via a monkeypatched failure instead, since a real
        # syntax error alone wouldn't actually raise inside chunk_file.
        import codeqa.indexing.pipeline as pipeline_module

        real_chunk_file = pipeline_module.chunk_file

        def flaky_chunk_file(spec, path, content):
            if path.endswith("bad.py"):
                raise ValueError("simulated parser failure")
            return real_chunk_file(spec, path, content)

        pipeline_module.chunk_file = flaky_chunk_file
        try:
            stats = index_repo(conn, repo, root, embedder)
        finally:
            pipeline_module.chunk_file = real_chunk_file

        assert stats.files_indexed == 1
        assert stats.files_failed == 1
        assert stats.errors[0][0] == "bad.py"

    def test_mark_indexed_sets_status_ready(self, conn, repo, embedder, tmp_path):
        root = make_repo_dir(tmp_path, {"app.py": "def f(): pass"})
        index_repo(conn, repo, root, embedder)

        with conn.cursor() as cur:
            cur.execute("SELECT status, last_indexed_at FROM repos WHERE id = %s", (repo,))
            status, last_indexed_at = cur.fetchone()
        assert status == "ready"
        assert last_indexed_at is not None


class TestIndexFlaskFixture:
    """The Phase 4 'Done when' bar: Flask indexes end-to-end and chunks is
    populated with vectors. Flask is also the Phase 9 eval anchor, so this
    doubles as confirmation the fixture is usable for that.

    Indexing all of Flask is the expensive part of this class, so it happens
    once (class-scoped) and every test asserts a different thing about the
    same result -- which needs its own class-scoped connection: a
    class-scoped fixture can't depend on the module's function-scoped conn
    fixture (pytest enforces this, correctly -- a function-scoped connection
    torn down between tests can't be shared across a whole class).
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def class_conn():
        dsn = os.environ.get(
            "CODEQA_TEST_DSN", "postgresql://codeqa:codeqa@localhost:5432/codeqa"
        )
        connection = psycopg.connect(dsn)
        register_vector(connection)
        yield connection
        connection.rollback()
        connection.close()

    @pytest.fixture(scope="class")
    @staticmethod
    def indexed(class_conn, embedder):
        slug = "flask-fixture-e2e"
        drop_repo_by_slug(class_conn, slug)
        repo_id = register_repo(
            class_conn, slug, "Flask", "local_path", str(FIXTURES / "flask"),
            "BAAI/bge-small-en-v1.5", EMBEDDING_DIM,
        )
        stats = index_repo(class_conn, repo_id, FIXTURES / "flask", embedder)
        yield repo_id, stats
        drop_repo_by_slug(class_conn, slug)

    def test_indexes_without_failures(self, indexed):
        _repo_id, stats = indexed
        assert stats.files_indexed > 20  # fixture has 26 files, some may be __init__-only
        assert stats.files_failed == 0
        assert stats.chunks_created > 100

    def test_every_chunk_has_a_correctly_sized_embedding(self, class_conn, indexed):
        repo_id, _stats = indexed
        with class_conn.cursor() as cur:
            cur.execute(
                "SELECT vector_dims(embedding) FROM chunks WHERE repo_id = %s", (repo_id,)
            )
            dims = {row[0] for row in cur.fetchall()}
        assert dims == {EMBEDDING_DIM}

    def test_the_multi_hop_request_chain_is_indexed_as_distinct_methods(
        self, class_conn, indexed
    ):
        # This exact chain (route -> ... -> full_dispatch_request ->
        # dispatch_request -> view) is the design doc's own example of what
        # naive RAG fails at and call-graph expansion (Phase 6-8) should
        # win on. It has to exist as addressable, correctly-qualified chunks
        # before there's anything for a call graph to connect.
        repo_id, _stats = indexed
        with class_conn.cursor() as cur:
            cur.execute(
                "SELECT qualified_name, kind FROM chunks "
                "WHERE repo_id = %s AND symbol_name = 'full_dispatch_request'",
                (repo_id,),
            )
            assert cur.fetchall() == [("Flask.full_dispatch_request", "method")]

            cur.execute(
                "SELECT qualified_name FROM chunks "
                "WHERE repo_id = %s AND symbol_name = 'dispatch_request' "
                "ORDER BY qualified_name",
                (repo_id,),
            )
            dispatch_names = [r[0] for r in cur.fetchall()]

        # Real Flask defines dispatch_request three times, on three different
        # classes. This is precisely why chunks carry qualified_name and not
        # just symbol_name: retrieving "dispatch_request" is ambiguous, and a
        # citation that can't say WHICH class it came from is a citation that
        # can't be verified. Asserted as a set membership rather than an exact
        # list so a future Flask version adding a fourth doesn't fail the test
        # for the wrong reason.
        assert {
            "Flask.dispatch_request",
            "View.dispatch_request",
            "MethodView.dispatch_request",
        } <= set(dispatch_names)
