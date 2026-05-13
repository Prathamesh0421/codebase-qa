"""clone_repo against a real local git repository -- no network, no
Postgres, just the real git binary and a repo built in tmp_path. Marked
integration because it shells out to a real subprocess, the same bar
test_naive_retrieval.py etc. use for "needs something real".
"""

import subprocess
from pathlib import Path

import pytest

from codeqa.indexing.clone import CloneFailed, clone_repo

pytestmark = pytest.mark.integration


def _run(*args: str, cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def local_repo(tmp_path) -> Path:
    """A real git repo with two commits, reachable via a file:// URL --
    clone_repo itself has no scheme restriction (that's validate_clone_url's
    job, tested separately), so file:// is a legitimate way to exercise real
    clone mechanics without a network dependency.
    """
    repo = tmp_path / "source"
    repo.mkdir()
    _run("git", "init", "-q", cwd=repo)
    _run("git", "config", "user.email", "test@example.com", cwd=repo)
    _run("git", "config", "user.name", "Test", cwd=repo)
    (repo / "a.txt").write_text("first")
    _run("git", "add", "a.txt", cwd=repo)
    _run("git", "commit", "-q", "-m", "first commit", cwd=repo)
    (repo / "a.txt").write_text("second")
    _run("git", "add", "a.txt", cwd=repo)
    _run("git", "commit", "-q", "-m", "second commit", cwd=repo)
    return repo


class TestCloneRepo:
    def test_clones_successfully_and_returns_the_head_sha(self, local_repo, tmp_path):
        dest = tmp_path / "clone"
        sha = clone_repo(f"file://{local_repo}", dest, timeout_seconds=30, max_mb=500)

        assert (dest / "a.txt").read_text() == "second"
        expected_sha = subprocess.run(
            ["git", "-C", str(local_repo), "rev-parse", "HEAD"],
            capture_output=True, check=True, text=True,
        ).stdout.strip()
        assert sha == expected_sha

    def test_depth_1_means_only_one_commit_is_present(self, local_repo, tmp_path):
        dest = tmp_path / "clone"
        clone_repo(f"file://{local_repo}", dest, timeout_seconds=30, max_mb=500)

        log = subprocess.run(
            ["git", "-C", str(dest), "log", "--oneline"],
            capture_output=True, check=True, text=True,
        ).stdout.strip().splitlines()
        assert len(log) == 1

    def test_a_clone_exceeding_the_size_cap_is_removed_and_raises(self, local_repo, tmp_path):
        (local_repo / "big.bin").write_bytes(b"\0" * (2 * 1024 * 1024))
        _run("git", "add", "big.bin", cwd=local_repo)
        _run("git", "commit", "-q", "-m", "add big file", cwd=local_repo)

        dest = tmp_path / "clone"
        with pytest.raises(CloneFailed, match="over the"):
            clone_repo(f"file://{local_repo}", dest, timeout_seconds=30, max_mb=1)
        assert not dest.exists()

    def test_an_existing_destination_is_refused_rather_than_overwritten(self, local_repo, tmp_path):
        dest = tmp_path / "clone"
        dest.mkdir()
        with pytest.raises(CloneFailed, match="already exists"):
            clone_repo(f"file://{local_repo}", dest, timeout_seconds=30, max_mb=500)

    def test_a_nonexistent_source_fails_cleanly_and_cleans_up(self, tmp_path):
        dest = tmp_path / "clone"
        missing = tmp_path / "does-not-exist"
        with pytest.raises(CloneFailed, match="git clone failed"):
            clone_repo(f"file://{missing}", dest, timeout_seconds=30, max_mb=500)
        assert not dest.exists()
