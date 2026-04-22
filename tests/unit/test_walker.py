"""File walker: ignore rules and the size cap.

Directory-pattern matching (node_modules/ matching nested files, but not
matching a similarly-named directory like mydist/) is exactly the subtlety
hand-rolled glob matching tends to get wrong -- verified here, not assumed
from pathspec's docs.
"""

from pathlib import Path

from codeqa.indexing.walker import MAX_FILE_SIZE_BYTES, walk_repo


def make_tree(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, content in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return tmp_path


class TestAlwaysIgnored:
    def test_git_directory_excluded(self, tmp_path):
        root = make_tree(tmp_path, {"app.py": "x = 1", ".git/config": "junk"})
        assert set(walk_repo(root)) == {Path("app.py")}

    def test_node_modules_excluded_even_when_nested(self, tmp_path):
        root = make_tree(
            tmp_path,
            {"index.js": "1", "src/node_modules/dep/lib.js": "2"},
        )
        assert set(walk_repo(root)) == {Path("index.js")}

    def test_similarly_named_directory_not_excluded(self, tmp_path):
        # mydist/ must not be caught by a naive substring match on "dist/".
        root = make_tree(tmp_path, {"mydist/real_code.py": "x = 1"})
        assert set(walk_repo(root)) == {Path("mydist/real_code.py")}

    def test_pycache_excluded(self, tmp_path):
        root = make_tree(
            tmp_path, {"app.py": "1", "__pycache__/app.cpython-314.pyc": "2"}
        )
        assert set(walk_repo(root)) == {Path("app.py")}


class TestRepoGitignore:
    def test_repos_own_gitignore_is_honored(self, tmp_path):
        root = make_tree(
            tmp_path,
            {
                ".gitignore": "generated/\n*.local.py\n",
                "app.py": "1",
                "generated/models.py": "2",
                "secrets.local.py": "3",
            },
        )
        # .gitignore itself has no registered language extension and would be
        # filtered by the caller (via detect_language) rather than the
        # walker -- walk_repo only applies ignore rules and the size cap.
        assert set(walk_repo(root)) == {Path("app.py"), Path(".gitignore")}

    def test_missing_gitignore_is_fine(self, tmp_path):
        root = make_tree(tmp_path, {"app.py": "1"})
        assert set(walk_repo(root)) == {Path("app.py")}


class TestSizeCap:
    def test_oversized_file_excluded(self, tmp_path):
        root = make_tree(
            tmp_path,
            {
                "normal.py": "x = 1",
                "vendored.js": "x" * (MAX_FILE_SIZE_BYTES + 1),
            },
        )
        assert set(walk_repo(root)) == {Path("normal.py")}

    def test_file_at_exact_boundary_is_included(self, tmp_path):
        root = make_tree(tmp_path, {"boundary.py": "x" * MAX_FILE_SIZE_BYTES})
        assert set(walk_repo(root)) == {Path("boundary.py")}


class TestGeneral:
    def test_empty_repo_yields_nothing(self, tmp_path):
        assert list(walk_repo(tmp_path)) == []

    def test_paths_are_relative_to_root(self, tmp_path):
        root = make_tree(tmp_path, {"a/b/c.py": "1"})
        result = list(walk_repo(root))
        assert result == [Path("a/b/c.py")]
        assert not result[0].is_absolute()

    def test_results_are_sorted(self, tmp_path):
        root = make_tree(tmp_path, {"z.py": "1", "a.py": "2", "m.py": "3"})
        result = list(walk_repo(root))
        assert result == sorted(result)
