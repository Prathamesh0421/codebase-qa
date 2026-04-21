"""Enumerate the files in a repo worth indexing.

Two filters, applied in order: ignore rules (this file), then "is this a
language we have a registered extension for" (checked by the caller via
codeqa.languages.detect_language). Files with no registered extension --
docs, configs, data, images -- are skipped entirely rather than given some
fallback chunk. This project indexes source code; a stated scope boundary,
not an oversight.
"""

from collections.abc import Iterator
from pathlib import Path

import pathspec

# Always ignored, on top of whatever the repo's own .gitignore says. These
# are directories no one asks "how does this code work" about: VCS metadata,
# dependency trees other tooling already owns, and build output that is a
# deterministic function of the source already being indexed.
_ALWAYS_IGNORE = pathspec.PathSpec.from_lines(
    "gitignore",
    [
        ".git/",
        "node_modules/",
        "__pycache__/",
        ".venv/",
        "venv/",
        "dist/",
        "build/",
        "*.egg-info/",
        ".mypy_cache/",
        ".pytest_cache/",
        ".ruff_cache/",
    ],
)

# A single source file past this size is almost always generated or vendored
# (a minified bundle, a lockfile-adjacent artifact) rather than something a
# human wrote and would ask about. Skipped rather than chunked slowly into one
# giant, low-value unit. Not a config knob: there's no per-project reason to
# tune it, and it exists to bound pathological inputs, not to be adjusted.
MAX_FILE_SIZE_BYTES = 512_000


def _load_gitignore(root: Path) -> pathspec.PathSpec:
    gitignore = root / ".gitignore"
    if not gitignore.is_file():
        return pathspec.PathSpec.from_lines("gitignore", [])
    return pathspec.PathSpec.from_lines("gitignore", gitignore.read_text().splitlines())


def walk_repo(root: Path) -> Iterator[Path]:
    """Yield indexable file paths under root, relative to root.

    Ignore rules are evaluated on the relative path, matching how
    .gitignore itself interprets patterns -- an absolute-path match would
    silently never fire since gitignore patterns are repo-relative.
    """
    root = root.resolve()
    repo_ignore = _load_gitignore(root)

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        rel = path.relative_to(root)
        rel_str = rel.as_posix()

        if _ALWAYS_IGNORE.match_file(rel_str) or repo_ignore.match_file(rel_str):
            continue

        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            continue

        yield rel
