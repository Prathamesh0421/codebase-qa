"""blob_sha is a pure function -- no DB needed, tested separately from the
integration-level pipeline tests in tests/integration/test_pipeline.py.
Public since Phase 13: indexing/incremental.py reuses it too.
"""

import hashlib
import subprocess

from codeqa.indexing.pipeline import blob_sha


def test_matches_real_git_hash_object(tmp_path):
    # Verified once by hand during design (docs/deep-dive.html); pinned here
    # so the algorithm can't silently drift from git's own.
    f = tmp_path / "sample.py"
    f.write_text("def greet(name):\n    return name\n")

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    real = subprocess.run(
        ["git", "hash-object", "sample.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert blob_sha(f.read_bytes()) == real


def test_empty_content():
    # git hash-object on an empty file is a well-known constant.
    assert blob_sha(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


def test_different_content_different_hash():
    assert blob_sha(b"a") != blob_sha(b"b")


def test_is_not_plain_sha1_of_content():
    # The git blob format prepends "blob {len}\0" before hashing -- confirms
    # the header is actually applied, not accidentally hashing raw content.
    content = b"hello"
    assert blob_sha(content) != hashlib.sha1(content).hexdigest()
