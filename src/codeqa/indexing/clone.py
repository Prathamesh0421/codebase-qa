"""Safely fetch a user-supplied git URL onto local disk.

Two separate, independently testable steps, deliberately not one function:

  validate_clone_url -- the SSRF guard. Rejects anything that isn't
    http(s) to an allowlisted host, before any network call happens.
  clone_repo         -- the actual `git clone`, with a subprocess timeout
    and a post-clone size check.

Splitting them means the guard can be unit-tested with no network and no
git binary (just string/host checks), and clone_repo's mechanics (depth,
timeout, size enforcement) can be tested against a real local repo without
needing a real SSRF attempt to construct. safe_clone composes both for the
one real call site (the worker) that needs the whole thing.

tree-sitter parsing arbitrary source is safe by construction -- it never
executes what it reads. Fetching a URL someone else supplied is not: it's
a live network request the server makes on the caller's behalf, which is
exactly what SSRF exploits.
"""

import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit


class UnsafeCloneURL(ValueError):
    pass


class CloneFailed(RuntimeError):
    pass


def validate_clone_url(url: str, allowed_hosts: tuple[str, ...]) -> None:
    """Raise UnsafeCloneURL unless url is http(s) to an allowlisted host.

    Host matching is exact and case-insensitive, not a suffix match --
    "github.com.attacker.example" must never pass because it ends in a
    string that contains "github.com".
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise UnsafeCloneURL(
            f"scheme {parts.scheme!r} is not allowed -- only http/https clone URLs are accepted"
        )
    host = (parts.hostname or "").lower()
    if host not in {h.lower() for h in allowed_hosts}:
        raise UnsafeCloneURL(f"host {host!r} is not in the allowed clone hosts {allowed_hosts}")


def _dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def clone_repo(url: str, dest: Path, timeout_seconds: int, max_mb: int) -> str:
    """git clone --depth 1 url into dest, then enforce a post-clone size cap.

    No pre-clone size cap exists -- git doesn't know a remote's total size
    before fetching it, so "clone, measure, delete-and-fail-if-too-big" is
    the only point this can actually be enforced, not a compromise. Returns
    the resolved HEAD commit SHA, which is what repos.last_indexed_sha
    needs to record what was actually indexed.

    dest must not already exist -- callers that want to re-clone are
    responsible for removing a stale directory first (see worker.py); this
    function only ever creates one, never silently overwrites one.
    """
    if dest.exists():
        raise CloneFailed(f"clone destination {dest} already exists")

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            timeout=timeout_seconds,
            capture_output=True,
            check=True,
            text=True,
        )
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(dest, ignore_errors=True)
        raise CloneFailed(f"clone of {url} exceeded {timeout_seconds}s timeout") from exc
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(dest, ignore_errors=True)
        raise CloneFailed(f"git clone failed: {exc.stderr.strip()}") from exc

    size_mb = _dir_size_mb(dest)
    if size_mb > max_mb:
        shutil.rmtree(dest, ignore_errors=True)
        raise CloneFailed(f"clone of {url} is {size_mb:.1f}MB, over the {max_mb}MB cap")

    sha_result = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        capture_output=True,
        check=True,
        text=True,
    )
    return sha_result.stdout.strip()


def safe_clone(
    url: str, dest: Path, allowed_hosts: tuple[str, ...], timeout_seconds: int, max_mb: int
) -> str:
    """validate_clone_url then clone_repo -- the one call real workers make."""
    validate_clone_url(url, allowed_hosts)
    return clone_repo(url, dest, timeout_seconds, max_mb)


def check_disk_quota(path: Path, min_free_mb: int) -> None:
    """Refuse to start a new clone at all if free space is already below
    threshold -- a check made before cloning, not a cap enforced during it,
    so one huge repo can't starve every other job's disk headroom.
    """
    usage = shutil.disk_usage(path)
    free_mb = usage.free / (1024 * 1024)
    if free_mb < min_free_mb:
        raise CloneFailed(f"only {free_mb:.0f}MB free at {path}, below the {min_free_mb}MB minimum")
