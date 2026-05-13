"""validate_clone_url: pure string/host checks, no network, no git binary."""

import pytest

from codeqa.indexing.clone import UnsafeCloneURL, validate_clone_url

_HOSTS = ("github.com", "gitlab.com", "bitbucket.org")


class TestValidateCloneUrl:
    def test_accepts_an_allowlisted_https_host(self):
        validate_clone_url("https://github.com/psf/requests.git", _HOSTS)

    def test_accepts_an_allowlisted_http_host(self):
        validate_clone_url("http://gitlab.com/group/project.git", _HOSTS)

    def test_rejects_a_host_not_in_the_allowlist(self):
        with pytest.raises(UnsafeCloneURL, match="not in the allowed"):
            validate_clone_url("https://evil.example.com/repo.git", _HOSTS)

    def test_rejects_localhost(self):
        with pytest.raises(UnsafeCloneURL):
            validate_clone_url("http://localhost:5432/repo.git", _HOSTS)

    def test_rejects_a_cloud_metadata_style_ip(self):
        with pytest.raises(UnsafeCloneURL):
            validate_clone_url("http://169.254.169.254/latest/meta-data", _HOSTS)

    def test_rejects_file_scheme(self):
        with pytest.raises(UnsafeCloneURL, match="scheme"):
            validate_clone_url("file:///etc/passwd", _HOSTS)

    def test_rejects_ssh_scheme(self):
        with pytest.raises(UnsafeCloneURL, match="scheme"):
            validate_clone_url("ssh://git@github.com/psf/requests.git", _HOSTS)

    def test_rejects_a_lookalike_host_with_the_real_host_as_a_suffix(self):
        # "github.com.attacker.example" contains "github.com" as a
        # substring but is a completely different host -- exact match only.
        with pytest.raises(UnsafeCloneURL):
            validate_clone_url("https://github.com.attacker.example/repo.git", _HOSTS)

    def test_host_matching_is_case_insensitive(self):
        validate_clone_url("https://GitHub.com/psf/requests.git", _HOSTS)

    def test_a_custom_allowlist_is_respected(self):
        validate_clone_url("https://git.internal.example/repo.git", ("git.internal.example",))
        with pytest.raises(UnsafeCloneURL):
            validate_clone_url("https://github.com/psf/requests.git", ("git.internal.example",))
