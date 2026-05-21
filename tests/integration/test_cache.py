"""api/cache.py's QueryCache against a real Redis instance.

The property that matters most here isn't "get/set round-trips" -- it's
that the key genuinely includes all three of repo, question, and indexed
commit, so none of them can silently leak an answer meant for a different
scope.
"""

import os

import pytest
import redis

from codeqa.api.cache import CachedAnswer, QueryCache

pytestmark = pytest.mark.integration


def _redis_url() -> str:
    return os.environ.get("CODEQA_TEST_REDIS_URL", "redis://localhost:6379/0")


@pytest.fixture
def redis_client():
    client = redis.Redis.from_url(_redis_url())
    yield client
    client.close()


@pytest.fixture
def cache(redis_client):
    return QueryCache(redis_client, ttl_seconds=60)


def _answer(text: str = "an answer") -> CachedAnswer:
    return CachedAnswer(
        answer=text,
        chunks=[{"citation": "a.py:1-2", "score": 0.9, "symbol": "f"}],
        citations_dropped=[],
    )


class TestQueryCache:
    def test_a_set_answer_is_retrieved_by_the_same_scope(self, cache):
        cache.set(1, "how does X work", "sha-a", _answer("cached text"))
        hit = cache.get(1, "how does X work", "sha-a")
        assert hit is not None
        assert hit.answer == "cached text"

    def test_a_miss_returns_none_not_an_error(self, cache):
        assert cache.get(999999, "never asked", "sha-x") is None

    def test_different_repos_do_not_share_a_cache_entry(self, cache):
        cache.set(1, "how does X work", "sha-a", _answer("repo 1's answer"))
        assert cache.get(2, "how does X work", "sha-a") is None

    def test_a_reindex_busts_the_cache_by_changing_the_key(self, cache):
        # No explicit invalidation logic -- the old sha's key just stops
        # being the one a post-reindex request asks for.
        cache.set(1, "how does X work", "sha-old", _answer("stale, pre-reindex answer"))
        assert cache.get(1, "how does X work", "sha-new") is None

    def test_question_normalization_makes_near_duplicates_hit(self, cache):
        cache.set(1, "How Does X Work?  ", "sha-a", _answer("normalized hit"))
        hit = cache.get(1, "  how   does x work?", "sha-a")
        assert hit is not None
        assert hit.answer == "normalized hit"

    def test_semantically_different_questions_do_not_collide(self, cache):
        # A naive normalization (e.g. stripping "not") could merge these --
        # deliberately not stemming or removing stopwords for this reason.
        cache.set(1, "how does X work", "sha-a", _answer("positive answer"))
        assert cache.get(1, "how does X not work", "sha-a") is None

    def test_an_unreachable_redis_degrades_to_a_miss_on_get(self):
        unreachable = redis.Redis(host="localhost", port=1, socket_connect_timeout=0.2)
        cache = QueryCache(unreachable, ttl_seconds=60)
        assert cache.get(1, "anything", "sha-a") is None

    def test_an_unreachable_redis_degrades_silently_on_set(self):
        unreachable = redis.Redis(host="localhost", port=1, socket_connect_timeout=0.2)
        cache = QueryCache(unreachable, ttl_seconds=60)
        cache.set(1, "anything", "sha-a", _answer())  # must not raise
