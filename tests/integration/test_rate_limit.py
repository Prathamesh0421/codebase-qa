"""api/rate_limit.py's RateLimiter against a real Redis instance.

The one thing this suite is specifically here to catch: HMSET writes
"tokens" as a Lua number that redis-py returns as a bytes string on the
next HMGET, and the Lua script's tonumber() has to round-trip that back
into a real float across calls. A naive implementation could silently
truncate a fractional refill to zero -- indistinguishable from "no time
passed" in a burst-then-refuse test alone, which is why refill is tested
as its own case with a real sleep, not inferred from the burst test.
"""

import os
import time

import pytest
import redis

from codeqa.api.rate_limit import RateLimiter

pytestmark = pytest.mark.integration


def _redis_url() -> str:
    return os.environ.get("CODEQA_TEST_REDIS_URL", "redis://localhost:6379/0")


@pytest.fixture
def redis_client():
    client = redis.Redis.from_url(_redis_url())
    yield client
    client.close()


@pytest.fixture
def limiter(redis_client):
    return RateLimiter(redis_client)


def _unique_key(request) -> int:
    # A distinct bucket per test, derived from the test name -- so tests
    # never share Redis state and don't need explicit cleanup (EXPIRE in
    # the Lua script reclaims it eventually).
    return hash(request.node.name) % 1_000_000


class TestRateLimiter:
    def test_a_burst_up_to_capacity_is_allowed_then_the_next_is_refused(
        self, limiter, request
    ):
        key = _unique_key(request)
        capacity = 3
        results = [limiter.allow(key, capacity) for _ in range(capacity)]
        assert results == [True, True, True]
        assert limiter.allow(key, capacity) is False

    def test_tokens_genuinely_refill_after_the_capacity_is_exhausted(self, limiter, request):
        # capacity_rpm=60 -> bucket holds 60 tokens, refilling at 1/second.
        # Drain the whole bucket, confirm the very next call is refused,
        # then wait past one refill interval and confirm exactly one more
        # request succeeds -- not just "eventually unstuck", which a bug
        # that refills in huge jumps (or never truncates to a huge number)
        # could also produce.
        key = _unique_key(request)
        capacity_rpm = 60
        for _ in range(capacity_rpm):
            assert limiter.allow(key, capacity_rpm) is True
        assert limiter.allow(key, capacity_rpm) is False

        time.sleep(1.1)
        assert limiter.allow(key, capacity_rpm) is True
        assert limiter.allow(key, capacity_rpm) is False

    def test_different_keys_have_independent_buckets(self, limiter, request):
        key_a = _unique_key(request)
        key_b = key_a + 1
        capacity = 1
        assert limiter.allow(key_a, capacity) is True
        assert limiter.allow(key_a, capacity) is False
        # key_b's bucket was never touched -- exhausting key_a must not
        # have leaked into it.
        assert limiter.allow(key_b, capacity) is True

    def test_an_unreachable_redis_fails_open_instead_of_raising(self, request):
        # A real connection failure -- nothing is listening on this port --
        # not a mock, so this exercises the actual redis-py exception path
        # rate_limit.py is written to catch.
        unreachable = redis.Redis(host="localhost", port=1, socket_connect_timeout=0.2)
        limiter = RateLimiter(unreachable)
        assert limiter.allow(_unique_key(request), 1) is True
