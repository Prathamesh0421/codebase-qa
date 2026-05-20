"""Redis token bucket rate limiting, per API key.

Token bucket over a fixed window: a fixed window ("60 requests per
clock-minute") lets a client send 60 requests in the last second of one
window and 60 more in the first second of the next -- 120 requests in under
two seconds, double the intended rate, purely from window-boundary timing.
A token bucket has no windows to align against -- tokens refill
continuously, so the maximum burst is always exactly the bucket's capacity,
regardless of when a request happens to land.

Implemented as a single Lua script run via EVAL so the
read-refill-consume-write sequence is atomic against concurrent requests
for the same key. A plain GET-then-SET from Python would race two
simultaneous requests into both reading the same starting token count and
both being allowed through.

Fails OPEN, not closed: a Redis outage means unmetered traffic, not a
service outage. For a Q&A service, availability is worth more than strict
enforcement -- the honest cost is that a Redis outage also removes abuse
protection for its duration, a choice, not an accident.
"""

import time

import redis
import structlog

_log = structlog.get_logger()

_TOKEN_BUCKET_SCRIPT = """
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

local bucket = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(bucket[1])
local ts = tonumber(bucket[2])

if tokens == nil then
  tokens = capacity
  ts = now
end

local elapsed = math.max(0, now - ts)
tokens = math.min(capacity, tokens + elapsed * refill_rate)

local allowed = 0
if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
end

redis.call('HMSET', KEYS[1], 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', KEYS[1], 3600)

return allowed
"""


class RateLimiter:
    def __init__(self, client: redis.Redis):
        self._client = client
        self._script = client.register_script(_TOKEN_BUCKET_SCRIPT)

    def allow(self, api_key_id: int, capacity_rpm: int) -> bool:
        try:
            refill_per_second = capacity_rpm / 60.0
            result = self._script(
                keys=[f"ratelimit:{api_key_id}"],
                args=[capacity_rpm, refill_per_second, time.time()],
            )
            return bool(result)
        except redis.RedisError as exc:
            _log.warning("rate_limit.degraded", error=str(exc))
            return True
