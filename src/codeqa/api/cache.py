"""Query result cache, keyed on (repo, normalized question, indexed commit).

Normalized question: lowercased, whitespace collapsed. Nothing more --
stemming or stopword removal would risk merging "how does X work" and "how
does X not work" into the same cache entry, a correctness bug in exchange
for a marginally higher hit rate.

Repo scope alone isn't enough: two different repos can get asked the exact
same question text with completely different correct answers. And question
text alone isn't enough either, even within one repo -- a re-index can
change or remove the exact chunks an old cached answer cited, so the commit
a repo was indexed AT (repos.last_indexed_sha) is part of the key too.
Without it, a cache hit after a re-index would serve an answer citing
chunk_ids that may no longer exist, and Phase 11's grounding check would
then flag citations that were true when written and are stale now -- a
caching bug that would surface, confusingly, as a grounding failure. Folding
the sha into the key instead makes a re-index a natural cache bust with no
invalidation logic required: the old key just stops being requested.

Fails open like rate_limit.py: a Redis outage means every request falls
through to a real cache miss (full retrieval + synthesis), not a 500.
"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass

import redis
import structlog

_log = structlog.get_logger()


@dataclass(frozen=True)
class CachedAnswer:
    answer: str
    chunks: list[dict[str, object]]
    citations_dropped: list[str]


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def _cache_key(repo_id: int, question: str, last_indexed_sha: str | None) -> str:
    q_hash = hashlib.sha256(normalize_question(question).encode()).hexdigest()
    return f"query_cache:{repo_id}:{last_indexed_sha or 'unindexed'}:{q_hash}"


class QueryCache:
    def __init__(self, client: redis.Redis, ttl_seconds: int):
        self._client = client
        self._ttl_seconds = ttl_seconds

    def get(self, repo_id: int, question: str, last_indexed_sha: str | None) -> CachedAnswer | None:
        try:
            raw = self._client.get(_cache_key(repo_id, question, last_indexed_sha))
        except redis.RedisError as exc:
            _log.warning("cache.degraded", op="get", error=str(exc))
            return None
        if raw is None:
            return None
        return CachedAnswer(**json.loads(raw))

    def set(
        self, repo_id: int, question: str, last_indexed_sha: str | None, answer: CachedAnswer
    ) -> None:
        try:
            self._client.set(
                _cache_key(repo_id, question, last_indexed_sha),
                json.dumps(asdict(answer)),
                ex=self._ttl_seconds,
            )
        except redis.RedisError as exc:
            # A failed write degrades to "this answer just won't be cached"
            # -- never lets a caching problem fail an otherwise-successful
            # query response that has already been computed.
            _log.warning("cache.degraded", op="set", error=str(exc))
