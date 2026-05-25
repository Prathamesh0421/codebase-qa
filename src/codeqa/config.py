"""Typed configuration.

Config is a validated object rather than scattered os.environ.get calls, for
three reasons that matter here:

  * Wrong values fail at startup with a field name, not at 3am inside a
    retrieval call with a TypeError.
  * EMBEDDING_DIM is an int the migration runner interpolates into DDL. A
    string that looks like an int would produce a schema nobody intended.
  * RETRIEVAL_STRATEGY is a closed set. The eval story depends on all three
    strategies staying selectable, so a typo must be a startup error rather
    than a silent fallback to the default.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RetrievalStrategy = Literal["naive", "hybrid", "hybrid_graph"]
EmbeddingProvider = Literal["local", "hosted"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CODEQA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )

    # ------------------------------------------------------------ storage
    database_url: PostgresDsn = Field(
        description="Postgres with the vector extension available.",
    )
    redis_url: RedisDsn = Field(
        default=RedisDsn("redis://localhost:6379/0"),
        description="Query cache and per-key rate-limit buckets.",
    )

    # ------------------------------------------------------------ embeddings
    # Provider is chosen by indexing throughput, not answer quality: a
    # rate-limited hosted API cannot bulk-embed thousands of chunks
    # reproducibly, and irreproducible eval runs are worse than marginally
    # weaker embeddings. Local for dev/CI/evals, hosted for the deployed image
    # where ~2GB of torch is not welcome.
    embedding_provider: EmbeddingProvider = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Interpolated into CREATE TABLE as vector(N). Changing this requires a new
    # migration and a re-index -- the runner refuses to boot on a mismatch
    # rather than let vectors from different models be compared.
    embedding_dim: int = 384

    embedding_batch_size: int = 64

    # Falls back to llm_api_key when unset -- see embedding_api_key property
    # below. Only needs its own value when embeddings and synthesis use
    # different hosted providers (e.g. OpenAI embeddings, Gemini synthesis).
    embedding_provider_api_key: str | None = None

    # ------------------------------------------------------------ retrieval
    # Never remove "naive". The entire measurement story is
    # naive -> hybrid -> hybrid_graph, and the baseline has to stay runnable
    # at any commit for the comparison to be reproducible.
    #
    # Default is "naive" until Phase 8 ships hybrid and hybrid_graph -- a
    # default pointing at a strategy that doesn't exist yet would make `ask`
    # fail out of the box with a NotImplementedError nobody asked for.
    # Becomes "hybrid_graph" once all three are implemented.
    retrieval_strategy: RetrievalStrategy = "naive"
    retrieval_top_k: int = 10

    # Bounds on call-graph expansion. Depth is the interview-relevant one: the
    # call graph is cyclic, so traversal terminates on depth plus the CTE's
    # CYCLE clause rather than on the graph being acyclic.
    graph_max_depth: int = 2
    graph_max_nodes: int = 40

    # ------------------------------------------------------------ llm
    llm_model: str = "gemini/gemini-3.6-flash"
    llm_api_key: str | None = None
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3

    # ------------------------------------------------------------ agents
    # Total locate attempts the trace->locate retry edge is allowed, INCLUDING
    # the first one -- 2 means one retry. Not the same knob as llm_max_retries
    # above, which is litellm's own retry-on-transient-failure count; this one
    # bounds a deliberate re-query loop over good responses that just weren't
    # enough context, not error recovery.
    agent_max_attempts: int = 2

    # ------------------------------------------------------------ ingestion
    clone_max_mb: int = 500
    clone_timeout_seconds: int = 300
    # Server-side cloning of user-supplied URLs is an SSRF surface. Parsing is
    # safe by construction -- tree-sitter never executes what it reads -- but
    # fetching is not. An allowlist of known public hosting domains sidesteps
    # resolving DNS and checking IP ranges entirely -- simpler and more
    # auditable than a private-IP check, and immune to the DNS-rebinding
    # trick that check has to guard against separately.
    allowed_clone_hosts: tuple[str, ...] = ("github.com", "gitlab.com", "bitbucket.org")

    # Where a git_url repo's clone persists on disk, keyed by repo_id. Not a
    # tempdir cleaned up after each job -- Phase 13's incremental re-index
    # needs the previous checkout to diff against, so a repo's clone has to
    # survive between indexing runs, not just one job's lifetime.
    clone_workdir: str = "./data/repos"
    # Checked before starting a new clone, not enforced during one -- one
    # huge repo shouldn't be able to starve every other job's disk headroom.
    disk_min_free_mb: int = 1024

    worker_poll_interval_seconds: float = 2.0
    job_heartbeat_interval_seconds: float = 5.0
    # A running job whose heartbeat is older than this is presumed to belong
    # to a dead worker and gets reclaimed back to queued. Comfortably above
    # job_heartbeat_interval_seconds so a live worker's normal heartbeat
    # cadence is never mistaken for a stall.
    job_stale_after_seconds: int = 60
    # Reclaiming a job that will never succeed (a permanently malformed URL,
    # say) forever would loop indefinitely instead of ever reporting failure.
    job_max_attempts: int = 3

    # ------------------------------------------------------------ api
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    rate_limit_rpm: int = 60
    cache_ttl_seconds: int = 3600

    # ------------------------------------------------------------ observability
    otel_endpoint: str | None = None
    log_level: str = "INFO"
    environment: Literal["dev", "test", "prod"] = "dev"

    @field_validator("embedding_dim")
    @classmethod
    def _dim_is_sane(cls, v: int) -> int:
        # Not a real bound, just a guard against a typo reaching DDL.
        if not 32 <= v <= 4096:
            raise ValueError(f"embedding_dim={v} is outside any plausible range")
        return v

    @property
    def embedding_api_key(self) -> str | None:
        """The key a hosted embedding provider should use.

        Falls back to llm_api_key: the common case is one provider (e.g.
        Gemini) serving both embeddings and synthesis, and requiring two
        separate keys for that case would be friction with no benefit.
        embedding_provider_api_key is the escape hatch for when they diverge.
        """
        return self.embedding_provider_api_key or self.llm_api_key

    @property
    def dsn(self) -> str:
        """psycopg wants a plain string, not a PostgresDsn."""
        return str(self.database_url)


@lru_cache
def get_settings() -> Settings:
    """Cached so config is parsed once per process.

    FastAPI dependencies and the CLI both call this; the cache also makes it a
    single override point in tests.
    """
    return Settings()  # type: ignore[call-arg]
