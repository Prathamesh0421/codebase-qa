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
        default="redis://localhost:6379/0",
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

    # ------------------------------------------------------------ retrieval
    # Never remove "naive". The entire measurement story is
    # naive -> hybrid -> hybrid_graph, and the baseline has to stay runnable
    # at any commit for the comparison to be reproducible.
    retrieval_strategy: RetrievalStrategy = "hybrid_graph"
    retrieval_top_k: int = 10

    # Bounds on call-graph expansion. Depth is the interview-relevant one: the
    # call graph is cyclic, so traversal terminates on depth plus the CTE's
    # CYCLE clause rather than on the graph being acyclic.
    graph_max_depth: int = 2
    graph_max_nodes: int = 40

    # ------------------------------------------------------------ llm
    llm_model: str = "gemini/gemini-2.0-flash"
    llm_api_key: str | None = None
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3

    # ------------------------------------------------------------ ingestion
    clone_max_mb: int = 500
    clone_timeout_seconds: int = 300
    # Server-side cloning of user-supplied URLs is an SSRF surface. Parsing is
    # safe by construction -- tree-sitter never executes what it reads -- but
    # fetching is not.
    allowed_clone_hosts: tuple[str, ...] = ("github.com", "gitlab.com", "bitbucket.org")

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
