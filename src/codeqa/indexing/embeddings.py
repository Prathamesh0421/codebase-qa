"""Turn chunk text into vectors, behind one interface with two implementations.

Provider choice is driven by indexing throughput, not answer quality:

  local  -- sentence-transformers. Free, deterministic, no rate limits.
            Used in dev, CI, and the eval harness, where reproducibility
            matters more than deployment footprint. Measured: batching
            (batch_size=64) is ~3.5x faster than one embed call per chunk on
            200 chunks with BAAI/bge-small-en-v1.5 -- see
            tests/unit/test_embeddings.py. That's why index_repo() in
            pipeline.py collects chunks across the whole repo before
            embedding, rather than embedding per file.

  hosted -- LiteLLM, provider-agnostic. Used for the deployed image, which
            omits the local-embeddings extra to keep ~2GB of torch out of the
            container. Correctness here is verified against LiteLLM's actual
            request-building code (confirmed its `dimensions` argument maps
            to Gemini's `outputDimensionality` field), and tested by mocking
            at the litellm.embedding() boundary -- there is no live API key
            in this environment, so real network behavior against Gemini or
            any other hosted provider is NOT verified here. State that
            honestly rather than claim more than was tested.

Every embed() call is dimension-checked against config before its result is
allowed to reach the pipeline -- the same invariant Phase 1's migration
runner enforces at the schema level (check_embedding_dim), caught here too
because a silent mismatch corrupts every downstream similarity score without
ever raising an exception.
"""

from typing import Protocol

import litellm


class DimensionMismatch(RuntimeError):
    """An embedding call returned vectors of the wrong size."""


class EmbeddingProvider(Protocol):
    dimension: int
    # Recorded per repo (repos.embedding_model) so a provider swap is
    # detected rather than silently mixing embeddings from two models that
    # happen to share a dimension. See store._check_embedder_matches_repo.
    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Order-preserving: result[i] is texts[i]'s vector."""
        ...


def _check_dimensions(vectors: list[list[float]], expected: int, provider: str) -> None:
    for i, v in enumerate(vectors):
        if len(v) != expected:
            raise DimensionMismatch(
                f"{provider} returned a {len(v)}-dim vector at index {i}, "
                f"expected {expected}. Embedding model or provider config "
                f"disagrees with settings.embedding_dim -- refusing to write "
                f"vectors that would silently corrupt similarity search."
            )


class LocalEmbedder:
    """sentence-transformers, loaded once and reused across calls.

    Requires the `local-embeddings` extra (sentence-transformers, torch) --
    intentionally not a hard dependency of the package, since the deployed
    image never imports this class.

    batch_size is a construction-time property, not a per-call argument, so
    embed()'s signature matches HostedEmbedder's exactly -- callers work
    against EmbeddingProvider without knowing which implementation they have.
    """

    def __init__(self, model_name: str, dimension: int, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.dimension = dimension
        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, batch_size=self._batch_size, show_progress_bar=False)
        result = vectors.tolist()
        _check_dimensions(result, self.dimension, "LocalEmbedder")
        return result


class HostedEmbedder:
    """LiteLLM-backed embedding call, provider chosen by the model string
    (e.g. "gemini/gemini-embedding-001"), never hardcoded to one vendor.

    No client-side request batching: the whole input list goes in one
    litellm.embedding() call. Real hosted APIs have per-request size limits
    that would eventually require splitting large inputs, but that number is
    provider-specific and unverifiable without a live key in this
    environment -- deferred rather than guessed at.
    """

    def __init__(self, model: str, dimension: int, api_key: str | None = None):
        self.model_name = model
        self.dimension = dimension
        self._api_key = api_key

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = litellm.embedding(
            model=self.model_name,
            input=texts,
            dimensions=self.dimension,
            api_key=self._api_key,
        )
        result = [item.embedding for item in response.data]
        _check_dimensions(result, self.dimension, "HostedEmbedder")
        return result


def build_embedder(
    provider: str,
    model: str,
    dimension: int,
    batch_size: int = 64,
    api_key: str | None = None,
) -> EmbeddingProvider:
    """Construct the configured provider. Single call site so config.py's
    embedding_provider setting is the only place this decision is made.
    """
    if provider == "local":
        return LocalEmbedder(model, dimension, batch_size)
    if provider == "hosted":
        return HostedEmbedder(model, dimension, api_key)
    raise ValueError(f"unknown embedding provider: {provider!r}")
