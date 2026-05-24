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
            omits the local-embeddings extra: sentence-transformers plus even
            a CPU-only torch peaks around 800MB RSS just loading the model
            and embedding a small batch (measured directly, PyTorch and ONNX
            Runtime backends both), over the 512MB free-tier RAM cap of the
            host this image targets. Provider is Cohere (embed-v4.0), not
            Gemini -- a real deployed run against a live Gemini key measured
            its free tier at 100 embed requests per MINUTE, which fails any
            repo bigger than ~100 chunks deterministically no matter how
            retries are tuned.

            Cohere's trial is better but still a real ceiling, not a fix:
            2,000 inputs/minute sounds generous, but the binding constraint
            turned out to be tokens, not request count -- 100,000 tokens per
            minute (found the same way, by actually indexing Flask against a
            live key: it fits comfortably under Gemini's 100-item cap in
            item count but blows well past Cohere's token cap, since Flask's
            source is ~200k+ tokens of chunk text). A repo has to fit under
            ~100k tokens to index in a single quota window on this tier --
            true for a small-to-medium repo, not for something Flask-sized.
            This is a stated scope limit of the deployed instance, not
            something batch_size or max_retries papers over: no client-side
            retry changes how many tokens fit through a per-minute cap.
            (Also found: litellm surfaces Cohere's 429 as a bare
            APIConnectionError rather than RateLimitError, unlike Gemini's
            proper RateLimitError -- num_retries may not even engage for it.
            Noted as a known limitation of the dependency, not fixed here,
            since pacing wouldn't fix the underlying ceiling anyway.)

            The Gemini run also surfaced a hard per-call cap ("at most 100
            requests can be in one batch"), which is what turned batch_size
            from a LocalEmbedder-only construction arg into something
            HostedEmbedder chunks requests by too -- Cohere's own per-call
            cap is 96, and a different provider might allow more or fewer
            still, which is exactly why it's a constructor argument and not
            a hardcoded constant. `dimensions` is verified against LiteLLM's
            actual request-building code for both providers (Gemini's
            `outputDimensionality`, Cohere's `output_dimension` -- the latter
            only valid on embed-v4 and newer, one of {256, 512, 1024, 1536}).

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
        # sentence-transformers' encode() is untyped (returns Any), so
        # tolist()'s result is too -- the annotation here is the actual
        # fix, not a cast: it's what tolist() on a 2D float array returns.
        result: list[list[float]] = vectors.tolist()
        _check_dimensions(result, self.dimension, "LocalEmbedder")
        return result


class HostedEmbedder:
    """LiteLLM-backed embedding call, provider chosen by the model string
    (e.g. "cohere/embed-v4.0"), never hardcoded to one vendor -- proved out
    by actually swapping it once already: the deployed model went from
    Gemini to Cohere as a one-line config change (model string + dimension),
    no code here changed, because nothing here is Gemini- or Cohere-specific.

    batch_size chunks the input list across multiple litellm.embedding()
    calls -- both hosted providers tried so far reject an unbounded single
    request (Gemini: "at most 100 requests can be in one batch"; Cohere:
    documented cap of 96 texts per call), found the first time by running a
    real repo-wide index against a live key rather than assumed ahead of
    time. Default matches LocalEmbedder's (64), comfortably under both
    limits; a different hosted provider with a different real limit is
    exactly why this is a constructor argument, not a hardcoded constant.

    max_retries matters more here than batch_size alone fixes for a
    per-minute-quota provider: a live Gemini run showed its free tier is
    limited per-MINUTE-of-items, not just per-call
    ("EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier",
    limit 100) -- indexing a few hundred chunks legitimately spans more
    than one quota window, 429s included, no matter how the batches are
    sized. num_retries is the kwarg litellm's generic retry-with-backoff
    wrapper reads (verified against litellm's own utils.py: unlike
    `max_retries`, which is silently dropped for non-OpenAI/Azure providers
    -- see synthesis.py's litellm.completion calls, which may be no-ops for
    Gemini for exactly that reason). Kept because it's a correct, real fix
    for a transient burst -- but Cohere's actual trial ceiling turned out to
    be a token-per-minute budget (100k), not request count, which no amount
    of retrying gets past: a repo whose chunk text exceeds that per minute
    hits a hard wall retries cannot solve, only pacing could (not
    implemented -- see this module's docstring). num_retries still matters
    for genuinely transient 429s under that ceiling; it just isn't a fix
    for exceeding the ceiling itself.
    """

    def __init__(
        self,
        model: str,
        dimension: int,
        api_key: str | None = None,
        batch_size: int = 64,
        max_retries: int = 0,
    ):
        self.model_name = model
        self.dimension = dimension
        self._api_key = api_key
        self._batch_size = batch_size
        self._max_retries = max_retries

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        result: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = litellm.embedding(
                model=self.model_name,
                input=batch,
                dimensions=self.dimension,
                api_key=self._api_key,
                num_retries=self._max_retries,
            )
            # LiteLLM's EmbeddingResponse.data items are not one consistent
            # shape across providers -- Gemini's are attribute-access
            # objects (item.embedding), Cohere's (verified directly) are
            # plain dicts (item["embedding"]). Handling both is the honest
            # fix; assuming either one is a latent provider-swap bug.
            result.extend(
                item["embedding"] if isinstance(item, dict) else item.embedding
                for item in response.data
            )
        _check_dimensions(result, self.dimension, "HostedEmbedder")
        return result


def build_embedder(
    provider: str,
    model: str,
    dimension: int,
    batch_size: int = 64,
    api_key: str | None = None,
    max_retries: int = 0,
) -> EmbeddingProvider:
    """Construct the configured provider. Single call site so config.py's
    embedding_provider setting is the only place this decision is made.

    max_retries is unused for "local" -- sentence-transformers runs
    in-process, nothing to retry against.
    """
    if provider == "local":
        return LocalEmbedder(model, dimension, batch_size)
    if provider == "hosted":
        return HostedEmbedder(model, dimension, api_key, batch_size, max_retries)
    raise ValueError(f"unknown embedding provider: {provider!r}")
