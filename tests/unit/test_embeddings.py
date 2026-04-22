"""Embedding providers.

LocalEmbedder is tested against the real model -- it's what CI and the eval
harness actually use, so faking it would test nothing meaningful. HostedEmbedder
is tested by mocking litellm.embedding() at the boundary this project owns:
there is no live API key in this environment, so these tests verify the call
contract (right arguments, correct response parsing, dimension enforcement),
not real network behavior against Gemini or any other provider. That
distinction is deliberate, not an oversight -- see embeddings.py.
"""

from unittest.mock import MagicMock, patch

import pytest

from codeqa.indexing.embeddings import (
    DimensionMismatch,
    HostedEmbedder,
    LocalEmbedder,
    build_embedder,
)


@pytest.fixture(scope="module")
def local_embedder():
    # Loaded once per test module -- constructing SentenceTransformer repeatedly
    # is the expensive part; encode() calls are fast once loaded.
    return LocalEmbedder("BAAI/bge-small-en-v1.5", dimension=384, batch_size=32)


class TestLocalEmbedder:
    def test_produces_correct_dimension(self, local_embedder):
        vectors = local_embedder.embed(["def greet(name): return name"])
        assert len(vectors) == 1
        assert len(vectors[0]) == 384

    def test_empty_input_returns_empty_list(self, local_embedder):
        assert local_embedder.embed([]) == []

    def test_preserves_order(self, local_embedder):
        # Not "are these embeddings good" (that's an eval-harness question) --
        # just that result[i] genuinely corresponds to texts[i], which
        # everything downstream (chunk <-> vector pairing) depends on.
        texts = ["def alpha(): pass", "def completely_different_beta(x, y, z): return x * y * z"]
        vectors = local_embedder.embed(texts)

        # Re-embedding "alpha" alone should land far closer to vectors[0]
        # than to vectors[1] if ordering is preserved.
        alpha_again = local_embedder.embed(["def alpha(): pass"])[0]

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b, strict=True))
            return dot  # bge embeddings are normalized; dot product == cosine

        assert cosine(alpha_again, vectors[0]) > cosine(alpha_again, vectors[1])

    def test_batch_size_does_not_change_output(self, local_embedder):
        # Batching is a throughput optimization; it must not change results.
        text = "def helper(x):\n    return x * 2"
        one = LocalEmbedder("BAAI/bge-small-en-v1.5", dimension=384, batch_size=1).embed([text])
        many = LocalEmbedder("BAAI/bge-small-en-v1.5", dimension=384, batch_size=32).embed(
            [text] * 5
        )
        assert one[0] == pytest.approx(many[0], abs=1e-5)

    def test_wrong_configured_dimension_raises(self):
        # Model genuinely produces 384-dim vectors; claiming 768 at
        # construction time must be caught, not silently written to a
        # differently-sized column.
        bad = LocalEmbedder("BAAI/bge-small-en-v1.5", dimension=768, batch_size=8)
        with pytest.raises(DimensionMismatch, match="384-dim.*expected 768"):
            bad.embed(["def f(): pass"])


class TestHostedEmbedder:
    def _mock_response(self, vectors: list[list[float]]):
        response = MagicMock()
        response.data = [MagicMock(embedding=v) for v in vectors]
        return response

    def test_calls_litellm_with_expected_arguments(self):
        embedder = HostedEmbedder("gemini/gemini-embedding-001", dimension=384, api_key="k")

        with patch("codeqa.indexing.embeddings.litellm.embedding") as mock_embed:
            mock_embed.return_value = self._mock_response([[0.1] * 384])
            embedder.embed(["def f(): pass"])

        mock_embed.assert_called_once_with(
            model="gemini/gemini-embedding-001",
            input=["def f(): pass"],
            dimensions=384,
            api_key="k",
        )

    def test_extracts_vectors_from_response(self):
        embedder = HostedEmbedder("gemini/gemini-embedding-001", dimension=3)

        with patch("codeqa.indexing.embeddings.litellm.embedding") as mock_embed:
            mock_embed.return_value = self._mock_response([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            result = embedder.embed(["a", "b"])

        assert result == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    def test_empty_input_never_calls_litellm(self):
        embedder = HostedEmbedder("gemini/gemini-embedding-001", dimension=384)
        with patch("codeqa.indexing.embeddings.litellm.embedding") as mock_embed:
            assert embedder.embed([]) == []
        mock_embed.assert_not_called()

    def test_provider_returning_wrong_dimension_raises(self):
        # Guards against a config drift where settings.embedding_dim disagrees
        # with what the hosted model/provider actually returns.
        embedder = HostedEmbedder("gemini/gemini-embedding-001", dimension=384)
        with patch("codeqa.indexing.embeddings.litellm.embedding") as mock_embed:
            mock_embed.return_value = self._mock_response([[0.1] * 768])
            with pytest.raises(DimensionMismatch, match="768-dim.*expected 384"):
                embedder.embed(["def f(): pass"])


class TestBuildEmbedder:
    def test_local_provider(self):
        e = build_embedder("local", "BAAI/bge-small-en-v1.5", 384, batch_size=16)
        assert isinstance(e, LocalEmbedder)
        assert e.dimension == 384

    def test_hosted_provider(self):
        e = build_embedder("hosted", "gemini/gemini-embedding-001", 384, api_key="k")
        assert isinstance(e, HostedEmbedder)
        assert e.dimension == 384

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="unknown embedding provider"):
            build_embedder("carrier-pigeon", "model", 384)
