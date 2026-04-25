"""Prompt construction and the streaming LLM call.

synthesize() is tested against litellm's real mock_response feature rather
than a hand-rolled mock of litellm.completion -- this exercises the actual
streaming chunk format litellm produces (verified in interactive testing:
CustomStreamWrapper, choices[0].delta.content per chunk), not a shape we
assumed. It verifies the plumbing -- prompt reaches the call, tokens stream
and reassemble correctly -- not real model output quality, which needs a
live key and is out of scope here. Same honesty boundary as HostedEmbedder
in Phase 4.
"""

from unittest.mock import patch

import litellm as real_litellm

from codeqa.retrieval.strategy import RetrievedChunk
from codeqa.synthesis import build_messages, synthesize


def make_chunk(**overrides) -> RetrievedChunk:
    defaults = dict(
        chunk_id=1,
        file_path="app.py",
        kind="method",
        symbol_name="dispatch_request",
        qualified_name="Flask.dispatch_request",
        start_line=969,
        end_line=994,
        content="def dispatch_request(self, ctx):\n    ...",
        score=0.87,
    )
    return RetrievedChunk(**{**defaults, **overrides})


class TestCitation:
    def test_format_is_path_colon_start_dash_end(self):
        c = make_chunk(file_path="flask/app.py", start_line=10, end_line=20)
        assert c.citation == "flask/app.py:10-20"


class TestBuildMessages:
    def test_system_message_present_first(self):
        messages = build_messages("how does routing work", [make_chunk()])
        assert messages[0]["role"] == "system"
        assert "cite" in messages[0]["content"].lower()

    def test_context_includes_citation_and_content(self):
        chunk = make_chunk(file_path="app.py", start_line=1, end_line=2, content="def f(): pass")
        messages = build_messages("q", [chunk])
        user_content = messages[1]["content"]
        assert "app.py:1-2" in user_content
        assert "def f(): pass" in user_content
        assert "Flask.dispatch_request" in build_messages("q", [chunk])[1]["content"]

    def test_question_is_included(self):
        messages = build_messages("how does dispatch work?", [make_chunk()])
        assert "how does dispatch work?" in messages[1]["content"]

    def test_empty_chunk_list_says_so_rather_than_omitting_context(self):
        messages = build_messages("q", [])
        assert "no relevant source excerpts" in messages[1]["content"].lower()

    def test_multiple_chunks_all_present(self):
        chunks = [
            make_chunk(symbol_name="a", start_line=1, end_line=2),
            make_chunk(symbol_name="b", start_line=10, end_line=20),
        ]
        content = build_messages("q", chunks)[1]["content"]
        assert "app.py:1-2" in content
        assert "app.py:10-20" in content


# Captured before any patching happens, so it's a snapshot of the real
# function -- codeqa.synthesis.litellm and this module's litellm are the
# same cached module object, so patching "codeqa.synthesis.litellm.completion"
# patches litellm.completion everywhere, including here. A reference taken
# after that patch is applied would resolve to the mock itself and recurse.
_real_completion = real_litellm.completion


def _patched_completion(mock_text: str, captured: dict | None = None):
    """Patches litellm.completion inside the synthesis module so that
    synthesize()'s own code runs unmodified, but the underlying call is
    litellm's real mock_response path -- realistic CustomStreamWrapper chunk
    objects, not a hand-rolled MagicMock shape we might get wrong."""

    def _delegate(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        kwargs.pop("api_key", None)
        return _real_completion(mock_response=mock_text, **kwargs)

    return patch("codeqa.synthesis.litellm.completion", side_effect=_delegate)


class TestSynthesizeStreaming:
    def test_yields_reassemblable_tokens(self):
        with _patched_completion("This answer cites app.py:969-994."):
            tokens = list(
                synthesize(
                    "how does dispatch work?", [make_chunk()], model="gemini/gemini-2.0-flash"
                )
            )
        assert "".join(tokens) == "This answer cites app.py:969-994."

    def test_prompt_actually_reaches_the_call(self):
        # A regression guard against synthesize() silently dropping context:
        # the mock model has no idea what the real answer should contain, but
        # this confirms the messages it receives include what build_messages
        # produced, not some other prompt.
        captured: dict = {}
        with _patched_completion("ok", captured):
            list(
                synthesize(
                    "how does dispatch work?", [make_chunk()], model="gemini/gemini-2.0-flash"
                )
            )

        assert any("how does dispatch work?" in str(m) for m in captured["messages"])
        assert captured["stream"] is True

    def test_returns_a_generator_not_a_list(self):
        # Even though every current call site immediately consumes it fully,
        # the contract is streaming -- Phase 14's SSE endpoint and Phase 10's
        # synthesize node both need to forward tokens as they arrive rather
        # than wait for the whole answer.
        import inspect

        assert inspect.isgeneratorfunction(synthesize)
