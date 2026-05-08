"""Pure logic for the locate -> trace -> synthesize graph: no database, no
LLM, no LangGraph. Same testing posture as test_fusion.py and
test_synthesis.py's TestBuildMessages -- control flow tested in isolation
from the I/O around it.
"""

from codeqa.agents.logic import (
    build_trace_messages,
    merge_chunks,
    parse_trace_response,
    route_after_trace,
)
from codeqa.agents.state import AgentState
from codeqa.retrieval.strategy import RetrievedChunk


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
        source="vector",
    )
    return RetrievedChunk(**{**defaults, **overrides})


class TestBuildTraceMessages:
    def test_system_message_explains_the_response_format(self):
        messages = build_trace_messages("how does routing work", [make_chunk()])
        assert messages[0]["role"] == "system"
        assert "SUFFICIENT" in messages[0]["content"]
        assert "INSUFFICIENT" in messages[0]["content"]

    def test_context_and_question_are_both_present(self):
        chunk = make_chunk(file_path="app.py", start_line=1, end_line=2, content="def f(): pass")
        messages = build_trace_messages("how does dispatch work?", [chunk])
        user_content = messages[1]["content"]
        assert "app.py:1-2" in user_content
        assert "def f(): pass" in user_content
        assert "how does dispatch work?" in user_content


class TestParseTraceResponse:
    def test_sufficient_with_no_reasoning(self):
        sufficient, next_query, reasoning = parse_trace_response("SUFFICIENT", "orig")
        assert sufficient is True
        assert reasoning == ""

    def test_sufficient_is_case_insensitive(self):
        sufficient, _, _ = parse_trace_response("sufficient", "orig")
        assert sufficient is True

    def test_sufficient_with_reasoning_on_following_lines(self):
        sufficient, _, reasoning = parse_trace_response(
            "SUFFICIENT\nThe dispatch_request method fully answers this.", "orig"
        )
        assert sufficient is True
        assert reasoning == "The dispatch_request method fully answers this."

    def test_insufficient_extracts_the_refined_query(self):
        sufficient, next_query, _ = parse_trace_response(
            "INSUFFICIENT: find the session signing logic", "orig"
        )
        assert sufficient is False
        assert next_query == "find the session signing logic"

    def test_insufficient_with_reasoning_on_following_lines(self):
        sufficient, next_query, reasoning = parse_trace_response(
            "INSUFFICIENT: find open_session\nThe signature check isn't shown yet.", "orig"
        )
        assert sufficient is False
        assert next_query == "find open_session"
        assert reasoning == "The signature check isn't shown yet."

    def test_insufficient_with_no_query_fails_safe_to_sufficient(self):
        # Malformed: INSUFFICIENT with no ": <query>" -- must never be
        # treated as a valid retry instruction with an empty query.
        sufficient, next_query, reasoning = parse_trace_response("INSUFFICIENT", "orig")
        assert sufficient is True
        assert next_query == "orig"
        assert "malformed" in reasoning.lower()

    def test_unrecognized_response_fails_safe_to_sufficient(self):
        sufficient, next_query, reasoning = parse_trace_response(
            "I think this looks pretty good actually.", "orig"
        )
        assert sufficient is True
        assert next_query == "orig"
        assert "unparseable" in reasoning.lower()

    def test_empty_response_fails_safe_to_sufficient(self):
        sufficient, next_query, _ = parse_trace_response("", "orig")
        assert sufficient is True
        assert next_query == "orig"


class TestMergeChunks:
    def test_no_existing_chunks_returns_new_ones(self):
        new = [make_chunk(chunk_id=1), make_chunk(chunk_id=2)]
        assert merge_chunks([], new) == new

    def test_new_chunks_already_present_are_not_duplicated(self):
        existing = [make_chunk(chunk_id=1)]
        new = [make_chunk(chunk_id=1), make_chunk(chunk_id=2)]
        merged = merge_chunks(existing, new)
        assert [c.chunk_id for c in merged] == [1, 2]

    def test_existing_chunks_come_first_in_order(self):
        existing = [make_chunk(chunk_id=5), make_chunk(chunk_id=3)]
        new = [make_chunk(chunk_id=1)]
        merged = merge_chunks(existing, new)
        assert [c.chunk_id for c in merged] == [5, 3, 1]


class TestRouteAfterTrace:
    def test_sufficient_routes_to_synthesize(self):
        state = AgentState(
            repo_id=1, question="q", current_query="q", sufficient=True, attempt=1, max_attempts=2
        )
        assert route_after_trace(state) == "synthesize"

    def test_insufficient_under_attempt_limit_routes_to_locate(self):
        state = AgentState(
            repo_id=1, question="q", current_query="q", sufficient=False, attempt=1, max_attempts=2
        )
        assert route_after_trace(state) == "locate"

    def test_insufficient_at_attempt_limit_routes_to_synthesize_anyway(self):
        # Bounded: never let a persistently-insufficient trace loop forever.
        state = AgentState(
            repo_id=1, question="q", current_query="q", sufficient=False, attempt=2, max_attempts=2
        )
        assert route_after_trace(state) == "synthesize"
