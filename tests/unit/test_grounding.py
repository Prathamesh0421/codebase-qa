"""Citation grounding: pure, no LLM, no database."""

from codeqa.grounding import Citation, find_citations, ground_answer, is_grounded
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


class TestFindCitations:
    def test_extracts_a_single_citation(self):
        citations = find_citations("See app.py:969-994 for the implementation.")
        expected = Citation(file="app.py", start_line=969, end_line=994, raw="app.py:969-994")
        assert citations == [expected]

    def test_extracts_a_citation_with_a_subdirectory_path(self):
        citations = find_citations("Defined in src/flask/app.py:10-20.")
        assert citations[0].file == "src/flask/app.py"

    def test_extracts_multiple_citations(self):
        text = "First see app.py:1-2, then helpers.py:100-120 for the rest."
        citations = find_citations(text)
        assert [c.raw for c in citations] == ["app.py:1-2", "helpers.py:100-120"]

    def test_no_citations_returns_empty_list(self):
        assert find_citations("This answer has no citations in it at all.") == []


class TestIsGrounded:
    def test_exact_match_to_a_context_chunk_is_grounded(self):
        chunk = make_chunk(file_path="app.py", start_line=969, end_line=994)
        citation = Citation(file="app.py", start_line=969, end_line=994, raw="app.py:969-994")
        assert is_grounded(citation, [chunk])

    def test_a_narrower_range_within_a_real_chunk_is_grounded(self):
        # Pointing at a few lines within a bigger function is a legitimate
        # citation, not a fabrication.
        chunk = make_chunk(file_path="app.py", start_line=969, end_line=994)
        citation = Citation(file="app.py", start_line=975, end_line=980, raw="app.py:975-980")
        assert is_grounded(citation, [chunk])

    def test_wrong_file_is_not_grounded(self):
        chunk = make_chunk(file_path="app.py", start_line=969, end_line=994)
        citation = Citation(
            file="helpers.py", start_line=969, end_line=994, raw="helpers.py:969-994"
        )
        assert not is_grounded(citation, [chunk])

    def test_a_fabricated_line_range_outside_any_chunk_is_not_grounded(self):
        chunk = make_chunk(file_path="app.py", start_line=969, end_line=994)
        citation = Citation(file="app.py", start_line=5000, end_line=5010, raw="app.py:5000-5010")
        assert not is_grounded(citation, [chunk])

    def test_a_range_extending_past_a_chunks_real_boundary_is_not_grounded(self):
        # Overlaps a real chunk but claims more than was actually retrieved
        # -- a partial fabrication, not a legitimate narrowing.
        chunk = make_chunk(file_path="app.py", start_line=969, end_line=994)
        citation = Citation(file="app.py", start_line=960, end_line=994, raw="app.py:960-994")
        assert not is_grounded(citation, [chunk])

    def test_an_inverted_range_is_never_grounded(self):
        # start > end doesn't describe a real span, regardless of whether a
        # chunk's bounds happen to numerically straddle both numbers.
        chunk = make_chunk(file_path="app.py", start_line=900, end_line=1000)
        citation = Citation(file="app.py", start_line=994, end_line=969, raw="app.py:994-969")
        assert not is_grounded(citation, [chunk])

    def test_empty_context_grounds_nothing(self):
        citation = Citation(file="app.py", start_line=969, end_line=994, raw="app.py:969-994")
        assert not is_grounded(citation, [])


class TestGroundAnswer:
    def test_a_grounded_citation_is_left_exactly_as_written(self):
        chunk = make_chunk(file_path="app.py", start_line=969, end_line=994)
        result = ground_answer("Dispatch happens in app.py:969-994.", [chunk])
        assert result.text == "Dispatch happens in app.py:969-994."
        assert len(result.grounded) == 1
        assert result.dropped == ()

    def test_an_ungrounded_citation_is_replaced_with_a_visible_marker(self):
        chunk = make_chunk(file_path="app.py", start_line=969, end_line=994)
        result = ground_answer("Dispatch happens in app.py:5000-5010.", [chunk])
        assert "app.py:5000-5010" not in result.text
        assert "[unverifiable citation]" in result.text
        assert len(result.dropped) == 1
        assert result.grounded == ()

    def test_a_fabricated_line_range_is_dropped_not_rendered(self):
        # The exact scenario the phase's "done when" bar names explicitly.
        chunk = make_chunk(file_path="sessions.py", start_line=249, end_line=260)
        text = "Sessions are opened in sessions.py:9999-10005."
        result = ground_answer(text, [chunk])
        assert result.text == "Sessions are opened in [unverifiable citation]."
        assert result.dropped[0].raw == "sessions.py:9999-10005"

    def test_mixed_grounded_and_ungrounded_citations_in_one_answer(self):
        chunk = make_chunk(file_path="app.py", start_line=969, end_line=994)
        text = "See app.py:969-994, and also app.py:5000-5010."
        result = ground_answer(text, [chunk])
        assert "app.py:969-994" in result.text
        assert "app.py:5000-5010" not in result.text
        assert len(result.grounded) == 1
        assert len(result.dropped) == 1

    def test_text_with_no_citations_is_returned_unchanged(self):
        result = ground_answer("This answer makes no citation claims.", [])
        assert result.text == "This answer makes no citation claims."
        assert result.grounded == ()
        assert result.dropped == ()
