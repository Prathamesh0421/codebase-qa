"""RRF fusion and identifier-candidate extraction: both pure, no database."""

import pytest

from codeqa.retrieval.fusion import (
    extract_identifier_candidates,
    filter_symbol_candidates,
    reciprocal_rank_fusion,
)


class TestReciprocalRankFusion:
    def test_single_ranking_preserves_relative_order(self):
        scores = reciprocal_rank_fusion([[10, 20, 30]])
        assert scores[10] > scores[20] > scores[30]

    def test_item_in_multiple_rankings_scores_higher(self):
        # Appears at rank 1 in both lists vs. an item at rank 1 in only one.
        scores = reciprocal_rank_fusion([[1, 2, 3], [1, 4, 5]])
        assert scores[1] > scores[2]
        assert scores[1] > scores[4]

    def test_never_compares_raw_scores_only_rank_position(self):
        # Two rankings with wildly different "meaning" per position (this
        # function has no idea one came from cosine similarity and the
        # other from ts_rank) -- fusion only sees rank order.
        vector_ranking = [100, 200, 300]
        lexical_ranking = [300, 100, 200]
        scores = reciprocal_rank_fusion([vector_ranking, lexical_ranking])
        # 100: rank 1 + rank 2 = 1/61 + 1/62
        # 300: rank 3 + rank 1 = 1/63 + 1/61
        # 200: rank 2 + rank 3 = 1/62 + 1/63
        assert scores[100] == pytest.approx(1 / 61 + 1 / 62)
        assert scores[300] == pytest.approx(1 / 63 + 1 / 61)
        assert scores[200] == pytest.approx(1 / 62 + 1 / 63)

    def test_empty_rankings_produce_empty_result(self):
        assert reciprocal_rank_fusion([]) == {}
        assert reciprocal_rank_fusion([[], []]) == {}

    def test_an_id_absent_from_all_rankings_is_absent_from_result(self):
        scores = reciprocal_rank_fusion([[1, 2]])
        assert 999 not in scores

    def test_k_constant_dampens_rank_differences(self):
        # A larger k flattens the score gap between adjacent ranks --
        # verifies k is actually wired through, not hardcoded internally.
        tight = reciprocal_rank_fusion([[1, 2]], k=1000)
        loose = reciprocal_rank_fusion([[1, 2]], k=1)
        tight_gap = tight[1] - tight[2]
        loose_gap = loose[1] - loose[2]
        assert tight_gap < loose_gap

    def test_default_k_is_the_conventional_60(self):
        explicit = reciprocal_rank_fusion([[1]], k=60)
        default = reciprocal_rank_fusion([[1]])
        assert explicit == default


class TestExtractIdentifierCandidates:
    def test_bare_identifier(self):
        bare, dotted = extract_identifier_candidates("how does dispatch_request work")
        assert "dispatch_request" in bare
        assert dotted == []

    def test_dotted_qualified_name(self):
        bare, dotted = extract_identifier_candidates("what does Flask.dispatch_request do")
        assert "Flask.dispatch_request" in dotted

    def test_dotted_name_also_produces_its_bare_parts(self):
        # Both checks run independently -- a bare match against a
        # differently-scoped chunk (e.g. View.dispatch_request) is still a
        # real signal, not noise.
        bare, _dotted = extract_identifier_candidates("Flask.dispatch_request")
        assert "Flask" in bare
        assert "dispatch_request" in bare

    def test_natural_language_words_are_returned_too(self):
        # No stopword filtering -- ordinary English words end up as bare
        # candidates, and the SQL exact-match WHERE clause is what filters
        # them out (they won't match any real symbol_name), not this function.
        bare, _dotted = extract_identifier_candidates("how does this work")
        assert "how" in bare
        assert "does" in bare

    def test_duplicate_tokens_appear_once_preserving_first_occurrence(self):
        bare, _dotted = extract_identifier_candidates("run run run walk")
        assert bare == ["run", "walk"]

    def test_empty_query_returns_empty_candidates(self):
        assert extract_identifier_candidates("") == ([], [])

    def test_punctuation_does_not_produce_spurious_tokens(self):
        bare, _dotted = extract_identifier_candidates("what does foo() return?")
        assert "foo" in bare
        assert "" not in bare


class TestFilterSymbolCandidates:
    def test_snake_case_token_is_kept(self):
        bare, _dotted = filter_symbol_candidates(["dispatch_request"], [])
        assert bare == ["dispatch_request"]

    def test_camel_or_pascal_case_token_is_kept(self):
        bare, _dotted = filter_symbol_candidates(["SessionManager"], [])
        assert bare == ["SessionManager"]

    def test_plain_lowercase_english_word_is_dropped(self):
        # The regression this exists to prevent: "Flask", "view", and
        # "request" are real symbol names (a class, two methods) in the
        # real Flask repo -- unfiltered, they out-ranked the actual answer.
        bare, _dotted = filter_symbol_candidates(["how", "does", "Flask", "view", "request"], [])
        assert bare == []

    def test_a_single_capitalized_word_is_dropped(self):
        # Capitalization alone (start-of-sentence casing, or a proper noun
        # like "Flask") isn't identifier evidence -- only an INTERNAL
        # capital (camelCase/PascalCase) is.
        bare, _dotted = filter_symbol_candidates(["Flask"], [])
        assert bare == []

    def test_dotted_candidates_are_never_filtered(self):
        _bare, dotted = filter_symbol_candidates([], ["Flask.dispatch_request"])
        assert dotted == ["Flask.dispatch_request"]

    def test_mixed_list_keeps_only_identifier_shaped_tokens(self):
        bare, _dotted = filter_symbol_candidates(
            ["how", "dispatch_request", "does", "SessionManager"], []
        )
        assert bare == ["dispatch_request", "SessionManager"]
