"""Pure precision/recall scoring, no database, no retrieval strategy."""

from codeqa.retrieval.strategy import RetrievedChunk
from evals.runners.metrics import GoldItem, chunk_matches_gold, matched_gold, score


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


class TestChunkMatchesGold:
    def test_matches_on_qualified_name_and_file_suffix(self):
        chunk = make_chunk(file_path="src/flask/app.py", qualified_name="Flask.dispatch_request")
        gold = GoldItem(file="app.py", symbol="Flask.dispatch_request")
        assert chunk_matches_gold(chunk, gold)

    def test_falls_back_to_symbol_name_when_no_qualified_name(self):
        chunk = make_chunk(qualified_name=None, symbol_name="url_for", file_path="helpers.py")
        gold = GoldItem(file="helpers.py", symbol="url_for")
        assert chunk_matches_gold(chunk, gold)

    def test_bare_symbol_name_does_not_satisfy_a_qualified_gold_symbol(self):
        # The exact disambiguation this project cares about: real Flask
        # defines dispatch_request on three different classes, so a bare
        # name match without the qualifying class would be a false positive.
        chunk = make_chunk(qualified_name=None, symbol_name="dispatch_request")
        gold = GoldItem(file="app.py", symbol="Flask.dispatch_request")
        assert not chunk_matches_gold(chunk, gold)

    def test_wrong_file_never_matches(self):
        chunk = make_chunk(file_path="views.py", qualified_name="Flask.dispatch_request")
        gold = GoldItem(file="app.py", symbol="Flask.dispatch_request")
        assert not chunk_matches_gold(chunk, gold)

    def test_file_match_is_a_suffix_match(self):
        chunk = make_chunk(file_path="src/flask/app.py", qualified_name="Flask.dispatch_request")
        gold = GoldItem(file="app.py", symbol="Flask.dispatch_request")
        assert chunk_matches_gold(chunk, gold)


class TestMatchedGold:
    def test_returns_only_satisfied_gold_items(self):
        chunks = [make_chunk(qualified_name="Flask.dispatch_request")]
        gold = (
            GoldItem(file="app.py", symbol="Flask.dispatch_request"),
            GoldItem(file="app.py", symbol="Flask.full_dispatch_request"),
        )
        assert matched_gold(chunks, gold) == {gold[0]}

    def test_empty_when_nothing_matches(self):
        chunks = [make_chunk(qualified_name="Flask.wsgi_app")]
        gold = (GoldItem(file="app.py", symbol="Flask.dispatch_request"),)
        assert matched_gold(chunks, gold) == set()


class TestScore:
    def test_perfect_retrieval(self):
        gold = (GoldItem(file="app.py", symbol="Flask.dispatch_request"),)
        chunks = [make_chunk(qualified_name="Flask.dispatch_request")]
        result = score(chunks, gold)
        assert result.precision == 1.0
        assert result.recall == 1.0

    def test_precision_penalized_by_irrelevant_results(self):
        gold = (GoldItem(file="app.py", symbol="Flask.dispatch_request"),)
        chunks = [
            make_chunk(qualified_name="Flask.dispatch_request"),
            make_chunk(chunk_id=2, qualified_name="Flask.wsgi_app"),
        ]
        result = score(chunks, gold)
        assert result.precision == 0.5
        assert result.recall == 1.0

    def test_recall_penalized_by_missing_gold_items(self):
        gold = (
            GoldItem(file="app.py", symbol="Flask.full_dispatch_request"),
            GoldItem(file="app.py", symbol="Flask.dispatch_request"),
        )
        chunks = [make_chunk(qualified_name="Flask.dispatch_request")]
        result = score(chunks, gold)
        assert result.precision == 1.0
        assert result.recall == 0.5

    def test_empty_result_list_scores_zero_precision_not_a_crash(self):
        gold = (GoldItem(file="app.py", symbol="Flask.dispatch_request"),)
        result = score([], gold)
        assert result.precision == 0.0
        assert result.recall == 0.0

    def test_result_and_gold_counts_are_recorded(self):
        gold = (GoldItem(file="app.py", symbol="Flask.dispatch_request"),)
        chunks = [
            make_chunk(qualified_name="Flask.dispatch_request"),
            make_chunk(chunk_id=2, qualified_name="Flask.wsgi_app"),
        ]
        result = score(chunks, gold)
        assert result.result_count == 2
        assert result.gold_count == 1
        assert result.matched_count == 1
