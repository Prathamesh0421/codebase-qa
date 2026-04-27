"""Call resolution logic, tested independent of the database.

_resolve_one is pure -- a dict index in, a (chunk_id, resolution) tuple out --
so every branch of the ordered heuristic (see graph/resolve.py's module
docstring) is testable without Postgres. The Flask case that motivated the
same-class heuristic (dispatch_request defined on Flask, View and
MethodView) is reproduced directly as a fixture here, not just asserted in
the integration test against the real fixture.
"""

from codeqa.graph.resolve import Candidate, PendingEdge, _resolve_one


def edge(callee_name="foo", caller_qualified_name=None, caller_file_id=1, **overrides):
    return PendingEdge(
        caller_chunk_id=overrides.get("caller_chunk_id", 999),
        caller_qualified_name=caller_qualified_name,
        caller_file_id=caller_file_id,
        callee_name=callee_name,
        call_line=overrides.get("call_line", 1),
    )


class TestZeroCandidates:
    def test_unresolved_when_callee_name_matches_nothing(self):
        chunk_id, resolution = _resolve_one(edge(callee_name="getattr"), {})
        assert chunk_id is None
        assert resolution == "unresolved"


class TestSingleCandidate:
    def test_exact_when_globally_unique(self):
        index = {
            "preprocess_request": [
                Candidate(chunk_id=42, qualified_name="Flask.preprocess_request", file_id=1)
            ]
        }
        chunk_id, resolution = _resolve_one(edge(callee_name="preprocess_request"), index)
        assert chunk_id == 42
        assert resolution == "exact"


class TestSameClassHeuristic:
    """The Flask motivating case: dispatch_request defined on three classes."""

    def _flask_index(self):
        return {
            "dispatch_request": [
                Candidate(chunk_id=1, qualified_name="Flask.dispatch_request", file_id=10),
                Candidate(chunk_id=2, qualified_name="View.dispatch_request", file_id=11),
                Candidate(chunk_id=3, qualified_name="MethodView.dispatch_request", file_id=12),
            ]
        }

    def test_resolves_to_the_callers_own_class(self):
        e = edge(
            callee_name="dispatch_request", caller_qualified_name="Flask.full_dispatch_request"
        )
        chunk_id, resolution = _resolve_one(e, self._flask_index())
        assert chunk_id == 1
        assert resolution == "approximate"

    def test_a_different_caller_class_resolves_to_its_own_match(self):
        e = edge(callee_name="dispatch_request", caller_qualified_name="MethodView.some_helper")
        chunk_id, resolution = _resolve_one(e, self._flask_index())
        assert chunk_id == 3
        assert resolution == "approximate"

    def test_caller_with_no_matching_class_falls_through_to_unresolved(self):
        # A method on a class that does NOT define dispatch_request itself,
        # and no same-file match either -- genuinely ambiguous.
        e = edge(
            callee_name="dispatch_request",
            caller_qualified_name="SomeUnrelatedClass.method",
            caller_file_id=999,
        )
        chunk_id, resolution = _resolve_one(e, self._flask_index())
        assert chunk_id is None
        assert resolution == "unresolved"

    def test_top_level_function_caller_has_no_class_to_match(self):
        # caller_qualified_name is None for a plain top-level function --
        # the same-class tier must not raise on this, just skip to the next.
        e = edge(callee_name="dispatch_request", caller_qualified_name=None, caller_file_id=999)
        chunk_id, resolution = _resolve_one(e, self._flask_index())
        assert chunk_id is None
        assert resolution == "unresolved"


class TestSameFileHeuristic:
    def test_resolves_to_the_callers_own_file_when_no_class_match_applies(self):
        index = {
            "helper": [
                Candidate(chunk_id=10, qualified_name=None, file_id=1),
                Candidate(chunk_id=20, qualified_name=None, file_id=2),
            ]
        }
        e = edge(callee_name="helper", caller_qualified_name=None, caller_file_id=2)
        chunk_id, resolution = _resolve_one(e, index)
        assert chunk_id == 20
        assert resolution == "approximate"

    def test_ambiguous_within_the_same_file_is_unresolved(self):
        # Pathological but must not crash or guess: two candidates claim the
        # same file id (this shouldn't happen with real data given the
        # UNIQUE(repo_id, path) constraint on files, but the resolver must
        # not silently pick one if it somehow did).
        index = {
            "helper": [
                Candidate(chunk_id=10, qualified_name=None, file_id=2),
                Candidate(chunk_id=11, qualified_name=None, file_id=2),
            ]
        }
        e = edge(callee_name="helper", caller_qualified_name=None, caller_file_id=2)
        chunk_id, resolution = _resolve_one(e, index)
        assert chunk_id is None
        assert resolution == "unresolved"


class TestNeverGuessesArbitrarily:
    def test_multiple_candidates_no_disambiguating_signal_is_unresolved(self):
        index = {
            "run": [
                Candidate(chunk_id=1, qualified_name="A.run", file_id=1),
                Candidate(chunk_id=2, qualified_name="B.run", file_id=2),
            ]
        }
        e = edge(callee_name="run", caller_qualified_name="C.method", caller_file_id=3)
        chunk_id, resolution = _resolve_one(e, index)
        assert chunk_id is None
        assert resolution == "unresolved"
