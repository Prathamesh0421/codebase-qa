"""smallest_enclosing: shared between chunker.py (class containment) and
graph/extraction.py (call-site containment). Tested independent of both
callers since it's a pure geometric question.
"""

from dataclasses import dataclass

from codeqa.spans import smallest_enclosing


@dataclass(frozen=True)
class Span:
    label: str
    start_byte: int
    end_byte: int


def test_returns_none_with_no_candidates():
    assert smallest_enclosing(Span("x", 10, 20), []) is None


def test_returns_none_when_nothing_contains_it():
    outer = Span("outer", 0, 5)
    assert smallest_enclosing(Span("x", 10, 20), [outer]) is None


def test_finds_single_enclosing_candidate():
    outer = Span("outer", 0, 100)
    assert smallest_enclosing(Span("x", 10, 20), [outer]) is outer


def test_picks_innermost_of_nested_candidates():
    outer = Span("outer", 0, 100)
    inner = Span("inner", 5, 50)
    target = Span("x", 10, 20)
    assert smallest_enclosing(target, [outer, inner]) is inner


def test_ignores_partially_overlapping_non_containing_candidates():
    # A candidate that overlaps but does not fully contain the span must
    # never be selected -- containment is strict on both ends.
    partial = Span("partial", 15, 25)
    target = Span("x", 10, 20)
    assert smallest_enclosing(target, [partial]) is None


def test_touching_boundaries_count_as_containing():
    exact = Span("exact", 10, 20)
    target = Span("x", 10, 20)
    assert smallest_enclosing(target, [exact]) is exact
