"""Byte-span containment: one small utility, two independent callers.

Phase 3's chunker uses it to find which class/interface/type span encloses a
function definition (to reclassify it as a method and build its qualified
name). Phase 6's call-graph extraction uses the identical logic to find which
function/method chunk encloses a call site (to attribute the call to its
caller). Both are "which of these candidate spans most tightly contains this
one" -- the same question over different node types, so it's one function
rather than two near-identical copies.
"""

from typing import Protocol


class HasByteSpan(Protocol):
    start_byte: int
    end_byte: int


def smallest_enclosing[T: HasByteSpan](span: HasByteSpan, candidates: list[T]) -> T | None:
    """The tightest candidate span strictly containing span, if any.

    Byte ranges rather than line numbers: precise even when two spans open on
    the same line, and it's what tree-sitter nodes and Chunks both carry
    natively. "Strictly containing" doesn't special-case span appearing in
    candidates itself -- callers never pass a span as a candidate for its own
    containment check, so identity vs. containment doesn't need to be
    distinguished here.
    """
    enclosing = [
        c for c in candidates if c.start_byte <= span.start_byte and span.end_byte <= c.end_byte
    ]
    if not enclosing:
        return None
    # Valid syntax never partially overlaps two candidate spans -- they're
    # always disjoint or nested -- so "smallest span" unambiguously picks the
    # innermost one.
    return min(enclosing, key=lambda c: c.end_byte - c.start_byte)
