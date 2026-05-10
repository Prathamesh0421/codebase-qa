"""Citation grounding: verify that a citation an LLM's answer claims
actually corresponds to a chunk that was genuinely in its context.

Deterministic, not another prompt -- synthesis.py already asks the model to
cite honestly, but a request is not a guarantee. This runs after generation
and checks each citation mechanically: does some chunk that was actually
retrieved have that exact file, with its real line range containing the
claimed one? A chunk's own start_line/end_line are already ground truth --
extracted by tree-sitter from real source (Phase 3) -- so a claimed range
contained inside one is automatically a real range in a real file. That's
why one containment check covers the whole "citation -> chunk in context ->
range exists in file" chain, not three separate lookups.

File matching is exact, not a suffix match like the eval harness's gold-set
matching (evals/runners/metrics.py) -- the model is shown c.citation
verbatim in its prompt context (synthesis.py's format_context), so a
faithful citation reproduces file_path exactly. A suffix match here would
forgive a shortened, partially-invented path as if it were the real one.

Stated honestly: this verifies the citation, not the claim. A citation can
point at a completely real line range and still support a sentence that
misrepresents what that code does -- checking that is a different, much
harder problem this function does not attempt.
"""

import re
from dataclasses import dataclass

from codeqa.retrieval.strategy import RetrievedChunk

# path:start-end, exactly the format RetrievedChunk.citation produces.
# Path characters kept deliberately narrow (word chars, dots, slashes,
# hyphens) -- wide enough for real repo-relative paths, narrow enough not to
# accidentally swallow surrounding prose punctuation into the match.
_CITATION = re.compile(r"([\w./-]+):(\d+)-(\d+)")

_UNGROUNDED_MARKER = "[unverifiable citation]"


@dataclass(frozen=True)
class Citation:
    file: str
    start_line: int
    end_line: int
    raw: str  # the exact matched substring, e.g. "app.py:969-994"


@dataclass(frozen=True)
class GroundingResult:
    text: str  # answer text with ungrounded citations replaced
    grounded: tuple[Citation, ...]
    dropped: tuple[Citation, ...]


def find_citations(text: str) -> list[Citation]:
    return [
        Citation(
            file=m.group(1), start_line=int(m.group(2)), end_line=int(m.group(3)), raw=m.group(0)
        )
        for m in _CITATION.finditer(text)
    ]


def is_grounded(citation: Citation, context_chunks: list[RetrievedChunk]) -> bool:
    """True if some chunk actually in context has this exact file and its
    real line range contains the claimed one. The claimed range may narrow
    a chunk's own bounds (pointing at a few lines within a bigger function
    is a legitimate citation) but never extend past them in either
    direction -- that would be a range partially invented, not narrowed.

    A malformed range (start > end) is never grounded, regardless of
    whether some chunk's bounds happen to numerically straddle both
    numbers -- it doesn't describe a real span in the first place.
    """
    if citation.start_line > citation.end_line:
        return False
    return any(
        c.file_path == citation.file
        and c.start_line <= citation.start_line
        and citation.end_line <= c.end_line
        for c in context_chunks
    )


def ground_answer(text: str, context_chunks: list[RetrievedChunk]) -> GroundingResult:
    """Every claimed citation in text, checked against context_chunks.
    Grounded citations are left exactly as written; ungrounded ones are
    replaced in the returned text with a visible marker -- dropped, not
    silently rendered as if they were verified, but not hidden either: a
    reader can still see a claim was made without a citation that checked
    out, which is more honest than deleting the claim's support invisibly.
    """
    grounded: list[Citation] = []
    dropped: list[Citation] = []

    def _replace(m: re.Match) -> str:
        citation = Citation(
            file=m.group(1), start_line=int(m.group(2)), end_line=int(m.group(3)), raw=m.group(0)
        )
        if is_grounded(citation, context_chunks):
            grounded.append(citation)
            return m.group(0)
        dropped.append(citation)
        return _UNGROUNDED_MARKER

    grounded_text = _CITATION.sub(_replace, text)
    return GroundingResult(text=grounded_text, grounded=tuple(grounded), dropped=tuple(dropped))
