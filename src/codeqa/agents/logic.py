"""Pure logic for the locate -> trace -> synthesize graph: no database, no
LLM call, no LangGraph import. Same separation Phase 5/8 used elsewhere
(synthesis.build_messages vs synthesize, fusion.py vs hybrid.py) -- the
control-flow decisions are what actually need to be trusted and tested in
isolation; the I/O around them is comparatively uninteresting.
"""

from codeqa.agents.state import AgentState
from codeqa.retrieval.strategy import RetrievedChunk
from codeqa.synthesis import format_context

_TRACE_SYSTEM_PROMPT = """\
You are deciding whether enough source code has been retrieved to fully and \
accurately answer a question about a codebase.

Respond with EXACTLY one of the following as your first line:
- "SUFFICIENT" if the excerpts below are enough to answer the question completely.
- "INSUFFICIENT: <a focused follow-up search query>" if something essential \
is missing. The query should describe what additional code needs to be \
found (a specific function, concept, or relationship not yet covered) -- \
not simply restate the original question.

You may add a brief reasoning explanation on the lines that follow.
"""


def build_trace_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    context = format_context(chunks)
    return [
        {"role": "system", "content": _TRACE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"# Source excerpts found so far\n\n{context}\n\n# Question\n\n{question}",
        },
    ]


def parse_trace_response(text: str, fallback_query: str) -> tuple[bool, str, str]:
    """(sufficient, next_query, reasoning). next_query is only meaningful
    when sufficient is False; callers should ignore it otherwise.

    A response that doesn't parse as either recognized keyword fails safe
    toward sufficient=True (fallback_query is returned but unused, proceed
    to synthesize with whatever's already been found) rather than toward
    retrying -- a parsing bug or an unexpected model response must never be
    able to spin the retry loop on its own.
    """
    lines = text.strip().splitlines()
    first = lines[0].strip() if lines else ""
    reasoning = "\n".join(lines[1:]).strip()

    if first.upper().startswith("SUFFICIENT"):
        return True, fallback_query, reasoning

    if first.upper().startswith("INSUFFICIENT"):
        _, _, rest = first.partition(":")
        refined = rest.strip()
        if refined:
            return False, refined, reasoning
        return True, fallback_query, f"malformed trace response, proceeding: {text!r}"

    return True, fallback_query, f"unparseable trace response, proceeding: {text!r}"


def merge_chunks(
    existing: list[RetrievedChunk], new: list[RetrievedChunk]
) -> list[RetrievedChunk]:
    """Union by chunk_id, preserving order (existing first). A retry's
    locate call should ADD context, not replace what a previous attempt
    already found -- the original results may still belong in a complete
    answer even though they weren't enough alone.
    """
    seen = {c.chunk_id for c in existing}
    merged = list(existing)
    for c in new:
        if c.chunk_id not in seen:
            merged.append(c)
            seen.add(c.chunk_id)
    return merged


def sort_for_display(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Chunks accumulated across multiple locate attempts (merge_chunks
    above) are in attempt order, not score order -- a later attempt's
    real-scored primary results can land after an earlier attempt's
    0.0-sentinel graph-expanded ones, since concatenation order is all
    merge_chunks promises. Sorts explicitly instead of trusting that order:
    real scores descending first; any chunk graph expansion touched (score
    is a 0.0 sentinel, never comparable to a real score -- see strategy.py's
    RetrievedChunk.score docstring) grouped at the end, in whatever relative
    order they were accumulated.
    """
    return sorted(chunks, key=lambda c: ("graph" in c.source.split("+"), -c.score))


def route_after_trace(state: AgentState) -> str:
    """The one real cycle in this project's AGENT graph -- distinct from the
    call GRAPH's cycles, which graph/traversal.py handles separately.
    Bounded by max_attempts so an LLM that keeps saying INSUFFICIENT can
    never loop forever.
    """
    if state.sufficient:
        return "synthesize"
    if state.attempt >= state.max_attempts:
        return "synthesize"
    return "locate"
