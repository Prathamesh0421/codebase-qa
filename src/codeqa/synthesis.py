"""Build a grounded prompt from retrieved chunks and stream an answer.

"Grounded" here is a prompt-level instruction, not yet an enforced guarantee
-- the LLM is told to cite path:start-end for every claim and to say when the
context doesn't answer the question, but nothing here checks that a citation
it produces actually corresponds to real content. That verification is a
deliberately separate, deterministic function (Phase 11): citation → chunk
actually in context → line range actually exists. Conflating "asked the model
to cite honestly" with "verified it did" would be exactly the kind of
unearned claim this project tries not to make.

Shared with Phase 10's synthesize agent node rather than duplicated -- the
same "chunks in, grounded streamed answer out" operation is needed in both
the single-shot CLI path and the locate/trace/synthesize pipeline.
"""

from collections.abc import Iterator

import litellm

from codeqa.retrieval.strategy import RetrievedChunk

_SYSTEM_PROMPT = """\
You are a code assistant answering questions about a specific codebase, \
using only the source excerpts provided below as context.

Rules:
- Cite the source of every factual claim about the code using the exact \
citation shown above each excerpt, in the form `path:start-end`.
- If the provided excerpts do not contain enough information to answer the \
question, say so plainly rather than guessing or relying on general \
knowledge about similar libraries.
- Do not invent file paths, line numbers, or function names that do not \
appear in the excerpts.
"""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no relevant source excerpts were found)"

    blocks = []
    for c in chunks:
        label = f"{c.qualified_name or c.symbol_name} ({c.kind})"
        blocks.append(f"### {c.citation} -- {label}\n```\n{c.content}\n```")
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    context = _format_context(chunks)
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"# Source excerpts\n\n{context}\n\n# Question\n\n{question}",
        },
    ]


def synthesize(
    question: str,
    chunks: list[RetrievedChunk],
    model: str,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> Iterator[str]:
    """Stream the answer token by token.

    Always a generator, even though a caller could pass stream=False to
    litellm -- the CLI, the eventual SSE endpoint (Phase 14), and Phase 10's
    synthesize node all want to consume tokens as they arrive, and a
    generator that yields one final string is a strict superset of a
    non-streaming call, not the other way around.
    """
    messages = build_messages(question, chunks)
    response = litellm.completion(
        model=model,
        messages=messages,
        api_key=api_key,
        timeout=timeout,
        stream=True,
    )
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
