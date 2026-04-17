"""Turn a file's Tags into Chunks: complete, addressable retrieval units.

The one invariant that matters more than any other in this module: a chunk's
content is always the verbatim, syntactically complete span of a single
definition. It is never truncated, never split at a fixed character count,
never re-windowed for an embedding model's context length. That is the
entire quality claim this project makes over naive RAG applied to code, and
it would be undone by a single "if the function is too long, slice it"
shortcut here.

A function that exceeds an embedding model's token limit is still chunked as
one complete unit. What happens to it at embedding time -- truncate the
input, use a longer-context model, whatever -- is Phase 4's decision, scoped
to the vector alone. It must never touch the stored content or the line span
a citation points at; grounding correctness depends on that span always being
real and complete.
"""

import hashlib
from dataclasses import dataclass

from codeqa.languages import LanguageSpec, Tag, extract_tags

ChunkKind = str  # "function" | "method" | "class" | "module" -- mirrors the DB enum

# Maps a tag's raw capture kind to one of the four chunkable kinds the schema
# recognizes. Deliberately not a 1:1 mirror of every capture grammars emit:
# interface/type definitions genuinely are class-shaped retrieval units (a
# Java interface or a Go struct is exactly the kind of thing "how does X
# work" should be able to find), so they fold into "class" rather than being
# invented as a fifth kind. Anything absent from this table (constants,
# fields, macros, and any future capture kind a new grammar introduces) is
# not an independently meaningful code unit and is not chunked at all.
_KIND_MAP: dict[str, ChunkKind] = {
    "definition.function": "function",
    "definition.method": "method",
    "definition.class": "class",
    "definition.interface": "class",
    "definition.type": "class",
    "definition.module": "module",
}

# Tag kinds whose span can enclose a method and give it a qualified name.
_CLASS_LIKE = frozenset({"definition.class", "definition.interface", "definition.type"})


@dataclass(frozen=True)
class Chunk:
    kind: ChunkKind
    symbol_name: str
    qualified_name: str | None
    start_byte: int
    end_byte: int
    # 1-indexed, inclusive -- matches chunks.start_line/end_line in
    # 0001_init.sql and the CHECK constraint on their ordering.
    start_line: int
    end_line: int
    content: str
    content_sha: str


def _smallest_enclosing(tag: Tag, class_like: list[Tag]) -> Tag | None:
    """The tightest class/interface/type span strictly containing tag, if any.

    Byte ranges rather than line numbers: precise even when a class and its
    first method open on the same line, and it's what tags already carry.
    "Strictly containing" excludes tag being compared against its own span --
    irrelevant here since tag is always a function/method capture and
    class_like entries are always separate class/interface/type captures,
    but stated for clarity: containment, not identity.
    """
    candidates = [
        c
        for c in class_like
        if c.start_byte <= tag.start_byte and tag.end_byte <= c.end_byte
    ]
    if not candidates:
        return None
    # Nested classes are always disjoint-or-nested, never partially
    # overlapping, in valid syntax -- so "smallest span" is well-defined and
    # picks the innermost enclosing class.
    return min(candidates, key=lambda c: c.end_byte - c.start_byte)


def _build_chunk(
    kind: ChunkKind,
    name: str,
    qualified_name: str | None,
    tag_span: tuple[int, int],
    source: bytes,
    start_line: int,
    end_line: int,
) -> Chunk:
    start_byte, end_byte = tag_span
    raw = source[start_byte:end_byte]
    return Chunk(
        kind=kind,
        symbol_name=name,
        qualified_name=qualified_name,
        start_byte=start_byte,
        end_byte=end_byte,
        start_line=start_line,
        end_line=end_line,
        content=raw.decode("utf-8", errors="replace"),
        content_sha=hashlib.sha256(raw).hexdigest(),
    )


def _whole_file_chunk(path: str, source: bytes) -> Chunk:
    """Fallback for a file with no chunkable definitions.

    Covers two real cases, not just tier3: a tier3 language (no tag query
    exists at all, so extract_tags always returns []), and a tier1/tier2 file
    that genuinely has none -- an __init__.py that's only imports, a
    constants-only module. Either way the file should still be retrievable as
    a whole rather than silently absent from the index.
    """
    name = path.rsplit("/", 1)[-1]
    line_count = source.count(b"\n") + (0 if source.endswith(b"\n") else 1)
    return _build_chunk("module", name, None, (0, len(source)), source, 1, max(line_count, 1))


def chunk_file(spec: LanguageSpec, path: str, source: bytes) -> list[Chunk]:
    """Extract Chunks from a file's content.

    path is used only to name the whole-file fallback chunk -- language
    detection already happened upstream (codeqa.languages.detect_language)
    to produce spec, and this function doesn't second-guess that.
    """
    tags = extract_tags(spec, source)
    definitions = [t for t in tags if t.kind.startswith("definition.")]
    class_like = [t for t in definitions if t.kind in _CLASS_LIKE]

    chunks: list[Chunk] = []
    for tag in definitions:
        kind = _KIND_MAP.get(tag.kind)
        if kind is None:
            # e.g. definition.constant, definition.field, definition.macro --
            # real captures, not retrieval-worthy units on their own.
            continue

        qualified_name = None
        if kind in ("function", "method"):
            enclosing = _smallest_enclosing(tag, class_like)
            if enclosing is not None:
                # Reclassify function -> method via containment rather than
                # trusting each grammar's own labeling: Python's tags.scm has
                # no definition.method at all (every function, top-level or
                # nested, is definition.function), and PHP's method_declaration
                # is tagged definition.function too. Containment catches both
                # uniformly instead of special-casing two languages.
                kind = "method"
                qualified_name = f"{enclosing.name}.{tag.name}"
            # No enclosing span found: either a genuine top-level function, or
            # a method whose language doesn't nest it inside its type at the
            # syntax level -- Go's receiver methods and Rust's impl-block
            # methods are both structural siblings of their struct/type, not
            # children of it (verified empirically, not assumed). Their kind
            # is still correctly "method" from the grammar's own tag; only
            # qualified_name is unavailable, and stays None rather than guessed.

        chunks.append(
            _build_chunk(
                kind, tag.name, qualified_name, (tag.start_byte, tag.end_byte),
                source, tag.start_line, tag.end_line,
            )
        )

    if not chunks and source.strip():
        chunks.append(_whole_file_chunk(path, source))

    return chunks
