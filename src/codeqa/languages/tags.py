"""Run a language's tag query against source and yield structured Tags.

This is the one code path every tier1/tier2 language goes through: parse,
run the tag query, pair each definition/reference capture with its @name
capture. Nothing here is language-specific -- language-specific knowledge
lives entirely in the query text (registry.py), not in this function.
"""

from dataclasses import dataclass

from tree_sitter import Parser, Query, QueryCursor

from codeqa.languages.registry import LanguageSpec, get_language

# Capture group prefixes this project acts on. Upstream tags.scm files also
# emit @doc (comment association), @local.scope, @reference.type,
# @reference.class and similar -- real captures, just not ones this system
# consumes. Filtering here rather than downstream keeps Tag a small, honest
# contract: every Tag yielded is either a definition or a call reference.
_KIND_PREFIXES = ("definition.", "reference.call")


@dataclass(frozen=True)
class Tag:
    kind: str  # e.g. "definition.function", "definition.method", "reference.call"
    name: str
    start_byte: int
    end_byte: int
    # 1-indexed, inclusive -- matches how editors and citations display
    # spans, and the CHECK constraint on chunks in 0001_init.sql.
    start_line: int
    end_line: int


def extract_tags(spec: LanguageSpec, source: bytes) -> list[Tag]:
    """Extract definitions and call references from source using spec's tag query.

    Returns [] for a tier3 language (spec.tags_query is None) rather than
    raising -- the file was still detected and its language reported, it
    simply has nothing to extract until a query is written for it.
    """
    if spec.tags_query is None:
        return []

    parser = Parser(get_language(spec))
    tree = parser.parse(source)

    query = Query(get_language(spec), spec.tags_query)
    cursor = QueryCursor(query)

    tags: list[Tag] = []
    for _pattern_index, captures in cursor.matches(tree.root_node):
        name_nodes = captures.get("name")
        if not name_nodes:
            # @doc-only matches and similar have no @name; nothing to record.
            continue
        name = source[name_nodes[0].start_byte : name_nodes[0].end_byte].decode(
            "utf-8", errors="replace"
        )

        for kind, nodes in captures.items():
            if not kind.startswith(_KIND_PREFIXES):
                continue
            node = nodes[0]
            tags.append(
                Tag(
                    kind=kind,
                    name=name,
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                )
            )

    return tags
