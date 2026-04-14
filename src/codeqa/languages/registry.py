"""Language registry: file extension -> grammar, capability tier, tag query.

"Support all languages" would normally mean one hand-written AST walker per
language -- the thing that makes multi-language support look infeasible for
one person. tree-sitter grammars ship queries/tags.scm, an S-expression query
declaring captures like @definition.function, @definition.class and
@reference.call -- the same mechanism GitHub uses for cross-language code
navigation. One query engine (tags.py) reads any grammar's tags query, so
adding a language means shipping a query, not writing code.

Support is not uniform across grammars, and that is declared rather than
hidden. Each LanguageSpec carries a tier that mirrors the `language_tier`
enum in 0001_init.sql:

  tier1 -- upstream tags.scm has @reference.call: real call edges.
  tier2 -- upstream tags.scm is missing definitions or calls: chunks plus
           an override query (overrides/) that adds what's missing, verified
           by parsing real code, not assumed from the grammar's reputation.
  tier3 -- no tags.scm at all: the file is detected, but nothing is
           extracted until a query is written for it.

Bulk grammar packages were considered and rejected. tree-sitter-language-pack
(PyPI, published by xberg-io) offers 371 languages behind one dependency, but
it downloads compiled native binaries at runtime (DownloadManager /
download_all) from a young, non-canonical publisher. This project clones and
parses arbitrary third-party repositories; a dependency that fetches and
loads native code over the network at runtime reintroduces exactly the
supply-chain risk that "tree-sitter parses but never executes" exists to
rule out. Grammars are pinned individually instead, each compiled into its
wheel at build time by the tree-sitter org or the grammar's own maintainers,
with no runtime fetch. See docs/deep-dive.html for the full comparison.
"""

import importlib.resources
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

from tree_sitter import Language

LanguageTier = Literal["tier1", "tier2", "tier3"]

_OVERRIDES = importlib.resources.files("codeqa.languages.overrides")


def _read_override(filename: str) -> str:
    return (_OVERRIDES / filename).read_text()


@dataclass(frozen=True)
class LanguageSpec:
    name: str
    tier: LanguageTier
    extensions: tuple[str, ...]
    # Zero-arg loader returning the raw grammar pointer each package exports
    # under a different name (language(), language_typescript(), ...) --
    # wrapped uniformly here so callers never see that inconsistency.
    _load_raw: Callable[[], object]
    # None means genuinely no tag query exists (tier3). Resolved eagerly at
    # registry-build time since these are just string reads/concatenations,
    # not parses -- there is nothing to defer.
    tags_query: str | None


@cache
def get_language(spec: LanguageSpec) -> Language:
    """Build (and cache) the tree_sitter.Language for a spec.

    Cached per spec rather than per name: a Language wraps a compiled grammar
    and there is no reason to rebuild it per file. lru_cache requires spec to
    be hashable, which a frozen dataclass with a plain function field is.
    """
    return Language(spec._load_raw())


def _tier1(name: str, extensions: tuple[str, ...], module) -> LanguageSpec:
    """Grammars whose upstream TAGS_QUERY constant already has @reference.call."""
    query = module.TAGS_QUERY
    assert query and "reference.call" in query, (
        f"{name}: expected upstream TAGS_QUERY to include @reference.call; "
        f"verify before registering as tier1"
    )
    return LanguageSpec(name, "tier1", extensions, module.language, query)


def _build_registry() -> tuple[LanguageSpec, ...]:
    import tree_sitter_cpp as ts_cpp
    import tree_sitter_go as ts_go
    import tree_sitter_java as ts_java
    import tree_sitter_javascript as ts_js
    import tree_sitter_php as ts_php
    import tree_sitter_python as ts_python
    import tree_sitter_ruby as ts_ruby
    import tree_sitter_rust as ts_rust
    import tree_sitter_typescript as ts_typescript

    # TypeScript's own tags.scm ships as a file, not a TAGS_QUERY constant.
    # importlib.resources rather than a __file__-relative path: works
    # regardless of how the wheel is unpacked or zipped.
    ts_tags_scm = (
        importlib.resources.files("tree_sitter_typescript") / "queries" / "tags.scm"
    ).read_text()
    ts_override = _read_override("typescript.scm")
    ts_combined = ts_tags_scm + "\n" + ts_override

    cpp_combined = ts_cpp.TAGS_QUERY + "\n" + _read_override("cpp.scm")

    import tree_sitter_c_sharp as ts_csharp

    return (
        _tier1("python", (".py",), ts_python),
        # tree-sitter-javascript's grammar parses JSX directly -- verified by
        # parsing a JSX snippet and checking for parse errors -- so .jsx uses
        # the same grammar and query as .js, no separate package needed.
        _tier1("javascript", (".js", ".mjs", ".cjs", ".jsx"), ts_js),
        _tier1("go", (".go",), ts_go),
        _tier1("java", (".java",), ts_java),
        _tier1("rust", (".rs",), ts_rust),
        _tier1("ruby", (".rb",), ts_ruby),
        # language_php parses a full PHP document (PHP possibly interleaved
        # with HTML), which is what a standalone .php file actually is.
        # language_php_only parses bare PHP with no surrounding-document
        # handling and exists for embedding PHP inside another grammar.
        LanguageSpec(
            "php", "tier1", (".php",), ts_php.language_php, ts_php.TAGS_QUERY
        ),
        # tier2: upstream tags.scm exists but is missing @reference.call
        # (and, for TypeScript, ordinary function/class/method definitions
        # too). Override queries are verified against real parse trees in
        # tests/unit/test_languages.py, not assumed from the grammar's
        # reputation.
        LanguageSpec(
            "typescript", "tier2", (".ts",), ts_typescript.language_typescript, ts_combined
        ),
        LanguageSpec(
            "tsx", "tier2", (".tsx",), ts_typescript.language_tsx, ts_combined
        ),
        LanguageSpec(
            "cpp", "tier2", (".cpp", ".cc", ".cxx", ".hpp", ".hh"), ts_cpp.language, cpp_combined
        ),
        # tier3: the grammar ships no tags.scm and no TAGS_QUERY at all.
        # Registered so the file is at least *detected* -- reported as
        # tier3, not silently skipped -- rather than writing a query blind.
        # See docs/deep-dive.html, Known Limits.
        LanguageSpec("c_sharp", "tier3", (".cs",), ts_csharp.language, ts_csharp.TAGS_QUERY),
    )


REGISTRY: tuple[LanguageSpec, ...] = _build_registry()

_EXTENSION_INDEX: dict[str, LanguageSpec] = {
    ext: spec for spec in REGISTRY for ext in spec.extensions
}


def detect_language(path: str | Path) -> LanguageSpec | None:
    """Map a file path to its LanguageSpec by extension. None if unrecognized."""
    return _EXTENSION_INDEX.get(Path(path).suffix.lower())
