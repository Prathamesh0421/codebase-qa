"""Language registry and tag extraction.

Every override query here was verified against a real parse tree before being
written (see docs/deep-dive.html, Phase 2) -- these tests pin that behaviour so
a future grammar upgrade that silently changes node shapes is caught, not
discovered downstream in chunking or call-graph extraction.
"""

import pytest

from codeqa.languages import REGISTRY, detect_language, extract_tags
from codeqa.languages.registry import _EXTENSION_INDEX


def spec(name: str):
    return next(s for s in REGISTRY if s.name == name)


class TestDetection:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("app.py", "python"),
            ("index.js", "javascript"),
            ("component.jsx", "javascript"),
            ("index.ts", "typescript"),
            ("component.tsx", "tsx"),
            ("main.go", "go"),
            ("App.java", "java"),
            ("lib.rs", "rust"),
            ("model.rb", "ruby"),
            ("index.php", "php"),
            ("widget.cpp", "cpp"),
            ("widget.hpp", "cpp"),
            ("Widget.cs", "c_sharp"),
        ],
    )
    def test_extension_maps_to_expected_language(self, path, expected):
        result = detect_language(path)
        assert result is not None
        assert result.name == expected

    def test_unknown_extension_returns_none(self):
        assert detect_language("data.xyz") is None
        assert detect_language("Makefile") is None

    def test_no_two_specs_claim_the_same_extension(self):
        # _EXTENSION_INDEX is built by a dict comprehension over all specs;
        # a silent collision would just mean the second spec wins, which is
        # exactly the kind of bug that shows up as "wrong language for a
        # file" three phases later with no obvious cause.
        seen: dict[str, str] = {}
        for s in REGISTRY:
            for ext in s.extensions:
                assert ext not in seen, f"{ext} claimed by both {seen[ext]} and {s.name}"
                seen[ext] = s.name
        assert seen == {k: v.name for k, v in _EXTENSION_INDEX.items()}


class TestTiers:
    def test_tier1_languages_have_call_references(self):
        for name in ("python", "javascript", "go", "java", "rust", "ruby", "php"):
            s = spec(name)
            assert s.tier == "tier1"
            assert s.tags_query is not None
            assert "reference.call" in s.tags_query

    def test_tier2_languages_have_query_via_override(self):
        for name in ("typescript", "tsx", "cpp"):
            s = spec(name)
            assert s.tier == "tier2"
            assert s.tags_query is not None
            assert "reference.call" in s.tags_query

    def test_tier3_language_has_no_query(self):
        s = spec("c_sharp")
        assert s.tier == "tier3"
        assert s.tags_query is None


class TestExtractTags:
    def test_python_function_and_call(self):
        src = b"def greet(name):\n    return say_hello(name)\n"
        tags = extract_tags(spec("python"), src)

        kinds = {t.kind for t in tags}
        assert "definition.function" in kinds
        assert "reference.call" in kinds

        fn = next(t for t in tags if t.kind == "definition.function")
        assert fn.name == "greet"
        assert fn.start_line == 1
        assert fn.end_line == 2

        call = next(t for t in tags if t.kind == "reference.call")
        assert call.name == "say_hello"

    def test_javascript_constructor_excluded_by_predicate(self):
        # Upstream tags.scm has (#not-eq? @name "constructor") -- this
        # confirms py-tree-sitter's QueryCursor actually honours real
        # predicates rather than silently ignoring them, which the
        # GitHub-style @doc/#strip! directives are (see registry.py).
        src = b"class Foo {\n  constructor() {}\n  bar() { return baz(1); }\n}\n"
        tags = extract_tags(spec("javascript"), src)

        methods = [t for t in tags if t.kind == "definition.method"]
        assert [m.name for m in methods] == ["bar"]

    def test_javascript_handles_jsx_without_error(self):
        # .jsx routes to the plain javascript grammar (no separate package);
        # this is the reason that routing is safe.
        src = b'function App() { return <div onClick={() => go()}>{hi("x")}</div>; }'
        tags = extract_tags(spec("javascript"), src)
        calls = {t.name for t in tags if t.kind == "reference.call"}
        assert calls == {"go", "hi"}

    def test_typescript_override_adds_missing_definitions_and_calls(self):
        # Upstream TS tags.scm alone would produce nothing for this snippet
        # at all -- no function_declaration, class_declaration or
        # reference.call. This is the override query proving its purpose.
        src = b"""
function greet(name: string): string {
  return sayHello(name);
}
class Greeter {
  greet(name: string) {
    return this.helper(name);
  }
}
"""
        tags = extract_tags(spec("typescript"), src)
        kinds = {t.kind for t in tags}
        assert {
            "definition.function",
            "definition.class",
            "definition.method",
            "reference.call",
        } <= kinds

        calls = {t.name for t in tags if t.kind == "reference.call"}
        assert calls == {"sayHello", "helper"}

    def test_tsx_override_survives_jsx_syntax(self):
        src = b"""
function App() {
  return <div onClick={() => helper()}>{greet("x")}</div>;
}
"""
        tags = extract_tags(spec("tsx"), src)
        calls = {t.name for t in tags if t.kind == "reference.call"}
        assert calls == {"helper", "greet"}

    def test_cpp_override_adds_missing_calls_dot_and_arrow(self):
        # Upstream C++ tags.scm has no @reference.call at all. Both dot and
        # arrow member calls resolve to the same field_expression node shape
        # in this grammar -- verified, not assumed -- so one pattern covers
        # both syntaxes.
        src = b"""
struct Widget {
  int run(int x) { return x; }
};
void f(Widget w, Widget* p) {
  w.run(1);
  p->run(2);
  helper(3);
}
"""
        tags = extract_tags(spec("cpp"), src)
        calls = {t.name for t in tags if t.kind == "reference.call"}
        assert calls == {"run", "helper"}

    def test_c_sharp_extracts_nothing(self):
        # tier3: the file is parseable and detected, but there is no query
        # to run. Returns [] rather than raising -- an unsupported language
        # is a known limitation, not an error condition.
        src = b"class Widget { void Run() { Helper(); } }"
        assert extract_tags(spec("c_sharp"), src) == []

    def test_empty_source_extracts_nothing(self):
        assert extract_tags(spec("python"), b"") == []

    def test_multiple_definitions_paired_with_correct_names(self):
        # Regression guard for the .captures() vs .matches() distinction:
        # .captures() returns a flat dict that can mispair @name with the
        # wrong @definition.function when there are multiple matches.
        # .matches() keeps each match's captures grouped correctly, which is
        # what extract_tags uses.
        src = b"""
def alpha():
    return beta(1)

def beta(x):
    return gamma(x)
"""
        tags = extract_tags(spec("python"), src)
        functions = {
            t.name: (t.start_line, t.end_line) for t in tags if t.kind == "definition.function"
        }
        assert functions == {"alpha": (2, 3), "beta": (5, 6)}

        calls = [t.name for t in tags if t.kind == "reference.call"]
        assert calls == ["beta", "gamma"]

    def test_doc_and_scope_captures_are_not_yielded_as_tags(self):
        # Upstream queries also emit @doc, @local.scope, @reference.type and
        # similar -- real captures this project doesn't act on. Confirms the
        # _KIND_PREFIXES filter actually excludes them rather than passing
        # everything through.
        src = b"""
/// Computes a greeting.
function greet(name: string): string {
  return sayHello(name);
}
"""
        tags = extract_tags(spec("typescript"), src)
        assert all(t.kind.startswith(("definition.", "reference.call")) for t in tags)
