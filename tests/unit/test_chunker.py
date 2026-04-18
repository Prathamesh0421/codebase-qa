"""AST-aware chunking.

Every case here corresponds to a design claim made in chunker.py's module
docstring, not just a happy path -- in particular the two verified gaps
(Go/Rust methods have no byte-range containment relationship to their
type) are pinned as expected behaviour, not treated as bugs to route around.
"""

import hashlib

from codeqa.indexing.chunker import chunk_file
from codeqa.languages import REGISTRY


def spec(name: str):
    return next(s for s in REGISTRY if s.name == name)


class TestBasicChunking:
    def test_top_level_function(self):
        src = b"def greet(name):\n    return f'hi {name}'\n"
        chunks = chunk_file(spec("python"), "greet.py", src)

        assert len(chunks) == 1
        c = chunks[0]
        assert c.kind == "function"
        assert c.symbol_name == "greet"
        assert c.qualified_name is None
        assert c.start_line == 1
        assert c.end_line == 2
        # The function_definition node's span ends at the last statement,
        # excluding the file's trailing newline -- verified, not assumed;
        # tree-sitter node spans are tight around syntax, not surrounding
        # whitespace. content is exactly that span, byte for byte.
        assert c.content == src.decode().rstrip("\n")

    def test_content_is_exact_verbatim_span_not_reformatted(self):
        # The whole quality claim rests on this: content is exactly
        # source[start_byte:end_byte], never touched.
        src = b"def   weird_spacing( x ,y ):\n    return   x+y\n"
        chunks = chunk_file(spec("python"), "f.py", src)
        assert chunks[0].content == src.decode().rstrip("\n")

    def test_content_sha_matches_content(self):
        src = b"def f():\n    pass\n"
        c = chunk_file(spec("python"), "f.py", src)[0]
        assert c.content_sha == hashlib.sha256(c.content.encode()).hexdigest()


class TestMethodReclassification:
    """Containment-based reclassification: the mechanism that handles both
    Python (no definition.method exists at all) and PHP (methods are tagged
    definition.function too) with one uniform rule instead of two hacks."""

    def test_python_method_reclassified_and_qualified(self):
        src = b"""
class Greeter:
    def hello(self, name):
        return name
"""
        chunks = chunk_file(spec("python"), "g.py", src)

        classes = [c for c in chunks if c.kind == "class"]
        methods = [c for c in chunks if c.kind == "method"]
        assert [c.symbol_name for c in classes] == ["Greeter"]
        assert [c.symbol_name for c in methods] == ["hello"]
        assert methods[0].qualified_name == "Greeter.hello"

    def test_php_method_reclassified_despite_upstream_tagging_it_function(self):
        src = b"""<?php
class Greeter {
    public $name;
    public function hello() {
        return $this->name;
    }
}
"""
        chunks = chunk_file(spec("php"), "g.php", src)

        methods = [c for c in chunks if c.kind == "method"]
        assert [c.symbol_name for c in methods] == ["hello"]
        assert methods[0].qualified_name == "Greeter.hello"
        # property_declaration is definition.field -- not one of the four
        # chunkable kinds, and must not silently become a chunk.
        assert all(c.symbol_name != "name" for c in chunks)

    def test_javascript_method_already_distinguished_by_grammar(self):
        # JS's own tags.scm already emits definition.method for
        # method_definition nodes -- containment should agree, not conflict.
        src = b"""
class Widget {
  run(x) { return x; }
}
"""
        chunks = chunk_file(spec("javascript"), "w.js", src)
        methods = [c for c in chunks if c.kind == "method"]
        assert [c.symbol_name for c in methods] == ["run"]
        assert methods[0].qualified_name == "Widget.run"

    def test_nested_class_picks_innermost_enclosing(self):
        src = b"""
class Outer:
    class Inner:
        def method(self):
            pass
"""
        chunks = chunk_file(spec("python"), "n.py", src)
        method = next(c for c in chunks if c.kind == "method")
        assert method.qualified_name == "Inner.method"


class TestStructuralSiblingGap:
    """Go receiver methods and Rust impl-block methods are verified (by
    parsing real code, not assumption) to be byte-range SIBLINGS of their
    type, not children of it. qualified_name must stay None here -- a stated
    limitation, not a bug to chase with heuristics."""

    def test_go_receiver_method_has_no_qualified_name(self):
        src = b"""package main

type Widget struct {
	Name string
}

func (w *Widget) Greet() string {
	return w.Name
}
"""
        chunks = chunk_file(spec("go"), "w.go", src)
        method = next(c for c in chunks if c.kind == "method")
        assert method.symbol_name == "Greet"
        assert method.qualified_name is None

        # The struct itself is still a correct, independently addressable
        # chunk -- the gap is only in linking the two together.
        struct_chunk = next(c for c in chunks if c.kind == "class")
        assert struct_chunk.symbol_name == "Widget"

    def test_rust_impl_block_method_has_no_qualified_name(self):
        src = b"""
struct Widget {
    name: String,
}

impl Widget {
    fn greet(&self) -> &str {
        &self.name
    }
}
"""
        chunks = chunk_file(spec("rust"), "w.rs", src)
        method = next(c for c in chunks if c.kind == "method")
        assert method.symbol_name == "greet"
        assert method.qualified_name is None


class TestKindMapping:
    def test_interface_maps_to_class(self):
        src = b"""
interface Greeter {
  greet(name: string): string;
}
"""
        chunks = chunk_file(spec("typescript"), "g.ts", src)
        assert any(c.kind == "class" and c.symbol_name == "Greeter" for c in chunks)

    def test_go_struct_type_maps_to_class(self):
        src = b"""package main

type Widget struct {
	Name string
}
"""
        chunks = chunk_file(spec("go"), "w.go", src)
        assert [c.kind for c in chunks] == ["class"]
        assert chunks[0].symbol_name == "Widget"

    def test_rust_mod_maps_to_module(self):
        src = b"""
mod utils {
    fn helper() {}
}
"""
        chunks = chunk_file(spec("rust"), "u.rs", src)
        assert any(c.kind == "module" and c.symbol_name == "utils" for c in chunks)


class TestFallbackAndEdgeCases:
    def test_file_with_no_definitions_gets_whole_file_module_chunk(self):
        src = b"import os\nimport sys\n\nDEBUG = True\n"
        chunks = chunk_file(spec("python"), "config.py", src)

        assert len(chunks) == 1
        assert chunks[0].kind == "module"
        assert chunks[0].symbol_name == "config.py"
        assert chunks[0].content == src.decode()
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 4

    def test_tier3_language_gets_whole_file_module_chunk(self):
        # No tags.scm exists for C# at all -- extract_tags always returns [],
        # so this exercises the exact same fallback path as a tier1 file
        # with no definitions, with zero special-case code for tier3.
        src = b"class Widget { void Run() { } }"
        chunks = chunk_file(spec("c_sharp"), "Widget.cs", src)

        assert len(chunks) == 1
        assert chunks[0].kind == "module"
        assert chunks[0].content == src.decode()

    def test_empty_file_produces_no_chunks(self):
        assert chunk_file(spec("python"), "empty.py", b"") == []

    def test_whitespace_only_file_produces_no_chunks(self):
        assert chunk_file(spec("python"), "blank.py", b"   \n\n  \n") == []

    def test_huge_function_is_one_unsplit_chunk(self):
        # The core invariant: no matter how large, a definition is never
        # windowed or truncated at chunking time. 500 statements is well
        # past any plausible embedding context window.
        body = "\n".join(f"    x{i} = {i}" for i in range(500))
        src = f"def big():\n{body}\n    return x0\n".encode()

        chunks = chunk_file(spec("python"), "big.py", src)

        assert len(chunks) == 1
        assert chunks[0].kind == "function"
        assert chunks[0].content == src.decode().rstrip("\n")
        assert chunks[0].end_line - chunks[0].start_line == 501
