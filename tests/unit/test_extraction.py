"""Call-site extraction and caller attribution.

Pure function of (spec, source, chunks) -- no database, same testing posture
as chunker.py. Uses the real chunker to produce the chunks fixture, since
extract_call_sites's caller attribution depends on genuine Chunk objects
with real byte spans, not hand-constructed ones that might not reflect how
chunk_file actually produces them.
"""

from codeqa.graph.extraction import extract_call_sites
from codeqa.indexing.chunker import chunk_file
from codeqa.languages import REGISTRY


def spec(name: str):
    return next(s for s in REGISTRY if s.name == name)


class TestCallerAttribution:
    def test_call_attributed_to_its_enclosing_function(self):
        src = b"def caller():\n    return callee()\n"
        chunks = chunk_file(spec("python"), "f.py", src)
        sites = extract_call_sites(spec("python"), src, chunks)

        assert len(sites) == 1
        assert sites[0].caller.symbol_name == "caller"
        assert sites[0].callee_name == "callee"

    def test_call_attributed_to_innermost_method_not_the_class(self):
        src = b"""
class Widget:
    def run(self):
        return self.helper()
"""
        chunks = chunk_file(spec("python"), "w.py", src)
        sites = extract_call_sites(spec("python"), src, chunks)

        assert len(sites) == 1
        assert sites[0].caller.kind == "method"
        assert sites[0].caller.qualified_name == "Widget.run"
        assert sites[0].callee_name == "helper"

    def test_multiple_calls_in_one_function_all_attributed(self):
        src = b"""
def caller():
    a()
    b()
    return c()
"""
        chunks = chunk_file(spec("python"), "f.py", src)
        sites = extract_call_sites(spec("python"), src, chunks)

        assert {s.callee_name for s in sites} == {"a", "b", "c"}
        assert all(s.caller.symbol_name == "caller" for s in sites)

    def test_calls_in_different_functions_attributed_separately(self):
        src = b"""
def one():
    return shared()

def two():
    return shared()
"""
        chunks = chunk_file(spec("python"), "f.py", src)
        sites = extract_call_sites(spec("python"), src, chunks)

        callers = {s.caller.symbol_name for s in sites}
        assert callers == {"one", "two"}
        assert all(s.callee_name == "shared" for s in sites)

    def test_module_level_call_with_no_enclosing_function_is_dropped(self):
        # app = Flask(__name__) at module scope: a real call, but not
        # attributable to any invocation context. Not a caller-less edge --
        # simply not represented, by design.
        src = b"x = factory()\n"
        chunks = chunk_file(spec("python"), "f.py", src)
        sites = extract_call_sites(spec("python"), src, chunks)
        assert sites == []

    def test_call_line_is_where_the_call_actually_appears(self):
        src = b"def caller():\n\n\n    return callee()\n"
        chunks = chunk_file(spec("python"), "f.py", src)
        sites = extract_call_sites(spec("python"), src, chunks)
        assert sites[0].call_line == 4

    def test_no_calls_produces_empty_list(self):
        src = b"def lonely():\n    return 1\n"
        chunks = chunk_file(spec("python"), "f.py", src)
        assert extract_call_sites(spec("python"), src, chunks) == []

    def test_call_via_containment_survives_across_languages(self):
        # Not a language-specific mechanism -- confirm it also works for a
        # language whose method/function distinction is already explicit
        # (unlike Python's, which chunker.py reclassifies via containment).
        src = b"""
class Widget {
  run() {
    return this.helper();
  }
}
"""
        chunks = chunk_file(spec("javascript"), "w.js", src)
        sites = extract_call_sites(spec("javascript"), src, chunks)
        assert len(sites) == 1
        assert sites[0].caller.qualified_name == "Widget.run"
        assert sites[0].callee_name == "helper"
