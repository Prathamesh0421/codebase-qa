"""Find every call site in a file and attribute it to its enclosing
function or method chunk.

No database access here -- this is a pure function of (spec, source, the
file's already-computed chunks), same testing posture as chunker.py.
Resolving the raw callee name to an actual chunk elsewhere in the repo is a
separate, repo-wide concern (resolve.py) that can only run once every file's
chunks exist.

Re-parses the file via extract_tags rather than threading tags through from
chunk_file: a deliberate, accepted cost (chunk_file's public API stays
unchanged, this module stays decoupled) rather than a premature optimization.
tree-sitter is fast; Flask indexed in 4.5s total in Phase 4 with embedding
dominating that time, not parsing.
"""

from dataclasses import dataclass

from codeqa.indexing.chunker import Chunk
from codeqa.languages import LanguageSpec, extract_tags
from codeqa.spans import smallest_enclosing


@dataclass(frozen=True)
class CallSite:
    caller: Chunk
    callee_name: str
    call_line: int


def extract_call_sites(spec: LanguageSpec, source: bytes, chunks: list[Chunk]) -> list[CallSite]:
    """Every @reference.call tag whose byte span falls inside one of chunks.

    A call with no enclosing function/method chunk -- at module level, or
    directly in a class body outside any method -- is dropped rather than
    attributed to a class or module chunk. Those aren't invocation contexts
    in the sense a call graph cares about, and forcing an attribution there
    would misrepresent module-level setup code as a "caller."
    """
    tags = extract_tags(spec, source)
    call_tags = [t for t in tags if t.kind == "reference.call"]
    func_chunks = [c for c in chunks if c.kind in ("function", "method")]

    sites = []
    for tag in call_tags:
        caller = smallest_enclosing(tag, func_chunks)
        if caller is None:
            continue
        sites.append(CallSite(caller=caller, callee_name=tag.name, call_line=tag.start_line))
    return sites
