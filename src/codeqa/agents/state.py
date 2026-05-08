"""Pydantic state schema for the locate -> trace -> synthesize graph.

Pydantic, not a TypedDict (LangGraph supports both) -- this project already
uses Pydantic for config (pydantic-settings) and this state gets exactly the
same benefit: real validation, not just a type hint nobody checks at runtime.
Verified interactively that a plain stdlib dataclass (RetrievedChunk) works
directly as a Pydantic field type with no extra config -- Pydantic v2
introspects it natively.
"""

from pydantic import BaseModel, ConfigDict

from codeqa.retrieval.strategy import RetrievedChunk


class AgentState(BaseModel):
    # frozen=False (the default): nodes return partial-update dicts that
    # LangGraph merges into a fresh state, so instances here are never
    # mutated in place -- but RetrievedChunk itself IS frozen, so keeping
    # this model mutable-by-default doesn't compromise the chunks' own
    # immutability.
    model_config = ConfigDict(arbitrary_types_allowed=False)

    repo_id: int
    question: str  # the original user question -- never mutated by any node
    current_query: str  # what `locate` searches for THIS attempt; may be refined by `trace`
    chunks: list[RetrievedChunk] = []
    attempt: int = 0
    # No default: config.agent_max_attempts is the one place this decision
    # gets made (mirroring get_strategy's own docstring reasoning) -- a
    # second default here would just be a second place it could drift from
    # that one, silently, the first time someone constructs AgentState
    # without reading config.py.
    max_attempts: int
    sufficient: bool = False
    trace_reasoning: str = ""
    answer: str = ""
