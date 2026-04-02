# CodeQA

Question answering over unfamiliar codebases, with call-graph context.

Generic RAG chunks source at arbitrary character boundaries, splitting functions
mid-body, and retrieves isolated snippets with no awareness of how they connect.
CodeQA parses with tree-sitter and chunks at function and class boundaries, so
every retrieved unit is syntactically complete — then expands along the call
graph, pulling in a function's callers and callees. That is what lets it answer
*"what happens between a request arriving and my view function running"* rather
than only *"what does this one function do"*.

> **Status: in development.** See [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md)
> for phase-by-phase progress. No retrieval-quality numbers are published here
> until the evaluation harness has measured them.

## How it works

| Layer | What it does |
|---|---|
| **Indexing** | tree-sitter parses the repo; chunks are emitted at function/class boundaries; caller→callee edges are extracted; chunks are embedded into Postgres/pgvector |
| **Retrieval** | Dense vector search fused with Postgres full-text and an exact symbol index via Reciprocal Rank Fusion, then expanded along the call graph |
| **Reasoning** | A LangGraph state machine — locate → trace → synthesize — where trace walks the cyclic call graph with bounded depth |
| **API** | FastAPI with SSE streaming, API-key auth, Redis rate limiting, OpenTelemetry spans per stage |

Three retrieval strategies stay permanently selectable by config — `naive`,
`hybrid`, `hybrid_graph` — because the project's central claim is that
call-graph expansion beats semantic similarity alone, and that claim is only
worth anything if the comparison is reproducible at any commit.

## Running it

```bash
docker compose up -d postgres redis jaeger

cp .env.example .env
pip install -e ".[dev,local-embeddings]"

codeqa migrate          # apply schema
codeqa config           # show resolved configuration
```

Jaeger UI is at `localhost:16686`.

## Tests

```bash
pytest -m "not integration"   # unit only
pytest                        # requires Postgres from the compose stack
```

## Design notes

- [`docs/deep-dive.html`](docs/deep-dive.html) — architecture, every design
  decision with the alternatives rejected, and the problems hit while building.
- [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) — phase plan and
  status.

Two decisions worth knowing before reading the code:

**`chunks` is partitioned by `repo_id`.** Not an optimization — a correctness
fix. HNSW is an approximate index and Postgres applies `WHERE repo_id = N`
*after* searching it, so a repo-scoped query can silently return fewer rows than
requested. Measured on 30.5k vectors across three repos, asking for the 10
nearest in the smallest: unpartitioned returned **0**, partitioned returned
**10** at default settings.

**There is no ORM.** The interesting queries — vector search, RRF fusion,
recursive-CTE graph traversal, partition management — are ones an ORM obscures
rather than helps.
