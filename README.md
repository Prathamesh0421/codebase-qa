# CodeQA

Question answering over unfamiliar codebases, with call-graph context.

Generic RAG chunks source at arbitrary character boundaries, splitting functions
mid-body, and retrieves isolated snippets with no awareness of how they connect.
CodeQA parses with tree-sitter and chunks at function and class boundaries, so
every retrieved unit is syntactically complete — then expands along the call
graph, pulling in a function's callers and callees. That is what lets it answer
*"what happens between a request arriving and my view function running"* rather
than only *"what does this one function do"*.

> **Status: deployed.** Retrieval is measured (below), the locate → trace →
> synthesize agent pipeline is live behind a streaming API, and a VS Code
> extension is the one remaining piece. See
> [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) for
> phase-by-phase progress.

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

## Retrieval quality

Measured on 25 hand-labeled Flask questions. Gold sets (which file/symbol a
correct answer must cite) were written by reading Flask's source directly,
*before* consulting `call_edges` — deliberately, so that measuring whether
graph expansion finds a question's gold chunks isn't circular with how the
gold chunks were chosen. See `evals/datasets/flask_qa.json` and
`evals/runners/retrieval_eval.py`.

| strategy | precision | recall | avg chunks returned |
|---|---|---|---|
| naive | 0.13 | 0.89 | 10.0 |
| hybrid | 0.13 | 0.89 | 10.0 |
| hybrid_graph | 0.08 | **1.00** | 22.6 |

The honest result: **hybrid's lexical + symbol fusion measured zero recall
improvement over naive dense search alone** — naive and hybrid tie on every
individual question in the set, not just on average. All of the measured
lift comes from call-graph expansion, and it's concentrated exactly where
this project's thesis predicts: 20 of 25 questions are single-function
lookups where naive already hits 100% recall (a ceiling, nothing left to
win), and the other 5 are multi-hop questions ("what happens between X and
Y") where naive and hybrid stall at 33–50% recall and `hybrid_graph` recovers
100%. Of the gold chunks that only graph expansion found, 4 of 9 were located
by *no other retrieval mechanism at any rank* — not a near-miss another
component almost ranked highly, but genuinely inaccessible without walking
the call graph.

The cost is real, not free: `hybrid_graph` returns more than double the
chunks on average (22.6 vs 10), which is lower precision and more context
for an LLM to read per answer.

## Running it

```bash
docker compose up -d postgres redis jaeger

cp .env.example .env
pip install -e ".[dev,local-embeddings]"

codeqa migrate          # apply schema
codeqa config           # show resolved configuration
```

Jaeger UI is at `localhost:16686`.

## Deploying

A genuinely free stack, not "free tier if you're careful": **Supabase**
(Postgres + pgvector, no card), **Upstash** (Redis, no card), **Render**
(the API, free web-service tier, no card), and a **GitHub Actions**
scheduled workflow standing in for a worker process — Render's free tier
has no free background-worker or cron option, confirmed against their own
docs, so `.github/workflows/worker.yml` runs `codeqa worker --once` (the
supervised one-shot mode `cli.py`'s `worker` command already had) every
10 minutes instead. `render.yaml` is a Render Blueprint: `New → Blueprint`
against this repo picks it up and prompts for four secrets
(`CODEQA_DATABASE_URL`, `CODEQA_REDIS_URL`, `CODEQA_LLM_API_KEY`,
`CODEQA_EMBEDDING_PROVIDER_API_KEY`) that never live in the file itself.
The same four need setting as this repo's Actions secrets too, for the
worker workflow.

**Use Supabase's Session pooler connection string, not the direct one.**
`db.<project>.supabase.co:5432` resolves to an IPv6-only address; GitHub
Actions runners (and possibly other hosts) don't reliably have IPv6
egress, which fails as `OperationalError: Network is unreachable` — a
real error hit deploying this, not a hypothetical. The pooler hostname
(`aws-<region>.pooler.supabase.com:5432`, **session** mode specifically,
not transaction mode — transaction mode doesn't support prepared
statements, which psycopg3 can use by default) is IPv4-compatible and
works from anywhere.

**Hosted embeddings are Cohere (`embed-v4.0`), not Gemini, and that
wasn't the first thing tried.** Local embeddings (`sentence-transformers`)
were tried first specifically to avoid any hosted quota at all, but a
live measurement (PyTorch and ONNX Runtime backends both) put peak memory
around 800MB just loading the model and embedding a handful of chunks —
over the 512MB Render's free tier budgets for the whole container.
Gemini's hosted embedding API was tried next and measured at a free-tier
cap of 100 requests/minute, which fails any repo bigger than ~100 chunks
outright. Cohere's trial tier raises that ceiling to about 100,000
tokens/minute — real headroom for a small-to-medium repo, not unlimited:
something Flask-sized (~450 chunks, 200k+ tokens) still doesn't fit in
one quota window on this stack. That's a stated scope limit of the
deployed instance, not a bug — index a large repo via the local CLI path
instead, where `local` embeddings have no such ceiling.

```bash
# once you have Supabase/Upstash/Render accounts and a Cohere + Gemini key:
CODEQA_DATABASE_URL="<supabase session pooler URL>" codeqa migrate
CODEQA_DATABASE_URL="<supabase session pooler URL>" codeqa keys create --name vscode

curl https://<your-render-service>.onrender.com/health
curl https://<your-render-service>.onrender.com/v1/repos \
  -H "Authorization: Bearer <key from codeqa keys create>" \
  -H "Content-Type: application/json" \
  -d '{"slug":"markupsafe","display_name":"MarkupSafe","source_kind":"git_url","source_ref":"https://github.com/pallets/markupsafe"}'
# then, to actually process the queued job before the next scheduled run:
gh workflow run worker.yml
```

`GET /health` needs no key and checks Postgres (gates the status code)
and Redis (reported, doesn't gate — see the Phase 14b notes in
`docs/deep-dive.html` on fail-open degradation). No Jaeger is deployed
alongside this; tracing stays a local/dev concern.

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
