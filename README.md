# CodeQA

Question answering over unfamiliar codebases, with call-graph context.

Generic RAG chunks source at arbitrary character boundaries, splitting functions
mid-body, and retrieves isolated snippets with no awareness of how they connect.
CodeQA parses with tree-sitter and chunks at function and class boundaries, so
every retrieved unit is syntactically complete — then expands along the call
graph, pulling in a function's callers and callees. That is what lets it answer
*"what happens between a request arriving and my view function running"* rather
than only *"what does this one function do."*

> **Status: deployed.** Retrieval is measured (below), the locate → trace →
> synthesize agent pipeline is live behind a streaming API on a free-tier
> stack, and a VS Code extension talks to it end to end. A published
> Marketplace listing is the one remaining piece.

## Contents

- [How it works](#how-it-works)
- [Retrieval quality](#retrieval-quality)
- [Tech stack](#tech-stack)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [CLI reference](#cli-reference)
- [Tests](#tests)
- [Deploying](#deploying)
- [VS Code extension](#vs-code-extension)
- [Design notes](#design-notes)

## How it works

| Layer | What it does |
|---|---|
| **Indexing** | tree-sitter parses the repo; chunks are emitted at function/class boundaries; caller→callee edges are extracted from `@reference.call` tag queries; chunks are embedded into Postgres/pgvector |
| **Retrieval** | Dense vector search fused with Postgres full-text and an exact symbol index via Reciprocal Rank Fusion, then expanded along the call graph via a recursive CTE |
| **Reasoning** | A LangGraph state machine — locate → trace → synthesize — where `trace` judges whether retrieved context is sufficient and can route back to `locate` with a refined query, bounded by a retry limit |
| **Grounding** | Every citation an answer makes is checked mechanically against what was actually retrieved — a claimed `path:start-end` has to correspond to a chunk really in context, with a real line range |
| **API** | FastAPI with SSE streaming (`POST /v1/query`), API-key auth, Redis rate limiting, OpenTelemetry spans per pipeline stage |
| **Ingestion** | `POST /v1/repos` clones a git URL (SSRF-guarded, size- and time-capped) or accepts a local path, enqueues a job, and a durable worker (heartbeats, stale-job reclaim) does the actual indexing |
| **Client** | A VS Code extension — ask a question in a webview panel, watch the answer stream token by token, click a citation to jump straight to that file and line range |

Three retrieval strategies stay permanently selectable by config — `naive`,
`hybrid`, `hybrid_graph` — because the project's central claim is that
call-graph expansion beats semantic similarity alone, and that claim is only
worth anything if the comparison is reproducible at any commit. The naive
baseline is never deleted.

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

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Parsing | tree-sitter + per-language grammar wheels | Tag queries (`@definition.*`, `@reference.call`) are the same mechanism GitHub uses for code navigation — one query-driven code path for every language instead of a hand-written AST walker per language |
| Vector store / DB | Postgres + pgvector | `chunks` is partitioned `BY LIST (repo_id)` — a correctness fix for filtered ANN search, not an optimization (see below) |
| Cache / rate limiting | Redis | A Lua `EVAL` token bucket for atomic rate limiting; query results cached on `(repo_id, question, last_indexed_sha)` so a re-index is a natural cache bust |
| Orchestration | LangGraph | The one real cycle — `trace` → `locate` on insufficient context — is what actually justifies a graph library over three function calls |
| LLM / embeddings | LiteLLM | One interface across providers; the deployed instance runs Gemini for synthesis and Cohere for embeddings, chosen after measuring real free-tier limits, not guessing |
| API | FastAPI + Server-Sent Events | One-directional token streaming without the complexity of WebSockets |
| Graph traversal | Postgres recursive CTE (networkx as a differential test oracle) | Edges already live in Postgres; `WITH RECURSIVE ... CYCLE` gives bounded-depth, cycle-safe traversal in one query |
| Observability | OpenTelemetry + structlog | Spans per pipeline stage, log lines correlated by trace ID |
| Client | TypeScript VS Code extension, esbuild-bundled | A webview panel is a thin render surface; all real work (the API key, the SSE parsing) happens in the extension host |
| No ORM | Raw `psycopg` + plain numbered SQL migrations | The vector, RRF-fusion, and recursive-CTE queries are the interesting part of this codebase — an ORM would obscure exactly the part worth reading |

## Repository layout

```
src/codeqa/
  db/          migrations + connection handling
  languages/   grammar registry, tag queries, capability tiers
  indexing/    chunking, call extraction, embedding, pipeline, worker
  retrieval/   strategy interface: naive | hybrid | hybrid_graph
  graph/       call-graph traversal (Postgres CTE + networkx oracle)
  agents/      LangGraph state graph (locate -> trace -> synthesize)
  api/         FastAPI routes, auth, rate limiting, caching
  obs/         OpenTelemetry + structlog wiring
extension/     VS Code extension (TypeScript)
evals/         labeled datasets + retrieval eval runner
tests/         unit + integration tests, fixtures
```

## Getting started

Requires Docker, Python 3.14+, and (for the extension) Node 18+.

```bash
git clone https://github.com/Prathamesh0421/codebase-qa
cd codebase-qa

docker compose up -d postgres redis jaeger

cp .env.example .env
pip install -e ".[dev,local-embeddings]"

codeqa migrate          # apply schema
codeqa config           # show resolved configuration
```

`local-embeddings` pulls in `sentence-transformers` for free, reproducible,
rate-limit-free embeddings in dev — the deployed image uses a hosted
provider instead (see [Deploying](#deploying)) to keep the container small.

Jaeger's UI is at `localhost:16686` for viewing traces once
`CODEQA_OTEL_ENDPOINT` is set.

Index a local repo and ask it a question:

```bash
codeqa index /path/to/some/repo --slug myrepo
codeqa ask --repo myrepo "how does X work here?"
codeqa ask --repo myrepo --agent "what happens between the route and the DB?"
```

`--agent` runs the full locate → trace → synthesize pipeline with the retry
edge; without it, `ask` does a single retrieve-then-synthesize pass.

## CLI reference

| Command | What it does |
|---|---|
| `codeqa migrate` | Apply pending database migrations |
| `codeqa index PATH` | Index a local repository: walk, chunk, embed, store |
| `codeqa ask QUESTION --repo SLUG [--agent]` | Ask a question and stream the answer |
| `codeqa config` | Print resolved configuration, with secrets redacted |
| `codeqa worker [--once]` | Run the durable indexing worker — claims queued jobs, clones `git_url` repos safely, indexes them |
| `codeqa keys create --name NAME` | Issue an API key for the HTTP API (shown once) |

## Tests

```bash
pytest -m "not integration"   # unit only
pytest                        # requires Postgres from the compose stack
```

CI (`.github/workflows/ci.yml`) runs `ruff check .`, `mypy src/codeqa`,
migrations, and the full suite against Postgres + Redis service containers
on every push.

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

**The LLM has the same shape of limit.** `gemini-3.6-flash`'s free tier
caps *chat* completions at 20 requests/day — a completely different
quota axis than the embedding limits above, and easy to exhaust in a
handful of questions once the agent's retry edge fires. The deployed
instance runs `gemini-3.5-flash` instead: an older, non-frontier model
with a meaningfully more generous free allocation.

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
and Redis (reported, doesn't gate it — a deliberate fail-open choice for
a Q&A service, availability over strict abuse enforcement). No Jaeger is
deployed alongside this; tracing stays a local/dev concern.

## VS Code extension

`extension/` is a TypeScript extension: a webview panel with a running
Q&A transcript, SSE token streaming from `POST /v1/query`, Markdown
rendering, and clickable `path:start-end` citations that jump straight to
that file and line range in the editor.

```bash
cd extension
npm install
npm run compile
```

Then launch a real Extension Development Host to try it:

```bash
code --extensionDevelopmentPath="$(pwd)" /path/to/some/workspace
```

Set `codeqa.backendUrl` and `codeqa.defaultRepo` in that workspace's
settings, run **CodeQA: Set API Key** (a key from `codeqa keys create`,
stored via VS Code's `SecretStorage`, never a plain setting) and
**CodeQA: Ask a Question**.

Architecture note: an extension runs in two contexts that can't call each
other directly — the extension host (Node, full network access, holds
the API key) and the webview (a sandboxed, CSP-restricted surface that
only renders what it's told). All real work happens host-side; the
webview is a deliberately thin render layer that talks to the host only
through `postMessage`.

```bash
npm test        # sseClient parsing + a jsdom-driven test of the real webview bundle
npm run typecheck
```

## Design notes

**`chunks` is partitioned by `repo_id`.** Not an optimization — a correctness
fix. HNSW is an approximate index and Postgres applies `WHERE repo_id = N`
*after* searching it, so a repo-scoped query can silently return fewer rows than
requested. Measured on 30.5k vectors across three repos, asking for the 10
nearest in the smallest: unpartitioned returned **0**, partitioned returned
**10** at default settings.

**Citation grounding is a deterministic function, not a prompt instruction.**
The model is asked to cite honestly, but a request isn't a guarantee — every
citation in a streamed answer is checked mechanically afterward: does some
chunk actually in context have that file, with a real line range containing
the claim. Approximate call edges are also never presented as certain —
they carry `exact` / `approximate` / `unresolved` and keep the raw callee
name so they stay auditable.

**Not multi-tenant, by design.** Multi-*repo* indexing is in — a `repos`
registry, `repo_id` scoping every retrieval path — but there's no per-org
isolation or auth boundary between users. This is built as a single-operator
tool you control access to via issued API keys, not a product other people
bring their own billing to.
