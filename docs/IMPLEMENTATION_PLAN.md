# Implementation Plan

Working plan for building the codebase Q&A assistant. Derived from
`codebase_qa_design.docx`, revised for decisions taken during design review
(see [Decisions](#decisions-taken) at the bottom).

## How we work through this

Each phase is sized to be reviewable in one sitting. For every phase:

1. I explain what I'm about to build and why, before writing code.
2. I build it.
3. I walk through what was built and the concepts behind it.
4. **You ask questions until you're satisfied. Nothing moves forward until you say so.**

Each phase below lists *Concepts* — the things worth understanding in that
phase, and fair game to quiz me on. These are also, not coincidentally, the
things an interviewer would probe.

Status: `[ ]` not started · `[~]` in progress · `[x]` done and reviewed

---

## Phase 0 — Scaffold and configuration `[x]`

**Goal:** a runnable skeleton with infrastructure up.

- Package layout under `src/codeqa/`, `pyproject.toml`, Python 3.14 (verified:
  torch 2.13, tree-sitter 0.26, langgraph 1.2 all have 3.14 wheels).
- `pydantic-settings` typed config, `.env.example`, no hardcoded secrets.
- Docker Compose: app + Postgres/pgvector + Redis + Jaeger.

**Concepts:** 12-factor config and why config is typed rather than
`os.environ.get`; why there is deliberately no ORM here.

**Done when:** `docker compose up` gives a healthy Postgres with the `vector`
extension, a Redis, and a Jaeger UI. ✅ *All three healthy in 6s; `codeqa
migrate` applies the schema against the compose Postgres and the dimension
guard rejects a mismatched config.*

**Built:** `pyproject.toml`, `config.py` (typed settings, closed-set retrieval
strategy), `.env.example`, `docker-compose.yml`, `Dockerfile` (multi-stage, no
torch in runtime), `cli.py` (`codeqa migrate` / `codeqa config`), `README.md`.

---

## Phase 1 — Schema and migrations `[x]`

**Goal:** the data model, which encodes most of the architecture.

- Tables: `repos`, `files`, `chunks`, `symbols`, `call_edges`, `index_jobs`, `api_keys`.
- Plain numbered SQL migrations plus a small runner (no Alembic — we have no ORM).

**Concepts:** why `repo_id` is on every retrievable row; why pgvector needs a
fixed dimension at DDL time and how we survive changing the embedding model;
what an HNSW index trades away versus exact search; why `tsvector` lives in the
same table as the vector; why `last_indexed_sha` has to exist on day one.

**Done when:** migrations apply cleanly to a fresh database and roll forward
idempotently. ✅ *9 integration tests green against Postgres 17.11 /
pgvector 0.8.6.*

**Built:** `0001_init.sql` (7 tables, `chunks` partitioned by `repo_id`) and
`db/migrate.py` (drift detection, advisory lock, `${EMBEDDING_DIM}`
substitution, deployed-dimension verification).

---

## Phase 2 — Language layer `[x]`

**Goal:** turn "support all languages" into one code path.

- Grammar registry built on tree-sitter `queries/tags.scm`.
- Capability tiering: **tier 1** = definitions + `@reference.call` (real call
  edges); **tier 2** = definitions only (approximate edges); **tier 3** = no tag
  queries (chunks + symbols only).
- Custom `.scm` overrides for grammars with gaps (TypeScript, C++, C#).

**Concepts:** what a concrete syntax tree is; tree-sitter's S-expression query
language; why tag queries generalize where hand-written AST walkers don't; why
declaring a capability tier is more honest than silently degrading.

**Done when:** given a file, we report its language and tier, and extract its
definitions. ✅ *28 unit tests green; 11 languages registered across 3 tiers.*

**Built:** `languages/registry.py` (grammar registry, tier assignment),
`languages/tags.py` (the one query-execution path every language goes
through), `languages/overrides/{typescript,cpp}.scm` (additive queries for
grammars with real gaps, each verified against a real parse tree before being
written). Rejected `tree-sitter-language-pack` (see decision below) in favor
of 10 individually pinned official grammar packages.

---

## Phase 3 — AST-aware chunking `[ ]`

**Goal:** the doc's single biggest quality claim.

- Emit one chunk per function/class/method with `(path, symbol, line span)`.
- Handle nesting (methods inside classes), module-level code, oversized
  functions that exceed the embedding context window.

**Concepts:** why fixed-window chunking breaks code specifically; how nested
definitions are attributed; what to do with a 2,000-line function.

**Done when:** chunking Flask produces syntactically complete chunks, and unit
tests pin the boundaries.

---

## Phase 4 — Embeddings and the indexing pipeline `[ ]`

**Goal:** chunks become searchable vectors.

- Embedding provider interface: local `sentence-transformers` (dev/CI/evals) and
  a hosted API (deployed). Model name + dimension recorded per repo.
- Full-index path: walk files → chunk → embed in batches → upsert.

**Concepts:** what an embedding actually is; why batching dominates indexing
throughput; why the model name is stored alongside the vectors; local vs hosted
as a deployment-size tradeoff, not a quality one.

**Done when:** Flask indexes end-to-end and `chunks` is populated with vectors.

---

## Phase 5 — Naive retrieval baseline + CLI `[ ]`

**Goal:** the first working answer — *and the eval baseline*.

- Single-shot cosine similarity → top-k → prompt → answer with citations.
- `codeqa ask "..."` prints a streamed answer.

> This phase exists to be **beaten**. The entire measurement story is
> naive → hybrid → +call-graph, so this code stays runnable forever. It is
> selected by config, never deleted.

**Concepts:** cosine similarity and why it suits embeddings; what "grounding" a
prompt means; exactly where naive RAG fails on code (this is the thesis).

**Done when:** you can ask Flask a question and get a cited answer.

---

## Phase 6 — Call-graph extraction `[ ]`

**Goal:** the differentiator.

- Extract `@reference.call` sites, resolve callee names to chunks.
- Classify each edge `exact` / `approximate` / `unresolved`; keep the raw callee
  name so approximations stay auditable.

**Concepts:** why call resolution is genuinely hard in dynamic languages; why an
approximate edge must never be presented as a certain call path; what we
deliberately don't attempt (no type inference, no dynamic dispatch).

**Done when:** Flask's `full_dispatch_request → dispatch_request → view` chain
exists in `call_edges`.

---

## Phase 7 — Graph traversal `[ ]`

**Goal:** walk the graph safely.

- Traversal behind an interface, two implementations:
  - **production:** Postgres `WITH RECURSIVE ... CYCLE` (bounded depth,
    built-in cycle detection, repo-scoped, no memory load).
  - **test oracle:** networkx BFS.
- Differential tests assert both produce identical traversals on fixtures.

**Concepts:** how a recursive CTE works; Postgres's `CYCLE` clause; why a
visited-set is mandatory on a cyclic call graph; differential testing as a
technique.

**Done when:** traversal terminates on a deliberately cyclic fixture, and both
implementations agree.

---

## Phase 8 — Hybrid retrieval and fusion `[ ]`

**Goal:** the full retrieval stack, swappable.

- Strategy interface with three modes: `naive`, `hybrid`, `hybrid+graph`.
- Fuse vector + `tsvector` lexical + exact symbol lookup via **Reciprocal Rank
  Fusion**; then call-graph expansion pulls in callers/callees.

**Concepts:** why hybrid beats dense-only for code (identifiers are lexical, not
semantic); how RRF fuses ranked lists without tuning score scales; the
context-window budget that caps expansion.

**Done when:** all three strategies run against the same query via config.

---

## Phase 9 — Evaluation harness ⭐ `[ ]`

**Goal:** find out whether the core hypothesis is actually true.

- ~25–30 hand-labeled Flask Q&A pairs with known-correct files/symbols.
- Runner reporting precision@k and recall@k for **naive vs hybrid vs +graph**.

> **Moved ahead of the agents deliberately.** "Call-graph expansion improves
> retrieval" is a *retrieval* claim — testable here, with no agents involved. If
> it's weak, that is the single most important finding this project can produce,
> and we need it now rather than with three days left. From here on, this
> harness guards every later change.

**Concepts:** precision@k vs recall@k and which matters here; what makes a
labeled set honest; why a modest measured number beats an impressive invented
one.

**Done when:** we have real numbers for all three strategies, and they go in the
README whatever they say.

⛔ **Checkpoint: we discuss the results before continuing.**

---

## Phase 10 — Multi-agent pipeline `[ ]`

**Goal:** locate → trace → synthesize in LangGraph.

- Pydantic state schema; explicit nodes and edges.
- **A real conditional edge:** if `trace` finds insufficient context, route back
  to `locate` to re-query, bounded by an attempt counter.

**Concepts:** what a state graph buys over three function calls (and the honest
answer to that challenge — the conditional edge is why); where the *agent*
graph's cycle is, as distinct from the *call* graph's cycles; bounding
non-termination.

**Done when:** the pipeline answers multi-hop questions, and the retry edge
demonstrably fires.

---

## Phase 11 — Citation grounding `[ ]`

**Goal:** make "withholds ungrounded claims" a real, tested mechanism.

- A deterministic function: claimed citation → chunk actually in context → line
  range actually exists in that file. Failures are dropped, not rendered.

**Concepts:** why this is code rather than a prompt instruction; what grounding
does and does not catch (it verifies the *citation*, not the *claim*) — stating
that limit precisely is the defensible position.

**Done when:** unit tests cover accept and reject paths, including a fabricated
line range.

---

## Phase 12 — Ingestion and the job worker `[ ]`

**Goal:** any repo, safely.

- `POST /v1/repos`: git URL (server clones) **or** client-pushed local path.
- Clone safety: SSRF guard, `--depth 1`, size cap, timeout, disk quota.
- Durable worker polling `index_jobs`, with heartbeats so a dead worker's job is
  reclaimed rather than stuck in `running`.

**Concepts:** why indexing can't live in a request; job-queue semantics without
Celery; why tree-sitter *parses but never executes*, which is what makes
indexing untrusted code safe.

**Done when:** a public git URL indexes end-to-end and the job survives a worker
restart.

---

## Phase 13 — Incremental re-indexing `[ ]`

**Goal:** make the resume bullet true.

- Diff against `last_indexed_sha`, re-embed only changed files, delete chunks
  for removed files, update the graph.

**Concepts:** why the blob SHA check beats trusting git's changed-path list;
what happens to call edges pointing at deleted chunks.

**Done when:** a one-file change re-embeds one file, asserted by test.

---

## Phase 14 — API surface and production hardening `[ ]`

**Goal:** the production story.

- `POST /v1/query` with SSE streaming; health endpoint.
- API-key auth (hashed), Redis per-key token bucket, query cache.
- Retries with backoff, graceful degradation, OTel spans per stage, structlog
  JSON correlated by trace ID.

**Concepts:** SSE vs WebSockets here; token bucket vs fixed window; what
"graceful degradation" means concretely per failure mode; why cache keys need
normalized questions *and* repo scope.

**Done when:** a query streams end-to-end and shows as one trace in Jaeger.

---

## Phase 15 — Tests and CI `[ ]`

- Unit: chunk boundaries, call extraction, grounding accept/reject, incremental
  file selection, traversal equivalence.
- Integration: index-then-ask on a fixture repo; cache hit avoids re-running the
  pipeline (asserted on mocked call counts); LLM failure mid-chain degrades cleanly.
- GitHub Actions on every push.

**Done when:** CI is green from a clean checkout.

---

## Phase 16 — Deploy and VS Code extension `[ ]`

- Deploy the backend with managed Postgres + Redis, so the extension works for
  someone who isn't you.
- TypeScript extension: webview panel, SSE streaming, clickable `file:line`
  citations that open the file via `vscode.open`.

**Concepts:** how the webview talks to the backend (this is the one
defensibility question the extension must survive); why it's deliberately a thin
client.

**Done when:** a published Marketplace listing works against the deployed backend.

---

## Decisions taken

Departures from the design doc, with reasons:

| Decision | Reason |
|---|---|
| **Multi-repo, not multi-tenant** | Doc cut "one codebase at a time"; you asked for all repos. Multi-repo indexing (`repo_id` everywhere) is in; per-org isolation stays cut. |
| **Tag queries over per-language parsers** | `queries/tags.scm` ships with grammars and defines `@definition.*` / `@reference.call`. One code path for all languages; adding a language ships a `.scm`, not a parser. |
| **Postgres recursive CTE over networkx** | Edges already live in Postgres; networkx is in-memory and would need loading per repo. `WITH RECURSIVE ... CYCLE` gives bounded depth and cycle detection natively. networkx demoted to test oracle. |
| **Evals moved before agents** | The hypothesis is a retrieval claim. Testing it at day 13 means discovering a weak thesis with 3 days left. |
| **Real conditional edge in the agent graph** | The doc justifies LangGraph with "cycles", but the cycles were in the *call* graph, not the agent graph. The retry edge closes that hole. |
| **RRF for fusion** | Doc says "reranked" without a method. RRF needs no model and no score-scale tuning. Cross-encoder becomes a *measured* upgrade. |
| **Local + hosted embeddings behind one interface** | Local is reproducible and free for CI/evals; hosted keeps ~2GB of torch out of the deployed image. |
| **Rejected `tree-sitter-language-pack`** | Downloads compiled native binaries at runtime from a young non-canonical publisher (org created Oct 2025). For a system parsing untrusted third-party repos, that reintroduces the exact supply-chain risk "parses but never executes" is meant to rule out. Pinned 10 official grammar packages individually instead. |
| **Anchor repo: Flask** | Real multi-hop flows (`route → add_url_rule → full_dispatch_request → dispatch_request → view`) — exactly what naive RAG fails and call-graph expansion should win. |
| **Python 3.14** | Verified: torch 2.13, tree-sitter 0.26, langgraph 1.2, psycopg 3.3 all install. |

## Scope-cut order

If time compresses, in this order: **1)** VS Code extension (CLI keeps a working
interface) · **2)** labeled set down to ~10–15 questions · **3)** config and
observability polish · **4)** approximate call graph instead of resolved edges,
limitation stated.

**Protected:** AST chunking, call-graph expansion, locate→trace→synthesize, and
at least one measured retrieval comparison.
