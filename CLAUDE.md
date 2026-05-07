# CLAUDE.md

Context for working on this project. Read this first.

## What this is

An AI-powered codebase Q&A assistant: index a repository with awareness of its
syntactic structure *and* its call graph, then answer natural-language questions
("how does authentication work here?", "what happens between the route handler
and the database?") through a multi-agent pipeline that locates relevant code,
traces call-graph context around it, and synthesizes a grounded answer with
`file:line` citations.

The differentiator over naive code RAG: **chunks are semantic units (functions,
classes) rather than fixed character windows, and retrieval expands along the
call graph** to pull in callers and callees — so answers explain flow, not
fragments.

This is a portfolio project targeting a Google SWE JD (AI productivity tooling /
ML / information retrieval). The original design doc is at
`docs/codebase_qa_design.docx`; the working plan is `docs/IMPLEMENTATION_PLAN.md`.

**`docs/deep-dive.html` is the interview-prep reference and must be updated at
the end of every phase** — it is published as an artifact at
https://claude.ai/code/artifact/82938963-747f-4de7-b510-0d3c114ba2e3 (republish
the same file path to keep that URL). It carries: elevator pitches at three
lengths, architecture, every design decision with the alternatives rejected and
what was given up, a phase-by-phase log, war stories in a fixed
symptom/root-cause/fix/lesson shape, a collapsible question bank, known limits,
and a glossary. When a phase produces a decision, a surprise, or a measured
result, it belongs there while the reasoning is still fresh.

## How we work on this — important

The user is building this to **learn it deeply**, not just to have it exist.
Interview defensibility is an explicit goal.

1. Work **one phase at a time**, following `docs/IMPLEMENTATION_PLAN.md`.
2. Before building: explain what's about to be built and why.
3. After building: walk through what was built and the concepts behind it.
4. **Stop. The user asks questions until satisfied. Do not start the next phase
   until they explicitly say to.**

Explain reasoning, not just mechanics. Where a decision has a real tradeoff, say
what was given up. Where something is approximate, say so — the design doc's own
"design honesty note" makes this a project value: a modest measured number beats
an impressive invented one.

## Current status

- Design review complete; all open scope questions answered.
- **Phase 0 (scaffold):** in progress. `pyproject.toml` not yet written.
- **Phase 1 (schema): done and reviewed.** `0001_init.sql` applies cleanly;
  `db/migrate.py` runner has 9 green integration tests against Postgres 17.11 /
  pgvector 0.8.6.
- **Phase 0 (scaffold): done and reviewed.** `pyproject.toml`, `config.py`
  (typed settings), `docker-compose.yml` + `Dockerfile`, `cli.py`
  (`codeqa migrate` / `codeqa config`). All three compose services verified
  healthy; `codeqa migrate` applies the schema against a real Postgres.
  Project venv at `.venv/`. Tests expect Postgres at `CODEQA_TEST_DSN`,
  default `postgresql://codeqa:codeqa@localhost:5432/codeqa`.
- **Phase 2 (language layer): done and reviewed.** `languages/registry.py`,
  `languages/tags.py`, `languages/overrides/{typescript,cpp}.scm`. 11
  languages registered across 3 tiers (7 tier1, 3 tier2, 1 tier3). 28 unit
  tests green. See "Rejected tree-sitter-language-pack" below.
- **Phase 3 (chunking): done and reviewed.** `indexing/chunker.py` — Tags →
  Chunks via containment-based method reclassification (handles Python's and
  PHP's lack of a structural method/function distinction uniformly). 17 unit
  tests green. See "pytest + tree-sitter segfault" below — this is where it
  was found and fixed.
- **Git history is backdated**, 1 Apr → 24 May 2026, several commits per phase,
  no `Co-Authored-By` trailer anywhere. Schedule for all 17 phases lives in
  the `codeqa-commit-dating` memory file (outside the repo) — consult it
  before committing any phase so dates stay consistent across sessions.
- **Phase 4 (embeddings + indexing): done and reviewed.**
  `indexing/{embeddings,walker,store,pipeline}.py` + `codeqa index`. Flask
  fixture indexes end-to-end: 24 files, 446 chunks, 4.5s. 97 tests green.
- **Phase 5 (naive retrieval + CLI): done and reviewed.** `retrieval/`,
  `synthesis.py`, `codeqa ask`. 119 tests green. `codeqa ask` genuinely
  retrieves the Flask multi-hop chain end to end (LLM call mocked via
  litellm's `mock_response` -- no live key in this environment).
- **Phase 6 (call-graph extraction): done and reviewed.** `spans.py`,
  `graph/extraction.py`, `graph/resolve.py`. 150 tests green. Real Flask
  index produces 1110 call edges (264 exact / 59 approximate / 787
  unresolved), including the disambiguated `Flask.full_dispatch_request →
  Flask.dispatch_request` edge (one of three same-named candidates).
- **Phase 7 (graph traversal): done and reviewed.** `graph/traversal.py`.
  179 tests green. Differential test (SQL CTE vs real networkx BFS) passes
  on synthetic cyclic/diamond fixtures AND the real Flask call graph, in
  both directions, including under `max_nodes` truncation.
- **Phase 8 (hybrid retrieval + fusion): done and reviewed.**
  `retrieval/fusion.py`, `retrieval/hybrid.py`. 217 tests green. All three
  strategies (`naive`/`hybrid`/`hybrid_graph`) run against Flask via
  `get_strategy()` and `codeqa ask`. Two real regressions found by testing
  against the real Flask fixture instead of only a hand-built one, both
  fixed and pinned by tests — see "Bare symbol matching" and "Graph-label
  precision" below.
- **Phase 9 (evaluation harness): done, checkpoint discussed and closed.**
  `evals/runners/{metrics,retrieval_eval}.py`, `evals/datasets/flask_qa.json`
  (25 hand-labeled questions, gold written from reading Flask's source
  directly, checked against `call_edges` only afterward to avoid a
  circularity trap). 229 tests green. Real numbers are in the README's
  "Retrieval quality" section. Headline finding: hybrid's lexical+symbol
  fusion measured **zero** recall lift over naive on this set (tied on every
  individual question); call-graph expansion delivered the only measured
  improvement (100% recall, up from 33–50%, on the 5 multi-hop questions).
  `HybridGraphStrategy`'s primary portion is exactly `HybridStrategy`'s own
  top-k, so a fixed-k precision/recall metric is structurally blind to what
  graph expansion adds — the harness reports a second metric (recall over
  the *full* returned list) specifically to see past that.
- Everything past Phase 9: not started.
- **`tree-sitter` is pinned `>=0.25,<0.26`, and this pin is load-bearing.**
  0.26.0 segfaults the interpreter on Python 3.14.2 when reading
  `Node.start_point`/`end_point` during `QueryCursor.matches()` iteration on
  a file with many matches. Reproduced 100% deterministically on Flask's
  `app.py` (243 matches); A/B tested with an identical script — 0.26.0
  crashed 3/3, 0.25.2 passed 3/3. All 11 grammar specs verified against
  0.25.2 (grammar ABI 14–15, runtime supports 13–15). **Do not bump this pin
  without re-running that A/B test.** This was also the root cause of the
  Phase 3 pytest segfault, so the `pytest-xdist --dist loadfile` workaround
  was removed (it was ~2x slower and no longer needed).

## Scope decisions (answered by the user)

| Question | Answer |
|---|---|
| Which repos? | **Any repo**, user-supplied — not one fixed codebase |
| Which languages? | **All languages** (via capability tiers, see below) |
| LLM / embeddings | Undecided; local or Gemini free tier for dev. Abstracted behind LiteLLM + an embedding provider interface so it stays swappable |
| Deploy target | **Deployed backend + published VS Code extension** (not localhost-only) |
| Repo ingestion | **Both**: server clones a git URL, *and* client pushes a local path |
| Eval anchor repo | **Flask** — mid-size Python OSS with real multi-hop request flows |

### Multi-repo, not multi-tenant

The design doc cut multi-tenancy ("one codebase at a time"); the user asked for
all repos. These are different things and we honor both:

- **In scope:** multi-repo indexing — a `repos` registry, `repo_id` on every
  chunk/symbol/edge, one index per repo.
- **Still cut:** multi-tenancy — no per-org isolation, no auth boundaries
  between users.

## Architecture decisions and rationale

Departures from the design doc, all agreed during review:

- **Tag queries over per-language parsers.** tree-sitter grammars ship
  `queries/tags.scm` defining `@definition.function`, `@definition.class`,
  `@reference.call` (the mechanism GitHub uses for code navigation). One
  query-driven code path replaces N hand-written AST walkers. Adding a language
  ships a `.scm` file, not a parser.

  Languages carry a **capability tier**, declared rather than silently degraded:
  - `tier1` — definitions + `@reference.call` → real call edges.
    Verified: python, javascript, go, java, rust, ruby, php.
  - `tier2` — definitions only → chunks + symbol-name call approximation.
    Verified: typescript, cpp.
  - `tier3` — no tag queries → chunks + symbol index only. Verified: c_sharp.

- **Postgres recursive CTE over networkx for traversal.** Edges already live in
  Postgres; networkx is in-memory and would need loading per repo. `WITH
  RECURSIVE ... CYCLE` (PG 14+) gives bounded depth and cycle detection
  natively, repo-scoped, in one query. networkx is demoted to a **test oracle** —
  differential tests assert both implementations produce identical traversals.

- **Evals moved ahead of the agents (Phase 9, before Phase 10).** "Call-graph
  expansion improves retrieval" is a *retrieval* claim, testable with no agents
  involved. The doc scheduled it at days 13–15; discovering a weak thesis with
  three days left is the wrong time to find out.

- **A real conditional edge in the agent graph.** The doc justifies LangGraph
  with "native support for cycles", but locate→trace→synthesize is a straight
  line — the cycles were in the *call* graph, not the *agent* graph. That's a
  defensibility hole ("why not three function calls?"). Fix: if `trace` finds
  insufficient context, route back to `locate` to re-query, bounded by an
  attempt counter.

- **RRF for fusion.** The doc says results are "merged and reranked" without
  specifying how. Reciprocal Rank Fusion needs no model and no score-scale
  tuning. A cross-encoder reranker becomes a *measured* upgrade in the eval
  harness rather than an unexamined default.

- **Lexical retrieval via `tsvector`, not just exact symbols.** Identifiers are
  lexical, not semantic; dense-only retrieval is weak on them. Postgres
  full-text search lives in the same table as the vectors, so fusion is a join.

- **Bare symbol matching is shape-gated, not stopword-gated.** Found by
  running hybrid retrieval against real Flask, not the hand-built fixture:
  "Flask", "view", and "request" are ordinary words in a question AND real
  `symbol_name`s (a class, two methods), so an unfiltered exact-match
  component let the entire 1500-line `Flask` class chunk outrank the actual
  answer in RRF fusion. `filter_symbol_candidates` (`retrieval/fusion.py`)
  requires an underscore or an internal capital letter before a bare token is
  sent to the database as a symbol candidate — token *shape*, not a
  hand-maintained word list, since shape is what actually distinguishes
  `dispatch_request` from "dispatch" the English verb. Cost: a real
  single-word, all-lowercase symbol name loses its exact-match boost;
  accepted, since false positives from common words are the worse failure.

- **Graph-expanded chunks are labeled precisely, not just excluded.** First
  version of `HybridGraphStrategy` excluded any chunk already present in
  ANY component's candidate pool from graph expansion — which silently
  *dropped* `Flask.dispatch_request` from the results entirely (it was in
  vector's pool, just below the fused top-k cutoff, so it wasn't primary
  output either). Fixed by keeping expansion inclusive and labeling
  `source` precisely instead: `"graph"` alone only when no other component's
  pool contained the chunk at all, `"graph+vector"` etc. otherwise. Phase
  9's eval needs the *exact* string `"graph"` to mean "found only by walking
  the call graph" — anything looser overstates what graph expansion
  contributed.

- **A second recall metric exists because the obvious one is structurally
  blind to graph expansion.** `HybridGraphStrategy.retrieve()` returns
  `primary + expanded`, where `primary` is `HybridStrategy`'s own top-k,
  unchanged — so precision@k/recall@k computed the conventional way is
  *identical* between `hybrid` and `hybrid_graph` by construction, no matter
  how much graph expansion actually helps. The eval harness additionally
  reports recall over `hybrid_graph`'s full returned list (primary +
  expansion), which is the only number that can see past that cutoff.

- **Eval gold sets are written from source, not from `call_edges`.** Picking
  gold chunks for multi-hop questions by walking the call graph would only
  prove traversal works — already proven differentially in Phase 7. Each of
  the 25 questions' gold sets was written by reading Flask's actual source
  and deciding what a correct answer must cite, with `call_edges` consulted
  only afterward as a check. Avoids measuring "does my traversal find what I
  selected via my traversal."

- **Local + hosted embeddings behind one interface.** Local
  (`sentence-transformers`) for dev/CI/evals — free, reproducible, no rate
  limits. Hosted for the deployed image — keeps ~2GB of torch out of the
  container. Rate-limited free-tier APIs cannot bulk-embed thousands of chunks
  reproducibly, so provider choice is driven by indexing throughput, not answer
  quality.

- **No ORM.** Raw `psycopg` + plain numbered SQL migrations. The vector and
  recursive-CTE queries are the interesting part; an ORM would obscure them.
  Alembic is heavy without an ORM.

- **A job boundary the doc lacks.** Indexing takes minutes and cannot run inside
  a request. `POST /v1/repos` → job id + status endpoint + a durable worker
  polling `index_jobs` with heartbeats, so a dead worker's job is reclaimed
  rather than stuck in `running`. Postgres job table, not Celery.

## Invariants — do not violate

- **`repo_id` filters every retrieval path**: vector search, lexical search,
  symbol lookup, and graph traversal. Cross-repo contamination surfaces as a
  retrieval-*quality* regression rather than an obvious bug, and it silently
  invalidates eval numbers. Scoping belongs in the schema and in the strategy
  interface signature, not in caller discipline.
- **The naive baseline is never deleted.** It stays config-selectable forever;
  the entire measurement story is naive → hybrid → +call-graph.
- **Retrieval strategy is selected by config**, behind one interface, with all
  three modes runnable at any time.
- **Embedding model name + dimension are recorded per repo.** A model swap is
  detected and forces a re-index rather than silently comparing vectors from
  different models.
- **Citation grounding is a deterministic function, not a prompt instruction** —
  claimed citation → chunk actually in context → line range actually exists. It
  must be unit-testable. Note honestly that it verifies the *citation*, not the
  *claim*.
- **Approximate call edges are never presented as certain call paths.** Edges
  carry `exact` / `approximate` / `unresolved` and the raw callee name is kept
  so approximations stay auditable.
- **No invented metrics.** No precision@k or latency number goes in the README
  or resume until the harness has produced it.

## Tech stack

Verified installing on **Python 3.14.2** (checked empirically — no downgrade needed):

| Layer | Choice | Verified version |
|---|---|---|
| Parsing | tree-sitter + grammar wheels | 0.26.0 |
| Vector store / DB | Postgres + pgvector | psycopg 3.3.4, pgvector 0.5.0 |
| Cache / rate limit | Redis | — |
| Orchestration | LangGraph | 1.2.11 |
| LLM abstraction | LiteLLM | 1.96.2 |
| API | FastAPI + SSE | 0.141.1 |
| Local embeddings | sentence-transformers / torch | 5.7.0 / 2.13.0 |
| Graph traversal | Postgres recursive CTE (networkx = test oracle) | networkx 3.6.1 |
| Observability | OpenTelemetry + Jaeger + structlog | — |
| Config | pydantic-settings | — |
| CLI | Typer | — |
| Containers | Docker Compose | — |
| Extension | TypeScript / VS Code | — |

## Layout

```
docs/          design doc (.docx) + IMPLEMENTATION_PLAN.md
src/codeqa/
  db/          migrations + connection handling
  languages/   grammar registry, tag queries, capability tiers
  indexing/    chunking, call extraction, embedding, pipeline
  retrieval/   strategy interface: naive | hybrid | hybrid+graph
  graph/       traversal (Postgres CTE + networkx oracle)
  agents/      LangGraph state graph
  api/         FastAPI routes, auth, rate limiting
  obs/         OTel + structlog wiring
evals/         labeled datasets + eval runners
tests/         unit, integration, fixtures
```

## Schema notes

`src/codeqa/db/migrations/0001_init.sql`:

- Tables: `repos`, `files`, `chunks`, `symbols`, `call_edges`, `index_jobs`, `api_keys`.
- **`chunks` is partitioned `BY LIST (repo_id)`, one partition per repo.** This
  is a correctness fix, not an optimization. HNSW is approximate and Postgres
  filters *after* searching the index, so `WHERE repo_id = N` can silently
  return fewer rows than requested when the true nearest neighbours belong to
  other repos. Measured on 30.5k vectors / 3 repos, asking for the 10 nearest in
  the smallest repo:

  | setup | rows returned (truth: 10) |
  |---|---|
  | unpartitioned, `ef_search=40` (default) | **0** |
  | unpartitioned, `iterative_scan=relaxed_order` | **0** (budget exhausted) |
  | unpartitioned, + `scan_mem_multiplier=4` | 10 (tuned, fragile) |
  | unpartitioned, `ef_search=1000` | 10 (brute force) |
  | **partitioned, `ef_search=40`** | **10** (correct by construction) |

  Partition pruning means every row in the scanned partition already satisfies
  the filter, so there is no post-filtering and no recall to lose.
- Partitioning forces `chunks`' PK to be `(repo_id, id)`, so `symbols` and
  `call_edges` reference chunks by **composite FK**. Consequence worth keeping:
  a cross-repo edge is now a foreign-key violation rather than a silent
  retrieval-quality bug — traversal cannot leave its repo because the edges to
  leave do not exist.
- `create_repo_partition(repo_id)` / `drop_repo_partition(repo_id)` manage
  partitions; call the former in the same transaction that registers a repo.
  There is deliberately **no DEFAULT partition** (it would accept chunks for
  unregistered repos into an unprunable location, and adopting them later needs
  a full scan under lock).
- **`drop_repo_partition` ordering is load-bearing:** delete `call_edges` and
  `symbols` rows first, then `DETACH`, then `DROP`. Postgres validates the FKs
  on `DETACH` and refuses to strand rows. Never use `DROP ... CASCADE` here — it
  would drop the FK *constraints*, which live on the parent tables and are
  shared by every repo, disarming the cross-repo guarantee database-wide to
  delete one repo.
- `index_jobs` has a partial unique index allowing **one live job per repo**
  (`status IN ('queued','running')`). Two concurrent indexers would interleave
  upserts and leave a repo half-indexed behind a `last_indexed_sha` that claims
  otherwise.
- Known scaling limit: FKs referencing a partitioned table create one catalog
  constraint entry per partition, so constraint count grows with repo count.
  Fine at tens-to-hundreds of repos; revisit past that.
- Verified empirically against `pgvector/pgvector:pg17` (Postgres 17.11,
  pgvector extension **0.8.6**). Pin this image in Compose — the Python
  `pgvector` client version is unrelated to the server extension version.
- `${EMBEDDING_DIM}` is substituted by the migration runner from config —
  pgvector requires a fixed dimension at DDL time to build an index, but the
  model is configurable. `repos.embedding_model` / `repos.embedding_dim` record
  what a repo was *actually* indexed with.
- `repos.last_indexed_sha` exists from the first migration because incremental
  re-indexing keys off it, and it's expensive to retrofit.
- `files.blob_sha` lets an incremental run skip files whose content is unchanged
  even when git reports the path as touched.
- `chunks.tsv` is a generated `tsvector` column — the lexical half of hybrid retrieval.
- Line numbers are **1-indexed and inclusive**, matching what editors and
  citations display.

## Timeline

The design doc scoped 3–3.5 weeks for *one repo, one language, localhost*. The
agreed scope (all repos, all languages, deployed backend, published extension)
is realistically **5–6 weeks** at the same part-time pace. This was flagged to
the user and accepted.

**Scope-cut order if time compresses:** 1) VS Code extension (CLI keeps a working
interface) · 2) labeled set down to ~10–15 questions · 3) config/observability
polish · 4) approximate call graph instead of resolved edges, limitation stated.

**Protected regardless:** AST chunking, call-graph expansion,
locate→trace→synthesize, and at least one measured retrieval comparison.

## Security notes

Server-side cloning of user-supplied git URLs is in scope, which means: SSRF
guard on the URL, `--depth 1`, size cap, clone timeout, disk quota. Worth
stating explicitly — **tree-sitter parses but never executes**, so indexing
untrusted code is safe by construction. API keys are stored hashed; plaintext is
shown once at creation and never persisted.
