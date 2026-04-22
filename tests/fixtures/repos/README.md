# Test fixture repositories

Vendored third-party source, used as realistic input for the indexing
pipeline. This is **input data, not project code** — it is excluded from ruff
(see `pyproject.toml`) because linting or reformatting it would both be wrong
and would destroy its value as a representative sample of a real codebase.

## `flask/`

- **Source:** https://github.com/pallets/flask — `src/flask/` only
- **License:** BSD-3-Clause, © Pallets. See the project's own LICENSE.txt.
- **Pruned to:** the `src/flask` package (no `.git`, tests, docs, or tooling).
  392K, 26 files.

Chosen as the anchor repo because it contains genuine multi-hop request flows
(`route` → `add_url_rule` → `full_dispatch_request` → `dispatch_request` →
view), which is exactly the structure naive RAG fails on and call-graph
expansion should win on. It is also the Phase 9 evaluation anchor, so the same
fixture backs both the end-to-end indexing test and the labeled Q&A set.

A useful accident worth preserving: Flask defines `dispatch_request` three
times, on `Flask`, `View` and `MethodView`. That ambiguity is why chunks carry
`qualified_name` and not just `symbol_name`, and it is asserted in
`tests/integration/test_pipeline.py`.
