"""Command-line entry point.

The CLI is the reliable interface floor for this project -- the VS Code
extension is a thin client over the same API, and if it is ever de-scoped
this still works.
"""

from pathlib import Path

import psycopg
import typer
from pgvector.psycopg import register_vector
from rich.console import Console

from codeqa.agents.graph import build_agent_graph
from codeqa.agents.logic import sort_for_display
from codeqa.agents.state import AgentState
from codeqa.api.auth import create_api_key
from codeqa.config import get_settings
from codeqa.db import migrate as migrations
from codeqa.grounding import ground_answer
from codeqa.indexing.embeddings import build_embedder
from codeqa.indexing.incremental import incremental_index_repo
from codeqa.indexing.pipeline import index_repo
from codeqa.indexing.store import RepoAlreadyExists, register_repo
from codeqa.indexing.worker import run_worker
from codeqa.retrieval.strategy import get_strategy
from codeqa.synthesis import synthesize

app = typer.Typer(
    name="codeqa",
    help="Question answering over source code with call-graph context.",
    no_args_is_help=True,
)
console = Console()

keys_app = typer.Typer(help="Manage API keys for the HTTP API.")
app.add_typer(keys_app, name="keys")


@app.command()
def migrate(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="List pending migrations without applying them."
    ),
) -> None:
    """Apply pending database migrations."""
    settings = get_settings()

    try:
        applied = migrations.migrate(
            settings.dsn, settings.embedding_dim, dry_run=dry_run
        )
    except migrations.MigrationDriftError as exc:
        # Drift means the files and the deployed schema disagree. Continuing
        # would run code against a schema it was not written for.
        console.print(f"[bold red]Migration drift[/] {exc}")
        raise typer.Exit(1) from exc
    except migrations.EmbeddingDimMismatch as exc:
        console.print(f"[bold red]Dimension mismatch[/] {exc}")
        raise typer.Exit(1) from exc

    if not applied:
        console.print("[green]Up to date[/] — no pending migrations.")
        return

    verb = "Pending" if dry_run else "Applied"
    for migration in applied:
        console.print(f"  [cyan]{verb}[/] {migration.label}")


@app.command()
def index(
    path: Path = typer.Argument(..., help="Path to a local repository to index."),  # noqa: B008
    slug: str = typer.Option(  # noqa: B008
        None, "--slug", help="Repo identifier. Defaults to the directory name."
    ),
    reindex: bool = typer.Option(  # noqa: B008
        False, "--reindex", help="Re-index a repo that is already registered."
    ),
) -> None:
    """Index a local repository: walk, chunk, embed, store."""
    settings = get_settings()
    path = path.resolve()
    if not path.is_dir():
        console.print(f"[bold red]Not a directory[/] {path}")
        raise typer.Exit(1)

    slug = slug or path.name

    embedder = build_embedder(
        settings.embedding_provider,
        settings.embedding_model,
        settings.embedding_dim,
        settings.embedding_batch_size,
        settings.embedding_api_key,
    )

    conn = psycopg.connect(settings.dsn)
    register_vector(conn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM repos WHERE slug = %s", (slug,))
            row = cur.fetchone()

        if row is None:
            repo_id = register_repo(
                conn, slug, path.name, "local_path", str(path),
                settings.embedding_model, settings.embedding_dim,
            )
            console.print(f"Registered [cyan]{slug}[/] (repo_id={repo_id})")
            stats = index_repo(conn, repo_id, path, embedder)
        elif reindex:
            repo_id = row[0]
            console.print(f"Re-indexing [cyan]{slug}[/] (repo_id={repo_id})")
            # incremental_index_repo degrades correctly to "index everything
            # as new" when a repo has no prior files rows (a first index
            # that never got past registration, say) -- always safe to use
            # here, not just an optimization for the common case.
            stats = incremental_index_repo(conn, repo_id, path, embedder)
        else:
            console.print(
                f"[yellow]{slug} is already registered.[/] "
                f"Pass --reindex to index it again."
            )
            raise typer.Exit(1)
    except RepoAlreadyExists as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(1) from exc
    finally:
        conn.close()

    console.print(
        f"\n  [green]{stats.files_indexed}[/] files, "
        f"[green]{stats.chunks_created}[/] chunks in {stats.duration_seconds:.1f}s"
    )
    total_edges = (
        stats.call_edges_exact + stats.call_edges_approximate + stats.call_edges_unresolved
    )
    if total_edges:
        console.print(
            f"  [green]{total_edges}[/] call edges: "
            f"[green]{stats.call_edges_exact}[/] exact, "
            f"[cyan]{stats.call_edges_approximate}[/] approximate, "
            f"[dim]{stats.call_edges_unresolved}[/] unresolved"
        )
    if stats.files_skipped_no_language:
        console.print(f"  [dim]{stats.files_skipped_no_language} skipped (no language)[/]")
    if stats.files_failed:
        console.print(f"  [yellow]{stats.files_failed} failed[/]")
        for file_path, error in stats.errors[:5]:
            console.print(f"    [dim]{file_path}: {error}[/]")
    # Only ever nonzero from incremental_index_repo -- the full pipeline has
    # no concept of "unchanged".
    if stats.files_unchanged or stats.files_removed or stats.chunks_preserved:
        console.print(
            f"  [dim]{stats.files_unchanged} files unchanged, "
            f"{stats.files_removed} removed, "
            f"{stats.chunks_preserved} chunks preserved[/]"
        )


@app.command()
def ask(
    question: str,
    slug: str = typer.Option(..., "--repo", help="Slug of an indexed repository."),  # noqa: B008
    top_k: int = typer.Option(None, "--top-k", help="Override settings.retrieval_top_k."),  # noqa: B008
    agent: bool = typer.Option(  # noqa: B008
        False,
        "--agent",
        help="Use the locate->trace->synthesize pipeline (Phase 10) instead of one-shot retrieval.",
    ),
) -> None:
    """Ask a question about an indexed repository and stream the answer."""
    settings = get_settings()
    top_k = top_k or settings.retrieval_top_k

    conn = psycopg.connect(settings.dsn)
    register_vector(conn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, embedding_model, embedding_dim FROM repos WHERE slug = %s", (slug,)
            )
            row = cur.fetchone()
        if row is None:
            console.print(f"[bold red]No repo registered with slug[/] {slug!r}")
            raise typer.Exit(1)
        repo_id, embedding_model, embedding_dim = row

        # The embedder used to embed the QUESTION must match what the repo's
        # chunks were embedded with -- same reasoning as
        # check_embedder_matches_repo in indexing/store.py, applied at query
        # time instead of index time. A query vector from a different model
        # is not comparable to the stored vectors at all.
        embedder = build_embedder(
            settings.embedding_provider, embedding_model, embedding_dim,
            settings.embedding_batch_size, settings.embedding_api_key,
        )
        strategy = get_strategy(
            settings.retrieval_strategy,
            graph_max_depth=settings.graph_max_depth,
            graph_max_nodes=settings.graph_max_nodes,
        )

        if agent:
            _ask_agent(conn, embedder, strategy, repo_id, question, top_k, settings)
        else:
            _ask_direct(conn, embedder, strategy, repo_id, question, top_k, settings)
    finally:
        conn.close()


def _print_grounding_warning(result) -> None:
    """Phase 11: nothing here can un-print tokens the terminal already
    showed live, so this flags what grounding found instead of silently
    editing scrollback. A non-streaming caller would show result.text
    directly and never render the raw fake citation at all -- see
    grounding.py.
    """
    if not result.dropped:
        return
    raw = ", ".join(c.raw for c in result.dropped)
    console.print(
        f"\n[yellow]{len(result.dropped)} citation(s) could not be verified "
        f"against retrieved context: {raw}[/]"
    )


def _ask_direct(
    conn, embedder, strategy, repo_id: int, question: str, top_k: int, settings
) -> None:
    """The Phase 5 path: one retrieval call, one synthesize call. Left
    unchanged by Phase 10's agent pipeline -- --agent is opt-in, not a
    replacement, the same "naive baseline never deleted" reasoning applied
    to the simpler of the two CLI paths.
    """
    chunks = strategy.retrieve(conn, repo_id, question, embedder, top_k)
    if not chunks:
        console.print("[yellow]No relevant chunks found.[/]")
        raise typer.Exit(1)

    # markup=False on the streamed tokens: rich's markup parser treats
    # "[...]" as a tag, and an LLM answering questions about code will
    # routinely emit real bracket syntax (list[int], chunks[0]) that must
    # never be silently stripped from what the user is shown. Verified this
    # is a real failure, not a theoretical one: "List[str]" prints as "List"
    # with markup parsing left on.
    console.print()  # never glue onto whatever printed before this (progress bars, warnings)
    tokens = []
    for token in synthesize(
        question,
        chunks,
        settings.llm_model,
        settings.llm_api_key,
        max_retries=settings.llm_max_retries,
    ):
        console.print(token, end="", markup=False, highlight=False)
        tokens.append(token)
    console.print()

    # Grounding runs after the tokens have already streamed to the terminal
    # -- it can flag an unverifiable citation, but it can't un-print
    # something already shown live. A non-streaming caller (a future batch
    # endpoint) would show result.text directly instead, which never
    # contains the raw fake citation at all.
    _print_grounding_warning(ground_answer("".join(tokens), chunks))

    console.print("\n[dim]Retrieved:[/]")
    for c in chunks:
        label = c.qualified_name or c.symbol_name
        console.print(f"  [dim]{c.score:.3f}[/]  {c.citation}  [cyan]{label}[/]")


def _ask_agent(conn, embedder, strategy, repo_id: int, question: str, top_k: int, settings) -> None:
    """The Phase 10 path: locate -> trace -> synthesize, with trace able to
    route back to locate on insufficient context. stream_mode=["updates",
    "custom"] carries both per-node progress (so the retry, if it fires, is
    visible) and the synthesize node's token-by-token output in one pass.
    """
    graph = build_agent_graph(
        conn,
        embedder,
        strategy,
        top_k,
        settings.llm_model,
        settings.llm_api_key,
        llm_max_retries=settings.llm_max_retries,
    )
    state = AgentState(
        repo_id=repo_id,
        question=question,
        current_query=question,
        max_attempts=settings.agent_max_attempts,
    )

    console.print()
    final_chunks: list = []
    final_answer = ""
    for mode, payload in graph.stream(state, stream_mode=["updates", "custom"]):
        if mode == "custom":
            # Same markup hazard as the direct path -- an LLM's own bracket
            # syntax must never be parsed as a rich markup tag.
            console.print(payload, end="", markup=False, highlight=False)
            continue
        for node_name, update in payload.items():
            if node_name == "locate":
                final_chunks = update["chunks"]
                console.print(
                    f"[dim]locate attempt {update['attempt']}: "
                    f"{len(update['chunks'])} chunks so far[/]"
                )
            elif node_name == "trace":
                verdict = "sufficient" if update["sufficient"] else "insufficient"
                console.print(f"[dim]trace: {verdict}[/]")
                if not update["sufficient"]:
                    console.print(f"[dim]  refining search: {update['current_query']}[/]")
            elif node_name == "synthesize":
                final_answer = update["answer"]
    console.print()

    # Same "flag, can't un-print" reasoning as the direct path -- the
    # synthesize node already streamed these tokens live via the custom
    # writer channel before this point.
    if final_answer:
        _print_grounding_warning(ground_answer(final_answer, final_chunks))

    if final_chunks:
        console.print("\n[dim]Retrieved:[/]")
        # Chunks accumulated across possibly-multiple locate attempts are in
        # attempt order, not score order -- sort explicitly rather than
        # printing a list where a later attempt's real scores can appear
        # after an earlier attempt's 0.0 graph-expansion sentinel.
        for c in sort_for_display(final_chunks):
            label = c.qualified_name or c.symbol_name
            console.print(f"  [dim]{c.score:.3f}[/]  {c.citation}  [cyan]{label}[/]")


@app.command()
def config() -> None:
    """Print resolved configuration, with secrets redacted."""
    settings = get_settings()
    redacted = {"llm_api_key", "database_url", "redis_url"}

    for name, value in settings.model_dump().items():
        shown = "[dim]<set>[/]" if name in redacted and value else value
        console.print(f"  [cyan]{name}[/] = {shown}")


@app.command()
def worker(
    once: bool = typer.Option(  # noqa: B008
        False, "--once", help="Process at most one job, then exit, instead of polling forever."
    ),
) -> None:
    """Run the durable indexing worker (Phase 12): claim queued jobs from
    index_jobs, clone git_url repos safely, and index them. Meant to run as
    a long-lived process (or a supervised one-shot with --once) alongside
    the API, not from an interactive session.
    """
    settings = get_settings()
    conn = psycopg.connect(settings.dsn)
    register_vector(conn)
    try:
        console.print("[dim]worker started[/]" + (" (--once)" if once else ""))
        run_worker(conn, settings, once=once)
    finally:
        conn.close()


@keys_app.command("create")
def keys_create(
    name: str = typer.Option(..., "--name", help="A label for this key, e.g. 'ci' or 'vscode'."),  # noqa: B008
    rate_limit_rpm: int = typer.Option(  # noqa: B008
        None, "--rate-limit-rpm", help="Override settings.rate_limit_rpm for this key."
    ),
) -> None:
    """Issue a new API key. The plaintext is shown exactly once, here --
    it is never stored and cannot be recovered later, only revoked and
    replaced with a new key.
    """
    settings = get_settings()
    conn = psycopg.connect(settings.dsn)
    try:
        created = create_api_key(conn, name, rate_limit_rpm or settings.rate_limit_rpm)
    finally:
        conn.close()

    console.print(f"[green]API key created:[/] {created.name} (id={created.id})")
    console.print(f"  [bold]{created.plaintext}[/]")
    console.print(
        "[yellow]This is shown once and cannot be retrieved again -- store it now.[/]"
    )


if __name__ == "__main__":
    app()
