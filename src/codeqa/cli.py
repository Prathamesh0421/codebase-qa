"""Command-line entry point.

`ask` arrives with the retrieval phase. The CLI is the reliable interface
floor for this project -- the VS Code extension is a thin client over the
same API, and if it is ever de-scoped this still works.
"""

from pathlib import Path

import psycopg
import typer
from pgvector.psycopg import register_vector
from rich.console import Console

from codeqa.config import get_settings
from codeqa.db import migrate as migrations
from codeqa.indexing.embeddings import build_embedder
from codeqa.indexing.pipeline import index_repo
from codeqa.indexing.store import RepoAlreadyExists, register_repo

app = typer.Typer(
    name="codeqa",
    help="Question answering over source code with call-graph context.",
    no_args_is_help=True,
)
console = Console()


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
        settings.llm_api_key,
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
        elif reindex:
            repo_id = row[0]
            console.print(f"Re-indexing [cyan]{slug}[/] (repo_id={repo_id})")
        else:
            console.print(
                f"[yellow]{slug} is already registered.[/] "
                f"Pass --reindex to index it again."
            )
            raise typer.Exit(1)

        stats = index_repo(conn, repo_id, path, embedder)
    except RepoAlreadyExists as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(1) from exc
    finally:
        conn.close()

    console.print(
        f"\n  [green]{stats.files_indexed}[/] files, "
        f"[green]{stats.chunks_created}[/] chunks in {stats.duration_seconds:.1f}s"
    )
    if stats.files_skipped_no_language:
        console.print(f"  [dim]{stats.files_skipped_no_language} skipped (no language)[/]")
    if stats.files_failed:
        console.print(f"  [yellow]{stats.files_failed} failed[/]")
        for file_path, error in stats.errors[:5]:
            console.print(f"    [dim]{file_path}: {error}[/]")


@app.command()
def config() -> None:
    """Print resolved configuration, with secrets redacted."""
    settings = get_settings()
    redacted = {"llm_api_key", "database_url", "redis_url"}

    for name, value in settings.model_dump().items():
        shown = "[dim]<set>[/]" if name in redacted and value else value
        console.print(f"  [cyan]{name}[/] = {shown}")


if __name__ == "__main__":
    app()
