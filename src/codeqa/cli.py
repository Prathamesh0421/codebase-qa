"""Command-line entry point.

Currently just the migration command; `ask` and `index` arrive with the
retrieval and ingestion phases. The CLI is the reliable interface floor for
this project -- the VS Code extension is a thin client over the same API, and
if it is ever de-scoped this still works.
"""

import typer
from rich.console import Console

from codeqa.config import get_settings
from codeqa.db import migrate as migrations

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
def config() -> None:
    """Print resolved configuration, with secrets redacted."""
    settings = get_settings()
    redacted = {"llm_api_key", "database_url", "redis_url"}

    for name, value in settings.model_dump().items():
        shown = "[dim]<set>[/]" if name in redacted and value else value
        console.print(f"  [cyan]{name}[/] = {shown}")


if __name__ == "__main__":
    app()
