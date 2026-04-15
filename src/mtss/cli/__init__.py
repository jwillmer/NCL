"""CLI entry point for MTSS - Email RAG Pipeline."""

from __future__ import annotations

import nest_asyncio

# Apply nest_asyncio to allow nested event loops (required for LlamaParse)
nest_asyncio.apply()

# Disable Langfuse tracing for CLI operations (only used for API)
# This prevents "[non-fatal] Tracing: server error 503" messages during ingest
import litellm

litellm.success_callback = [cb for cb in litellm.success_callback if "langfuse" not in cb]
litellm.failure_callback = [cb for cb in litellm.failure_callback if "langfuse" not in cb]

import typer

# Create the main app and sub-apps
app = typer.Typer(
    name="MTSS",
    help="MTSS - Email RAG Pipeline for processing EML files with attachments",
)
vessels_app = typer.Typer(help="Vessel registry management")
app.add_typer(vessels_app, name="vessels")
topics_app = typer.Typer(help="Topic management for categorization and filtering")
app.add_typer(topics_app, name="topics")

# Register commands from each module
from . import admin_cmd, entities_cmd, import_cmd, ingest_cmd, maintenance_cmd, query_cmd

ingest_cmd.register(app)
import_cmd.register(app)
query_cmd.register(app)
admin_cmd.register(app)
maintenance_cmd.register(app)
entities_cmd.register(app, vessels_app, topics_app)

if __name__ == "__main__":
    app()
