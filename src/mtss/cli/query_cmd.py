"""Query and search CLI commands."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.table import Table

from ._common import console


def register(app: typer.Typer):
    """Register query and search commands on the app."""

    @app.command()
    def query(
        question: str = typer.Argument(..., help="Question to ask"),
        top_k: int = typer.Option(20, "--top-k", "-k", help="Candidates for reranking"),
        threshold: float = typer.Option(0.5, "--threshold", "-t", help="Similarity threshold"),
        rerank_top_n: int = typer.Option(5, "--rerank-top-n", "-n", help="Final results after reranking"),
        no_rerank: bool = typer.Option(False, "--no-rerank", help="Disable reranking"),
    ):
        """Query the RAG system with a question.

        Uses two-stage retrieval: vector search + reranking for 20-35% better accuracy.
        """
        asyncio.run(_query(question, top_k, threshold, rerank_top_n, not no_rerank))

    @app.command()
    def search(
        query_text: str = typer.Argument(..., help="Search query"),
        top_k: int = typer.Option(20, "--top-k", "-k", help="Candidates for reranking"),
        threshold: float = typer.Option(0.5, "--threshold", "-t", help="Similarity threshold"),
        rerank_top_n: int = typer.Option(10, "--rerank-top-n", "-n", help="Final results after reranking"),
        no_rerank: bool = typer.Option(False, "--no-rerank", help="Disable reranking"),
    ):
        """Search for relevant documents without generating an answer.

        Uses two-stage retrieval: vector search + reranking for better results.
        """
        asyncio.run(_search(query_text, top_k, threshold, rerank_top_n, not no_rerank))


async def _query(
    question: str,
    top_k: int,
    threshold: float,
    rerank_top_n: int,
    use_rerank: bool,
):
    """Async implementation of query command."""
    from ..rag.query_engine import RAGQueryEngine

    engine = RAGQueryEngine()

    try:
        status_msg = "Searching, reranking, and generating answer..." if use_rerank else "Searching and generating answer..."
        with console.status(status_msg):
            response = await engine.query(
                question=question,
                top_k=top_k,
                similarity_threshold=threshold,
                rerank_top_n=rerank_top_n,
                use_rerank=use_rerank,
            )

        console.print(response.answer)

    finally:
        await engine.close()


async def _search(
    query_text: str,
    top_k: int,
    threshold: float,
    rerank_top_n: int,
    use_rerank: bool,
):
    """Async implementation of search command."""
    from ..rag.query_engine import RAGQueryEngine

    engine = RAGQueryEngine()

    try:
        status_msg = "Searching and reranking..." if use_rerank else "Searching..."
        with console.status(status_msg):
            sources = await engine.search_only(
                question=query_text,
                top_k=top_k,
                similarity_threshold=threshold,
                rerank_top_n=rerank_top_n,
                use_rerank=use_rerank,
            )

        if not sources:
            console.print("[yellow]No results found[/yellow]")
            return

        table = Table(title=f"Search Results for: {query_text}")
        table.add_column("#", style="dim")
        table.add_column("File", style="cyan")
        table.add_column("Subject", style="green")
        table.add_column("Relevance", justify="right")
        table.add_column("Preview", max_width=50)

        for i, source in enumerate(sources, 1):
            # Show rerank score if available, otherwise similarity score
            relevance = source.rerank_score if source.rerank_score is not None else source.score
            relevance_label = f"{relevance:.1%}" + (" \u2713" if source.rerank_score is not None else "")
            file_name = Path(source.file_path).name if source.file_path else "-"
            table.add_row(
                str(i),
                file_name,
                source.email_subject or "-",
                relevance_label,
                source.text[:100] + "..." if source.text else "-",
            )

        console.print(table)

    finally:
        await engine.close()
