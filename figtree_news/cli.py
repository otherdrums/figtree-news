"""figtree-news CLI.

Thin dispatcher over the library. Each command loads the model + store, calls
the matching module, and prints/JSON-dumps the result. No news logic leaks
into the core library.
"""

from __future__ import annotations

import json

import typer

from figtree import FigmentStore, connect, load_model

from .config import SourceRegistry
from . import ingest as ingest_mod
from . import trust as trust_mod
from . import query as query_mod
from . import export as export_mod
from . import eval as eval_mod

app = typer.Typer(help="Source-aware news aggregator built on Figtree figments.")


def _load(model_id: str, db: str, sources: str):
    model, tokenizer = load_model(model_id)
    store: FigmentStore = connect(db)
    registry = SourceRegistry.load(sources)
    return model, tokenizer, store, registry


@app.command("ingest-feed")
def ingest_feed_cmd(
    uri: str = typer.Argument(..., help="RSS/Atom URL or local feed file"),
    source_id: str = typer.Option(..., "--source", help="Source id (e.g. reuters)"),
    db: str = typer.Option("./news.lance", help="LanceDB store path"),
    sources: str = typer.Option("./sources.json", help="Source registry JSON"),
    model_id: str = typer.Option("unsloth/Qwen3-4B-bnb-4bit"),
    compute_kv: bool = False,
    summarize: bool = False,
):
    """Fetch a feed and ingest every entry as an article."""
    model, tokenizer, store, registry = _load(model_id, db, sources)
    stats = ingest_mod.ingest_feed(
        model, tokenizer, store, registry, source_id, uri,
        compute_kv=compute_kv, summarize_images=summarize,
    )
    typer.echo(json.dumps(stats, indent=2))


@app.command("ingest-file")
def ingest_file_cmd(
    path: str = typer.Argument(..., help="JSON/JSONL file of article dicts"),
    db: str = typer.Option("./news.lance"),
    sources: str = typer.Option("./sources.json"),
    model_id: str = typer.Option("unsloth/Qwen3-4B-bnb-4bit"),
    compute_kv: bool = False,
    summarize: bool = False,
):
    """Ingest articles from a local JSON/JSONL file (no network)."""
    model, tokenizer, store, registry = _load(model_id, db, sources)
    stats = ingest_mod.ingest_file(
        model, tokenizer, store, registry, path,
        compute_kv=compute_kv, summarize_images=summarize,
    )
    typer.echo(json.dumps(stats, indent=2))


@app.command("update-trust")
def update_trust_cmd(
    db: str = typer.Option("./news.lance"),
    sources: str = typer.Option("./sources.json"),
    dedupe: bool = False,
):
    """Build edges and persist adjusted per-source trust."""
    _, _, store, _ = _load("unsloth/Qwen3-4B-bnb-4bit", db, sources)
    out = trust_mod.update_trust(store, dedupe=dedupe)
    typer.echo(json.dumps(out["updates"], indent=2))


@app.command("show-source-trust")
def show_trust_cmd(
    db: str = typer.Option("./news.lance"),
    sources: str = typer.Option("./sources.json"),
    json_out: bool = False,
):
    """Print the per-source trust report."""
    _, _, store, _ = _load("unsloth/Qwen3-4B-bnb-4bit", db, sources)
    rows = trust_mod.show_source_trust(store)
    if json_out:
        typer.echo(json.dumps(rows, indent=2))
        return
    for r in rows:
        typer.echo(
            f"{r['source_id']:12s} trust={r['adjusted_trust']:.2f} "
            f"(base {r['base_trust']:.2f}) contradicted_by={r['contradicting']}"
        )


@app.command("query")
def query_cmd(
    prompt: str = typer.Argument(..., help="Question to answer"),
    db: str = typer.Option("./news.lance"),
    sources: str = typer.Option("./sources.json"),
    model_id: str = typer.Option("unsloth/Qwen3-4B-bnb-4bit"),
    k: int = typer.Option(8),
    min_trust: float = typer.Option(0.0, help="Drop figments below this source trust"),
    faithful: bool = False,
    max_new_tokens: int = typer.Option(200),
):
    """Retrieve trusted figments and generate an answer."""
    model, tokenizer, store, _ = _load(model_id, db, sources)
    res = query_mod.query(
        model, tokenizer, store, prompt,
        k=k, min_trust=min_trust, faithful=faithful, max_new_tokens=max_new_tokens,
    )
    typer.echo(res.get("text", ""))


@app.command("export-graph")
def export_cmd(
    db: str = typer.Option("./news.lance"),
    sources: str = typer.Option("./sources.json"),
    out: str = typer.Option("./graph.json"),
):
    """Export the figment graph as JSON (nodes + edges)."""
    _, _, store, _ = _load("unsloth/Qwen3-4B-bnb-4bit", db, sources)
    graph = export_mod.export_graph(store, out)
    typer.echo(f"exported {len(graph['nodes'])} nodes, {len(graph['edges'])} edges -> {out}")


@app.command("eval")
def eval_cmd(
    db: str = typer.Option("./news.lance"),
    sources: str = typer.Option("./sources.json"),
    model_id: str = typer.Option("unsloth/Qwen3-4B-bnb-4bit"),
    source_id: str | None = typer.Option(None),
    out: str = typer.Option("./eval_report.json"),
    max_new_tokens: int = typer.Option(400),
):
    """Evaluate per-source recall + trust/contradiction state."""
    model, tokenizer, store, _ = _load(model_id, db, sources)
    report = eval_mod.evaluate(
        model, tokenizer, store, source_id=source_id, max_new_tokens=max_new_tokens
    )
    eval_mod.write_report(report, out)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":  # pragma: no cover
    app()
