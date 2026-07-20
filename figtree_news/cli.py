"""figtree-news CLI.

Thin dispatcher over the library + the news modules. Each command loads the
model + store as needed, calls the matching module, and prints/serves the
result. No news logic leaks into the core library.
"""

from __future__ import annotations

import json
import os
import time

import typer

from figtree import FigmentStore, connect, load_model

from .config import SourceRegistry
from . import ingest as ingest_mod
from . import trust as trust_mod
from . import query as query_mod
from . import export as export_mod
from . import eval as eval_mod
from . import crawler as crawler_mod
from . import lineage as lineage_mod
from . import pipeline as pipeline_mod

app = typer.Typer(help="Source-aware news aggregator built on Figtree figments.")


def _load(model_id: str, db: str, sources: str):
    model, tokenizer = load_model(model_id)
    store: FigmentStore = connect(db)
    registry = SourceRegistry.load(sources)
    return model, tokenizer, store, registry


def _load_store(db: str, sources: str):
    """Connect to the store without loading the (GPU) model. CPU-only commands."""
    store: FigmentStore = connect(db)
    registry = SourceRegistry.load(sources)
    return store, registry


def _crawl_config(sources: str):
    """Read feeds/seeds from sources.json if present (top-level keys)."""
    feeds: dict[str, str] = {}
    seeds: list[str] = []
    if sources and os.path.exists(sources):
        try:
            with open(sources, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            feeds = raw.get("feeds", {})
            seeds = raw.get("seeds", [])
        except Exception:
            pass
    return feeds, seeds


def _parse_feeds(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for v in values or []:
        if "=" in v:
            s, u = v.split("=", 1)
            out[s.strip()] = u.strip()
    return out


@app.command("ingest-feed")
def ingest_feed_cmd(
    uri: str = typer.Argument(..., help="RSS/Atom URL or local feed file"),
    source_id: str = typer.Option(..., "--source", help="Source id (e.g. reuters)"),
    db: str = typer.Option("./news.lance"),
    sources: str = typer.Option("./sources.json"),
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


@app.command("crawl")
def crawl_cmd(
    db: str = typer.Option("./news.lance"),
    sources: str = typer.Option("./sources.json"),
    feed: list[str] = typer.Option([], "--feed", help="source=url (repeatable)"),
    seed: list[str] = typer.Option([], "--seed", help="seed URL (repeatable)"),
    loop: bool = typer.Option(False, "--loop", help="Run continuously"),
    interval: int = typer.Option(3600, "--interval", help="Seconds between ticks"),
    max_depth: int = typer.Option(1, "--max-depth"),
    max_pages: int = typer.Option(50, "--max-pages"),
    seen_path: str = typer.Option("./seen_urls.json"),
    model_id: str = typer.Option("unsloth/Qwen3-4B-bnb-4bit"),
    compute_kv: bool = False,
    summarize: bool = True,
    no_summaries: bool = False,
):
    """Crawl feeds + seed URLs, then run the trust/lineage/summary pipeline."""
    model, tokenizer, store, registry = _load(model_id, db, sources)
    cfg_feeds, cfg_seeds = _crawl_config(sources)
    feeds = {**cfg_feeds, **_parse_feeds(feed)}
    seeds = list(cfg_seeds) + list(seed)

    crawler = crawler_mod.Crawler(
        model, tokenizer, store, registry,
        seen_path=seen_path, max_depth=max_depth, max_pages=max_pages,
        compute_kv=compute_kv, summarize_images=summarize,
    )

    def tick():
        s = crawler.run_once(feeds, seeds)
        p = pipeline_mod.run_pipeline(
            model, tokenizer, store, do_summaries=not no_summaries, do_brief=True
        )
        typer.echo(json.dumps({"crawl": s, "pipeline": p}, indent=2))

    if loop:
        typer.echo("Starting continuous crawl. Ctrl-C to stop.")
        while True:
            try:
                tick()
            except Exception as exc:  # pragma: no cover
                typer.echo(f"tick error: {exc}")
            time.sleep(interval)
    else:
        tick()


@app.command("update-trust")
def update_trust_cmd(
    db: str = typer.Option("./news.lance"),
    sources: str = typer.Option("./sources.json"),
    dedupe: bool = False,
):
    """Build edges and persist adjusted per-source trust."""
    store, _ = _load_store(db, sources)
    out = trust_mod.update_trust(store, dedupe=dedupe)
    typer.echo(json.dumps(out["updates"], indent=2))


@app.command("lineage")
def lineage_cmd(
    db: str = typer.Option("./news.lance"),
    sources: str = typer.Option("./sources.json"),
):
    """Compute and persist narrative / derivative lineage (CPU only)."""
    store, _ = _load_store(db, sources)
    out = lineage_mod.compute_lineage(store)
    typer.echo(json.dumps(out, indent=2))


@app.command("show-source-trust")
def show_trust_cmd(
    db: str = typer.Option("./news.lance"),
    sources: str = typer.Option("./sources.json"),
    json_out: bool = False,
):
    """Print the per-source trust report."""
    store, _ = _load_store(db, sources)
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
    store, _ = _load_store(db, sources)
    graph = export_mod.export_graph(store, out)
    typer.echo(f"exported {len(graph['nodes'])} nodes, {len(graph['edges'])} edges -> {out}")


@app.command("build-newspaper")
def build_newspaper_cmd(
    db: str = typer.Option("./news.lance"),
    sources: str = typer.Option("./sources.json"),
    out: str = typer.Option("./newspaper.json"),
):
    """Dump a static front-page snapshot (narratives, agenda, articles) as JSON."""
    store, _ = _load_store(db, sources)
    snap = {
        "narratives": lineage_mod.get_narratives(store),
        "agenda": lineage_mod.source_agenda(store),
        "articles": [
            {
                "id": a.figment_id,
                "title": a.meta.get("title") or a.text[:80],
                "source": a.meta.get("source_id"),
                "url": a.meta.get("url"),
                "summary": a.meta.get("summary", ""),
            }
            for a in store.all()
            if a.meta.get("is_image") and a.meta.get("source_id") and not a.is_edge()
        ],
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(snap, fh, indent=2)
    typer.echo(f"wrote {out}")


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


@app.command("serve")
def serve_cmd(
    db: str = typer.Option("./news.lance"),
    sources: str = typer.Option("./sources.json"),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
):
    """Serve the interactive web newspaper (FastAPI)."""
    import uvicorn

    from .web.serve import create_app

    app_instance = create_app(db=db, sources=sources)
    uvicorn.run(app_instance, host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    app()
