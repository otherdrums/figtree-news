"""figtree-news CLI.

Thin dispatcher over the library + the news modules. Each command loads the
model + store as needed, calls the matching module, and prints/serves the
result. No news logic leaks into the core library.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

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


def _setup_logging(log_dir: str = "./logs"):
    """Set up logging to both stdout and a timestamped log file."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"figtree_news_{timestamp}.log")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging to {log_file}")
    return log_file


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


def _fts_path(db: str) -> str:
    """FTS index path derived from the LanceDB path (must match the store)."""
    return db.replace(".lance", "_fts.db")


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
        fts_path=_fts_path(db),
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
        fts_path=_fts_path(db),
    )
    typer.echo(json.dumps(stats, indent=2))


@app.command("crawl")
def crawl_cmd(
    db: str = typer.Option("./news.lance"),
    sources: str = typer.Option("./sources.json"),
    feed: list[str] = typer.Option([], "--feed", help="source=url (repeatable)"),
    seed: list[str] = typer.Option([], "--seed", help="seed URL (repeatable)"),
    loop: bool = typer.Option(True, "--loop", help="Run continuously (default)"),
    once: bool = typer.Option(False, "--once", help="Run a single tick and exit"),
    interval: int = typer.Option(0, "--interval", help="Seconds between ticks (0 = no pause)"),
    max_depth: int = typer.Option(1, "--max-depth"),
    max_pages: int = typer.Option(50, "--max-pages"),
    seen_path: str = typer.Option("./seen_urls.json"),
    max_articles: int = typer.Option(40, "--max-articles", help="Cap articles ingested per run"),
    max_stories: int = typer.Option(0, "--max-stories", help="Cap narratives generated (0 = unlimited)"),
    since: str = typer.Option("", "--since", help="Only ingest articles published after this date (YYYY-MM-DD)"),
    before: str = typer.Option("", "--before", help="Only ingest articles published before this date (YYYY-MM-DD)"),
    backfill: bool = typer.Option(False, "--backfill", help="Deep initial crawl (max_articles=200) for empty stores"),
    model_id: str = typer.Option("unsloth/Qwen3-4B-bnb-4bit"),
    compute_kv: bool = False,
    summarize: bool = True,
    no_summaries: bool = False,
):
    """Crawl feeds + seed URLs, then run the trust/lineage/summary pipeline.

    Runs continuously by default (use --once for a single tick). Pair with the
    systemd service so it starts at boot and restarts on failure.
    """
    if once:
        loop = False
    if backfill:
        max_articles = 200
        loop = False
        typer.echo("Backfill mode: deep initial crawl with 200 article cap")
    model, tokenizer, store, registry = _load(model_id, db, sources)
    cfg_feeds, cfg_seeds = _crawl_config(sources)
    feeds = {**cfg_feeds, **_parse_feeds(feed)}
    seeds = list(cfg_seeds) + list(seed)

    crawler = crawler_mod.Crawler(
        model, tokenizer, store, registry,
        seen_path=seen_path, max_depth=max_depth, max_pages=max_pages,
        compute_kv=compute_kv, summarize_images=summarize,
        fts_path=_fts_path(db),
    )

    def tick():
        s = crawler.run_once(feeds, seeds, max_articles=max_articles, since=since, before=before)
        p = pipeline_mod.run_pipeline(
            model, tokenizer, store, do_summaries=not no_summaries, do_brief=True,
            max_stories=max_stories,
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


@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="Search query"),
    db: str = typer.Option("./news.lance"),
    sources: str = typer.Option("./sources.json"),
    model_id: str = typer.Option("unsloth/Qwen3-4B-bnb-4bit"),
    max_results: int = typer.Option(20, "--max", help="Max results per query"),
    pages: int = typer.Option(1, "--pages", help="Number of result pages"),
    time_range: str = typer.Option("", "--time-range", help="day|week|month|year|''"),
    categories: str = typer.Option("news", "--categories", help="SearXNG categories"),
    seen_path: str = typer.Option("./seen_urls.json"),
):
    """Search the web via SearXNG and ingest articles."""
    from .searxng import SearxngConfig
    model, tokenizer, store, registry = _load(model_id, db, sources)
    # Override searxng config from CLI args
    if registry.searxng:
        registry.searxng.enabled = True
        if time_range:
            registry.searxng.time_range = time_range
        if categories:
            registry.searxng.categories = categories
    else:
        registry.searxng = SearxngConfig(
            enabled=True, time_range=time_range,
            categories=categories, max_results=max_results, pages=pages,
        )
    crawler = crawler_mod.Crawler(
        model, tokenizer, store, registry, seen_path=seen_path,
        fts_path=_fts_path(db),
    )
    added = crawler.search_searxng(
        query, categories=categories, time_range=time_range,
        max_results=max_results, pages=pages,
    )
    typer.echo(json.dumps({"query": query, "articles_added": added}, indent=2))


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
    typer.echo(res.get("generated_text", ""))


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
            if a.kind == "article" and a.meta.get("source_id")
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


@app.command("boundary-threshold")
def boundary_threshold_cmd(
    db: str = typer.Option("./news.lance"),
    log: str = typer.Option("", "--log", help="Path to boundary_data.jsonl (default: derived from db)"),
    thresholds: str = typer.Option("0.80,0.85,0.90,0.95,0.98", "--thresholds", help="Comma-separated thresholds to evaluate"),
):
    """Analyze boundary cosine data collected from LLM pair evaluations.

    Reads the JSONL log of boundary comparisons (a1_id, a2_id, boundary_cos,
    llm_verdict) and prints distribution statistics and threshold quality metrics.
    """
    import json
    import statistics
    from collections import Counter

    path = log if log else db.replace(".lance", "_boundary_data.jsonl")
    if not os.path.exists(path):
        typer.echo(f"Boundary data not found at: {path}")
        typer.echo("Run the pipeline with LLM evaluation enabled to collect data.")
        raise typer.Exit(1)

    records = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not records:
        typer.echo("No boundary comparison records found.")
        raise typer.Exit(0)

    same = [r for r in records if r.get("llm_verdict") is True]
    diff = [r for r in records if r.get("llm_verdict") is False]
    unknown = [r for r in records if r.get("llm_verdict") is None]

    typer.echo(f"\n{'='*60}")
    typer.echo("Boundary Comparison Analysis")
    typer.echo(f"{'='*60}")
    typer.echo(f"  Total records:      {len(records)}")
    typer.echo(f"  Same-event (merge): {len(same)}")
    typer.echo(f"  Different-event:    {len(diff)}")
    typer.echo(f"  Unknown:            {len(unknown)}")

    if same:
        same_cos = [r["boundary_cos"] for r in same]
        typer.echo("\n  SAME-EVENT boundary cosine:")
        typer.echo(f"    Mean:   {statistics.mean(same_cos):.4f}")
        typer.echo(f"    Median: {statistics.median(same_cos):.4f}")
        if len(same_cos) > 1:
            typer.echo(f"    Stdev:  {statistics.stdev(same_cos):.4f}")
        typer.echo(f"    Min:    {min(same_cos):.4f}")
        typer.echo(f"    Max:    {max(same_cos):.4f}")

    if diff:
        diff_cos = [r["boundary_cos"] for r in diff]
        typer.echo("\n  DIFFERENT-EVENT boundary cosine:")
        typer.echo(f"    Mean:   {statistics.mean(diff_cos):.4f}")
        typer.echo(f"    Median: {statistics.median(diff_cos):.4f}")
        if len(diff_cos) > 1:
            typer.echo(f"    Stdev:  {statistics.stdev(diff_cos):.4f}")
        typer.echo(f"    Min:    {min(diff_cos):.4f}")
        typer.echo(f"    Max:    {max(diff_cos):.4f}")

    typer.echo(f"\n  {'='*50}")
    typer.echo("  Threshold Quality (if used as boundary fallback)")
    typer.echo(f"  {'='*50}")

    # Source diversity
    source_pairs = Counter()
    for r in records:
        pair = tuple(sorted([r.get("source1","?"), r.get("source2","?")]))
        source_pairs[pair] += 1
    same_source = sum(c for p, c in source_pairs.items() if p[0] == p[1])
    cross_source = sum(c for p, c in source_pairs.items() if p[0] != p[1])
    typer.echo(f"  Same-source pairs:  {same_source}")
    typer.echo(f"  Cross-source pairs: {cross_source}")

    threshold_list = [float(t.strip()) for t in thresholds.split(",") if t.strip()]
    typer.echo(f"\n  {'Threshold':>10s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>10s}  {'TP':>5s}  {'FP':>5s}  {'FN':>5s}")
    typer.echo(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*5}  {'-'*5}  {'-'*5}")
    for t in threshold_list:
        tp = sum(1 for r in same if r["boundary_cos"] >= t)
        fn = len(same) - tp
        fp = sum(1 for r in diff if r["boundary_cos"] >= t)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        typer.echo(f"  {t:>10.2f}  {prec:>10.4f}  {rec:>10.4f}  {f1:>10.4f}  {tp:>5d}  {fp:>5d}  {fn:>5d}")

    typer.echo("\n  Best F1 threshold: ", nl=False)
    best_t, best_f1 = threshold_list[0], 0.0
    for t in threshold_list:
        tp = sum(1 for r in same if r["boundary_cos"] >= t)
        fn = len(same) - tp
        fp = sum(1 for r in diff if r["boundary_cos"] >= t)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    typer.echo(f"threshold={best_t:.2f}  F1={best_f1:.4f}")

    # Also show origin breakdown
    origins = Counter(r.get("origin", "?") for r in records)
    typer.echo("\n  Origin breakdown:")
    for origin, count in origins.most_common():
        typer.echo(f"    {origin:25s}: {count}")


@app.command("serve")
def serve_cmd(
    db: str = typer.Option("./news.lance"),
    sources: str = typer.Option(str(Path(__file__).parent.parent / "demo/sources.json")),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    device: str = typer.Option("auto", "--device", help="auto|cpu|none (none = viewer-only, no model)"),
):
    """Serve the interactive web newspaper (FastAPI).

    The server owns the (GPU) model and runs the crawl + pipeline loop as one
    background task — there is no separate crawler process anymore.
    """
    import uvicorn

    from .web.serve import create_app

    _setup_logging()
    app_instance = create_app(db=db, sources=sources, device=device)
    uvicorn.run(app_instance, host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    app()
