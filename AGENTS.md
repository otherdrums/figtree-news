# Figtree-News

> **EXPERIMENTAL** — Rapid iteration, breaking changes, incomplete features.

Source-aware news aggregator built on Figtree figments. Articles are decomposed
into WHO/WHAT/WHERE/WHEN/WHY/HOW role figments that are reused across narratives,
enabling structured search, trust-aware reasoning, and self-correcting LLM eval.

## Build / Test

```bash
pip install -e .                    # install (from figtree-news/ root)
python3 -m pytest tests/ -v         # run all tests (CPU-only, no GPU needed)
python3 -m pytest tests/test_web.py -v  # specific test file
```

## Architecture

```
figtree_news/
├── cli.py              # Typer CLI: crawl, serve, search, query, lineage, export, eval
├── config.py           # SourceRegistry + SearxngConfig + LlmConfig
├── searxng.py          # SearXNG client: search → article dicts with image/video
├── ingest.py           # Feed/article → figments with provenance
├── crawler.py          # RSS + SearXNG + BFS link-follower (thread-safe pending queue)
├── pipeline.py         # 8-phase orchestration: trust → lineage → eval → summaries → brief
├── lineage.py          # Entity-based narrative clustering + frame shift + derivative edges
├── trust.py            # Source trust propagation
├── decompose.py        # WHO/WHAT/WHERE/WHEN/WHY/HOW extraction (3 background workers)
├── cogitate.py         # Periodic consolidation + insight generation
├── evaluate.py         # External LLM: cluster validation, frame shift, brief review
├── correct.py          # Self-correction: confirmation threshold + auto-apply
├── llm_config.py       # External LLM configuration
├── summarize_news.py   # Per-article summaries + world brief
├── query.py            # Embed query → nearest figments → generate
├── search_index.py     # SQLite FTS5 full-text search
├── eval.py             # Per-source faithful-recall eval
├── export.py           # Graph export as JSON
└── web/
    ├── serve.py        # FastAPI: HTML pages + JSON API + WebSocket + background crawl loop
    ├── templates/      # Jinja2 HTML
    └── static/         # CSS (dark theme)
```

## Key Commands

```bash
# Serve web UI + background crawler
figtree-news serve --db demo/news.lance --sources demo/sources.json --host 0.0.0.0 --port 8000

# Standalone crawl
figtree-news crawl --interval 0 --max-articles 40

# SearXNG search
figtree-news search "AI regulation" --time-range day --max 10

# Run pipeline only (trust + lineage + summaries + brief)
# Triggered via web UI or: POST /api/pipeline/run
```

## Key Design Details

- **CLI entry point**: `figtree-news` → `figtree_news.cli:app`
- **Server**: `serve` starts FastAPI (port 8000) + background crawler loop
- **External LLM**: Qwen3.6-35B at configurable URL
  - **Critical**: Must pass `chat_template_kwargs: {"enable_thinking": false}` in API payload
  - `chat_json()` strips `<think>` tags and reads `reasoning_content` as fallback
  - Context size: 32K tokens minimum (cluster eval sends up to ~13K tokens for 20 articles)
- **SearXNG**: Runs in the web crawl tick (not just CLI) with independent budget
- **Entity extraction**: `_article_entities()` uses **title** (not body) for clustering
- **Jaccard threshold**: 0.30 with >= 2 shared entities required for clustering
- **Crawler thread safety**: `ingest_article()` appends to `_pending_decompose` list (thread-safe);
  async caller drains it via `drain_pending_decompose()` after `to_thread` returns
- **Tests**: CPU-only, use `tmp_path` for isolation

## Data Flow

```
RSS/SearXNG/Seeds → crawl_feed/search → ingest_articles (figmentize)
    → crawl_seeds → pipeline:
        Phase 1: Trust propagation
        Phase 2: Lineage (entity clustering → narratives)
        Phase 2.5: LLM labeling (if enabled)
        Phase 3-4: LLM eval + correction (if enabled)
        Phase 5: Article summaries
        Phase 6: World brief
        Phase 7: Brief review (if enabled)
        Phase 8: Queue decomposition (background)
```

## Source Configuration

`sources.json`:
- `"feeds"`: `{source_id: rss_url}` — RSS/Atom feeds
- `"sources"`: `{source_id: {name, base_trust, url, kind, logo_url}}` — source metadata
- `"searxng"`: `{url, enabled, queries, categories, time_range, max_results, pages}`
- `"llm"`: `{url, model, timeout, enabled, ...}` — external LLM config
- Unknown domains auto-registered with `base_trust=0.7`

## Common Pitfalls

1. **asyncio in crawler**: `ingest_article()` runs in a thread via `asyncio.to_thread`;
   never call `asyncio.create_task()` from there — use `_pending_decompose` list instead
2. **Numpy truth value**: Never `if not array:` on numpy — use `if array is None:`
3. **Entity clustering**: Uses title, not body text; titles must share 2+ named entities
4. **Qwen3.6 thinking**: LLM puts ALL output in `reasoning_content` unless `enable_thinking: false`
5. **SearXNG**: Requires JSON format enabled in its settings.yml; may need restart
