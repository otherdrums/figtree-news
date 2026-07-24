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
├── lineage.py          # Entity-based narrative clustering + frame shift + derivative + role assignment
├── trust.py            # Source trust propagation (accepts all_figs to avoid redundant store.all())
├── decompose.py        # WHO/WHAT/WHERE/WHEN/WHY/HOW extraction (3 background workers)
├── cogitate.py         # Periodic consolidation + insight generation
├── evaluate.py         # External LLM: cluster validation, frame shift, brief review
├── correct.py          # Self-correction: confirmation threshold + auto-apply
├── llm_config.py       # External LLM configuration
├── summarize_news.py   # Per-article summaries + world brief (accepts all_figs)
├── query.py            # Embed query → nearest figments → generate
├── search_index.py     # SQLite FTS5 full-text search
├── export.py           # Graph export as JSON
└── web/
    ├── serve.py        # FastAPI: HTML pages + JSON API + WebSocket + background crawl loop
    ├── templates/      # Jinja2 HTML (index.html with tabbed UI: Top Stories | WHO/WHAT/WHERE/WHY/HOW)
    └── static/         # CSS (dark theme)
```
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

## Key Design Details (v2)

- **Every article = a story**: `lineage.py` no longer filters out single-article clusters.
  Every article becomes a `narrative:{key}` figment, whether solo or multi-source.
- **Role-to-narrative linkage**: After lineage re-computes, `assign_roles_to_narratives()`
  stamps each role figment with `story_id` pointing to its parent narrative, enabling
  the WHO/WHAT/WHERE/WHY/HOW tabbed views.
- **Recency of coverage sorting**: Narratives sorted by `latest_article_date DESC`
  (newest article in the story floats to top, not creation date).
- **Tabbed front page**: Top Stories (default) | WHO | WHAT | WHERE | WHY | HOW.
  WHEN date range selector (today/yesterday/last_week/last_month/last_year/all) filters all tabs.
- **Smart crawl mode**: Forward mode (current feeds) auto-switches to backward mode
  (expanded time range: day → week → month → year → all) when no new articles found
  for 2 consecutive ticks. Toggled in control panel.
- **Find More**: Each story card has a "Find More" button that POSTs to
  `/api/story/{nid}/find-more` to search SearXNG for the story's entities and
  returns matching articles for ingestion.
- **Role API**: `GET /api/roles?role=who&range=last_week` returns entities
  grouped by role text with associated story IDs.
- **SearXNG auto-queries**: No manual query list in config. Each crawl tick extracts
  top keywords from newly ingested RSS article titles/summaries (built-in stopword
  filtering) and uses those as SearXNG queries. Falls back to generic news queries
  if no articles were added. This ensures web coverage mirrors feed topics.
- **Persisted crawler state**: All control panel settings + runtime state (mode,
  consecutive empty ticks, last run timestamp) saved to `sources.json` under
  `"crawler"` key. Survives server restarts; multi-user ready.
- **Responsive slide panel**: Control panel widened to `min(900px, 95vw)` with
  horizontal scroll for long feed URLs. Sticky action bar at top keeps
  Run Once / Start Continuous / Stop buttons always visible.

## Data Flow

```
RSS/SearXNG/Seeds → crawl_feed/search → ingest_articles (figmentize)
    → crawl_seeds → pipeline:
        Phase 1: Trust propagation
        Phase 2: Lineage (entity clustering → narratives; single-article = valid story)
        Phase 2: assign_roles_to_narratives (role figments → story_id)
        Phase 2.5: LLM labeling (if enabled)
        Phase 3-4: LLM eval + correction (if enabled)
        Phase 5: Article summaries
        Phase 6: World brief
        Phase 7: Brief review (if enabled)
        Phase 8: Queue decomposition (background)
```

## Source Configuration

`sources.json` — **single unified config file** containing all settings:
- `"feeds"`: `{source_id: rss_url}` — RSS/Atom feeds
- `"sources"`: `{source_id: {name, base_trust, url, kind, logo_url}}` — source metadata
- `"searxng"`: `{url, enabled, categories, time_range, max_results, pages}` — SearXNG settings
  - **No `queries` field** — queries are auto-derived from RSS article keywords each tick
- `"llm"`: `{url, model, timeout, enabled, ...}` — external LLM config
- `"crawler"`: `{continuous, smart_crawl, interval, max_articles, max_stories, llm_enabled, searxng_enabled, searxng_time_range, searxng_categories, mode, consecutive_empty_ticks, last_run, backward_time_range}` — **persisted crawler state** (survives restarts)
- Unknown domains auto-registered with `base_trust=0.7`

## Common Pitfalls

1. **asyncio in crawler**: `ingest_article()` runs in a thread via `asyncio.to_thread`;
   never call `asyncio.create_task()` from there — use `_pending_decompose` list instead
2. **Numpy truth value**: Never `if not array:` on numpy — use `if array is None:`
3. **Entity clustering**: Uses title, not body text; titles must share 2+ named entities
4. **Qwen3.6 thinking**: LLM puts ALL output in `reasoning_content` unless `enable_thinking: false`
5. **SearXNG**: Requires JSON format enabled in its settings.yml; may need restart
