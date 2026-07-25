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
├── config.py           # SourceRegistry + CrawlerState + SourceConfig dataclasses
├── searxng.py          # SearXNG client: search → article dicts with image/video
├── ingest.py           # Feed/article → figments with provenance
├── crawler.py          # RSS + SearXNG + BFS link-follower (thread-safe ingestion)
├── pipeline.py         # Parallel pipeline orchestration (ThreadPoolExecutor)
├── lineage.py          # Role figment clustering + frame shift + derivatives (expands through associations)
├── trust.py            # Source trust propagation
├── decompose.py        # WHO/WHAT/WHERE/WHEN/WHY/HOW extraction + inline cogitation + auto-associations
├── cogitate.py         # Periodic insight generation
├── evaluate.py         # External LLM: cluster validation, frame shift, brief review
├── correct.py          # Self-correction: confirmation threshold + auto-apply
├── llm_config.py       # External LLM configuration
├── summarize_news.py   # Per-article summaries + world brief
├── query.py            # Embed query → nearest figments → generate
├── search_index.py     # SQLite FTS5 full-text search (thread-safe)
├── associations.py     # Co-reference layer: association figments for surface-form variants
├── intersection.py     # Multi-role intersection query: find narratives containing all specified roles
├── context.py          # Context materialization: assemble structured context packages from narratives
├── export.py           # Graph export as JSON
├── eval.py             # Per-source faithful-recall eval
└── web/
├── evaluate.py         # External LLM: cluster validation, frame shift, brief review
├── correct.py          # Self-correction: confirmation threshold + auto-apply
├── llm_config.py       # External LLM configuration
├── summarize_news.py   # Per-article summaries + world brief
├── query.py            # Embed query → nearest figments → generate
├── search_index.py     # SQLite FTS5 full-text search
├── eval.py             # Per-source faithful-recall eval
├── export.py           # Graph export as JSON
└── web/
    ├── serve.py        # FastAPI + auto-crawl + parallel crawl tick
    ├── templates/      # Jinja2 HTML (base.html, index.html, article.html, etc.)
    └── static/
        ├── style.css   # Dark theme CSS
        └── app.js      # Client-side JS (WebSocket, crawl control, stats)
```

## Key Commands

```bash
# Serve web UI + auto-starts continuous crawl
figtree-news serve --db demo/news.lance --sources demo/sources.json --host 0.0.0.0 --port 8000

# Standalone crawl
figtree-news crawl --interval 300 --max-articles 15

# SearXNG search
figtree-news search "AI regulation" --time-range day --max 10

# Run pipeline only
# Triggered via web UI or: POST /api/pipeline/run
```

## Execution Model

### Crawl Tick (serve.py `_run_crawl_tick`)
```
Phase 1: Feeds + Seeds → asyncio.gather (all feeds + seeds in parallel)
Phase 2: SearXNG queries → sequential (each query triggers GPU ingestion)
Phase 3: Pipeline → single thread (pipeline handles its own parallelism)
```

### Pipeline (pipeline.py `run_pipeline`)
```
Phase 1: Decomposition (external LLM — wait for new articles)
Phase 2: Trust + Lineage (parallel, CPU) — lineage uses role figments
Phase 3: Summaries (GPU, sequential, max 10 per tick)
Phase 4: Brief (GPU) + Eval (I/O) — parallel
Phase 5: Corrections + Brief review
```

### Narrative Clustering (lineage.py)
Clusters articles by shared role figments (WHO/WHAT/WHERE/WHEN/WHY/HOW).
Two articles share a narrative if they share >= 2 role figment IDs.
Role figments are deduplicated by `hash(role + normalized_text)` — exact
semantic matching regardless of how different outlets phrase headlines.

Falls back to boundary similarity (cosine > 0.95 within 48h) when the
external LLM is not configured.

### Decomposition (decompose.py)
3 async workers consume from `asyncio.Queue`. Each article decomposition includes
**inline cogitation**: relationship edges between co-occurring role figments are
created/strengthened immediately — no separate consolidation pass needed.

### Auto-Crawl
Server startup checks `sources.json` crawler state. If `continuous=true` or the
store has < 50 articles, crawling starts automatically with persisted interval
(default 300s). No manual "Run" button needed.

## Key Design Details

- **Role figment clustering**: Narratives built from shared role figments, not text heuristics
- **Decomposition before clustering**: Pipeline waits for external LLM to extract roles
- **Auto-crawl on startup**: Server starts continuous crawling immediately if configured
- **Parallel feed crawling**: All feeds + seeds via asyncio.gather
- **Sequential SearXNG**: Queries run one at a time (each triggers GPU ingestion)
- **Thread-safe ingestion**: `Crawler._model_lock` serializes GPU, `_ingest_lock` protects shared state
- **VRAM management**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, cleared per article
- **VRAM check**: Pipeline skips summaries/brief if < 200MB GPU free
- **Inline cogitation**: Relationship discovery during decomposition
- **Default budget**: 15 articles per tick (1 per feed), 300s interval
- **External LLM**: Qwen3.6-35B — must pass `chat_template_kwargs: {"enable_thinking": false}`
- **SearXNG auto-queries**: Keywords extracted from newly ingested articles; no manual query list

## Data Flow

```
Startup → auto-start continuous crawl (if configured)
    │
    ▼ (every 300s or manual trigger)
    │
    ├─ Phase 1 (parallel): Feed crawling + Seed crawling
    │   Each feed/seed → ingest_articles → figmentize → store
    │   VRAM cleared before each article ingestion
    │
    ├─ Phase 2 (sequential): SearXNG queries (keywords from Phase 1)
    │   Each query → search + fetch + ingest
    │
    ├─ Phase 3 (pipeline):
    │   1. Decomposition (external LLM) → role figments
    │   2. Trust + Lineage (parallel) → role figment clustering
    │   3. Summaries (GPU, max 10)
    │   4. Brief (GPU) + Eval (I/O)
    │   5. Corrections + Brief review
    │
    └─ Tick complete → sleep 300s → next tick
```

## Source Configuration

`sources.json` — **single unified config file**:
- `"feeds"`: `{source_id: rss_url}` — RSS/Atom feeds
- `"sources"`: `{source_id: {name, base_trust, url, kind, logo_url}}` — source metadata
- `"searxng"`: `{url, enabled, categories, time_range, max_results, pages}` — SearXNG settings
  - **No `queries` field** — queries are auto-derived from RSS article keywords each tick
- `"llm"`: `{url, model, timeout, enabled, ...}` — external LLM config
- `"crawler"`: `{continuous, smart_crawl, interval, max_articles, max_stories, llm_enabled, ...}` — **persisted crawler state**
- Unknown domains auto-registered with `base_trust=0.7`

## Common Pitfalls

1. **asyncio in crawler**: `ingest_article()` runs in a thread via `asyncio.to_thread`;
   never call `asyncio.create_task()` from there — use `_pending_decompose` list instead
2. **Thread safety**: `Crawler._model_lock` serializes GPU; `_ingest_lock` protects `seen`/`_new_articles`
3. **Numpy truth value**: Never `if not array:` on numpy — use `if array is None:`
4. **Role figment clustering**: Requires decomposition to run first; boundary fallback if no LLM
5. **Qwen3.6 thinking**: LLM puts ALL output in `reasoning_content` unless `enable_thinking: false`
6. **SearXNG**: Requires JSON format enabled in its settings.yml; may need restart
7. **VRAM**: 3GB GPU is tight — max 10 summaries per tick, skip brief if low VRAM
8. **Pipeline ThreadPoolExecutor**: Do not call GPU operations from pipeline thread pool — only CPU/IO work

## New Capabilities (Phase 1–3)

### Association Figments (Co-Reference Layer)

`associations.py` links surface-form variants of the same entity so that
``"Donald Trump"``, ``"Trump"``, and ``"DJT"`` are treated as the same WHO node.

- Associations are `association` edge figments stored in LanceDB alongside everything else.
- Auto-proposals at decompose time (boundary similarity > 0.90, string overlap > 0.50, edit similarity > 0.85).
- Bounded expansion (default 2 hops) at query time.
- `expand_associations(store, role_figment_id)` returns all variant IDs.
- `get_association_groups(store)` returns all clusters.

### Multi-Role Intersection Query

`intersection.py` implements the core retrieval primitive: ``find_narratives(store, roles [...])``.

```python
from figtree_news.intersection import find_narratives

narratives = find_narratives(
    store,
    roles=[{"role": "who", "text": "Donald Trump"}, {"role": "where", "text": "Disney World"}],
    expand_associations=True,   # covers Trump / Donald Trump / DJT
    require_all=True,
    ranking="trust_recency",
)
```

Returns ranked narratives with ``role_matches``, ``trust_score``, and ``source_count``.

### Context Materialization

`context.py` assembles a provenance-preserving context package from narrative IDs.

```python
from figtree_news.context import materialize_context

ctx = materialize_context(
    store,
    [n["narrative_id"] for n in narratives],
    include_text=True,
    max_articles_per_narrative=10,
)
# ctx["context_text"] is ready for FigmentGenerator.generate()
# ctx["trust_profile"] preserves source credibility
# ctx["chronological_order"] gives temporal ordering
```

### The Dream Query (now working end-to-end)

```python
narratives = find_narratives(store, [{"role": "who", "text": "Donald Trump"}, {"role": "where", "text": "Disney World"}])
ctx = materialize_context(store, [n["narrative_id"] for n in narratives])
# Feed ctx["context_text"] into FigmentGenerator.generate() for faithful, source-attributed output
```

## Bug Fixes (Phase 0)

- **SQLite thread safety** (`search_index.py`): Added `threading.Lock()` around all DB operations. The `check_same_thread=False` connection was subject to ``NULL without setting an exception`` errors from concurrent write access across crawl/pipeline/web threads.
- **VRAM OOM guard** (`serve.py`): SearXNG query loop now checks free VRAM (< 500MB stops queries). Added `torch.cuda.empty_cache()` and 1s pause between queries to let CUDA reclaim fragmented memory.
- **Decomposition queue dedup** (`decompose.py`): `DecompositionEngine` now tracks ``self._queued`` to prevent the same article from being scheduled for decomposition multiple times.
- **Keyword extraction** (`crawler.py`): Strips URLs and URL fragments before word extraction, producing meaningful SearXNG queries instead of ``https``/``www``/``com``.
- **MSN article fetch** (`crawler.py`): `ingest_article` now attempts ``trafilatura`` on short articles with URLs before discarding them, improving coverage from JS-heavy news sites.
