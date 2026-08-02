# Figtree-News

> **EXPERIMENTAL** — Rapid iteration, breaking changes, incomplete features.

Source-aware news aggregator built on Figtree figments. Articles are decomposed
into WHO/WHAT/WHERE/WHEN/WHY/HOW role figments that are reused across narratives,
enabling structured search, trust-aware reasoning, and faithful generation.

## Build / Test

```bash
pip install -e .                    # install (from figtree-news/ root)
python3 -m pytest tests/ -v         # run all tests (CPU-only, no GPU needed)
python3 -m pytest tests/test_web.py -v  # specific test file
ruff check figtree_news/ tests/
```

## Architecture

```
figtree_news/
├── cli.py              # Typer CLI: crawl, serve, search, query, lineage, export, eval, boundary-threshold
├── config.py           # SourceRegistry + CrawlerState + SourceConfig dataclasses
├── searxng.py          # SearXNG client: search → article dicts (SearxngConfig lives here)
├── ingest.py           # Feed/article → figments (+ single-pass role extraction, FTS indexing)
├── crawler.py          # RSS + SearXNG + BFS link-follower (thread-safe ingestion, FTS dedup)
├── pipeline.py         # In-process pipeline: trust + lineage (parallel) → summaries → brief → compaction
├── lineage.py          # Role figment clustering (min 2 shared) + derivative edges (atomic rebuild)
├── trust.py            # Source trust propagation (uses FigtreeGraph, no O(n²) dedup)
├── normalize.py        # Shared entity-normalization (honorific-stripping, lowercase, punctuation-removal)
├── summarize_news.py   # Per-article summaries + world brief
├── query.py            # Embed query → nearest figments → faithful generate
├── search_index.py     # SQLite FTS5 full-text search (thread-safe)
├── associations.py     # Thin wrapper over figtree.identity (co-reference merge engine)
├── intersection.py     # Multi-role intersection query: find narratives containing all specified roles
├── context.py          # Context materialization: structured context packages from narratives
├── export.py           # Graph export as JSON
├── eval.py             # Per-source faithful-recall eval
├── model_lock.py       # Shared RLock for GPU model forward passes (serializes all GPU work)
└── web/
    ├── serve.py        # FastAPI: single process owns model + crawl + pipeline + web
    ├── templates/      # Jinja2 HTML (base.html, index.html, article.html, etc.)
    └── static/         # Dark theme CSS + client-side JS (WebSocket, crawl control, stats)
```

**Deleted (Phase 2)**: `decompose.py`, `evaluate.py`, `correct.py`, `cogitate.py`,
`association_worker.py`, `llm_config.py` — the external 35B LLM stack. Nothing
worked end-to-end (every Jul 30 run OOM-killed within ~20 min; one 4.3h
decompose run per tick). The local Qwen3-4B model handles role extraction
(single-pass decode at ingest) + summaries + brief. `--boundary-threshold`
CLI + `dedup_obs` figments remain as tuning-data collection.

## Key Commands

```bash
# Serve web UI + auto-starts continuous crawl (single process, owns the model)
figtree-news serve --db news.lance --sources demo/sources.json --host 0.0.0.0 --port 8000
figtree-news serve --device none    # viewer-only: read the store, never load a model

# Standalone single-tick crawl + pipeline
figtree-news crawl --once --max-articles 15

# SearXNG search
figtree-news search "AI regulation" --time-range day --max 10

# Run pipeline only (CPU phases or with an already-loaded model)
figtree-news lineage && figtree-news update-trust

# Install as one systemd service (replaces the old crawler+web pair)
./systemd/install_systemd.sh
```

## Execution Model (single process)

`serve.py` runs **one process** that owns the (GPU) model. Crawl + pipeline are
a background asyncio task inside the same uvicorn process — there is no second
crawler loop to contend with (the old dual-loop design OOM-killed the box every
~20 minutes). `--device none` gives a viewer-only mode (reads the shared store,
never loads a model) for boxes without the GPU or when a separate crawler owns it.

### Crawl Tick (serve.py `_run_tick`)
```
0.  Load model once (cached; `_load_model_cached`, executor + asyncio.Lock)
1.  Phase 1: Feeds + Seeds → asyncio.gather (parallel, 300s timeout each)
2.  Phase 2: SearXNG queries → sequential (VRAM guard < 500MB free stops them)
3.  Pipeline in-process (run_pipeline on the pipeline executor, with the SAME
    model object — no reload)
4.  Broadcast crawl_status over WebSocket; invalidate the data cache
```

`_run_loop` runs ticks every `interval` (min 60s), checks `stop_requested`
every second, and starts automatically at startup when `crawler.continuous`
or the store has < 50 articles.

### Pipeline (pipeline.py `run_pipeline`)
```
Phase 1: Trust (CPU) + Lineage (CPU) in parallel, ONE store.all() snapshot
Phase 2: Role → narrative linking on the FRESH lineage output (stale-snapshot fix)
Phase 3: VRAM check (< 200MB free skips GPU phases)
Phase 4: Summaries (GPU, sequential, max 10)
Phase 5: World brief (GPU, top_n=8)
Phase 6: store.optimize() compaction + _force_free()
```

No external-LLM phases: role figments arrive at ingest (single-pass decode),
trust/lineage are CPU, summaries/brief use the local model.

### Narrative Clustering (lineage.py `compute_lineage`)
Clusters articles by shared role figment IDs (WHO/WHAT/WHERE/WHEN/WHY/HOW).
Two articles share a narrative if they share >= 2 role figment IDs. Role
figment IDs are deterministic (`sha256(f"role:{role}:{normalized}")[:16]`), so
exact semantic matches dedupe regardless of phrasing.

**Atomic rebuild**: new narrative/derivative figments are fully computed and
upserted BEFORE stale ones are deleted. A crash mid-rebuild (OOM, Ctrl-C) can
never leave the store with zero narratives — the regression test
(`test_lineage_atomic_on_crash`) pins this. `assign_roles_to_narratives` takes
the fresh lineage output directly (no stale store re-read).

### Association Nodes (figtree.identity)

`associations.py` is a thin wrapper over the library's `figtree.identity`
merge engine: `propose_identity_merges` (heuristics: boundary cosine,
token overlap, edit similarity, co-occurrence), `assert_identity`
(edge_type="association" figments), and `merge_role_figments` (canonical-node
promotion + rewrite of every reference: article/paragraph `role_figments`,
`sentence.children`, relationship edges, association edges, dedup_obs; then
deletes the variant rows). When "Donald Trump", "Trump", and "DJT" are
confirmed equivalent, one ID is canonical and all references point at it — no
expansion hop needed at query time.

## Key Design Details

- **Single process owns the model**: no separate crawler/pipeline process; `--device none` = viewer-only
- **Single-pass role extraction**: roles extracted during the ingest forward pass (appended decode prompt, `enable_thinking=False`)
- **Atomic lineage**: upsert-new-then-delete-stale; crash never yields 0 narratives
- **FTS path threading**: every `get_index()` call passes the derived `<db>.lance → <db>_fts.db` path (ingest/crawler/serve/cli all take `fts_path`) — a hardcoded default once indexed the wrong store
- **Parallel feed crawling**: all feeds + seeds via asyncio.gather, 300s timeout
- **Sequential SearXNG**: one query at a time, VRAM guard (500MB free) stops them, 1s pause between
- **Thread-safe ingestion**: `Crawler._ingest_lock` protects `seen`/`_new_articles`; `model_lock` serializes GPU
- **VRAM management**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128`; `_force_free()` after pipeline; 200MB gate skips summaries/brief
- **Default budget**: 12 articles per tick, 900s interval, searxng disabled (demo/sources.json)
- **SearXNG auto-queries**: keywords extracted from newly ingested articles; no manual query list
- **Idempotent trust**: `update_trust` persists `trust:{source_id}` figments; base_trust is never read back (no drift)
- **Store compaction**: `store.optimize(cleanup_older_than=0)` each pipeline run keeps `store.all()` latency bounded

## Source Configuration

`sources.json` — **single unified config file**:
- `"feeds"`: `{source_id: rss_url}` — RSS/Atom feeds (may include YouTube video feeds)
- `"sources"`: `{source_id: {name, base_trust, url, kind, logo_url}}` — source metadata
- `"searxng"`: `{url, enabled, categories, time_range, max_results, pages}` — SearXNG settings (no `queries` field; auto-derived)
- `"crawler"`: `{continuous, smart_crawl, interval, max_articles, max_stories, ...}` — persisted crawler state
- The `"llm"` section is GONE (external LLM stack removed). Unknown domains auto-register with `base_trust=0.7`.

## Common Pitfalls

1. **Never re-add the external-LLM loop**: the local model + single-pass extraction is the design; 35B decompose runs (4.3h/tick) made completion impossible on this box
2. **asyncio in crawler**: `ingest_article()` runs in a thread via the crawl executor; never `asyncio.create_task` from there
3. **Thread safety**: `Crawler._ingest_lock` protects shared state; `model_lock` serializes GPU
4. **Numpy truth value**: never `if not array:` on numpy — use `if array is None:`
5. **FTS path**: always derive the FTS db from the LanceDB path (`db.replace(".lance", "_fts.db")`) — do not call `get_index()` bare
6. **Qwen3 thinking**: keep `enable_thinking=False` in the ChatML template (see `figtree.kernel.prompt.build_prompt_ids`)
7. **VRAM**: 4GB GPU is tight — max 10 summaries per tick, skip brief if < 200MB free
8. **Pipeline executor**: do not call GPU ops from the pipeline thread pool — only CPU/IO work
9. **Lineage ordering**: never mutate the store between `compute_lineage` and `assign_roles_to_narratives` — pass the fresh output

## Multi-Role Intersection Query

`intersection.py` implements the core retrieval primitive: `find_narratives(store, roles [...])`.

```python
from figtree_news.intersection import find_narratives

narratives = find_narratives(
    store,
    roles=[{"role": "who", "text": "Donald Trump"}, {"role": "where", "text": "Disney World"}],
    require_all=True,
    ranking="trust_recency",
)
```

Returns ranked narratives with `role_matches`, `trust_score`, and `source_count`.

## Context Materialization

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
```

## The Dream Query

```python
narratives = find_narratives(store, [{"role": "who", "text": "Donald Trump"}, {"role": "where", "text": "Disney World"}])
ctx = materialize_context(store, [n["narrative_id"] for n in narratives])
# Feed ctx["context_text"] into FigmentGenerator.generate() for faithful, source-attributed output
```

## Bug Fixes (Phase 2)

- **FTS path mismatch**: `ingest.py`/`crawler.py` called bare `get_index()` (default `demo/news_fts.db`) even when the server wrote a custom store — search silently returned 0 hits. `fts_path` is now threaded everywhere.
- **SearXNG `max_results` kwarg**: `searxng.search()` takes `pageno`, not `max_results` — the find-more route was fixed.
- **Keyword-only args**: `ensure_article_summaries(limit=)` / `build_world_brief(top_n=)` are keyword-only; the regenerate route passed them positionally.
- **OOM dual-loop**: crawl and pipeline both loaded the model — merged into one process; verified 3 ticks at RSS < 2.5GB, VRAM ~2.8GB, no OOM.
