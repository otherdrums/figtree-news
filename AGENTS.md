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
├── cli.py              # Typer CLI: crawl, serve, search, query, lineage, export, eval, boundary-threshold
├── config.py           # SourceRegistry + CrawlerState + SourceConfig dataclasses
├── searxng.py          # SearXNG client: search → article dicts with image/video
├── ingest.py           # Feed/article → figments with provenance (+ single-pass role extraction)
├── crawler.py          # RSS + SearXNG + BFS link-follower (thread-safe ingestion)
├── pipeline.py         # Parallel pipeline orchestration (ThreadPoolExecutor)
├── lineage.py          # Role figment clustering (min 2 shared) + frame shift + derivative edges
├── trust.py            # Source trust propagation (uses FigtreeGraph, no O(n²) dedup)
├── decompose.py        # Legacy per-paragraph extraction (Phase 1 fallback for non-inline articles)
├── association_worker.py  # Background worker: merges variant role figments → canonical nodes
├── normalize.py        # Shared entity-normalization (honorific-stripping, lowercase, punctuation-removal)
├── cogitate.py         # Periodic insight generation + merge/consolidation
├── evaluate.py         # External LLM: cluster validation, frame shift, brief review
├── correct.py          # Self-correction: confirmation threshold + auto-apply
├── llm_config.py       # External LLM configuration (optional — uses local model by default)
├── summarize_news.py   # Per-article summaries + world brief
├── query.py            # Embed query → nearest figments → generate
├── search_index.py     # SQLite FTS5 full-text search (thread-safe)
├── associations.py     # Co-reference layer: association figments for surface-form variants
├── intersection.py     # Multi-role intersection query: find narratives containing all specified roles
├── context.py          # Context materialization: assemble structured context packages from narratives
├── export.py           # Graph export as JSON
├── eval.py             # Per-source faithful-recall eval
├── model_lock.py       # Shared RLock for GPU model forward passes (serializes all GPU work)
└── web/
    ├── serve.py        # FastAPI + auto-crawl (ingest only; pipeline loop owns summaries/brief)
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
  └─ Each ingest call: single-pass forward → boundaries + role extraction
     (decode prompt appended to the same cache, no separate decompose pass)
```

### Pipeline (pipeline.py `run_pipeline`)
```
Phase 1: Decomposition (local model) — automatically skips articles
         already decomposed by single-pass ingest (role_figments + decomposed flags)
Phase 2a: Trust (CPU) + 2b: Lineage (CPU, parallel) — lineage uses role figments
Phase 2c: LLM clustering eval → merge/split narratives
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
external LLM is not configured.  (Note: boundary fallback was removed in
v0.2; singletons + LLM merge is the current design.  Boundary data is
collected via ``--boundary-threshold`` CLI for future re-tuning.)

### Decomposition (decompose.py)
Legacy per-paragraph extraction used as a fallback. Since v0.3, role figments are
extracted during the ingest forward pass (single-pass decode), so decompose runs
only for articles that predate this change or lack role_figments.

### Association-Node Worker (association_worker.py)
Runs as a background asyncio task (started in ``serve.py`` startup).  Each tick
scans for unprocessed role figments, finds same-role candidates by heuristics
(boundary, containment, edit similarity), and calls the external 35B LLM arbiter
to confirm equivalence.  On confirmation, ``merge_role_figments()`` promotes the
more-established figment to a canonical node (`is_association=True`), rewrites
all references (article/paragraph role_figments, sentence.children, relationship
edges, association edges, dedup_obs), and deletes the variant rows.  This keeps
role IDs canonical — no expansion hop needed at query time.

### Auto-Crawl
Server startup checks `sources.json` crawler state. If `continuous=true` or the
store has < 50 articles, crawling starts automatically with persisted interval
(default 300s). No manual "Run" button needed.

## Key Design Details

- **Role figment clustering**: Narratives built from shared role figments, not text heuristics
- **Single-pass role extraction**: Roles are extracted during the ingest forward pass (appended decode prompt), eliminating a separate GPU pass
- **Association-node dedup**: Confirmed-equivalent role variants are hard-replaced by a canonical node; all references rewritten, variant rows deleted
- **Auto-crawl on startup**: Server starts continuous crawling immediately if configured
- **Parallel feed crawling**: All feeds + seeds via asyncio.gather
- **Sequential SearXNG**: Queries run one at a time (each triggers GPU ingestion)
- **Thread-safe ingestion**: `Crawler._model_lock` serializes GPU, `_ingest_lock` protects shared state
- **VRAM management**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, cleared per article
- **VRAM check**: Pipeline skips summaries/brief if < 200MB GPU free
- **Default budget**: 15 articles per tick (1 per feed), 300s interval
- **External LLM**: Qwen3.6-35B — must pass `chat_template_kwargs: {"enable_thinking": false}`
- **SearXNG auto-queries**: Keywords extracted from newly ingested articles; no manual query list
- **Pipeline loop**: Independent background task runs full pipeline (summaries, brief) on cadence; crawl tick only ingests
- **Light store.all()**: The `boundaries` per-layer column (92,160 floats/row) is no longer persisted or loaded; `store.all()` is ~200MB not 6GB
- **Background workers disabled**: When external LLM is configured, background decompose workers are NOT started after crawl ticks. The pipeline's Phase 1 handles ALL decomposition sequentially with a shared `created` dict, enabling cross-article dedup. Background workers would create unmerged role figments (each worker has its own local `created` dict).
- **Entity dedup**: Three-layer dedup strategy: (1) exact normalized-text hash match, (2) intra-article heuristic-only (containment >= 0.66 or edit_sim >= 0.70), (3) cross-article LLM arbiter (textual overlap prefilter first). Dedup_obs figments record heuristic scores + LLM verdict for threshold analysis.
- **Cross-article heuristic prefilter**: Requires `containment >= 0.4` or `(jaccard >= 0.25 and edit_sim >= 0.25)` — boundary sim alone is too noisy for cross-article comparison.
- **Extraction prompt**: Uses few-shot canonicalization with explicit "SINGLE entity only" constraint to prevent comma-separated lists. Includes a negative example showing the wrong (list) and correct (single) format.
- **SearXNG**: Enabled in sources.json (`searxng.enabled: true`, `crawler.searxng_enabled: true`). Queries auto-derived from RSS article keywords each tick.
- **Brief quality**: Pipeline uses `top_n=8` (not 2). Review prompt reports actual article count, not total store articles, preventing false negatives.
- **Narrative merging**: `label_article_pairs` samples both within-cluster (split detection) and cross-cluster (merge detection) pairs. Pipeline automatically merges/splits narratives from LLM labels via `merge_narratives_by_llm_labels` / `split_narratives_by_llm_labels`.
- **Config schema**: `demo/config_schema.json` documents all config sections (sources, feeds, seeds, searxng, llm, crawler).

## Data Flow

```
Startup → auto-start continuous crawl (if configured)
    │
    ▼ (every 300s or manual trigger)
    │
    ├─ Phase 1 (parallel): Feed crawling + Seed crawling
    │   Each feed/seed → ingest_articles → figmentize → store
    │   VRAM cleared before each article ingestion
    │   └─ Single-pass decode appends role extraction → role figments stored
    │
    ├─ Phase 2 (sequential): SearXNG queries (keywords from Phase 1)
    │   Each query → search + fetch + ingest (same single-pass role extraction)
    │
    ├─ Phase 3 (pipeline):
    │   1. Decomposition (local model) → skip already-decomposed articles
    │   2. Trust + Lineage (parallel) → role figment clustering
    │   3. Summaries (GPU, max 10)
    │   4. Brief (GPU) + Eval (I/O)
    │   5. Corrections + Brief review
    │
    ├─ Background (continuous):
    │   Assoc worker → finds unprocessed role figments, confirms equivalence
    │   via LLM arbiter, merges variants into canonical nodes (hard rewrite)
    │
    └─ Tick complete → sleep 300s → next tick
```

## Source Configuration

`sources.json` — **single unified config file**:
- `"feeds"`: `{source_id: rss_url}` — RSS/Atom feeds
- `"sources"`: `{source_id: {name, base_trust, url, kind, logo_url}}` — source metadata
- `"searxng"`: `{url, enabled, categories, time_range, max_results, pages}` — SearXNG settings
  - **No `queries` field** — queries are auto-derived from RSS article keywords each tick
- `"llm"`: `{url, model, timeout, enabled, decompose, find_missed_merges, review_brief, ...}` — external LLM config; `decompose: false` skips both Phase 1 decomposition (legacy) and the association worker's LLM arbiter
- `"crawler"`: `{continuous, smart_crawl, interval, max_articles, max_stories, llm_enabled, ...}` — **persisted crawler state**
- Unknown domains auto-registered with `base_trust=0.7`

## Common Pitfalls

1. **asyncio in crawler**: `ingest_article()` runs in a thread via `asyncio.to_thread`;
   never call `asyncio.create_task()` from there — use `_pending_decompose` list instead
2. **Thread safety**: `Crawler._model_lock` serializes GPU; `_ingest_lock` protects `seen`/`_new_articles`
3. **Numpy truth value**: Never `if not array:` on numpy — use `if array is None:`
4. **Role figment clustering**: Requires decomposition to run first; singletons + LLM merge if no roles
5. **Qwen3.6 thinking**: LLM puts ALL output in `reasoning_content` unless `enable_thinking: false`
6. **SearXNG**: Requires JSON format enabled in its settings.yml; may need restart
7. **VRAM**: 3GB GPU is tight — max 10 summaries per tick, skip brief if low VRAM
8. **Pipeline ThreadPoolExecutor**: Do not call GPU operations from pipeline thread pool — only CPU/IO work

## New Capabilities

### Association Nodes (Canonical Entity Model)

`association_worker.py` merges confirmed-equivalent role variants into a single
canonical node via ``merge_role_figments()`` in ``associations.py``.  When
``"Donald Trump"``, ``"Trump"``, and ``"DJT"`` are confirmed to refer to the
same entity, one is promoted to a canonical node (`is_association=True`) and
all references (article/paragraph ``role_figments``, ``sentence.children``,
relationship edges, association edges, dedup_obs) are rewritten to point at it.
The variant rows are deleted.

- **Hard replacement**: No expansion hop needed at query time — ``article.role_figments`` already holds the canonical ID.
- **LLM arbiter**: The external 35B confirms equivalence before any merge; heuristic-only fallback when no LLM is configured.
- **``merge_role_figments(store, keep_id, remove_ids)``**: The rewrite engine — handles all six reference locations.
- **``AssociationWorker``**: Background asyncio task (started in ``serve.py`` startup), runs every 10s with a 2-call semaphore.
- **Seamless with ``_cluster_by_roles``**: Clustering no longer expands associations — role IDs are already canonical.

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

Because the association-node worker rewrites all references to canonical IDs,
clustering and query both see a single shared ID per entity — no expansion
hop needed.  ``find_narratives`` with ``expand_associations=True`` is now a
no-op for internal logic (the parameter is kept for API compatibility).

## Bug Fixes (Phase 0)

- **SQLite thread safety** (`search_index.py`): Added `threading.Lock()` around all DB operations. The `check_same_thread=False` connection was subject to ``NULL without setting an exception`` errors from concurrent write access across crawl/pipeline/web threads.
- **VRAM OOM guard** (`serve.py`): SearXNG query loop now checks free VRAM (< 500MB stops queries). Added `torch.cuda.empty_cache()` and 1s pause between queries to let CUDA reclaim fragmented memory.
- **Decomposition queue dedup** (`decompose.py`): `DecompositionEngine` now tracks ``self._queued`` to prevent the same article from being scheduled for decomposition multiple times.
- **Keyword extraction** (`crawler.py`): Strips URLs and URL fragments before word extraction, producing meaningful SearXNG queries instead of ``https``/``www``/``com``.
- **MSN article fetch** (`crawler.py`): `ingest_article` now attempts ``trafilatura`` on short articles with URLs before discarding them, improving coverage from JS-heavy news sites.
