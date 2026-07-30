# figtree-news

> **EXPERIMENTAL** — This system is under active development and subject to
> breaking changes, rapid iteration, and incomplete features. APIs, data
> formats, and config files may change without notice. Not production-ready.

A **source-aware news aggregator** built on [figtree](https://github.com/otherdrums/figtree)
(figment-based memory for language models). Articles are decomposed into
reusable semantic primitives (WHO/WHAT/WHERE/WHEN/WHY/HOW) that link across
narratives — enabling structured search, trust-aware reasoning, and
self-correcting LLM evaluation.

## How It Works

News articles are decomposed into reusable semantic primitives
(WHO/WHAT/WHERE/WHEN/WHY/HOW **role figments**) that link across narratives.
The system treats news as a graph of figments rather than independent documents.

```
                    ┌─────────────┐
                    │  RSS Feeds  │
                    │  SearXNG    │
                    │  Seed URLs  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────────┐
                    │  Crawl & Ingest │  One forward pass yields both
                    │  (single-pass)  │  boundaries + role extraction
                    │                 │  (appended decode prompt with
                    │                 │   enable_thinking=False)
                    └──────┬──────────┘
                           │
                    ┌──────▼──────┐
                    │  LanceDB    │  Article figment has role_figments +
                    │  Store      │  decomposed flag — ready for clustering
                    └──────┬──────┘
                           │
              ┌────────────┼─────────────────┐
              │            │                  │
       ┌──────▼────────┐  │     ┌────────────▼───────────┐
       │ Association   │  │     │  Pipeline Loop          │
       │ Worker        │  │     │  (independent cadence)  │
       │ (background)  │  │     └────────────┬────────────┘
       │ heuristic +   │  │                  │
       │ LLM arbiter   │  │     ┌────────────▼────────────┐
       │ → canonical   │  │     │ Phase 1: Decompose      │
       │   nodes       │  │     │ (fallback for legacy    │
       └──────┬────────┘  │     │  articles only)         │
              │           │     └────────────┬────────────┘
              │           │                  │
              │     ┌─────┴──────────────────▼────────────┐
              │     │  Trust (CPU) + Lineage (parallel)   │
              │     │  — role figment clustering,         │
              │     │    frame shift, derivative edges    │
              │     └─────┬──────────────────┬────────────┘
              │           │                  │
              │     ┌─────▼──────────┐ ┌────▼───────────┐
              │     │ LLM merge/split│ │  Summaries     │
              │     │ narratives     │ │  (local model) │
              │     └─────┬──────────┘ │  max 10/tick   │
              │           │           └────┬────────────┘
              │           │                │
              │     ┌─────┴────────────────▼──────┐
              │     │ Brief (local model) + Eval  │
              │     │ (external LLM cluster       │
              │     │  review) + Corrections      │
              │     │ + Brief review              │
              │     └─────┬───────────────────────┘
              │           │
              │     ┌─────▼──────────────────────┐
              └─────►     Web UI / API           │
                    └────────────────────────────┘
```

### Core Concepts

**Figments** are the atomic unit. An article is an "image" figment containing
child sentence figments. Role figments (WHO, WHAT, WHERE, WHEN, WHY, HOW) are
extracted **during ingest** via a single-pass decode prompt appended to the
forward-pass cache — no separate decomposition pass for new articles. The same
role figment (e.g. "Joe Biden" as WHO) reuses across multiple articles via
`hash(role + normalized_text)` deduplication, creating a bipartite graph where
narrative relationships emerge from figment overlap. The article figment is
stamped with `role_figments` and `decomposed=True` at ingest time.

**Association nodes** resolve surface-form variants ("Donald Trump" ↔ "Trump"
↔ "DJT") into a single canonical role figment. A background worker
(`association_worker.py`) scans unprocessed role figments, finds same-role
candidates by heuristics (boundary, containment, edit similarity), and calls
the external 35B LLM to confirm equivalence. On confirmation,
`merge_role_figments()` promotes one variant to a canonical node
(`is_association=True`), rewrites all references (article/paragraph
role_figments, sentence.children, relationship edges, association edges,
dedup_obs), and deletes the variant rows. Role IDs stay canonical — no
expansion hop needed at query time. When no external LLM is configured, the
worker uses heuristic-only matching.

**Multi-role intersection** finds narratives containing ALL specified roles
(e.g. WHO:Trump AND WHERE:Disney World). Because role IDs are already
canonical (the association worker has already merged variants), query-time
expansion is not needed. Results are ranked by trust, recency, source
diversity, and frame shift.

**Context materialization** assembles a provenance-preserving context package
from intersection results — source attribution, trust scores, chronological
ordering, frame information — ready for faithful model-native generation.

**Boundary vectors** (~10KB float32) are captured from the model's hidden state
during ingestion. They enable similarity search, dedup, and frame-shift
detection without re-running the model.

**Narrative clustering** groups articles by shared role figments. Two articles
share a narrative if they share >= 2 role figment IDs — exact semantic matching
regardless of how different outlets phrase headlines. Singletons with no shared
roles fall through to LLM-based merge detection (when the external LLM is
configured), and clustering is periodically refined via
`merge_narratives_by_llm_labels` / `split_narratives_by_llm_labels`.

**Self-correction** — an external LLM (Qwen 3.6 35B) reviews narrative
clusters, flags miscategorized articles, and suggests corrections.
Corrections accumulate across eval runs and auto-apply at a configurable
confirmation threshold (default 2).

## Install

```bash
pip install -e .   # from figtree-news/ root (also install figtree first)
```

## Quick Start

```bash
# 1. Configure sources (edit demo/sources.json — add LLM URL, feeds, etc.)
cp demo/sources.json ./sources.json

# 2. Serve the web newspaper (auto-starts continuous crawl)
figtree-news serve --db demo/news.lance --sources demo/sources.json --host 0.0.0.0 --port 8000

# 3. Standalone crawl (CLI)
figtree-news crawl --interval 300 --max-articles 15

# 4. Standalone search
figtree-news search "AI regulation" --time-range day --max 10
```

## Features

### Web Newspaper (FastAPI)

- Front page with world brief, narrative comparison cards, source trust board
- Article detail pages with images, video embeds, author bylines
- Source pages with all articles + trust scores
- Narrative pages with all source versions + frame-shift badges
- Full-text search (SQLite FTS5) with date range filters
- Responsive slide-out control panel for all crawl + search settings
- Sticky crawl action bar — Run Once / Start Continuous / Stop + mode indicator
- WebSocket live updates — page auto-refreshes on new content
- Dark theme (benthic.io style)
- JavaScript extracted to `static/app.js` for clean separation

### Crawl Control Panel

| Control | Default | Description |
|---------|---------|-------------|
| Max articles | 15 | Cap per tick (1 per feed) |
| Max stories | 0 (unlimited) | Cap narratives per pipeline run |
| Pause between ticks | 300 | Seconds between ticks (continuous mode) |
| Compute KV cache | off | Cache K/V for boundary-based generation |
| Enable LLM Review | on | External LLM cluster validation + self-correction |
| Smart crawl | on | Auto-switch: forward when new articles found, backward when stuck |
| **Web Search (SearXNG)** | | |
| Enable web search | on | Toggle SearXNG search |
| Time range | Day | day / week / month / year / anytime |
| Categories | News | news / general / general,news |
| **Search queries** | **auto** | **Keywords extracted from RSS article titles/text each tick** |

### Web Search (SearXNG)

Articles from across the web via a local [SearXNG](https://docs.searxng.org/)
instance. Full text fetched via trafilatura, deduplicated by URL + title.
Unknown domains auto-registered as sources with `base_trust=0.7`.

**Queries are auto-derived**: each crawl tick extracts top keywords from newly
ingested RSS article titles/summaries (with built-in stopword filtering) and
uses those as SearXNG queries. This ensures web coverage mirrors the topics
currently appearing in feeds — no manual query maintenance needed. Falls back
to generic news queries if no articles were added.

### Role Extraction

Role figments are extracted **during the ingest forward pass**, not in a
separate stage. The prompt is a few-shot canonicalization instruction with
`SINGLE entity only` constraint; it is appended to the forward-pass cache
via `past_key_values` reuse and decoded with greedy sampling. The model's
chain-of-thought reasoning is suppressed by pre-seeding an empty closed
`<think>` block (`build_prompt_ids(enable_thinking=False)`).

For articles that predate this feature or lack `role_figments`, a fallback
Phase 1 runs in the pipeline using the external LLM when configured, or the
local model otherwise. The article figment `decomposed` flag prevents
double-processing.

### Association-Node Worker

A background asyncio task (`association_worker.py`, started in `serve.py`
startup) scans for unprocessed role figments each tick. For each same-role
candidate pair it computes heuristics (boundary containment, edit similarity,
Jaccard overlap) and — when the external LLM is configured — submits them
to the 35B arbiter for equivalence confirmation. On confirmation,
`merge_role_figments()` promotes one figment to a canonical node
(`is_association=True`), rewrites all six reference locations (article/paragraph
role_figments, sentence.children, relationship edges, association edges,
dedup_obs), and deletes the variant rows. Without an external LLM, heuristic-only
merges are applied at a higher similarity threshold.

### Cogitation Engine

Periodic insight generation (default 30min interval):
1. Relationship discovery — co-occurrence patterns across role figments
2. Insight generation — LLM-generated landscape insights

### LLM Evaluation & Self-Correction

External LLM (Qwen 3.6 35B) validates pipeline output:
- **Cluster validation** — flags miscategorized articles in narratives
- **Frame shift verification** — confirms genuine framing divergence
- **Brief review** — critiques world brief for accuracy
- **Self-correction** — corrections accumulate, auto-apply at threshold (default 2)

## Pipeline

```
crawl → ingest (single-pass: boundaries + roles) → [assoc-worker background] →
pipeline loop: decompose-fallback → trust ∥ lineage → LLM merge/split →
summaries → brief ∥ eval → corrections + review
```

1. **Crawl**: RSS feeds + SearXNG search + bounded link-follower (URL + title dedup)
2. **Ingest**: One forward pass → boundaries + role extraction via appended decode
   prompt (local model, `enable_thinking=False`). Article stamped with
   `role_figments` and `decomposed=True` at ingest time.
3. **Association worker** (background, continuous): scans for unprocessed roles,
   finds same-role candidates by heuristics, confirms via LLM arbiter, merges
   variants into canonical nodes.
4. **Pipeline Phase 1 — Decompose fallback**: runs only for legacy articles that
   lack `role_figments` — uses external LLM when configured, local model otherwise.
5. **Phase 2a — Trust**: Source trust propagation (idempotent, CPU).
6. **Phase 2b — Lineage**: Narrative clustering via shared role figment IDs
   (>= 2 shared), frame shift detection, derivative edge computation (CPU).
7. **Phase 2c — LLM merge/split**: `label_article_pairs` samples within-cluster
   (split) and cross-cluster (merge) pairs; automatically merges/splits narratives.
8. **Phase 3 — Summaries**: Per-article summaries (local model, max 10 per tick).
9. **Phase 4 — Brief + Eval**: World brief (local model) + external LLM cluster
   review + frame shift check (parallel I/O + GPU).
10. **Phase 5 — Corrections**: Self-correction accumulation + brief review.
    Corrections auto-apply at configurable confirmation threshold (default 2).

## Architecture

```
figtree_news/
├── cli.py              # Typer CLI: 14 commands (crawl, serve, search, ingest, query, etc.)
├── config.py           # SourceRegistry: source config + SearXNG + LLM settings
├── searxng.py          # SearXNG client + article extraction
├── ingest.py           # Feed/article → figments with provenance (+ single-pass role extraction)
├── crawler.py          # Crawler: feeds + SearXNG + BFS link-follower (thread-safe)
├── pipeline.py         # Parallel pipeline orchestration (ThreadPoolExecutor)
├── lineage.py          # Role figment clustering (≥2 shared IDs) + frame shift + derivatives
├── trust.py            # Source trust propagation (idempotent, store-persisted)
├── decompose.py        # Legacy fallback: WHO/WHAT/WHERE/WHEN/WHY/HOW extraction
├── association_worker.py  # Background worker: merges variant role figments → canonical nodes
├── normalize.py        # Shared entity-normalization (honorific-stripping, lowercase, punctuation-removal)
├── cogitate.py         # Periodic insight generation
├── evaluate.py         # External LLM evaluation: clusters, frame shift, brief
├── correct.py          # Self-correction: confirmation threshold + auto-apply
├── llm_config.py       # External LLM configuration
├── summarize_news.py   # Per-article summaries + world brief
├── query.py            # Embed query → nearest figments → generate
├── search_index.py     # SQLite FTS5 full-text search (thread-safe)
├── associations.py     # merge_role_figments(): canonical-node rewrite engine + heuristics
├── intersection.py     # Multi-role intersection query: find narratives containing all specified roles
├── context.py          # Context materialization: assemble structured context packages from narratives
├── export.py           # Graph export as JSON
├── eval.py             # Per-source faithful-recall eval
├── model_lock.py       # Shared RLock for GPU model forward passes (serializes all GPU work)
└── web/
    ├── serve.py         # FastAPI app: HTML pages + JSON API + WebSocket + auto-crawl
    ├── templates/       # Jinja2 HTML templates
    └── static/          # CSS + JS (dark theme, extracted app.js)
```

### Source Registry

`sources.json` maps `source_id → {name, base_trust, url, kind, logo_url}`.
Demo ships with 15 feeds (12 RSS + 3 video: dwnews_yt, france24_yt,
pbsnewshour) + SearXNG + external LLM config.  Unknown domains auto-register
at runtime with `base_trust=0.7`.

### Data Storage

- **LanceDB** — all figments (articles, narratives, edges, trust, role figments)
- **SQLite FTS5** — full-text search index (`{db}_fts.db`)
- **seen_urls.json** — URL dedup (runtime, gitignored)
- **KV cache** (optional) — quantized K/V for boundary-based generation
- **sources.json** — feeds, seeds, SearXNG/LLM config, crawler state

## CLI Reference

```bash
figtree-news crawl [OPTIONS]
  --feed source=url         Add feed (repeatable)
  --seed url                Add seed URL (repeatable)
  --interval N              Seconds between ticks (default: 0)
  --max-articles N          Cap articles per tick (default: 40)
  --max-stories N           Cap narratives, 0 = unlimited (default: 0)
  --since YYYY-MM-DD        Only ingest after this date
  --before YYYY-MM-DD       Only ingest before this date
  --backfill                Deep crawl (200 cap)
  --once                    Single tick then exit
  --compute-kv              Enable KV cache persistence
  --no-summaries            Skip summaries

figtree-news serve [OPTIONS]
  --db PATH                 LanceDB path (default: ./news.lance)
  --sources PATH            sources.json path (default: ./sources.json)
  --host HOST               Bind address (default: 127.0.0.1)
  --port PORT               Bind port (default: 8000)

figtree-news search QUERY [OPTIONS]
  --max N                   Max results (default: 20)
  --time-range RANGE        day|week|month|year (default: from sources.json)
  --categories CATS         SearXNG categories (default: news)

figtree-news ingest-feed SOURCE_ID URL     # Ingest a single feed entry
figtree-news ingest-file PATH              # Ingest a local text file
figtree-news update-trust                  # Recompute trust for all sources
figtree-news show-source-trust [SOURCE]    # Show trust scores
figtree-news lineage                       # Run narrative clustering (standalone)
figtree-news query "QUERY" [--max N]       # Embed → nearest figments → generate
figtree-news export-graph                  # Export graph as JSON
figtree-news eval                          # Per-source faithful-recall eval
figtree-news boundary-threshold            # Collect boundary similarity data for threshold tuning
figtree-news purge-derived                 # Delete derived figments (narratives, etc.)
```

## Tests

```bash
python3 -m pytest tests/ -v
```

64 tests covering all modules. CPU-only (no GPU required).

## Model

- **Local (ingestion + role extraction + summaries/brief)**: Qwen3-4B
  (unsloth bnb-4bit), ~3GB VRAM. Single-pass decode appends the role extraction
  prompt to the forward-pass cache with `enable_thinking=False`.
- **External (eval + association arbiter + legacy decompose)**: Qwen 3.6 35B
  at configurable URL. Self-correction, cluster validation, frame-shift
  verification, and brief review.

## GPU Memory Management

The system runs on a 3GB GPU with careful VRAM management:
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces fragmentation
- VRAM cleared before each article ingestion
- SearXNG queries run sequentially (not parallel) to avoid GPU spikes
- VRAM check before summaries/brief — skips if < 200MB free
- Max 10 summaries per tick to limit GPU work

## License

Research use. See parent [figtree](https://github.com/otherdrums/figtree) repo.
