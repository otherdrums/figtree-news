# figtree-news

> **EXPERIMENTAL** — This system is under active development and subject to
> breaking changes, rapid iteration, and incomplete features. APIs, data
> formats, and config files may change without notice. Not production-ready.

A **source-aware news aggregator** built on [figtree](https://github.com/otherdrums/figtree)
(figment-based memory for language models). Articles are decomposed into
reusable semantic primitives (WHO/WHAT/WHERE/WHEN/WHY/HOW) that link across
narratives — enabling structured search, trust-aware reasoning, and faithful,
source-attributed generation.

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
                    ┌──────▼────────────────────┐
                    │  Pipeline (in-process)    │  ONE process owns the model
                    │  ─────────────────────    │  (crawl + pipeline + web)
                    │  Trust (CPU) ∥ Lineage    │
                    │  (CPU, atomic rebuild)    │
                    │  Role → narrative link    │
                    │  Summaries (GPU, ≤10)     │
                    │  World brief (GPU)        │
                    │  store.optimize()         │
                    └──────┬────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Web UI /   │  pages + JSON API + WebSocket
                    │  API        │  (viewer-only via --device none)
                    └─────────────┘
```

### Core Concepts

**Figments** are the atomic unit. An article is an "image" figment containing
child sentence figments. Role figments (WHO, WHAT, WHERE, WHEN, WHY, HOW) are
extracted **during ingest** via a single-pass decode prompt appended to the
forward-pass cache — no separate decomposition pass. The same role figment
(e.g. "Joe Biden" as WHO) reuses across multiple articles via
`sha256(f"role:{role}:{normalized}")[:16]` deduplication, creating a bipartite
graph where narrative relationships emerge from figment overlap.

**Identity nodes** resolve surface-form variants ("Donald Trump" ↔ "Trump"
↔ "DJT") into a single canonical role figment using the library's
[figtree.identity](https://github.com/otherdrums/figtree) merge engine.
`propose_identity_merges` finds same-role candidates by heuristics (boundary
similarity, token overlap, edit similarity) — no LLM arbiter needed.
`merge_role_figments()` promotes one variant to a canonical node
(`is_association=True`), rewrites all references (article/paragraph
role_figments, sentence.children, relationship edges, association edges,
dedup_obs), and deletes the variant rows. Role IDs stay canonical — no
expansion hop needed at query time.

**Multi-role intersection** finds narratives containing ALL specified roles
(e.g. WHO:Trump AND WHERE:Disney World). Results are ranked by trust, recency,
source diversity, and frame shift.

**Context materialization** assembles a provenance-preserving context package
from intersection results — source attribution, trust scores, chronological
ordering, frame information — ready for faithful model-native generation.

**Boundary vectors** (~10KB float32) are captured from the model's hidden state
during ingestion. They enable similarity search, dedup, and frame-shift
detection without re-running the model.

**Narrative clustering** groups articles by shared role figments. Two articles
share a narrative if they share >= 2 role figment IDs — exact semantic matching
regardless of how different outlets phrase headlines. Lineage rebuild is
**atomic**: new narrative figments are upserted before stale ones are deleted,
so a crash can never leave the store with zero narratives.

**Source trust** — each source has an immutable `base_trust`; `update_trust`
propagates corroboration/contradiction into persisted `trust:{source_id}`
figments, shown as the trust board on the front page.

## Install

```bash
pip install -e .   # from figtree-news/ root (also install figtree first)
```

## Quick Start

```bash
# 1. Configure sources (edit demo/sources.json — feeds, trust, crawler state)
cp demo/sources.json ./sources.json

# 2. Serve the web newspaper (auto-starts continuous crawl; ONE process owns the model)
figtree-news serve --db news.lance --sources demo/sources.json --host 0.0.0.0 --port 8000
figtree-news serve --device none    # viewer-only: read the store, never load a model

# 3. Standalone single-tick crawl + pipeline
figtree-news crawl --once --max-articles 15

# 4. Standalone search
figtree-news search "AI regulation" --time-range day --max 10

# 5. One systemd service (replaces the old crawler+web pair)
./systemd/install_systemd.sh
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

### Crawl Control Panel

| Control | Default | Description |
|---------|---------|-------------|
| Max articles | 12 | Cap per tick |
| Max stories | 0 (unlimited) | Cap narratives per pipeline run |
| Pause between ticks | 900 | Seconds between ticks (continuous mode) |
| Compute KV cache | off | Cache K/V for boundary-based generation |
| Smart crawl | on | Auto-switch: forward when new articles found, backward when stuck |
| **Web Search (SearXNG)** | | |
| Enable web search | off | Toggle SearXNG search (demo ships disabled) |
| Time range | Day | day / week / month / year / anytime |
| Categories | News | news / general / general,news |
| **Search queries** | **auto** | **Keywords extracted from RSS article titles/text each tick** |

### Web Search (SearXNG)

Articles from across the web via a local [SearXNG](https://docs.searxng.org/)
instance. Full text fetched via trafilatura, deduplicated by URL + title.
Unknown domains auto-registered as sources with `base_trust=0.7`.

**Queries are auto-derived**: each crawl tick extracts top keywords from newly
ingested RSS article titles/summaries (with built-in stopword filtering) and
uses those as SearXNG queries — no manual query maintenance needed. SearXNG
queries run sequentially with a VRAM guard (< 500MB free stops them).

### Role Extraction

Role figments are extracted **during the ingest forward pass**, not in a
separate stage. The prompt is a few-shot canonicalization instruction with
`SINGLE entity only` constraint; it is appended to the forward-pass cache
via `past_key_values` reuse and decoded with greedy sampling. The model's
chain-of-thought reasoning is suppressed via `build_prompt_ids(enable_thinking=False)`.

### Identity Merging (figtree.identity)

`figtree_news/associations.py` thin-wraps the library's `figtree.identity`:
- `propose_associations()` — same-role variant pairs scored by boundary cosine,
  token overlap, edit similarity, co-occurrence
- `assert_association()` — creates the `edge_type="association"` figment
- `merge_role_figments()` — canonical-node promotion + rewrite of all six
  reference locations, then deletes the variant rows
- `expand_associations()` / `get_association_groups()` — query-time traversal

### Faithful Per-Source Eval

`eval.py` measures per-source recall (every figure of a source reproduced in a
generated summary) via `FigmentGenerator.generate_faithful`, with a
`recall_score` and `missing_atoms` report (`figtree-news eval`).

## Pipeline

```
crawl → ingest (single-pass: boundaries + roles, FTS index) →
pipeline: trust ∥ lineage → role→narrative link → summaries → brief → compaction
```

1. **Crawl**: RSS feeds + seed URLs in parallel (asyncio.gather, 300s timeout),
   then SearXNG queries sequentially (VRAM-guarded). URL + title dedup.
2. **Ingest**: One forward pass → boundaries + role extraction via appended
   decode prompt (local model, `enable_thinking=False`). Article stamped with
   `role_figments` and `decomposed=True`. FTS index updated with the derived
   `{db}_fts.db` path.
3. **Phase 1 — Trust + Lineage (parallel, CPU)**: `update_trust` (idempotent,
   store-persisted) ∥ `compute_lineage` (role clustering >= 2 shared IDs,
   frame shift, derivative edges, atomic rebuild).
4. **Phase 2 — Role linking**: `assign_roles_to_narratives` on the FRESH
   lineage output (single snapshot, no stale store re-read).
5. **Phase 3 — Summaries**: Per-article summaries (local model, max 10 per
   tick), VRAM gate (< 200MB free skips GPU phases).
6. **Phase 4 — World brief**: top-8 narrative brief (local model).
7. **Phase 5 — Compaction**: `store.optimize()` + `_force_free()`.

## Architecture

```
figtree_news/
├── cli.py              # Typer CLI: crawl, serve, search, query, lineage, export, eval, boundary-threshold
├── config.py           # SourceRegistry + CrawlerState + SourceConfig dataclasses
├── searxng.py          # SearXNG client + article extraction (SearxngConfig lives here)
├── ingest.py           # Feed/article → figments with provenance (+ single-pass role extraction, FTS index)
├── crawler.py          # Crawler: feeds + SearXNG + BFS link-follower (thread-safe)
├── pipeline.py         # In-process pipeline: trust + lineage (parallel) → summaries → brief → compaction
├── lineage.py          # Role figment clustering (≥2 shared IDs) + frame shift + derivatives (atomic)
├── trust.py            # Source trust propagation (idempotent, store-persisted)
├── normalize.py        # Shared entity-normalization (honorific-stripping, lowercase, punctuation-removal)
├── summarize_news.py   # Per-article summaries + world brief
├── query.py            # Embed query → nearest figments → faithful generate
├── search_index.py     # SQLite FTS5 full-text search (thread-safe)
├── associations.py     # Thin wrapper over figtree.identity (co-reference merge engine)
├── intersection.py     # Multi-role intersection query: find narratives containing all specified roles
├── context.py          # Context materialization: assemble structured context packages from narratives
├── export.py           # Graph export as JSON
├── eval.py             # Per-source faithful-recall eval
├── model_lock.py       # Shared RLock for GPU model forward passes (serializes all GPU work)
└── web/
    ├── serve.py        # FastAPI: ONE process owns model + crawl + pipeline + web (--device none = viewer)
    ├── templates/      # Jinja2 HTML templates
    └── static/         # CSS + JS (dark theme)
```

The external-LLM stack (decompose/evaluate/correct/cogitate/association_worker/
llm_config) was **removed in Phase 2** — nothing ran end-to-end (the 35B
decompose pass took 4.3h per tick and every run OOM-killed within ~20 min).
Everything now runs on the local Qwen3-4B in one process.

### Source Registry

`sources.json` maps `source_id → {name, base_trust, url, kind, logo_url}`.
Demo ships with 15 feeds (12 RSS + 3 video: dwnews_yt, france24_yt,
pbsnewshour) + SearXNG config (disabled by default). Unknown domains
auto-register at runtime with `base_trust=0.7`.

### Data Storage

- **LanceDB** — all figments (articles, narratives, edges, trust, role figments)
- **SQLite FTS5** — full-text search index (`{db}_fts.db`)
- **seen_urls.json** — URL dedup (runtime, gitignored)
- **KV cache** (optional) — quantized K/V for boundary-based generation
- **sources.json** — feeds, seeds, SearXNG config, crawler state

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
  --device MODE             auto|cpu|none (none = viewer-only, no model)

figtree-news search QUERY [OPTIONS]
  --max N                   Max results (default: 20)
  --time-range RANGE        day|week|month|year (default: from sources.json)
  --categories CATS         SearXNG categories (default: news)

figtree-news ingest-feed SOURCE_ID URL     # Ingest a single feed entry
figtree-news ingest-file PATH              # Ingest a local article file
figtree-news update-trust                  # Recompute trust for all sources
figtree-news show-source-trust             # Show trust scores
figtree-news lineage                       # Run narrative clustering (standalone)
figtree-news query "QUERY" [--max N]       # Embed → nearest figments → generate
figtree-news export-graph                  # Export graph as JSON
figtree-news eval                          # Per-source faithful-recall eval
figtree-news boundary-threshold            # Boundary similarity threshold tuning (data collection)
```

## Tests

```bash
python3 -m pytest tests/ -v
```

35 tests covering all modules. CPU-only (no GPU required). Includes the
atomic-lineage crash regression and viewer-mode (device=none) tests.

## Model

**Local (ingestion + role extraction + summaries/brief)**: Qwen3-4B
(unsloth bnb-4bit), ~2.4GB VRAM on a 4GB Quadro T1000. Single-pass decode
appends the role extraction prompt to the forward-pass cache with
`enable_thinking=False`. One process owns the model — no separate crawler
process (the old dual-loop OOM-killed the box every ~20 minutes).

## GPU Memory Management

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128` reduces fragmentation
- VRAM cleared before each article ingestion
- SearXNG queries run sequentially (not parallel) with a 500MB-free VRAM guard
- VRAM check before summaries/brief — skips if < 200MB free
- Max 10 summaries per tick; `_force_free()` after the pipeline
- Verified: 3 consecutive ticks at RSS < 2.5GB, VRAM ~2.8GB, no OOM

## License

Research use. See parent [figtree](https://github.com/otherdrums/figtree) repo.
