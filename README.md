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

The system treats news as a graph of reusable **figments** rather than
independent documents. Here's the data flow:

```
                    ┌─────────────┐
                    │  RSS Feeds  │
                    │  SearXNG    │
                    │  Seed URLs  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Crawl &   │  Fetch, extract text/images, dedup
                    │   Ingest    │  URL + title dedup, robots.txt
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Figmentize │  Article → sentences → atomic figments
                    │  (boundary) │  Each sentence gets a boundary vector
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Decompose   │  WHO/WHAT/WHERE/WHEN/WHY/HOW role figments
                    │ (external   │  via external LLM, 3 parallel workers
                    │  LLM)       │
                    └──────┬──────┘
                           │
               ┌───────────┼───────────┐
               │                       │
        ┌──────▼──────┐         ┌─────▼─────┐
        │   Trust     │         │ Lineage   │
        │             │         │           │
        └──────┬──────┘         └─────┬─────┘
               │                      │
               │    ┌─────────────────▼─────┐
               └───►│    Narratives         │
                    │    (role figment      │
                    │     clustering)       │
                    └──────────┬────────────┘
                               │
               ┌───────────────┼───────────────┐
               │               │               │
        ┌──────▼──────┐ ┌─────▼─────┐  ┌──────▼──────┐
        │  Evaluate   │ │ Summaries │  │   Brief     │
        │  (LLM)      │ │ (local    │  │   (local    │
        │             │ │  model)   │  │    model)   │
        └──────┬──────┘ └───────────┘  └─────────────┘
               │
        ┌──────▼──────────────────────┐
        │     Web UI / API            │
        └─────────────────────────────┘
```

### Core Concepts

**Figments** are the atomic unit. An article is an "image" figment containing
child sentence figments. Each sentence can be decomposed into role figments
(WHO, WHAT, WHERE, WHEN, WHY, HOW). The same role figment (e.g. "Joe Biden" as
WHO) reuses across multiple articles via `hash(role + normalized_text)`
deduplication, creating a bipartite graph where narrative relationships emerge
from figment overlap.

**Association figments** link surface-form variants of the same entity
("Donald Trump" ↔ "Trump" ↔ "DJT") so co-reference does not depend on
perfect LLM normalization during decomposition. Associations are first-class,
traversable figments. Auto-proposals fire at decompose time based on
boundary similarity + string overlap + co-occurrence. Expansion walks
association edges at query time (bounded, default 2 hops).

**Multi-role intersection** finds narratives containing ALL specified roles
(e.g. WHO:Trump AND WHERE:Disney World), expanding through associations
first so variant forms resolve to the same entity. Results are ranked by
trust, recency, source diversity, and frame shift.

**Context materialization** assembles a provenance-preserving context package
from intersection results — source attribution, trust scores, chronological
ordering, frame information — ready for faithful model-native generation.

**Boundary vectors** (~10KB float32) are captured from the model's hidden state
during ingestion. They enable similarity search, dedup, and frame-shift
detection without re-running the model.

**Narrative clustering** groups articles by shared role figments. Two articles
share a narrative if they share >= 2 role figment IDs — exact semantic matching
regardless of how different outlets phrase headlines. Falls back to boundary
similarity (cosine > 0.95 within 48h) when the external LLM is not configured.

**Self-correction** — an external LLM (Qwen 3.6 35B) reviews narrative clusters,
flags miscategorized articles, and suggests corrections. Corrections accumulate
across eval runs and auto-apply at a configurable confirmation threshold.

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

### Decomposition Engine

Background extraction of WHO/WHAT/WHERE/WHEN/WHY/HOW role figments from
each sentence, using 3 parallel workers and the external LLM. Runs as
Phase 1 of the pipeline (before clustering), so narratives are built from
the semantic role graph rather than text heuristics.

### Cogitation Engine

Periodic insight generation (default 30min interval):
1. Duplicate merging — merge semantically similar role figments
2. Relationship discovery — co-occurrence patterns
3. Insight generation — LLM-generated landscape insights

### LLM Evaluation & Self-Correction

External LLM (Qwen 3.6 35B) validates pipeline output:
- **Cluster validation** — flags miscategorized articles in narratives
- **Frame shift verification** — confirms genuine framing divergence
- **Brief review** — critiques world brief for accuracy
- **Self-correction** — corrections accumulate, auto-apply at threshold (default 2)

## Pipeline

```
crawl → ingest → decompose (external LLM) → trust → lineage (role figment clustering) → summaries → brief → eval
```

1. **Crawl**: RSS feeds + SearXNG search + bounded link-follower (URL dedup, robots.txt)
2. **Ingest**: Articles → figments (sentence-level + image + video), VRAM cleared per article
3. **Decompose**: WHO/WHAT/WHERE/WHEN/WHY/HOW role figments (external LLM, 3 workers)
4. **Trust**: Source trust propagation (idempotent, store-persisted)
5. **Lineage**: Narrative clustering via role figment overlap + frame shift detection
6. **Summaries**: Per-article summaries (local model, max 10 per tick)
7. **Brief**: World brief (2-3 sentences, local model)
8. **Eval**: LLM cluster review + frame shift check + self-correction

## Architecture

```
figtree_news/
├── cli.py              # Typer CLI: crawl, serve, search, query, lineage, export, eval
├── config.py           # SourceRegistry: source config + SearXNG + LLM settings
├── searxng.py          # SearXNG client + article extraction
├── ingest.py           # Feed/article → figments with provenance
├── crawler.py          # Crawler: feeds + SearXNG + BFS link-follower (thread-safe)
├── pipeline.py         # Parallel pipeline orchestration (ThreadPoolExecutor)
├── lineage.py          # Role figment clustering + frame shift + derivatives (expands through associations)
├── trust.py            # Source trust propagation
├── decompose.py        # WHO/WHAT/WHERE/WHEN/WHY/HOW extraction + inline cogitation + auto-associations
├── cogitate.py         # Periodic insight generation
├── evaluate.py         # External LLM evaluation: clusters, frame shift, brief
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
    ├── serve.py         # FastAPI app: HTML pages + JSON API + WebSocket + auto-crawl
    ├── templates/       # Jinja2 HTML templates
    └── static/          # CSS + JS (dark theme, extracted app.js)
```

### Source Registry

`sources.json` maps `source_id → {name, base_trust, url, kind, logo_url}`.
Demo ships with 15 sources (7 RSS + 8 YouTube) + SearXNG + external LLM config.

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
  --interval N              Seconds between ticks (default: 300)
  --max-articles N          Cap articles per tick (default: 15)
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
```

## Tests

```bash
python3 -m pytest tests/ -v
```

All tests run CPU-only (no GPU required).

## Model

- **Local (ingestion/summaries)**: Qwen3-4B (unsloth bnb-4bit), ~3GB VRAM
- **External (eval/decomposition)**: Qwen 3.6 35B at configurable URL

## GPU Memory Management

The system runs on a 3GB GPU with careful VRAM management:
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces fragmentation
- VRAM cleared before each article ingestion
- SearXNG queries run sequentially (not parallel) to avoid GPU spikes
- VRAM check before summaries/brief — skips if < 200MB free
- Max 10 summaries per tick to limit GPU work

## License

Research use. See parent [figtree](https://github.com/otherdrums/figtree) repo.
