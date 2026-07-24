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
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐ ┌──▼────┐ ┌─────▼─────┐
       │ Decompose   │ │ Trust │ │ Lineage   │
       │ (background)│ │       │ │           │
       └──────┬──────┘ └──┬────┘ └─────┬─────┘
              │           │            │
              │    ┌──────▼──────┐     │
              │    │  Narratives │─────┘
              │    │  (clusters) │  Entity overlap → story clusters
              │    └──────┬──────┘
              │           │
       ┌──────▼──────┐ ┌──▼──────┐
       │  Evaluate   │ │ Brief   │
       │  (LLM)      │ │ (LLM)  │
       └──────┬──────┘ └──┬──────┘
              │           │
       ┌──────▼───────────▼──────┐
       │     Web UI / API        │
       └─────────────────────────┘
```

### Core Concepts

**Figments** are the atomic unit. An article is an "image" figment containing
child sentence figments. Each sentence can be decomposed into role figments
(WHO, WHAT, WHERE, WHEN, WHY, HOW). The same role figment (e.g. "Trump" as WHO)
reuses across multiple articles, creating a bipartite graph where narrative
relationships emerge from figment overlap.

**Boundary vectors** (~10KB float32) are captured from the model's hidden state
during ingestion. They enable similarity search, dedup, and frame-shift
detection without re-running the model.

**Narrative clustering** groups articles by entity overlap (Jaccard >= 0.30 with
>= 2 shared entities from titles). Each cluster becomes a "story" with first-
reporter detection, derivative echo chains, and cross-source trust scores.

**Self-correction** — an external LLM (Qwen 3.6 35B) reviews narrative clusters,
flags miscategorized articles, and suggests corrections. Corrections accumulate
across eval runs and auto-apply at a configurable confirmation threshold.

## Install

```bash
pip install -e .   # from figtree-news/ root (also install figtree first)
```

## Quick Start

```bash
# 1. Configure sources
cp demo/sources.json ./sources.json

# 2. Serve the web newspaper (includes background crawler)
figtree-news serve --db demo/news.lance --sources demo/sources.json --host 0.0.0.0 --port 8000

# 3. Standalone crawl (CLI)
figtree-news crawl --interval 0 --max-articles 40

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
- Slide-out control panel (600px) for all crawl + search settings
- WebSocket live updates — page auto-refreshes on new content
- Dark theme (benthic.io style)

### Crawl Control Panel

| Control | Default | Description |
|---------|---------|-------------|
| Max articles | 40 | Cap per tick |
| Max stories | 0 (unlimited) | Cap narratives per pipeline run |
| Since / Before | (empty) | Date range filter |
| Pause between ticks | 0 | Seconds between ticks (0 = continuous) |
| Compute KV cache | off | Cache K/V for boundary-based generation |
| Generate summaries | on | Per-article summaries + world brief |
| Enable LLM Review | off | External LLM cluster validation + self-correction |
| **Web Search (SearXNG)** | | |
| Enable web search | on | Toggle SearXNG search |
| Time range | Last day | day / week / month / year / anytime |
| Categories | News | news / general / general,news |
| Search queries | (from sources.json) | One per line; results ingested per tick |

### Web Search (SearXNG)

Articles from across the web via a local [SearXNG](https://docs.searxng.org/)
instance. Full text fetched via trafilatura, deduplicated by URL + title.
Unknown domains auto-registered as sources with `base_trust=0.7`.

### Decomposition Engine

Background extraction of WHO/WHAT/WHERE/WHEN/WHY/HOW role figments from
each sentence, using 3 parallel workers and the external LLM. Enables
structured search: "who was involved in X?" → find all WHO figments.

### Cogitation Engine

Periodic "dreaming" phase (default 6h interval):
1. Duplicate merging — merge semantically similar figments
2. Relationship discovery — co-occurrence patterns
3. Insight generation — LLM-generated landscape insights

### LLM Evaluation & Self-Correction

External LLM (Qwen 3.6 35B) validates pipeline output:
- **Cluster validation** — flags miscategorized articles in narratives
- **Frame shift verification** — confirms genuine framing divergence
- **Brief review** — critiques world brief for accuracy
- **Self-correction** — corrections accumulate, auto-apply at threshold (default 2)

## Pipeline (8 phases)

```
crawl → ingest → decompose → trust → lineage → evaluate → summaries → brief
```

1. **Crawl**: RSS feeds + SearXNG search + bounded link-follower (URL dedup, robots.txt)
2. **Ingest**: Articles → figments (sentence-level + image + video)
3. **Decompose**: WHO/WHAT/WHERE/WHEN/WHY/HOW role figments (background, 3 workers)
4. **Trust**: Source trust propagation (idempotent, store-persisted)
5. **Lineage**: Narrative clustering via entity overlap + frame shift detection
6. **Evaluate**: LLM cluster review + frame shift check
7. **Summaries**: Per-article summaries
8. **Brief**: World brief (2-3 sentences)

## Architecture

```
figtree_news/
├── cli.py              # Typer CLI: crawl, serve, search, query, lineage, export, eval
├── config.py           # SourceRegistry: source config + SearXNG + LLM settings
├── searxng.py          # SearXNG client + article extraction
├── ingest.py           # Feed/article → figments with provenance
├── crawler.py          # Crawler: feeds + SearXNG + BFS link-follower
├── pipeline.py         # 8-phase pipeline orchestration
├── lineage.py          # Narrative clustering + frame shift + derivative edges
├── trust.py            # Source trust propagation
├── decompose.py        # WHO/WHAT/WHERE/WHEN/WHY/HOW extraction (3 workers)
├── cogitate.py         # Background consolidation + insights
├── evaluate.py         # External LLM evaluation: clusters, frame shift, brief
├── correct.py          # Self-correction: confirmation threshold + auto-apply
├── llm_config.py       # External LLM configuration
├── summarize_news.py   # Per-article summaries + world brief
├── query.py            # Embed query → nearest figments → generate
├── search_index.py     # SQLite FTS5 full-text search
├── eval.py             # Per-source faithful-recall eval
├── export.py           # Graph export as JSON
└── web/
    ├── serve.py         # FastAPI app: HTML pages + JSON API + WebSocket
    ├── templates/       # Jinja2 HTML templates
    └── static/          # CSS (dark theme)
```

### Source Registry

`sources.json` maps `source_id → {name, base_trust, url, kind, logo_url}`.
Demo ships with 15 sources (7 RSS + 8 YouTube).

### Data Storage

- **LanceDB** — all figments (articles, narratives, edges, trust, role figments)
- **SQLite FTS5** — full-text search index (`{db}_fts.db`)
- **seen_urls.json** — URL dedup (runtime, gitignored)
- **KV cache** (optional) — quantized K/V for boundary-based generation

## CLI Reference

```bash
figtree-news crawl [OPTIONS]
  --feed source=url         Add feed (repeatable)
  --seed url                Add seed URL (repeatable)
  --interval N              Seconds between ticks, 0 = continuous (default: 0)
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
```

## Tests

```bash
python3 -m pytest tests/ -v
```

All tests run CPU-only (no GPU required).

## Model

- **Local (ingestion/summaries)**: Qwen3-4B (unsloth bnb-4bit), ~3GB VRAM
- **External (eval/decomposition)**: Qwen 3.6 35B at configurable URL

## License

Research use. See parent [figtree](https://github.com/otherdrums/figtree) repo.
