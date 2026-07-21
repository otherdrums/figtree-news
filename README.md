# figtree-news

A **source-aware news aggregator** built on [figtree](https://github.com/otherdrums/figtree)
(figment-based memory for language models). It ingests articles from RSS/Atom
feeds or local files, stores them as *figments* (sentences → atomic figments
under an article *image*), and uses figtree's source-based trust model to track
which outlets corroborate or contradict each other — then answers questions only
from figments whose source clears a credibility bar.

This repo is an **application** on top of the figtree *library*. No news-specific
logic lives in the core library; everything here is composition of the public API
(`Figment`, `ingest_text_to_figments`, `FigmentGenerator`, `Figtree`,
`FigmentStore`, `connect`, `load_model`, `recall_score`).

## Install

`figtree` is not on PyPI yet — install it from its repo first, then this package:

```bash
# from the figtree repo root
pip install -e .

# from this repo root
pip install -e .        # installs figtree-news + pulls in figtree's deps
# (or) pip install -r requirements.txt   # after figtree is installed
```

Runtime deps: `typer`, `feedparser` (for feeds), `httpx` (for crawling),
`trafilatura` (for article extraction), and transitively `torch`,
`transformers`, `lancedb` (via figtree).

## Quick start

```bash
# 1. Point a source registry at your outlets (initial trust per source)
cp demo/sources.json ./sources.json
#    edit base_trust: e.g. reuters 0.9, a random blog 0.5
#    add logo_url for branding in the web UI

# 2. Crawl the web continuously (feeds + bounded link-follower). Needs a GPU.
figtree-news crawl \
  --feed reuters="https://example.com/rss" \
  --seed "https://news.example.org/" \
  --loop --interval 3600

# 3. Deep initial crawl (backfill mode, 200 article cap)
figtree-news crawl --backfill --once

# 4. Serve the interactive web newspaper (FastAPI)
figtree-news serve --port 8000
#   open http://127.0.0.1:8000  (front page, per-source, per-narrative, search, lineage)

# 5. Ask, only trusting figments from sources above a credibility bar
figtree-news query "What happened at Davos?" --min-trust 0.6 --faithful

# 6. Inspect trust, lineage, and export the graph
figtree-news show-source-trust
figtree-news lineage
figtree-news export-graph --out graph.json
```

## Features

### Web Newspaper (FastAPI)

- **Front page** with world brief, newspaper-style narrative comparison cards
  (headline + source badges + lead paragraph + dates + article images)
- **Source logos** displayed next to source badges throughout the UI
- **Article images** from RSS `media_content`/`media_thumbnail` and `og:image` extraction
- **Author bylines** on articles where available
- **Full-text search** (SQLite FTS5) with date range filters and BM25 ranking
- **Slide-out control panel** for managing feeds, crawl settings, and actions
- **Live stats bar** (articles, stories, sources, brief, UTC clock, crawler status)
- **WebSocket progress** for real-time crawl status updates
- **Responsive design** with benthic.io dark theme

### Media & Branding

- **Article images** extracted from RSS feeds (`media_content`, `media_thumbnail`, `enclosures`) and web pages (`og:image`, `twitter:image` via trafilatura + regex fallback)
- **Source logos** configured per-source in `sources.json` via `logo_url` field
- Images displayed in narrative cards, article pages, feed items, and source pages
- Graceful fallback: images fail silently with `onerror="this.parentElement.style.display='none'"`

### Trust & Lineage

- Per-source `base_trust` in `sources.json` → figtree trust propagation
- **Narrative clustering** via entity overlap (Jaccard ≥ 0.25)
- **First reporter detection** (earliest published/first_seen)
- **Derivative edge detection** (echo chains between outlets)
- **Frame shift detection** — cosine boundary similarity < 0.85 between newest and first article marks a story as framing-shifted
- **Contradiction awareness** across sources

### Pipeline

```
crawl → ingest → trust → lineage → summaries → world brief
  │        │       │         │          │            └─ /api/stats
  │        │       │         │          └─ per-article summaries
  │        │       │         └─ narrative + derivative + frame shift
  │        │       └─ source trust propagation (idempotent, store-persisted)
  │        └─ articles → figments (sentence-level + image)
  └─ RSS feeds + bounded link-follower (URL dedup, robots.txt)
```

## Architecture

```
figtree_news/
├── config.py              # SourceRegistry: source_id → {name, base_trust, url, logo_url}
├── ingest.py              # Feed/article → figments with full provenance + image extraction
├── crawler.py             # Continuous web crawler: feeds + BFS link-follower
├── pipeline.py            # Orchestration: trust → lineage → summaries → brief
├── lineage.py             # Narrative clustering, frame shift detection, derivative edges
├── trust.py               # Source trust propagation
├── summarize_news.py      # Per-article summaries + world brief
├── query.py               # Embed query → nearest figments → generate
├── search_index.py        # SQLite FTS5 full-text search index
├── eval.py                # Per-source faithful-recall eval
├── export.py              # Graph export as JSON
├── cli.py                 # Typer CLI: crawl, serve, query, lineage, export, eval
└── web/
    ├── serve.py            # FastAPI app: HTML pages + JSON API + WebSocket
    ├── templates/          # Jinja2: base, index, article, source, narrative, lineage, search
    └── static/style.css    # benthic.io dark theme
```

### Source registry

`sources.json` is a map of `source_id -> {name, base_trust, url, kind, logo_url}`.

`base_trust` is the outlet's starting credibility and feeds figtree's trust
model. `logo_url` is the source's brand logo (displayed in the web UI).

```json
{
  "bbc": {
    "name": "BBC News",
    "base_trust": 0.9,
    "url": "https://www.bbc.com/news",
    "kind": "news",
    "logo_url": "https://static.files.bbci.co.uk/core/website/assets/static/img/bbc-news/bbc-news-wordmark-black.png"
  },
  "feeds": { "bbc": "http://feeds.bbci.co.uk/news/rss.xml" },
  "seeds": []
}
```

### Data storage

- **LanceDB** — all figments (articles, narratives, edges, trust assertions)
- **SQLite FTS5** — full-text search index (auto-created at `{db}.replace('.lance', '_fts.db')`)
- **seen_urls.json** — URL dedup for crawl idempotency
- **KV cache** (optional) — quantized K/V blobs for boundary-based generation

### Web API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Front page (brief, narratives, articles, trust board) |
| `/article/{fid}` | GET | Article detail page |
| `/source/{sid}` | GET | Source page with articles and narratives |
| `/narrative/{nid}` | GET | Narrative detail with all source versions |
| `/lineage` | GET | First-reporter / echo-chain view |
| `/search` | GET | Full-text search with date range filters |
| `/api/search?q=&range=&sort=&page=` | GET | JSON search results (BM25) |
| `/api/stats` | GET | Store stats (articles, narratives, sources, brief) |
| `/api/narratives` | GET | JSON list of all narratives |
| `/api/sources` | GET | JSON source trust agenda |
| `/api/crawl/status` | GET | Crawler state |
| `/api/crawl/run` | POST | Start crawl (continuous or once) |
| `/api/crawl/stop` | POST | Stop continuous crawl |
| `/api/pipeline/run` | POST | Run trust → lineage → summaries |
| `/ws` | WebSocket | Live crawl status updates |

## Tests

CPU-only tests (no model/GPU) cover the registry, graph export, trust
read-back, the crawler (URL dedup + bounded traversal with mocked fetch),
lineage (first-reporter + derivative detection, idempotency), and the FastAPI
web app (page render + JSON API via `TestClient`):

```bash
pytest tests/
```

## Deployment with systemd (no containers)

Two services share the same LanceDB store: a continuous **crawler** (GPU) and
the **web** viewer (CPU).

```bash
# from the repo root
./systemd/install_systemd.sh            # user services (~/.config/systemd/user)
# or, as root, for a system-wide service:
./systemd/install_systemd.sh --system   # writes to /etc/systemd/system

# then:
systemctl --user enable --now figtree-news-crawler figtree-news-web
# (for user services, also: loginctl enable-linger "$USER"
#  so they keep running without an active login session)
```

* The crawler runs continuously by default (`figtree-news crawl` == `--loop`);
  pass `--once` for a single tick.
* The web service binds `127.0.0.1:8000`; put nginx/caddy in front to expose
  it. It is CPU-only and reads the shared store, so the crawler can hold the
  GPU without affecting browsing.
* Both services restart automatically on failure (`Restart=on-failure`).

## Model

Default: Qwen3-4B (unsloth bnb-4bit, cached at ~/.cache/huggingface/hub/)
GPU: Quadro T1000 (3GB VRAM)

## Scope

This repo is one consumer of figtree. The library remains a general figment
substrate; a different app (legal docs, personal notes, scientific literature)
would look the same and reuse the same store + trust machinery.
