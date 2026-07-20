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

Runtime deps: `typer`, `feedparser` (for feeds), and transitively `torch`,
`transformers`, `lancedb` (via figtree).

## Quick start

```bash
# 1. Point a source registry at your outlets (initial trust per source)
cp examples/sample_sources.json ./sources.json
#    edit base_trust: e.g. reuters 0.9, a random blog 0.5

# 2. Crawl the web continuously (feeds + bounded link-follower). Needs a GPU.
figtree-news crawl \
  --feed reuters="https://example.com/rss" \
  --seed "https://news.example.org/" \
  --loop --interval 3600

# 3. Inspect trust, lineage, and export the graph
figtree-news show-source-trust
figtree-news lineage
figtree-news export-graph --out graph.json

# 4. Serve the interactive web newspaper (FastAPI)
figtree-news serve --port 8000
#   open http://127.0.0.1:8000  (front page, per-source, per-narrative, lineage)

# 5. Ask, only trusting figments from sources above a credibility bar
figtree-news query "What happened at Davos?" --min-trust 0.6 --faithful

# 6. Offline snapshot / eval
figtree-news build-newspaper --out newspaper.json
figtree-news eval --out eval_report.json
```

## Architecture

```
article (text)  + provenance (url, published, first_seen, title)
   └─ ingest_text_to_figments ─► Image figment (parent)
                                    ├─ atomic figment (sentence)  source_id set
                                    ├─ atomic figment (sentence)  base_trust stamped
                                    └─ ...

crawler (GPU)  ──►  trust ──►  lineage ──►  summaries/brief  ──►  LanceDB store
   feeds + bounded link-follower, URL-deduped, robots-respecting

serve (FastAPI, CPU)  reads the store and renders the newspaper:
   front page · per-article · per-source · per-narrative · lineage view
```

### Provenance & lineage (the "who broke it first" engine)

Every figment carries its `url`, `published` date, and `first_seen` timestamp, so
the newspaper always links back to the original outlet. After each crawl tick,
`figtree_news.lineage` clusters articles by shared entities and, per cluster:

* marks the **first reporter** (earliest `published`/`first_seen`);
* marks later articles covering the same story as **derivative** of it (an echo
  chain), persisted as `derivative:{orig}:{der}` edge figments;
* groups the cluster into a **narrative** figment listing the outlets involved,
  the first reporter, and the stance lean.

`figtree_news.pipeline.run_pipeline` ties it together: trust → lineage →
per-article summaries → a single "world brief", all pre-generated on the
crawler so the web viewer needs no GPU.

Each command maps to a module:

| Command | Module | What it does |
|---------|--------|--------------|
| `ingest-feed` / `ingest-file` | `figtree_news.ingest` | Articles → figments (reuses library ingest; tags `source_id`, stamps `base_trust`, stamps provenance) |
| `crawl` | `figtree_news.crawler` + `pipeline` | Feeds + bounded link-follower → ingest → trust → lineage → summaries → world brief |
| `update-trust` / `show-source-trust` | `figtree_news.trust` | Build edges + persist/show adjusted per-source trust |
| `lineage` | `figtree_news.lineage` | Compute/persist narrative + derivative (first-reporter) lineage — CPU only |
| `query` | `figtree_news.query` | Embed query → nearest figments → filter by trust → generate |
| `export-graph` | `figtree_news.export` | Dump nodes+edges as JSON (no graph lib needed) |
| `serve` | `figtree_news.web.serve` | Interactive FastAPI newspaper (front page, article, source, narrative, lineage, JSON API) |
| `build-newspaper` | `figtree_news.cli` | Dump a static front-page snapshot as JSON |
| `eval` | `figtree_news.eval` | Per-source faithful-recall score + trust shift + contradictions |

## Source registry

`sources.json` is a map of `source_id -> {name, base_trust, url, kind}`.
`base_trust` is the outlet's starting credibility and feeds figtree's trust
model. This is the only place an editorial judgement about a source lives; the
library stays agnostic.

## Tests

CPU-only tests (no model/GPU) cover the registry, graph export, trust
read-back, the crawler (URL dedup + bounded traversal with mocked fetch),
lineage (first-reporter + derivative detection, idempotency), and the FastAPI
web app (page render + JSON API via `TestClient`):

```bash
pytest tests/
```

End-to-end crawl/ingest/summarize/query/eval require a GPU and the reference
model. For 24/7 operation on a home server or cloud box, run the `crawl --loop`
process (GPU) and the `serve` process (CPU) under a process manager
(systemd / docker-compose / pm2); both share the same LanceDB store.

## Scope

This repo is one consumer of figtree. The library remains a general figment
substrate; a different app (legal docs, personal notes, scientific literature)
would look the same and reuse the same store + trust machinery.
