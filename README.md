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

# 2. Ingest a feed (needs a GPU + the Qwen3-4B model by default)
figtree-news ingest-feed "https://example.com/rss" --source reuters

# 3. Build cross-source edges and persist adjusted trust
figtree-news update-trust

# 4. Ask, only trusting figments from sources above a credibility bar
figtree-news query "What happened at Davos?" --min-trust 0.6 --faithful

# 5. Inspect trust and export the graph
figtree-news show-source-trust
figtree-news export-graph --out graph.json
figtree-news eval --out eval_report.json
```

## Architecture

```
article (text)
   └─ ingest_text_to_figments ─► Image figment (parent)
                                   ├─ atomic figment (sentence)  source_id set
                                   ├─ atomic figment (sentence)  base_trust stamped
                                   └─ ...
   └─ trust figment created per article (about=image)

Figtree.propagate_trust(store)
   └─ edges SUPPORTS / SAME_ENTITY / CONTRADICTS from entity overlap + sentiment
   └─ adjusted_trust per source = 0.6·base + 0.4·corroborated, ×0.85 if contradicted
   └─ persisted as trust:{source_id} figment (idempotent, store-backed)
```

Each command maps to a module:

| Command | Module | What it does |
|---------|--------|--------------|
| `ingest-feed` / `ingest-file` | `figtree_news.ingest` | Articles → figments (reuses library ingest; tags `source_id`, stamps `base_trust`) |
| `update-trust` | `figtree_news.trust` | Build edges + persist adjusted per-source trust |
| `show-source-trust` | `figtree_news.trust` | Print trust report (score, corroboration, contradictions) |
| `query` | `figtree_news.query` | Embed query → nearest figments → filter by trust → generate |
| `export-graph` | `figtree_news.export` | Dump nodes+edges as JSON (no graph lib needed) |
| `eval` | `figtree_news.eval` | Per-source faithful-recall score + trust shift + contradictions |

## Source registry

`sources.json` is a map of `source_id -> {name, base_trust, url, kind}`.
`base_trust` is the outlet's starting credibility and feeds figtree's trust
model. This is the only place an editorial judgement about a source lives; the
library stays agnostic.

## Tests

CPU-only tests (no model/GPU) cover the registry, graph export, and trust
read-back:

```bash
pytest tests/
```

End-to-end ingestion/query/eval require a GPU and the reference model.

## Scope

This repo is one consumer of figtree. The library remains a general figment
substrate; a different app (legal docs, personal notes, scientific literature)
would look the same and reuse the same store + trust machinery.
