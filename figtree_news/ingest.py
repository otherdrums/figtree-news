"""Feed ingestion: turn articles into Figments.

Each article becomes an Image Figment (parent) whose children are the
sentence-level figments produced by ``figtree.ingest_text_to_figments``. The
library already tags every figment with ``source_id`` and stamps the image
with ``base_trust``, so the source's initial credibility flows into Figtree's
trust propagation without any news-specific code living in the core library.
"""

from __future__ import annotations

import json
from typing import Any

from figtree import FigmentStore, ingest_text_to_figments

from .config import SourceRegistry


def _read_feed(uri: str, source_id: str) -> list[dict[str, Any]]:
    """Parse an RSS/Atom feed (URL or local file) into article dicts.

    Requires ``feedparser`` (optional dependency). Falls back to parsing the
    URI as a JSON/JSONL file of article dicts if feedparser is unavailable or
    the content is not a feed.
    """
    try:
        import feedparser  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "feed ingestion requires feedparser: pip install feedparser"
        ) from exc

    parsed = feedparser.parse(uri)
    articles: list[dict[str, Any]] = []
    for entry in getattr(parsed, "entries", []):
        text = (entry.get("summary") or "") + "\n\n" + (entry.get("content") or "")
        if isinstance(entry.get("content"), list):
            text = (entry.get("summary") or "") + "\n\n" + " ".join(
                c.get("value", "") for c in entry["content"]
            )
        if not text.strip():
            continue
        articles.append(
            {
                "source_id": source_id,
                "text": text.strip(),
                "url": entry.get("link"),
                "author": entry.get("author"),
                "published": entry.get("published"),
            }
        )
    return articles


def _read_article_file(path: str) -> list[dict[str, Any]]:
    """Load articles from a JSON list or JSONL file (one article dict per line)."""
    with open(path, "r", encoding="utf-8") as fh:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in fh if line.strip()]
        data = json.load(fh)
    if isinstance(data, dict):
        # allow {"source_id": [...articles...]} shape
        if "articles" in data:
            return data["articles"]
        return [data]
    return data


def ingest_articles(
    model,
    tokenizer,
    store: FigmentStore,
    registry: SourceRegistry,
    articles: list[dict[str, Any]],
    kv_manager=None,
    compute_kv: bool = False,
    summarize_images: bool = False,
) -> dict[str, Any]:
    """Ingest a list of article dicts into the store.

    Each article dict needs at least ``source_id`` and ``text``. Returns a
    small stats dict (article/figment counts, sources touched).
    """
    stats = {"articles": 0, "figments": 0, "sources": set()}
    for art in articles:
        sid = art["source_id"]
        text = art["text"]
        if not text or not text.strip():
            continue
        base = registry.base_trust(sid)
        figments = ingest_text_to_figments(
            model,
            tokenizer,
            text,
            source_id=sid,
            trust=base,
            store=store,
            kv_manager=kv_manager,
            compute_kv=compute_kv,
            summarize_images=summarize_images,
        )
        stats["articles"] += 1
        stats["figments"] += len(figments)
        stats["sources"].add(sid)
    stats["sources"] = sorted(stats["sources"])
    return stats


def ingest_feed(
    model,
    tokenizer,
    store: FigmentStore,
    registry: SourceRegistry,
    source_id: str,
    uri: str,
    **kwargs,
) -> dict[str, Any]:
    """Fetch a feed and ingest every entry as an article."""
    articles = _read_feed(uri, source_id)
    return ingest_articles(model, tokenizer, store, registry, articles, **kwargs)


def ingest_file(
    model,
    tokenizer,
    store: FigmentStore,
    registry: SourceRegistry,
    path: str,
    **kwargs,
) -> dict[str, Any]:
    """Ingest articles from a local JSON/JSONL file (no network needed)."""
    articles = _read_article_file(path)
    return ingest_articles(model, tokenizer, store, registry, articles, **kwargs)
