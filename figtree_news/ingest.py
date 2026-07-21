"""Feed ingestion: turn articles into Figments with full provenance.

Each article becomes an Image Figment (parent) whose children are the
sentence-level figments produced by ``figtree.ingest_text_to_figments``. The
library already tags every figment with ``source_id`` and stamps the image
with ``base_trust``, so the source's initial credibility flows into Figtree's
trust propagation without any news-specific code living in the core library.

This module additionally stamps **provenance** onto every figment returned by
the library (``url``, ``published``, ``title``, ``first_seen``) and re-persists
it, so the generated newspaper can always link back to the original article.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from figtree import FigmentStore, ingest_text_to_figments

from .config import SourceRegistry
from .search_index import get_index


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        summary = entry.get("summary") or ""
        content = entry.get("content")
        if isinstance(content, list):
            body = " ".join(
                c.get("value", "") for c in content if isinstance(c, dict)
            )
        elif isinstance(content, str):
            body = content
        else:
            body = ""
        text = (summary + "\n\n" + body).strip() if body else summary
        if not text.strip():
            continue

        # Extract article image from feed metadata
        image_url = _extract_feed_image(entry)

        articles.append(
            {
                "source_id": source_id,
                "text": text.strip(),
                "url": entry.get("link"),
                "title": entry.get("title"),
                "author": entry.get("author"),
                "published": entry.get("published"),
                "image_url": image_url,
            }
        )
    return articles


def _extract_feed_image(entry) -> str | None:
    """Extract the best image URL from an RSS/Atom feed entry."""
    # 1. media_content (media RSS namespace)
    for media in entry.get("media_content", []):
        if media.get("medium") == "image" or (
            media.get("url", "").lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif"))
        ):
            return media["url"]
    # 2. media_thumbnail
    for thumb in entry.get("media_thumbnail", []):
        if thumb.get("url"):
            return thumb["url"]
    # 3. enclosures with image type
    for enc in entry.get("enclosures", []):
        href = enc.get("href") or enc.get("url", "")
        etype = enc.get("type", "")
        if href and (etype.startswith("image/") or href.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif"))):
            return href
    # 4. media:thumbnail as single dict (some feeds)
    mt = entry.get("media_thumbnail")
    if isinstance(mt, dict) and mt.get("url"):
        return mt["url"]
    # 5. enclosure as single dict
    enc = entry.get("enclosure")
    if isinstance(enc, dict):
        href = enc.get("href") or enc.get("url", "")
        etype = enc.get("type", "")
        if href and (etype.startswith("image/") or href.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif"))):
            return href
    return None


def _read_article_file(path: str) -> list[dict[str, Any]]:
    """Load articles from a JSON list or JSONL file (one article dict per line)."""
    with open(path, "r", encoding="utf-8") as fh:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in fh if line.strip()]
        data = json.load(fh)
    if isinstance(data, dict):
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
    stamp_provenance: bool = True,
) -> dict[str, Any]:
    """Ingest a list of article dicts into the store.

    Each article dict needs at least ``source_id`` and ``text``. When
    ``stamp_provenance`` is True (default), ``url``/``published``/``title`` are
    attached to every resulting figment and re-persisted.

    Returns a small stats dict (article/figment counts, sources touched, urls).
    """
    stats = {"articles": 0, "figments": 0, "sources": set(), "urls": []}
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
        if stamp_provenance:
            url = art.get("url")
            published = art.get("published")
            title = art.get("title")
            author = art.get("author", "")
            image_url = art.get("image_url") or ""
            for f in figments:
                f.meta["url"] = url
                f.meta["published"] = published
                f.meta["title"] = title
                f.meta["first_seen"] = _now_iso()
                f.meta["author"] = author
                f.meta["syndication"] = art.get("source", "")
                f.meta["image_url"] = image_url
            hidden = figments[0].boundary.shape[0]
            store.upsert(figments, hidden_size=hidden)

            # Index in FTS for text search
            image_fig = figments[0]  # the image figment
            idx = get_index()
            idx.index_article(
                article_id=image_fig.figment_id,
                title=title or "",
                text=text,
                author=author or "",
                source_id=sid,
                published=published or "",
                first_seen=image_fig.meta.get("first_seen", ""),
            )

        stats["articles"] += 1
        stats["figments"] += len(figments)
        stats["sources"].add(sid)
        if art.get("url"):
            stats["urls"].append(art["url"])
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
