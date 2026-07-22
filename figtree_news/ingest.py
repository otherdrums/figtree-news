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


def _read_feed(uri: str, source_id: str, since: str = "", before: str = "") -> list[dict[str, Any]]:
    """Parse an RSS/Atom feed (URL or local file) into article dicts.

    Requires ``feedparser`` (optional dependency). Falls back to parsing the
    URI as a JSON/JSONL file of article dicts if feedparser is unavailable or
    the content is not a feed.

    since/before: ISO date strings or RFC 2822 dates to filter articles by published date.
    """
    try:
        import feedparser  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "feed ingestion requires feedparser: pip install feedparser"
        ) from exc

    parsed = feedparser.parse(uri)
    articles: list[dict[str, Any]] = []

    # Parse since/before into comparable datetimes
    _since = _parse_date_param(since) if since else None
    _before = _parse_date_param(before) if before else None
    if _before and len(before.strip()) <= 10:
        _before = _before.replace(hour=23, minute=59, second=59)

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

        # Date range filtering
        pub_raw = entry.get("published") or ""
        if (_since or _before) and pub_raw:
            pub_dt = _parse_date_param(pub_raw)
            if pub_dt:
                if _since and pub_dt < _since:
                    continue
                if _before and pub_dt > _before:
                    continue

        # Extract article image from feed metadata
        image_url = _extract_feed_image(entry)
        video_url = _extract_video_url(entry)

        articles.append(
            {
                "source_id": source_id,
                "text": text.strip(),
                "url": entry.get("link"),
                "title": entry.get("title"),
                "author": entry.get("author"),
                "published": pub_raw,
                "image_url": image_url,
                "video_url": video_url,
            }
        )
    return articles


def _parse_date_param(s: str) -> datetime | None:
    """Parse a date string (ISO or RFC 2822) into a timezone-aware datetime."""
    from email.utils import parsedate_to_datetime
    if not s:
        return None
    try:
        dt = parsedate_to_datetime(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        pass
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


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
    # 3. YouTube media_group → media_thumbnail
    for group in entry.get("media_group", []):
        if isinstance(group, dict):
            for thumb in group.get("media_thumbnail", []):
                if isinstance(thumb, dict) and thumb.get("url"):
                    return thumb["url"]
    # 4. enclosures with image type
    for enc in entry.get("enclosures", []):
        href = enc.get("href") or enc.get("url", "")
        etype = enc.get("type", "")
        if href and (etype.startswith("image/") or href.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif"))):
            return href
    # 5. media:thumbnail as single dict (some feeds)
    mt = entry.get("media_thumbnail")
    if isinstance(mt, dict) and mt.get("url"):
        return mt["url"]
    # 6. enclosure as single dict
    enc = entry.get("enclosure")
    if isinstance(enc, dict):
        href = enc.get("href") or enc.get("url", "")
        etype = enc.get("type", "")
        if href and (etype.startswith("image/") or href.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif"))):
            return href
    return None


def _extract_video_url(entry) -> str | None:
    """Extract a video embed or watch URL from a feed entry (YouTube, Rumble, etc.)."""
    link = entry.get("link") or ""
    # YouTube watch URL → embed URL
    import re
    yt_match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]+)', link)
    if yt_match:
        return f"https://www.youtube.com/embed/{yt_match.group(1)}"
    # Rumble URL → embed
    rumble_match = re.search(r'rumble\.com/(?:embed/)?v([\w-]+)', link)
    if rumble_match:
        return f"https://rumble.com/embed/{rumble_match.group(0).split('/')[-1]}"
    # Check media_content for video type
    for media in entry.get("media_content", []):
        if media.get("medium") == "video" or media.get("type", "").startswith("video/"):
            return media.get("url")
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
            video_url = art.get("video_url") or ""
            for f in figments:
                f.meta["url"] = url
                f.meta["published"] = published
                f.meta["title"] = title
                f.meta["first_seen"] = _now_iso()
                f.meta["author"] = author
                f.meta["syndication"] = art.get("source", "")
                f.meta["image_url"] = image_url
                f.meta["video_url"] = video_url
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
    since: str = "",
    before: str = "",
    **kwargs,
) -> dict[str, Any]:
    """Fetch a feed and ingest every entry as an article."""
    articles = _read_feed(uri, source_id, since=since, before=before)
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
