"""SearXNG metasearch client for web article discovery.

Queries a local SearXNG instance and converts results into article dicts
compatible with the existing ingestion pipeline.  Each search result URL
is fetched via trafilatura (in the crawler) for full-text extraction; the
snippet from SearXNG is only a preview.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx


@dataclass
class SearxngConfig:
    """SearXNG connection + default search parameters."""

    url: str = "http://192.168.10.202:8081"
    enabled: bool = True
    queries: list[str] = field(default_factory=list)
    categories: str = "news"
    time_range: str = "week"  # day | week | month | year | ""
    language: str = "en"
    max_results: int = 20
    pages: int = 1
    timeout: int = 15

    @classmethod
    def from_sources_json(cls, path: str) -> SearxngConfig:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return cls(enabled=False)
        sec = raw.get("searxng")
        if not isinstance(sec, dict):
            return cls(enabled=False)
        return cls(
            url=str(sec.get("url", cls.url)),
            enabled=bool(sec.get("enabled", True)),
            queries=[str(q) for q in sec.get("queries", [])],
            categories=str(sec.get("categories", cls.categories)),
            time_range=str(sec.get("time_range", cls.time_range)),
            language=str(sec.get("language", cls.language)),
            max_results=int(sec.get("max_results", cls.max_results)),
            pages=int(sec.get("pages", cls.pages)),
            timeout=int(sec.get("timeout", cls.timeout)),
        )


# ---------------------------------------------------------------------------
# SearXNG API
# ---------------------------------------------------------------------------

def search(
    config: SearxngConfig,
    query: str,
    *,
    pageno: int = 1,
    categories: str | None = None,
    time_range: str | None = None,
    language: str | None = None,
) -> list[dict]:
    """Query SearXNG ``/search`` endpoint and return raw result dicts.

    Returns the ``results`` list from the JSON response.  Each dict has
    keys like ``url``, ``title``, ``content``, ``publishedDate``,
    ``engines``, ``thumbnail``, ``score``, ``metadata``, etc.
    """
    params: dict[str, str | int] = {
        "q": query,
        "format": "json",
        "categories": categories or config.categories,
        "language": language or config.language,
        "pageno": pageno,
    }
    tr = time_range if time_range is not None else config.time_range
    if tr:
        params["time_range"] = tr

    try:
        resp = httpx.get(
            f"{config.url.rstrip('/')}/search",
            params=params,
            timeout=config.timeout,
            follow_redirects=True,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
    except Exception as exc:
        print(f"[searxng] search error: {exc}")
        return []


# ---------------------------------------------------------------------------
# Result → article conversion
# ---------------------------------------------------------------------------

# Domains that serve video embeds — detected for video_url extraction.
_VIDEO_DOMAINS = {
    "youtube.com": "youtube",
    "youtu.be": "youtube",
    "rumble.com": "rumble",
    "dailymotion.com": "dailymotion",
    "vimeo.com": "vimeo",
}


def _domain_from_url(url: str) -> str:
    """Extract the domain from a URL, stripping ``www.`` prefix."""
    try:
        netloc = urlparse(url).netloc
    except Exception:
        return ""
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc.lower()


def _source_id_from_domain(domain: str) -> str:
    """Map a domain to a source_id suitable for the SourceRegistry.

    Examples:
        "nbcnews.com"   → "nbcnews.com"
        "www.msn.com"   → "msn.com"
        "bbc.co.uk"     → "bbc.co.uk"
        "reuters.com"   → "reuters.com"
    """
    return domain  # domain is already stripped of www. and lowercased


def _extract_video_url(url: str) -> str | None:
    """Return an embed URL if the result points to a known video platform."""
    domain = _domain_from_url(url)
    kind = _VIDEO_DOMAINS.get(domain)
    if kind == "youtube":
        m = re.search(r"(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]+)", url)
        if m:
            return f"https://www.youtube.com/embed/{m.group(1)}"
    elif kind == "rumble":
        m = re.search(r"rumble\.com/(?:embed/)?v([\w-]+)", url)
        if m:
            return f"https://rumble.com/embed/{m.group(0).split('/')[-1]}"
    return None


def _parse_published_date(raw: str | None) -> str | None:
    """Best-effort parse of SearXNG's ``publishedDate`` field.

    SearXNG returns ISO-8601 strings like
    ``2026-07-21T08:16:54.141000+00:00`` or ``null``.
    """
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    # Already ISO-ish — return as-is (pipeline will parse it)
    return raw


def _thumbnail_from_result(result: dict) -> str | None:
    """Extract the best image URL from a SearXNG result dict.

    Priority: ``img_src`` > ``thumbnail`` > ``image``.
    """
    for key in ("img_src", "thumbnail", "image"):
        val = result.get(key)
        if val and isinstance(val, str) and val.startswith("http"):
            return val
    return None


def _author_from_metadata(metadata: str | None) -> str:
    """Try to pull an author / source name from the ``metadata`` field.

    SearXNG sets ``metadata`` to things like ``"19 hours ago | NBC News"``
    or ``"World"`` or ``None``.
    """
    if not metadata or not isinstance(metadata, str):
        return ""
    parts = metadata.split("|")
    if len(parts) >= 2:
        return parts[-1].strip()
    return ""


def results_to_articles(results: list[dict]) -> list[dict]:
    """Convert SearXNG JSON results to article dicts for ingestion.

    Each dict has the keys expected by ``Crawler.ingest_article()``:
    ``url``, ``title``, ``text`` (snippet), ``published``, ``author``,
    ``image_url``, ``video_url``, ``source_id``.

    Note: ``text`` is the short snippet from SearXNG.  The crawler will
    replace it with the full trafilatura-extracted text via ``fetch_page()``.
    """
    articles: list[dict] = []
    seen_urls: set[str] = set()

    for result in results:
        url = result.get("url", "").strip()
        if not url or not url.startswith("http"):
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)

        title = result.get("title", "").strip()
        snippet = result.get("content", "").strip()
        if not title and not snippet:
            continue

        domain = _domain_from_url(url)
        source_id = _source_id_from_domain(domain)

        article: dict = {
            "url": url,
            "title": title,
            "text": snippet,  # placeholder — replaced by fetch_page()
            "published": _parse_published_date(result.get("publishedDate")),
            "author": _author_from_metadata(result.get("metadata")),
            "image_url": _thumbnail_from_result(result),
            "video_url": _extract_video_url(url),
            "source_id": source_id,
        }
        articles.append(article)

    return articles
