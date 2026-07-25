"""Continuous web crawler for the news aggregator.

Two ingestion paths, both feeding the same ``ingest_articles`` pipeline:

* **Feeds** — RSS/Atom URLs mapped to a ``source_id``.
* **Bounded link-follower** — start from seed URLs, extract article text with
  ``trafilatura``, and follow same-domain links up to ``max_depth`` to discover
  more articles.

Every ingested URL is de-duplicated against a persisted ``seen`` index so
re-crawls are cheap and idempotent. Crawling is polite: a User-Agent is sent,
``robots.txt`` is honoured via ``RobotFileParser``, and fetches are rate-limited
per host.
"""

from __future__ import annotations

import os
import re
import threading
from collections import Counter
from functools import lru_cache
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

from figtree import FigmentStore

from .config import SourceRegistry
from .ingest import _read_feed, ingest_articles
from .search_index import get_index

# Built-in stopwords for keyword extraction (zero deps)
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "is", "was", "are", "were", "been", "be", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "can", "this", "that", "these", "those", "it", "its", "their", "they", "them", "he", "she", "we", "you", "i", "said", "says", "according", "report", "reports", "reported", "news", "new", "today", "week", "month", "year", "time", "times", "day", "days", "month", "months", "year", "years", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "first", "last", "next", "previous", "new", "old", "big", "small", "large", "great", "good", "bad", "high", "low", "long", "short", "early", "late", "right", "left", "up", "down", "out", "in", "over", "under", "again", "also", "just", "now", "then", "than", "more", "most", "some", "any", "all", "each", "every", "other", "another", "such", "only", "even", "still", "already", "ever", "never", "always", "often", "sometimes", "usually", "breaking", "update", "more", "latest", "current",
}


_USER_AGENT = "figtree-news/0.1 (+https://github.com/otherdrums/figtree-news; research crawler)"

_BLOCKED_DOMAINS = {
    "msn.com", "msn.co.uk", "msn.es", "msn.fr", "msn.it", "msn.de",
}


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _same_domain(a: str, b: str) -> bool:
    return _domain(a) == _domain(b)


@lru_cache(maxsize=64)
def _robot_parser(netloc: str) -> RobotFileParser:
    rp = RobotFileParser()
    rp.set_url(f"https://{netloc}/robots.txt")
    try:
        rp.read()
    except Exception:
        pass
    return rp


def _can_fetch(url: str) -> bool:
    netloc = _domain(url)
    if not netloc:
        return False
    try:
        return _robot_parser(netloc).can_fetch(_USER_AGENT, url)
    except Exception:
        return True


def _extract_og_image(html: str) -> str | None:
    """Extract og:image or twitter:image from raw HTML."""
    for pattern in [
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
        r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:image["\']',
    ]:
        m = re.search(pattern, html, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


_URL_FRAGMENTS = {
    "https", "http", "www", "com", "org", "net", "html", "php", "asp",
    "aspx", "jsp", "cgi", "xml", "json", "api", "svc", "ftp", "ssh",
    "gov", "edu", "uk", "jp", "de", "fr", "au", "ca", "br", "in", "cn",
}

def _extract_keywords(articles: list[dict], top_n: int = 10) -> list[str]:
    """Extract meaningful multi-word phrases from article titles for SearXNG queries.

    Builds frequency-weighted bigrams and trigrams from titles,
    filtering out generic/stopword pairs so queries are specific
    enough to return targeted news results.
    """
    import re
    from itertools import combinations

    phrase_counts = Counter()
    for art in articles:
        title = art.get("title", "")
        if not title:
            continue
        words = re.findall(r"[a-zA-Z]{3,}", title.lower())
        words = [w for w in words if w not in _STOPWORDS and w not in _URL_FRAGMENTS]
        # Build bigrams and trigrams from remaining meaningful words
        for n in (2, 3):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i + n])
                if any(w in _STOPWORDS for w in words[i:i + n]):
                    continue
                phrase_counts[phrase] += 1

    if not phrase_counts:
        return [
            "breaking news",
            "world news today",
            "technology news",
            "business markets",
            "politics government",
        ]
    return [p for p, _ in phrase_counts.most_common(top_n)]


class Crawler:
    def __init__(
        self,
        model,
        tokenizer,
        store: FigmentStore,
        registry: SourceRegistry,
        seen_path: str | None = None,
        max_depth: int = 1,
        max_pages: int = 50,
        compute_kv: bool = False,
        summarize_images: bool = False,
        kv_manager=None,
        decompose_engine=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.store = store
        self.registry = registry
        self.seen_path = seen_path
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.compute_kv = compute_kv
        self.summarize_images = summarize_images
        self.kv_manager = kv_manager
        self.decompose_engine = decompose_engine
        self.seen: set[str] = self._load_seen()
        self._pending_decompose: list[str] = []
        self._pending_lock = threading.Lock()
        self._ingest_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._new_articles: list[dict] = []

    # -- URL de-duplication ------------------------------------------------ #
    def _load_seen(self) -> set[str]:
        if self.seen_path and os.path.exists(self.seen_path):
            try:
                import json as _json

                with open(self.seen_path, "r", encoding="utf-8") as fh:
                    return set(_json.load(fh))
            except Exception:
                pass
        return set()

    def _save_seen(self) -> None:
        if not self.seen_path:
            return
        import json as _json

        with open(self.seen_path, "w", encoding="utf-8") as fh:
            _json.dump(sorted(self.seen), fh)

    def _already(self, url: str) -> bool:
        with self._ingest_lock:
            return url in self.seen

    def _mark(self, url: str) -> None:
        if url:
            with self._ingest_lock:
                self.seen.add(url)
                self._save_seen()

    # -- extraction -------------------------------------------------------- #
    def _can_fetch(self, url: str) -> bool:
        # Instance method so tests/override can patch politeness easily.
        return _can_fetch(url)

    def fetch_page(self, url: str) -> dict:
        """Extract article text, metadata and same-domain links from a URL."""
        try:
            resp = httpx.get(
                url, headers={"User-Agent": _USER_AGENT}, follow_redirects=True, timeout=20
            )
            html = resp.text
        except Exception as exc:  # pragma: no cover
            return {"url": url, "text": None, "error": str(exc)}

        text = None
        title = None
        published = None
        author = ""
        image_url = None
        try:
            import trafilatura  # type: ignore

            text = trafilatura.extract(html, url=url)
            meta = trafilatura.extract_metadata(html, url=url)
            if meta:
                title = getattr(meta, "title", None)
                published = getattr(meta, "date", None)
                author = getattr(meta, "author", "") or ""
                image_url = getattr(meta, "image", None)
        except Exception:
            pass

        # Fallback: extract og:image directly from HTML if trafilatura missed it
        if not image_url:
            image_url = _extract_og_image(html or "")

        links = set()
        for m in re.findall(r'href=["\']([^"\']+)["\']', html or ""):
            abs_url = urljoin(url, m)
            if abs_url.startswith("http") and _same_domain(abs_url, url):
                links.add(abs_url.split("#")[0])

        return {
            "url": url,
            "text": text,
            "title": title,
            "published": published,
            "author": author,
            "image_url": image_url,
            "links": sorted(links),
            "error": None,
        }

    # -- ingestion --------------------------------------------------------- #
    def ingest_article(self, source_id: str, article: dict) -> bool:
        """Ingest a single article dict if its URL is new. Returns True if added."""
        url = article.get("url")
        if url and self._already(url):
            return False

        url_domain = (url.split("/")[2] if url else "")
        if any(d in url_domain for d in _BLOCKED_DOMAINS):
            if url:
                self._mark(url)
            return False

        text = article.get("text") or ""
        # If text is short and we have a URL, try trafilatura to get the full article
        if text.strip() and len(text.strip()) < 200 and url:
            try:
                full = self.fetch_page(url)
                if full and full.get("text") and len(full["text"].strip()) > len(text.strip()):
                    article["text"] = full["text"]
                    text = article["text"]
                    if full.get("title"):
                        article["title"] = full["title"]
                    if full.get("image_url"):
                        article["image_url"] = full["image_url"]
                    if full.get("published"):
                        article["published"] = full["published"]
            except Exception:
                pass

        if not text.strip() or len(text.strip()) < 40:
            if url:
                self._mark(url)
            return False

        # Title-based dedup: skip if near-duplicate title from same source exists
        title = article.get("title") or ""
        if title and get_index().title_exists(title, source_id):
            if url:
                self._mark(url)
            return False

        # VRAM guard: refuse new ingestion if GPU memory is critically low
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_mb = torch.cuda.mem_get_info()[0] // (1024 * 1024)
            if free_mb < 300:
                if url:
                    self._mark(url)
                return False

        with self._model_lock:
            ingest_articles(
                self.model,
                self.tokenizer,
                self.store,
                self.registry,
                [article],
                compute_kv=self.compute_kv,
                summarize_images=self.summarize_images,
                kv_manager=self.kv_manager,
            )
        
        # Queue for background decomposition (thread-safe: append to list,
        # the async caller drains it after to_thread returns)
        if self.decompose_engine and url:
            # Find the article figment we just created
            all_figs = self.store.all()
            for fig in reversed(all_figs):
                if (fig.meta.get("is_image") and 
                    fig.meta.get("source_id") == source_id and
                    fig.meta.get("url") == url):
                    with self._pending_lock:
                        self._pending_decompose.append(fig.figment_id)
                    break
        
        if url:
            self._mark(url)
        title = article.get("title") or ""
        text = article.get("text") or ""
        if title or text:
            with self._ingest_lock:
                self._new_articles.append({"title": title, "text": text})
        return True

    def drain_pending_decompose(self) -> list[str]:
        """Return and clear the list of article IDs queued for decomposition (thread-safe)."""
        with self._pending_lock:
            ids = list(self._pending_decompose)
            self._pending_decompose.clear()
        return ids

    def crawl_feed(self, source_id: str, feed_uri: str, max_articles: int | None = None,
                   since: str = "", before: str = "") -> int:
        try:
            articles = _read_feed(feed_uri, source_id, since=since, before=before)
        except Exception as e:
            print(f"[crawler] {source_id}: failed to read feed - {e}")
            return 0
            
        total_in_feed = len(articles)
        added = 0
        skipped_dedup = 0
        skipped_short = 0

        for art in articles:
            if max_articles is not None and added >= max_articles:
                break
            if self.ingest_article(source_id, art):
                added += 1
            else:
                url = art.get("url")
                text = art.get("text", "")
                if url and self._already(url):
                    skipped_dedup += 1
                elif len(text.strip()) < 40:
                    skipped_short += 1

        if total_in_feed > 0:
            print(f"[crawler] {source_id}: in_feed={total_in_feed}  added={added}  dedup={skipped_dedup}  short={skipped_short}")
        return added

    def crawl_seeds(self, seeds: list[str], source_id: str | None = None) -> int:
        """Bounded BFS from seed URLs, ingesting discovered articles."""
        added = 0
        queue = [(s, 0) for s in seeds if self._can_fetch(s)]
        visited_local: set[str] = set()
        pages = 0
        while queue and pages < self.max_pages:
            url, depth = queue.pop(0)
            norm = url.split("#")[0]
            if norm in visited_local or (url and self._already(url)):
                continue
            visited_local.add(norm)
            pages += 1
            if not self._can_fetch(url):
                continue
            page = self.fetch_page(url)
            if page.get("error"):
                continue
            sid = source_id or _domain(url)
            self.registry.ensure(sid, name=sid, base_trust=0.5)
            art = {
                "source_id": sid,
                "text": page.get("text") or "",
                "url": url,
                "title": page.get("title"),
                "published": page.get("published"),
                "image_url": page.get("image_url"),
                "video_url": page.get("video_url"),
            }
            if self.ingest_article(sid, art):
                added += 1
            if depth < self.max_depth:
                for link in page.get("links", []):
                    if link not in visited_local and self._can_fetch(link):
                        queue.append((link, depth + 1))
        return added

    # -- SearXNG web search -------------------------------------------------- #
    def search_searxng(
        self, query: str, categories: str = "news", time_range: str = "week",
        max_results: int = 20, pages: int = 1,
    ) -> int:
        """Search SearXNG, fetch full text via trafilatura, ingest.

        Returns the number of articles successfully added.
        """
        from .searxng import search as searxng_search, results_to_articles

        cfg = self.registry.searxng
        if not cfg or not cfg.enabled:
            return 0

        all_results: list[dict] = []
        for page in range(1, pages + 1):
            results = searxng_search(cfg, query, pageno=page,
                                     categories=categories, time_range=time_range)
            all_results.extend(results)
            if len(all_results) >= max_results:
                break

        articles = results_to_articles(all_results[:max_results])
        added = 0
        for article in articles:
            # Auto-register unknown domain as a source
            sid = article.get("source_id", "")
            if sid:
                self.registry.ensure(sid, name=sid, base_trust=0.7)
            url = article["url"]
            url_domain = (url.split("/")[2] if url else "")
            if any(d in url_domain for d in _BLOCKED_DOMAINS):
                continue
            # Fetch full article text via trafilatura (replaces snippet)
            full = self.fetch_page(url)
            if full and full.get("text") and len(full["text"].strip()) >= 40:
                article["text"] = full["text"]
                if full.get("title"):
                    article["title"] = full["title"]
                if full.get("image_url"):
                    article["image_url"] = full["image_url"]
                if full.get("author"):
                    article["author"] = full["author"]
                if full.get("published"):
                    article["published"] = full["published"]
            # ingest_article handles URL dedup + title dedup + full ingestion
            if self.ingest_article(sid or _domain(url), article):
                added += 1
        if added:
            print(f"[crawler] search '{query}': {added} articles ingested")
        return added

    # -- orchestration ----------------------------------------------------- #
    def run_once(
        self, feeds: dict[str, str], seeds: list[str], max_articles: int | None = None,
        since: str = "", before: str = "",
    ) -> dict:
        # Reset tracking for new articles this run
        self._new_articles.clear()
        
        stats = {"feeds_added": 0, "seeds_added": 0, "search_added": 0, "sources": set()}
        # Spread the budget across feeds so no single source dominates a run.
        per = None
        if max_articles is not None and feeds:
            per = max(1, max_articles // len(feeds))
        budget = max_articles
        for sid, uri in feeds.items():
            if budget is not None and budget <= 0:
                break
            got = self.crawl_feed(sid, uri, max_articles=per, since=since, before=before)
            stats["feeds_added"] += got
            if budget is not None:
                budget -= got
            stats["sources"].add(sid)
        if seeds and (budget is None or budget > 0):
            got = self.crawl_seeds(seeds)
            stats["seeds_added"] = got
            if budget is not None:
                budget -= got
        # SearXNG web search — use keywords from newly ingested RSS articles
        cfg = self.registry.searxng
        if cfg and cfg.enabled:
            # Extract keywords from articles added this run
            queries = _extract_keywords(self._new_articles, top_n=10)
            # Fallback to generic news queries if no articles were added
            if not queries:
                queries = [
                    "breaking news",
                    "world news today",
                    "technology news",
                    "business markets",
                    "politics government",
                ]
            per_q = max(1, budget // len(queries)) if budget else cfg.max_results
            for q in queries:
                if budget is not None and budget <= 0:
                    break
                got = self.search_searxng(
                    q, categories=cfg.categories, time_range=cfg.time_range,
                    max_results=per_q, pages=cfg.pages,
                )
                stats["search_added"] += got
                if budget is not None:
                    budget -= got
        stats["sources"] = sorted(stats["sources"])
        return stats
