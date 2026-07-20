"""CPU tests for the crawler (no model / network)."""

from __future__ import annotations

from figtree_news.config import SourceRegistry
from figtree_news.crawler import Crawler, _domain, _same_domain


def _make(db=None, seen_path=None, max_depth=1, max_pages=10):
    reg = SourceRegistry({})
    return Crawler(
        None, None, db, reg,
        seen_path=seen_path, max_depth=max_depth, max_pages=max_pages,
    )


def test_domain_and_same_domain():
    assert _domain("https://example.com/x") == "example.com"
    assert _same_domain("https://example.com/a", "http://example.com/b")
    assert not _same_domain("https://a.com", "https://b.com")


def test_seen_persistence(tmp_path):
    p = tmp_path / "seen.json"
    c = _make(seen_path=str(p))
    c._mark("http://a.com/1")
    assert c._already("http://a.com/1")
    c2 = _make(seen_path=str(p))
    assert c2._already("http://a.com/1")


def test_crawl_seeds_traversal(tmp_path):
    c = _make(seen_path=str(tmp_path / "seen.json"), max_depth=1, max_pages=20)
    ingested = []
    c.ingest_article = lambda sid, art: (ingested.append(art) or True)  # type: ignore
    c._can_fetch = lambda u: True  # type: ignore
    c.fetch_page = lambda url: {  # type: ignore
        "url": url,
        "text": "A news article about the Election and the Economy.",
        "title": "T", "published": None,
        "links": [f"http://seed.com/{i}" for i in range(3)],
    }
    added = c.crawl_seeds(["http://seed.com/start"], source_id="seed.com")
    assert added == 4  # start + 3 discovered links at depth 1
    assert len(ingested) == 4


def test_crawl_seeds_respects_depth(tmp_path):
    c = _make(seen_path=str(tmp_path / "seen.json"), max_depth=0, max_pages=20)
    ingested = []
    c.ingest_article = lambda sid, art: (ingested.append(art) or True)  # type: ignore
    c._can_fetch = lambda u: True  # type: ignore
    c.fetch_page = lambda url: {  # type: ignore
        "url": url, "text": "News about the Election.", "title": "T",
        "published": None, "links": [f"http://seed.com/{i}" for i in range(3)],
    }
    added = c.crawl_seeds(["http://seed.com/start"], source_id="seed.com")
    assert added == 1  # depth 0 -> only the seed itself
