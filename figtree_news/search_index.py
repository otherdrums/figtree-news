"""Persistent full-text search index (sqlite FTS5) for articles.

Lives alongside the LanceDB store. Articles are indexed at ingest time and
removed when the URL-based dedup layer discards them (never for ingested
articles — the index is append-only for now).

Provides:
* ``index_article(id, title, text, author, source_id, published, first_seen)``
  — add or update (idempotent) an article in the FTS index.
* ``search(q, range, sort, page, limit)`` — text search with date-range
  filter, sort-by-date or sort-by-relevance, and pagination.
* ``title_exists(title, source_id)`` — near-duplicate title check (dedup
  helper).  Returns True if a very similar title from the same source
  already exists.
"""

from __future__ import annotations

import math
import os
import sqlite3
import time
from difflib import SequenceMatcher
from typing import Any


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def _range_clause(range_str: str) -> tuple[str, str | None]:
    """Return (sql_clause, param) for a date-range filter."""
    now_ts = time.time()
    if range_str == "today":
        since = time.strftime("%Y-%m-%d", time.gmtime())
        return ("published >= ?", since)
    elif range_str == "yesterday":
        since = time.strftime("%Y-%m-%d", time.gmtime(now_ts - 86400))
        until = time.strftime("%Y-%m-%d", time.gmtime())
        return ("published >= ? AND published < ?", (since, until))
    elif range_str == "last_week":
        since = time.strftime("%Y-%m-%d", time.gmtime(now_ts - 7 * 86400))
        return ("published >= ?", since)
    elif range_str == "last_month":
        since = time.strftime("%Y-%m-%d", time.gmtime(now_ts - 30 * 86400))
        return ("published >= ?", since)
    elif range_str == "last_year":
        since = time.strftime("%Y-%m-%d", time.gmtime(now_ts - 365 * 86400))
        return ("published >= ?", since)
    return ("1=1", None)  # "all"


def _parse_search_date(raw: str | None) -> str | None:
    """Try to extract a YYYY-MM-DD prefix from arbitrary date strings."""
    if not raw:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%a, %d %b %Y %H:%M:%S %z",
    ):
        try:
            return time.strftime("%Y-%m-%d", time.strptime(raw, fmt) if "struct_time" not in dir() else None)
        except (ValueError, TypeError):
            continue
    # last resort: try dateutil-like heuristic
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(raw)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    return None


class SearchIndex:
    """SQLite FTS5 index for article text search."""

    def __init__(self, db_path: str = "demo/news_fts.db"):
        self.db_path = db_path
        self._con: sqlite3.Connection | None = None

    def _conn(self) -> sqlite3.Connection:
        if self._con is None:
            self._con = sqlite3.connect(self.db_path, check_same_thread=False)
            self._con.execute("PRAGMA journal_mode=WAL")
            self._con.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5("
                "  article_id, title, text, author, source_id,"
                "  published, first_seen,"
                "  tokenize='porter unicode61'"
                ")"
            )
        return self._con

    def index_article(
        self,
        article_id: str,
        title: str,
        text: str,
        author: str = "",
        source_id: str = "",
        published: str = "",
        first_seen: str = "",
    ) -> None:
        """Add or replace an article in the FTS index (idempotent on article_id)."""
        con = self._conn()
        pub = _parse_search_date(published) or published or ""
        seen = first_seen or _now_iso()
        # Delete old row then insert (FTS5 has no upsert)
        con.execute("DELETE FROM articles_fts WHERE article_id = ?", (article_id,))
        con.execute(
            "INSERT INTO articles_fts(article_id, title, text, author, source_id, published, first_seen) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (article_id, title, text[:3000], author, source_id, pub, seen),
        )
        con.commit()

    def delete_article(self, article_id: str) -> None:
        con = self._conn()
        con.execute("DELETE FROM articles_fts WHERE article_id = ?", (article_id,))
        con.commit()

    def search(
        self,
        q: str = "",
        range: str = "all",
        sort: str = "date_desc",
        page: int = 1,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Search articles by text + date range. Returns paginated results."""
        con = self._conn()
        where_clause, where_param = _range_clause(range)

        results: list[str] = []
        total = 0

        if q.strip():
            # FTS5 MATCH with BM25 ranking
            query = " OR ".join(f'"{t}"' for t in q.split() if len(t) > 1) or q
            try:
                if where_param is not None:
                    if isinstance(where_param, tuple):
                        rows = con.execute(
                            f"SELECT article_id, rank FROM articles_fts WHERE articles_fts MATCH ? "
                            f"AND {where_clause} ORDER BY rank LIMIT ? OFFSET ?",
                            (query, *where_param, limit, (page - 1) * limit),
                        ).fetchall()
                        count_row = con.execute(
                            f"SELECT COUNT(*) FROM articles_fts WHERE articles_fts MATCH ? AND {where_clause}",
                            (query, *where_param),
                        ).fetchone()
                    else:
                        rows = con.execute(
                            f"SELECT article_id, rank FROM articles_fts WHERE articles_fts MATCH ? "
                            f"AND {where_clause} ORDER BY rank LIMIT ? OFFSET ?",
                            (query, where_param, limit, (page - 1) * limit),
                        ).fetchall()
                        count_row = con.execute(
                            f"SELECT COUNT(*) FROM articles_fts WHERE articles_fts MATCH ? AND {where_clause}",
                            (query, where_param),
                        ).fetchone()
                else:
                    rows = con.execute(
                        "SELECT article_id, rank FROM articles_fts WHERE articles_fts MATCH ? "
                        "ORDER BY rank LIMIT ? OFFSET ?",
                        (query, limit, (page - 1) * limit),
                    ).fetchall()
                    count_row = con.execute(
                        "SELECT COUNT(*) FROM articles_fts WHERE articles_fts MATCH ?",
                        (query,),
                    ).fetchone()
                results = [r[0] for r in rows]
                total = count_row[0] if count_row else 0
            except sqlite3.OperationalError:
                # FTS5 query syntax error — fall back to no results
                results = []
                total = 0
        else:
            # No query text — date-filtered browse
            if where_param is not None:
                if isinstance(where_param, tuple):
                    rows = con.execute(
                        f"SELECT article_id FROM articles_fts WHERE {where_clause} "
                        f"ORDER BY published DESC LIMIT ? OFFSET ?",
                        (*where_param, limit, (page - 1) * limit),
                    ).fetchall()
                    count_row = con.execute(
                        f"SELECT COUNT(*) FROM articles_fts WHERE {where_clause}",
                        where_param,
                    ).fetchone()
                else:
                    rows = con.execute(
                        f"SELECT article_id FROM articles_fts WHERE {where_clause} "
                        f"ORDER BY published DESC LIMIT ? OFFSET ?",
                        (where_param, limit, (page - 1) * limit),
                    ).fetchall()
                    count_row = con.execute(
                        f"SELECT COUNT(*) FROM articles_fts WHERE {where_clause}",
                        (where_param,),
                    ).fetchone()
            else:
                rows = con.execute(
                    "SELECT article_id FROM articles_fts ORDER BY published DESC LIMIT ? OFFSET ?",
                    (limit, (page - 1) * limit),
                ).fetchall()
                count_row = con.execute("SELECT COUNT(*) FROM articles_fts").fetchone()
            results = [r[0] for r in rows]
            total = count_row[0] if count_row else 0

        return {
            "article_ids": results,
            "total": total,
            "page": page,
            "total_pages": max(1, math.ceil(total / limit)) if total else 0,
        }

    def title_exists(self, title: str, source_id: str, threshold: float = 0.85) -> bool:
        """Check if a similar title from the same source already exists."""
        if not title:
            return False
        title_lower = title.lower().strip()
        con = self._conn()
        try:
            rows = con.execute(
                "SELECT title FROM articles_fts WHERE source_id = ? LIMIT 50",
                (source_id,),
            ).fetchall()
        except sqlite3.OperationalError:
            return False
        for (existing,) in rows:
            if not existing:
                continue
            ratio = SequenceMatcher(None, title_lower, existing.lower().strip()).ratio()
            if ratio >= threshold:
                return True
        return False

    def article_count(self) -> int:
        try:
            row = self._conn().execute("SELECT COUNT(*) FROM articles_fts").fetchone()
            return row[0] if row else 0
        except sqlite3.OperationalError:
            return 0


_index: SearchIndex | None = None


def get_index(db_path: str = "demo/news_fts.db") -> SearchIndex:
    global _index
    if _index is None or _index.db_path != db_path:
        _index = SearchIndex(db_path)
    return _index
