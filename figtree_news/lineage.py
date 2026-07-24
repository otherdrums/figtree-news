"""Lineage: who broke it first, who is derivative, and narrative/agenda maps.

Pure CPU analysis over stored figments — no model required. Runs after
ingestion (e.g. at the end of each crawler tick) and persists its findings as
figments so the web UI can query them directly:

* ``narrative:{key}``  — one figment per cluster of articles about the same
  entities (the "story"). Carries the member article ids, the sources, the
  first reporter, and the stance lean.
* ``derivative:{orig}:{der}`` — an edge figment marking ``der`` as published
  later than ``orig`` while covering the same story (an echo / derivative).

Deterministic ids make the whole step idempotent: re-running overwrites the
previous lineage rather than duplicating it.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from figtree import Figment, FigmentStore, Figtree

_ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z]{2,})\b")
_STOP = {
    # Articles / determiners / pronouns
    "The", "This", "That", "These", "Those", "We", "They", "He", "She", "It",
    # Days / months (low-signal for clustering)
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
    # Source names (not entities)
    "Reuters", "AP", "BBC", "Guardian", "NPR", "Aljazeera", "CNN", "NYT",
    "New", "York", "Times",
    # Generic news verbs / roles (not unique entities)
    "President", "Prime", "Minister", "Government", "Official",
    "World", "News", "Report", "Story", "Article", "People", "Some", "Amid",
    "After", "Before", "During", "Following", "Because", "Despite",
    "About", "Over", "Under", "Between", "Against", "Through", "Into",
    "According", "Said", "Says", "Told", "Old", "First", "Last",
    "More", "Most", "Many", "Much", "Several", "One", "Two", "Three",
    "Watch", "Video", "Live", "Updated", "Breaking",
    # Generic directions
    "South", "North", "East", "West",
    # Common verbs/adjectives that get capitalized in headlines
    "Hits", "Slams", "Blasts", "Hails", "Urges", "Calls", "Makes",
    "Gets", "Set", "May", "Could", "Would", "Should", "Will",
    "Reportedly", "Allegedly",
    # Question words / sentence starters that get extracted as entities
    "Who", "What", "Where", "When", "Why", "How", "Which",
}


def _entities(text: str) -> set[str]:
    if not text:
        return set()
    toks = {t.strip(".") for t in _ENTITY_RE.findall(text)}
    result = set()
    for t in toks:
        if len(t) < 3:
            continue
        # Only filter single-word entities against stop list
        if " " not in t and t in _STOP:
            continue
        result.add(t)
        # For multi-word entities, also add individual words as entities
        # This helps match "Saudi Arabia" with "Saudi" across different headlines
        if " " in t:
            for word in t.split():
                if len(word) >= 3 and word not in _STOP:
                    result.add(word)
    return result


def _normalize_source(source_id: str) -> str:
    """Collapse same-org feed variants (e.g. france24 + france24_yt -> france24)."""
    # Strip _yt/_rss/_tw suffixes that mark different feed types from the same org
    for suffix in ("_yt", "_rss", "_tw", "_fb"):
        if source_id.endswith(suffix):
            return source_id[: -len(suffix)]
    return source_id


def _article_entities(art: Figment) -> set[str]:
    """Extract entities from title, falling back to text.
    
    Prefer title because it's more specific per-article. Body text tends
    to contain generic entities (country names, common figures) that cause
    over-clustering.
    """
    title = art.meta.get("title", "")
    if title:
        return _entities(title)
    return _entities(art.text)


def _parse_time(fig: Figment) -> datetime | None:
    for key in ("published", "first_seen"):
        raw = fig.meta.get(key)
        if not raw:
            continue
        try:
            if key == "first_seen":
                return datetime.fromisoformat(raw)
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            continue
    return None


def _articles(store: FigmentStore, *, all_figs: list | None = None) -> list[Figment]:
    return [
        f
        for f in (all_figs if all_figs is not None else store.all())
        if f.meta.get("is_image") and f.meta.get("source_id") and not f.is_edge()
    ]


def _cluster(articles: list[Figment], min_shared: int = 2, min_jaccard: float = 0.30) -> list[list[Figment]]:
    """Group articles by entity overlap using inverted index.
    
    Uses a combined check: articles must share >= min_shared entities AND
    have >= min_jaccard Jaccard similarity. This prevents mega-clusters from
    single shared generic terms (Jaccard requirement) while allowing articles
    with very similar entity sets to cluster even if small (shared count).
    
    Uses an inverted index for O(n * avg_entities) instead of O(n²).
    """
    by_id = {f.figment_id: f for f in articles}
    ent_sets = {f.figment_id: _article_entities(f) for f in articles}
    parent = {f.figment_id: f.figment_id for f in articles}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def jaccard(a, b):
        sa, sb = ent_sets[a], ent_sets[b]
        inter = len(sa & sb)
        union_len = len(sa | sb)
        return inter / union_len if union_len else 0.0

    # Build inverted index: entity -> set of article IDs
    inv_index: dict[str, set[str]] = {}
    for fid, ents in ent_sets.items():
        for ent in ents:
            inv_index.setdefault(ent, set()).add(fid)

    # For each article, find candidates via shared entities, then check both
    checked: set[tuple[str, str]] = set()
    for fid, ents in ent_sets.items():
        candidates: set[str] = set()
        for ent in ents:
            candidates |= inv_index.get(ent, set())
        candidates.discard(fid)
        
        for cand in candidates:
            pair = tuple(sorted([fid, cand]))
            if pair in checked:
                continue
            checked.add(pair)
            shared = len(ent_sets[fid] & ent_sets[cand])
            if shared >= min_shared and jaccard(fid, cand) >= min_jaccard:
                union(fid, cand)

    groups: dict[str, list[Figment]] = {}
    for fid in parent:
        groups.setdefault(find(fid), []).append(by_id[fid])
    return [g for g in groups.values() if len(g) >= 2]



def compute_lineage(store: FigmentStore, max_stories: int = 0) -> dict[str, Any]:
    """Recompute lineage figments from the current store. Idempotent.
    
    Uses entity-based clustering (fast inverted index) as primary.
    Boundary search only used within clusters for frame shift detection.
    
    max_stories: if > 0, keep only the top N narratives by member count.
    """
    all_figs = store.all()
    # Purge old narrative/derivative figments so stale clusters don't linger.
    for f in all_figs:
        if f.meta.get("edge_type") in ("narrative", "derivative"):
            store.delete(f.figment_id)

    articles = _articles(store, all_figs=all_figs)
    
    # Entity-based clustering (fast: inverted index, O(n * avg_entities))
    print(f"\n[lineage] Clustering {len(articles)} articles...")
    entity_clusters = _cluster(articles)
    print(f"[lineage]   Entity-based: {len(entity_clusters)} clusters")
    
    # Use entity clustering as primary
    clusters = entity_clusters
    
    figments: list[Figment] = []
    summaries: list[dict[str, Any]] = []

    # Sort clusters by size (largest first) so max_stories keeps the most-covered stories
    clusters.sort(key=lambda g: len(g), reverse=True)
    if max_stories > 0:
        clusters = clusters[:max_stories]

    for group in clusters:
        group = sorted(group, key=lambda f: _parse_time(f) or datetime.max.replace(tzinfo=timezone.utc))
        times = [(f, _parse_time(f)) for f in group]
        first = min(times, key=lambda ft: ft[1] or datetime.max.replace(tzinfo=timezone.utc))[0]
        members = [f.figment_id for f in group]
        sources = sorted({_normalize_source(f.meta.get("source_id")) for f in group})
        key = hashlib.sha1("|".join(members).encode()).hexdigest()[:12]
        narrative_id = f"narrative:{key}"

        # Mark first reporter + derivatives on the article figments themselves.
        updated_articles: list[Figment] = []
        for f in group:
            if f.figment_id == first.figment_id:
                f.meta["first_reporter"] = True
            else:
                f.meta["derivative_of"] = first.figment_id
                deriv_id = f"deriv:{first.figment_id}:{f.figment_id}"
                figments.append(
                    Figment.create(
                        text=f"{f.meta.get('source_id')} echoed a story first reported by "
                             f"{first.meta.get('source_id')}",
                        boundary=first.boundary.copy(),
                        meta={
                            "edge_type": "derivative",
                            "original": first.figment_id,
                            "original_url": first.meta.get("url"),
                            "derivative": f.figment_id,
                            "derivative_url": f.meta.get("url"),
                        },
                        figment_id=deriv_id,
                        sources=[first.figment_id],
                        children=[f.figment_id],
                    )
                )
            updated_articles.append(f)

        # Use first reporter's headline as the narrative title; fall back to first sentence.
        narrative_title = first.meta.get("title") or first.text.split(".")[0].strip()
        narrative_text = narrative_title

        # Detect frame shift: compare newest article's boundary to first reporter's
        newest = group[-1]
        frame_shift = False
        frame_shift_score = None
        if len(group) >= 2 and newest.figment_id != first.figment_id:
            import numpy as np
            a = first.boundary.astype(np.float64)
            b = newest.boundary.astype(np.float64)
            cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
            frame_shift = cos_sim < 0.85
            frame_shift_score = cos_sim

        narrative = Figment.create(
            text=narrative_text,
            boundary=first.boundary.copy(),
            meta={
                "edge_type": "narrative",
                "title": narrative_title,
                "members": members,
                "sources": sources,
                "first_reporter": first.figment_id,
                "first_reporter_source": first.meta.get("source_id"),
                "first_reporter_url": first.meta.get("url"),
                "entities": sorted(set().union(*[_article_entities(f) for f in group]))[:12],
                "frame_shift": frame_shift,
                "frame_shift_score": frame_shift_score,
                "frame_shift_note": (
                    f"Boundary similarity {frame_shift_score:.2f} < 0.85 threshold "
                    f"(first: {first.meta.get('source_id')}, latest: {newest.meta.get('source_id')})"
                    if frame_shift else ""
                ),
            },
            figment_id=narrative_id,
        )
        figments.append(narrative)
        summaries.append(
            {
                "narrative_id": narrative_id,
                "sources": sources,
                "members": members,
                "first_reporter": first.meta.get("source_id"),
                "first_reporter_url": first.meta.get("url"),
                "size": len(group),
            }
        )

        # Persist article meta updates (first_reporter / derivative_of).
        hidden = group[0].boundary.shape[0]
        store.upsert(updated_articles, hidden_size=hidden)

    if figments:
        hidden = figments[0].boundary.shape[0]
        store.upsert(figments, hidden_size=hidden)

    return {
        "narratives": summaries, 
        "edges": len(figments) - len(summaries),
    }


def get_narratives(store: FigmentStore, *, all_figs: list | None = None) -> list[dict[str, Any]]:
    """Read persisted narrative figments for display."""
    out = []
    for f in all_figs if all_figs is not None else store.all():
        if f.meta.get("edge_type") == "narrative":
            out.append(
                {
                    "narrative_id": f.figment_id,
                    "title": f.meta.get("title", ""),
                    "sources": f.meta.get("sources", []),
                    "members": f.meta.get("members", []),
                    "first_reporter": f.meta.get("first_reporter_source"),
                    "first_reporter_url": f.meta.get("first_reporter_url"),
                    "entities": f.meta.get("entities", []),
                    "text": f.text,
                    "frame_shift": f.meta.get("frame_shift", False),
                    "frame_shift_score": f.meta.get("frame_shift_score"),
                    "frame_shift_note": f.meta.get("frame_shift_note", ""),
                }
            )
    return out


def get_derivatives(store: FigmentStore, *, all_figs: list | None = None) -> list[dict[str, Any]]:
    out = []
    for f in all_figs if all_figs is not None else store.all():
        if f.meta.get("edge_type") == "derivative":
            out.append(
                {
                    "original_url": f.meta.get("original_url"),
                    "derivative_url": f.meta.get("derivative_url"),
                    "derivative": f.meta.get("derivative"),
                    "original": f.meta.get("original"),
                }
            )
    return out


def source_agenda(store: FigmentStore, *, all_figs: list | None = None) -> dict[str, dict[str, Any]]:
    """Per-source agenda lean: stories led vs echoed, and trust."""
    figs = all_figs if all_figs is not None else store.all()
    graph = Figtree(figs, store=store)
    analysis = graph.analyze_sources()
    narrs = get_narratives(store, all_figs=figs)
    led = {}
    echoed = {}
    for n in narrs:
        fr = _normalize_source(n["first_reporter"])
        led.setdefault(fr, 0)
        led[fr] += 1
        for s in n["sources"]:
            ns = _normalize_source(s)
            if ns != fr:
                echoed.setdefault(ns, 0)
                echoed[ns] += 1
    agenda = {}
    for src, info in analysis.items():
        agenda[src] = {
            "adjusted_trust": info["adjusted_trust"],
            "base_trust": info["base_trust"],
            "led": led.get(src, 0),
            "echoed": echoed.get(src, 0),
            "contradicting": info["contradicting"],
            "agreeing": info["agreeing"],
        }
    return agenda
