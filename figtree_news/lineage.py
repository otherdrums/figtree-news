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
    "The", "This", "That", "These", "Those", "We", "They", "He", "She", "It",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December", "Reuters", "AP", "BBC",
    "Guardian", "NPR", "Aljazeera", "CNN", "NYT", "New York Times",
    "United States", "US", "USA", "U.S.", "U.S.A.", "America", "American",
    "United Kingdom", "UK", "U.K.", "Britain", "British", "England",
    "Canada", "Australian", "China", "Chinese",
    "Russia", "Russian", "Ukraine", "Ukrainian", "Iran", "Iranian",
    "Israel", "Israeli", "Palestinian", "Palestine", "Gaza", "Hamas",
    "President", "Prime", "Minister", "Government", "Official",
    "World", "News", "Report", "Story", "Article", "People", "Some", "Amid",
    "After", "Before", "During", "Following", "Because", "Despite",
    "About", "Over", "Under", "Between", "Against", "Through", "Into",
    "According", "Said", "Says", "Told", "New", "Old", "First", "Last",
    "More", "Most", "Many", "Much", "Several", "One", "Two", "Three",
    "South", "North", "East", "West",
    "Watch", "Video", "Live", "Updated", "Breaking",
}


def _entities(text: str) -> set[str]:
    if not text:
        return set()
    toks = {t.strip(".") for t in _ENTITY_RE.findall(text)}
    return {t for t in toks if t not in _STOP and len(t) > 2}


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


def _cluster(articles: list[Figment]) -> list[list[Figment]]:
    """Group articles by Jaccard entity overlap (>= 0.25) instead of single shared entity."""
    by_id = {f.figment_id: f for f in articles}
    ent_sets = {f.figment_id: _entities(f.text) for f in articles}
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

    fids = list(ent_sets.keys())
    for i in range(len(fids)):
        for j in range(i + 1, len(fids)):
            if jaccard(fids[i], fids[j]) >= 0.25:
                union(fids[i], fids[j])

    groups: dict[str, list[Figment]] = {}
    for fid in parent:
        groups.setdefault(find(fid), []).append(by_id[fid])
    return [g for g in groups.values() if len(g) >= 2]


def compute_lineage(store: FigmentStore) -> dict[str, Any]:
    """Recompute lineage figments from the current store. Idempotent."""
    all_figs = store.all()
    # Purge old narrative/derivative figments so stale clusters don't linger.
    for f in all_figs:
        if f.meta.get("edge_type") in ("narrative", "derivative"):
            store.delete(f.figment_id)

    articles = _articles(store, all_figs=all_figs)
    clusters = _cluster(articles)
    figments: list[Figment] = []
    summaries: list[dict[str, Any]] = []

    for group in clusters:
        group = sorted(group, key=lambda f: _parse_time(f) or datetime.max.replace(tzinfo=timezone.utc))
        times = [(f, _parse_time(f)) for f in group]
        first = min(times, key=lambda ft: ft[1] or datetime.max.replace(tzinfo=timezone.utc))[0]
        members = [f.figment_id for f in group]
        sources = sorted({f.meta.get("source_id") for f in group})
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
        if len(group) >= 2 and newest.figment_id != first.figment_id:
            import numpy as np
            a = first.boundary.astype(np.float64)
            b = newest.boundary.astype(np.float64)
            cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
            frame_shift = cos_sim < 0.85

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
                "entities": sorted(set().union(*[_entities(f.text) for f in group]))[:12],
                "frame_shift": frame_shift,
                "frame_shift_note": "Coverage framing shifted from first report" if frame_shift else "",
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

    return {"narratives": summaries, "edges": len(figments) - len(summaries)}


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
        fr = n["first_reporter"]
        led.setdefault(fr, 0)
        led[fr] += 1
        for s in n["sources"]:
            if s != fr:
                echoed.setdefault(s, 0)
                echoed[s] += 1
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
