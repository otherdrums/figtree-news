"""Lineage: narrative clustering via role figments, who broke it first, derivatives.

Clustering uses role figments (WHO/WHAT/WHERE/WHEN/WHY/HOW) extracted by the
external LLM decomposition engine. Two articles share a narrative if they share
>= 2 role figments (exact semantic match via normalized text deduplication).

Falls back to boundary similarity when the external LLM is not configured and
no role figments exist.

Persists findings as figments so the web UI can query them directly:
* ``narrative:{key}`` — one figment per cluster of articles about the same story.
* ``derivative:{orig}:{der}`` — edge marking ``der`` as echoing ``orig``.

Deterministic ids make the whole step idempotent.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import numpy as np

from figtree import Figment, FigmentStore, Figtree


def _normalize_source(source_id: str) -> str:
    """Collapse same-org feed variants (e.g. france24 + france24_yt -> france24)."""
    for suffix in ("_yt", "_rss", "_tw", "_fb"):
        if source_id.endswith(suffix):
            return source_id[: -len(suffix)]
    return source_id


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


def _cluster_by_roles(articles: list[Figment], min_shared: int = 2) -> list[list[Figment]]:
    """Cluster articles by shared role figments.

    Two articles share a narrative if they share >= min_shared role figment IDs.
    Role figments are deduplicated by hash(role + normalized_text), so sharing
    a role figment ID means semantic identity.

    Articles without role figments (not yet decomposed) are left as singletons.
    """
    by_id = {f.figment_id: f for f in articles}
    article_roles: dict[str, set[str]] = {}

    for f in articles:
        role_ids = set(f.meta.get("role_figments", []))
        article_roles[f.figment_id] = role_ids

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

    role_to_articles: dict[str, list[str]] = {}
    for aid, roles in article_roles.items():
        for rid in roles:
            role_to_articles.setdefault(rid, []).append(aid)

    for rid, aids in role_to_articles.items():
        aids_unique = list(set(aids))
        for i in range(len(aids_unique)):
            for j in range(i + 1, len(aids_unique)):
                a, b = aids_unique[i], aids_unique[j]
                if find(a) == find(b):
                    continue
                shared = article_roles[a] & article_roles[b]
                if len(shared) >= min_shared:
                    union(a, b)

    groups: dict[str, list[Figment]] = {}
    for fid in parent:
        groups.setdefault(find(fid), []).append(by_id[fid])
    return list(groups.values())


def _cluster_by_boundary(articles: list[Figment], threshold: float = 0.95, hours: int = 48) -> list[list[Figment]]:
    """Fallback: cluster by boundary cosine similarity within time window.

    Used only when no role figments exist (external LLM not configured).
    Threshold is tunable — start conservative (0.95) and adjust based on results.
    """
    by_id = {f.figment_id: f for f in articles}
    times = {f.figment_id: _parse_time(f) for f in articles}
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

    def _hours_apart(a: str, b: str) -> float | None:
        ta, tb = times.get(a), times.get(b)
        if ta is None or tb is None:
            return None
        return abs((ta - tb).total_seconds()) / 3600.0

    def _boundary_cos(a: str, b: str) -> float:
        ba = by_id[a].boundary.astype(np.float64)
        bb = by_id[b].boundary.astype(np.float64)
        return float(np.dot(ba, bb) / (np.linalg.norm(ba) * np.linalg.norm(bb) + 1e-10))

    fids = list(by_id.keys())
    for i in range(len(fids)):
        for j in range(i + 1, len(fids)):
            a, b = fids[i], fids[j]
            if find(a) == find(b):
                continue
            h = _hours_apart(a, b)
            if h is not None and h <= hours:
                cos = _boundary_cos(a, b)
                if cos >= threshold:
                    union(a, b)

    groups: dict[str, list[Figment]] = {}
    for fid in parent:
        groups.setdefault(find(fid), []).append(by_id[fid])
    return list(groups.values())


def _has_role_figments(articles: list[Figment]) -> bool:
    """Check if any articles have role figments (decomposition completed)."""
    return any(f.meta.get("role_figments") for f in articles)


def _extract_role_entities(articles: list[Figment], all_figs: list[Figment]) -> list[str]:
    """Extract entity names from role figments for narrative metadata."""
    by_id = {f.figment_id: f for f in all_figs}
    entities: set[str] = set()
    for art in articles:
        for rid in art.meta.get("role_figments", []):
            role_fig = by_id.get(rid)
            if role_fig:
                text = role_fig.meta.get("normalized") or role_fig.text
                if text and len(text) >= 3:
                    entities.add(text)
    return sorted(entities)[:12]


def compute_lineage(store: FigmentStore, max_stories: int = 0) -> dict[str, Any]:
    """Recompute lineage figments from the current store. Idempotent.

    Uses role figment clustering when available, boundary fallback otherwise.
    """
    all_figs = store.all()
    for f in all_figs:
        if f.meta.get("edge_type") in ("narrative", "derivative"):
            store.delete(f.figment_id)

    articles = _articles(store, all_figs=all_figs)

    has_roles = _has_role_figments(articles)
    print(f"\n[lineage] Clustering {len(articles)} articles...")

    if has_roles:
        print("[lineage]   Using role figment clustering")
        clusters = _cluster_by_roles(articles)
    else:
        print("[lineage]   No role figments — using boundary similarity fallback")
        clusters = _cluster_by_boundary(articles)

    print(f"[lineage]   {len(clusters)} clusters")

    figments: list[Figment] = []
    summaries: list[dict[str, Any]] = []

    clusters.sort(
        key=lambda g: max((_parse_time(f) or datetime.max.replace(tzinfo=timezone.utc)) for f in g),
        reverse=True,
    )
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

        narrative_title = first.meta.get("title") or first.text.split(".")[0].strip()

        newest = group[-1]
        frame_shift = False
        frame_shift_score = None
        if len(group) >= 2 and newest.figment_id != first.figment_id:
            a = first.boundary.astype(np.float64)
            b = newest.boundary.astype(np.float64)
            cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
            frame_shift = cos_sim < 0.85
            frame_shift_score = cos_sim

        latest_time = max((ft[1] for ft in times if ft[1]), default=None)
        latest_iso = latest_time.isoformat() if latest_time else ""

        now_utc = datetime.now(timezone.utc)
        day_ago = now_utc - timedelta(days=1)
        new_count = sum(1 for ft in times if ft[1] and ft[1] >= day_ago)
        first_seen_iso = now_utc.isoformat()

        entities = _extract_role_entities(group, all_figs) if has_roles else []

        narrative = Figment.create(
            text=narrative_title,
            boundary=first.boundary.copy(),
            meta={
                "edge_type": "narrative",
                "title": narrative_title,
                "members": members,
                "sources": sources,
                "first_reporter": first.figment_id,
                "first_reporter_source": first.meta.get("source_id"),
                "first_reporter_url": first.meta.get("url"),
                "entities": entities,
                "frame_shift": frame_shift,
                "frame_shift_score": frame_shift_score,
                "frame_shift_note": (
                    f"Boundary similarity {frame_shift_score:.2f} < 0.85 threshold "
                    f"(first: {first.meta.get('source_id')}, latest: {newest.meta.get('source_id')})"
                    if frame_shift else ""
                ),
                "latest_article_date": latest_iso,
                "first_seen": first_seen_iso,
                "last_updated": latest_iso,
                "new_article_count": new_count,
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
                "latest_article_date": latest_iso,
                "first_seen": first_seen_iso,
                "last_updated": latest_iso,
                "new_article_count": new_count,
            }
        )

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
                    "latest_article_date": f.meta.get("latest_article_date", ""),
                    "first_seen": f.meta.get("first_seen", ""),
                    "last_updated": f.meta.get("last_updated", ""),
                    "new_article_count": f.meta.get("new_article_count", 0),
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


def assign_roles_to_narratives(store: FigmentStore, *, all_figs: list | None = None) -> int:
    """Link role figments to their parent narrative via story_id meta key."""
    figs = all_figs if all_figs is not None else store.all()
    narrs = get_narratives(store, all_figs=figs)
    updated = 0
    for n in narrs:
        member_set = set(n["members"])
        for f in figs:
            article_id = f.meta.get("article_id")
            if not article_id or article_id not in member_set:
                continue
            refs = f.meta.get("references", [])
            if member_set.isdisjoint(refs):
                continue
            existing = f.meta.get("story_id")
            if existing == n["narrative_id"]:
                continue
            f.meta["story_id"] = n["narrative_id"]
            store.upsert([f], hidden_size=f.boundary.shape[0])
            updated += 1
    return updated
