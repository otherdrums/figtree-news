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
import re
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


_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    re.compile(r'subscribe\s+(to|now)', re.I),
    re.compile(r'download\s+the\s.*app', re.I),
    re.compile(r'become\s+a\s.*patriot', re.I),
    re.compile(r'click\s+here', re.I),
    re.compile(r'watch\s+(more|24/7|247)', re.I),
    re.compile(r'sign\s+up', re.I),
    re.compile(r'newsletter', re.I),
    re.compile(r'licensing@', re.I),
    re.compile(r'all rights reserved', re.I),
    re.compile(r'fox news channel \(fnc\)|fnc is', re.I),
    re.compile(r'ms\s+now', re.I),
    re.compile(r'my\s+source\s+for\s+news', re.I),
    re.compile(r'cbs news 24.?7', re.I),
    re.compile(r'available for archive', re.I),
    re.compile(r'by emailing', re.I),
    re.compile(r'©', re.I),
    re.compile(r'be part of it', re.I),
    re.compile(r'touch to listen to cbs news', re.I),
]


def _is_boilerplate(text: str) -> bool:
    lower = text.lower()
    for pat in _BOILERPLATE_PATTERNS:
        if pat.search(lower):
            return True
    url_chars = sum(1 for c in text if c in ':/?#[]@!$&()*+,;=')
    if len(text) > 20 and url_chars / max(len(text), 1) > 0.15:
        return True
    return False


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
        if f.kind == "article" and f.meta.get("source_id")
    ]


def _cluster_by_roles(
    store: FigmentStore | None,
    articles: list[Figment],
    min_shared: int = 2,
) -> list[list[Figment]]:
    """Cluster articles by shared role figments.

    Two articles share a narrative if they share >= min_shared role figment IDs.
    Role figments are deduplicated by exact normalized-text hash, so sharing a
    role figment ID means semantic identity.  When the association-node worker
    has merged variants, all references already point at the canonical node ID,
    so no expansion is needed.

    Articles without role figments (not yet decomposed) are left as singletons.
    The LLM-based split step (``split_narratives_by_llm_labels``) is the safety
    net for false positives from a low ``min_shared``.
    """
    by_id = {f.figment_id: f for f in articles}
    article_roles: dict[str, set[str]] = {}

    for f in articles:
        role_ids = set(f.meta.get("role_figments", []))
        article_roles[f.figment_id] = role_ids

    # ── Role filtering ────────────────────────────────────────────────────
    # Build role DF and fetch role figment texts for boilerplate/self-ref checks.
    role_df: dict[str, int] = {}
    for roles in article_roles.values():
        for rid in roles:
            role_df[rid] = role_df.get(rid, 0) + 1

    source_name_map: dict[str, str] = {}
    for f in articles:
        sid = (f.meta.get("source_id") or "").replace("_", " ").replace("-", " ").lower().strip()
        source_name_map[f.figment_id] = sid

    role_text: dict[str, str] = {}
    if store is not None:
        try:
            all_figs = store.all()
            for f in all_figs:
                rid = f.figment_id
                if rid in role_df:
                    role_text[rid] = f.text or ""
        except Exception:
            pass

    n_articles = len(articles)
    df_threshold = max(5, int(0.20 * n_articles))

    for aid in list(article_roles.keys()):
        filtered: set[str] = set()
        src_name = source_name_map.get(aid, "")
        for rid in article_roles[aid]:
            # high document frequency → generic/bridge role
            if role_df.get(rid, 0) > df_threshold:
                continue
            # self‑referential: role text names the source itself
            rtext = role_text.get(rid, "")
            if src_name and rtext and src_name in rtext.lower():
                continue
            # boilerplate pattern match on role text
            if _is_boilerplate(rtext):
                continue
            filtered.add(rid)
        article_roles[aid] = filtered

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


def _cluster_by_boundary(articles: list[Figment], threshold: float = 0.98, hours: int = 48) -> list[list[Figment]]:
    """Fallback: cluster by boundary cosine similarity within time window.

    Used only when no role figments exist (external LLM not configured).
    Boundary vectors are unreliable for very short snippets (e.g. YouTube
    video descriptions), so we require a minimum text length on both articles.
    Threshold is tunable — start conservative (0.95) and adjust based on results.
    """
    by_id = {f.figment_id: f for f in articles}
    times = {f.figment_id: _parse_time(f) for f in articles}
    parent = {f.figment_id: f.figment_id for f in articles}

    def _text_len(f: Figment) -> int:
        return len((f.meta.get("title") or f.text or "").strip())

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
            # Skip short snippets; boundary vectors are not discriminative enough
            if _text_len(by_id[a]) < 120 or _text_len(by_id[b]) < 120:
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


def _singleton_narrative(article: Figment) -> tuple[Figment, dict[str, Any]]:
    """Create a one-article narrative for a single article."""
    now_utc = datetime.now(timezone.utc)
    narrative_id = f"narrative:{article.figment_id[:12]}"
    narrative_title = article.meta.get("title") or article.text.split(".")[0].strip()
    narrative = Figment.create(
        text=narrative_title,
        boundary=article.boundary.copy(),
        meta={
            "edge_type": "narrative",
            "title": narrative_title,
            "members": [article.figment_id],
            "sources": sorted({_normalize_source(article.meta.get("source_id"))}),
            "first_reporter": article.figment_id,
            "first_reporter_source": article.meta.get("source_id"),
            "first_reporter_url": article.meta.get("url"),
            "entities": [],
            "frame_shift": False,
            "frame_shift_score": None,
            "frame_shift_note": "",
            "latest_article_date": "",
            "first_seen": now_utc.isoformat(),
            "last_updated": "",
            "new_article_count": 0,
        },
        figment_id=narrative_id,
        kind="edge",
    )
    summary = {
        "narrative_id": narrative_id,
        "sources": sorted({_normalize_source(article.meta.get("source_id"))}),
        "members": [article.figment_id],
        "first_reporter": article.meta.get("source_id"),
        "first_reporter_url": article.meta.get("url"),
        "size": 1,
        "latest_article_date": "",
        "first_seen": now_utc.isoformat(),
        "last_updated": "",
        "new_article_count": 0,
    }
    return narrative, summary


def compute_lineage(store: FigmentStore, max_stories: int = 0, all_figs: list | None = None) -> dict[str, Any]:
    """Recompute lineage figments from the current store. Idempotent.

    Uses role figment clustering when available. When no roles exist (articles
    not yet decomposed), falls back to one narrative per article — the external
    LLM phases (Phase 2c, Phase 4b) handle merging via semantic evaluation.

    Boundary similarity is NOT used for clustering decisions. It is recorded
    separately for data-collection purposes (see ``boundary_data.jsonl``).
    """
    all_figs = all_figs if all_figs is not None else store.all()
    for f in all_figs:
        if f.meta.get("edge_type") in ("narrative", "derivative"):
            store.delete(f.figment_id)

    articles = _articles(store, all_figs=all_figs)

    has_roles = _has_role_figments(articles)
    print(f"\n[lineage] {len(articles)} articles, has_roles={has_roles}")

    if has_roles:
        print("[lineage]   Using role figment clustering")
        clusters = _cluster_by_roles(store, articles)
    else:
        # No boundary fallback — create singletons and let the LLM merge them.
        print("[lineage]   No role figments — creating singletons (LLM will merge)")
        clusters = [[a] for a in articles]

    print(f"[lineage]   {len(clusters)} clusters")

    figments: list[Figment] = []
    summaries: list[dict[str, Any]] = []

    if not clusters:
        print("[lineage]   No clusters; falling back to one narrative per article")
        for article in articles:
            narrative, summary = _singleton_narrative(article)
            figments.append(narrative)
            summaries.append(summary)
            article.meta["first_reporter"] = True
        if figments:
            hidden = figments[0].boundary.shape[0]
            store.upsert(articles, hidden_size=hidden)
            store.upsert(figments, hidden_size=hidden)
        return {"narratives": summaries, "edges": 0}

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
                        kind="edge",
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
            kind="edge",
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
            existing = f.meta.get("story_id")
            if existing == n["narrative_id"]:
                continue
            f.meta["story_id"] = n["narrative_id"]
            store.upsert([f], hidden_size=f.boundary.shape[0])
            updated += 1
    return updated


def _rebuild_narrative_from_articles(
    articles: list[Figment],
    old_narratives: list[Figment],
    *,
    suffix: str = "",
) -> tuple[Figment, list[Figment], dict[str, Any]]:
    """Build a new narrative figment and derivative edges from a set of articles.

    ``old_narratives`` is used to preserve the earliest ``first_seen`` timestamp.
    Returns (narrative_figment, list_of_derivative_edges, summary_dict).
    """
    if not articles:
        raise ValueError("articles must not be empty")
    articles = sorted(
        articles,
        key=lambda f: _parse_time(f) or datetime.max.replace(tzinfo=timezone.utc),
    )
    first = articles[0]
    sources = sorted({_normalize_source(f.meta.get("source_id", "unknown")) for f in articles})
    members = [f.figment_id for f in articles]
    key = hashlib.sha1("|".join(members).encode()).hexdigest()[:12]
    narrative_id = f"narrative:{key}{suffix}"

    title = first.meta.get("title") or first.text.split(".")[0].strip()
    times = [(_parse_time(f), f) for f in articles]
    latest_time = max((t for t, _ in times if t), default=None)
    latest_iso = latest_time.isoformat() if latest_time else ""
    now_utc = datetime.now(timezone.utc)
    first_seen = min(
        (nf.meta.get("first_seen") or now_utc.isoformat() for nf in old_narratives if nf),
        default=now_utc.isoformat(),
    )
    new_count = sum(1 for t, _ in times if t and t >= now_utc - timedelta(days=1))

    derivatives: list[Figment] = []
    for f in articles[1:]:
        deriv_id = f"deriv:{first.figment_id}:{f.figment_id}"
        deriv = Figment.create(
            text=f"{f.meta.get('source_id', 'unknown')} echoed a story first reported by {first.meta.get('source_id', 'unknown')}",
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
            kind="edge",
        )
        derivatives.append(deriv)
        f.meta["derivative_of"] = first.figment_id

    narrative = Figment.create(
        text=title,
        boundary=first.boundary.copy(),
        meta={
            "edge_type": "narrative",
            "title": title,
            "members": members,
            "sources": sources,
            "first_reporter": first.figment_id,
            "first_reporter_source": first.meta.get("source_id"),
            "first_reporter_url": first.meta.get("url"),
            "entities": [],
            "frame_shift": False,
            "frame_shift_score": None,
            "frame_shift_note": "",
            "latest_article_date": latest_iso,
            "first_seen": first_seen,
            "last_updated": latest_iso,
            "new_article_count": new_count,
        },
        figment_id=narrative_id,
        kind="edge",
    )
    summary = {
        "narrative_id": narrative_id,
        "title": title,
        "sources": sources,
        "members": members,
        "first_reporter": first.meta.get("source_id"),
        "first_reporter_url": first.meta.get("url"),
        "entities": [],
        "text": title,
        "frame_shift": False,
        "frame_shift_score": None,
        "frame_shift_note": "",
        "latest_article_date": latest_iso,
        "first_seen": first_seen,
        "last_updated": latest_iso,
        "new_article_count": new_count,
    }
    return narrative, derivatives, summary


def merge_narratives_by_llm_labels(
    store: FigmentStore,
    labels: list[dict[str, Any]],
    *,
    all_figs: list | None = None,
) -> dict[str, Any]:
    """Merge existing narrative clusters when the LLM says two articles are same-event.

    Operates on the narrative figments already persisted by ``compute_lineage``.
    For every label with ``same_event=True``, find the narratives containing the
    two articles and union them into a single narrative. Old narratives are
    deleted and replaced with merged ones.

    Returns the merged narrative summaries in the same shape as
    ``compute_lineage`` (``{"narratives": [...], "edges": int}``).
    """
    figs = all_figs if all_figs is not None else store.all()
    narr_figs = [f for f in figs if f.meta.get("edge_type") == "narrative"]
    if not narr_figs or not labels:
        return {"narratives": [], "edges": 0}

    by_id = {f.figment_id: f for f in figs}
    article_to_narrative: dict[str, str] = {}
    for nf in narr_figs:
        for mid in nf.meta.get("members", []):
            article_to_narrative[mid] = nf.figment_id

    parent = {nf.figment_id: nf.figment_id for nf in narr_figs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    merges = 0
    for label in labels:
        if not label.get("same_event"):
            continue
        a1 = label.get("a1")
        a2 = label.get("a2")
        if not a1 or not a2:
            continue
        n1 = article_to_narrative.get(a1)
        n2 = article_to_narrative.get(a2)
        if n1 and n2 and n1 != n2:
            union(n1, n2)
            merges += 1

    if merges == 0:
        return {"narratives": [], "edges": 0}

    groups: dict[str, list[Figment]] = {}
    for nf in narr_figs:
        groups.setdefault(find(nf.figment_id), []).append(nf)

    hidden_size = narr_figs[0].boundary.shape[0]
    new_narratives: list[Figment] = []
    new_derivatives: list[Figment] = []
    merged_summaries: list[dict[str, Any]] = []
    all_affected_articles: list[Figment] = []

    for root, group in groups.items():
        if len(group) == 1:
            continue
        member_ids: list[str] = []
        for nf in group:
            member_ids.extend(nf.meta.get("members", []))
        member_ids = sorted(set(member_ids))
        if len(member_ids) < 2:
            continue

        group_articles = [by_id[mid] for mid in member_ids if mid in by_id]
        if not group_articles:
            continue

        for nf in group:
            store.delete(nf.figment_id)

        narrative, derivatives, summary = _rebuild_narrative_from_articles(
            group_articles, group
        )
        new_narratives.append(narrative)
        new_derivatives.extend(derivatives)
        merged_summaries.append(summary)
        all_affected_articles.extend(group_articles)

    if new_narratives:
        store.upsert(new_narratives, hidden_size=hidden_size)
    if new_derivatives:
        store.upsert(new_derivatives, hidden_size=hidden_size)
    if all_affected_articles:
        store.upsert(all_affected_articles, hidden_size=hidden_size)

    deleted_count = sum(len(g) for g in groups.values() if len(g) > 1)
    print(f"[lineage] LLM merge: {len(merged_summaries)} merged narratives from {deleted_count} originals")
    return {
        "narratives": merged_summaries,
        "edges": len(new_derivatives),
    }


def split_narratives_by_llm_labels(
    store: FigmentStore,
    labels: list[dict[str, Any]],
    *,
    all_figs: list | None = None,
) -> dict[str, Any]:
    """Split existing narrative clusters when the LLM says two articles are different events.

    Operates on the narrative figments already persisted by ``compute_lineage``.
    For every label with ``same_event=False``, if the two articles currently live
    in the same narrative, that narrative is dissolved and each article becomes
    its own single-article narrative. This is the primary defense against the
    boundary-similarity fallback clustering unrelated short snippets.

    Returns the split narrative summaries in the same shape as
    ``compute_lineage`` (``{"narratives": [...], "edges": int}``).
    """
    figs = all_figs if all_figs is not None else store.all()
    narr_figs = [f for f in figs if f.meta.get("edge_type") == "narrative"]
    if not narr_figs or not labels:
        return {"narratives": [], "edges": 0}

    by_id = {f.figment_id: f for f in figs}
    article_to_narrative: dict[str, str] = {}
    for nf in narr_figs:
        for mid in nf.meta.get("members", []):
            article_to_narrative[mid] = nf.figment_id

    # Find which narratives need to be split because of a different-event label
    split_narrative_ids: set[str] = set()
    for label in labels:
        if label.get("same_event"):
            continue
        a1 = label.get("a1")
        a2 = label.get("a2")
        if not a1 or not a2:
            continue
        n1 = article_to_narrative.get(a1)
        n2 = article_to_narrative.get(a2)
        if n1 and n2 and n1 == n2:
            split_narrative_ids.add(n1)

    if not split_narrative_ids:
        return {"narratives": [], "edges": 0}

    hidden_size = narr_figs[0].boundary.shape[0]
    new_narratives: list[Figment] = []
    new_derivatives: list[Figment] = []
    split_summaries: list[dict[str, Any]] = []
    all_affected_articles: list[Figment] = []
    member_ids_set = {mid for nf in narr_figs for mid in nf.meta.get("members", [])}

    for nf in narr_figs:
        if nf.figment_id not in split_narrative_ids:
            continue
        member_ids = nf.meta.get("members", [])
        nf_articles = [by_id[mid] for mid in member_ids if mid in by_id]
        if len(nf_articles) < 2:
            continue

        # Clear derivative_of from members so they can stand alone
        for art in nf_articles:
            art.meta.pop("derivative_of", None)

        store.delete(nf.figment_id)
        # Delete derivative edges where both original AND derivative belong
        # to this dissolved narrative (over-broad deletion was the bug).
        for df in figs:
            if df.meta.get("edge_type") != "derivative":
                continue
            orig = df.meta.get("original")
            deriv = df.meta.get("derivative")
            if orig in member_ids_set and deriv in member_ids_set:
                store.delete(df.figment_id)

        for art in nf_articles:
            narrative, derivatives, summary = _rebuild_narrative_from_articles(
                [art], [nf]
            )
            new_narratives.append(narrative)
            new_derivatives.extend(derivatives)
            split_summaries.append(summary)

        all_affected_articles.extend(nf_articles)

    if new_narratives:
        store.upsert(new_narratives, hidden_size=hidden_size)
    if new_derivatives:
        store.upsert(new_derivatives, hidden_size=hidden_size)
    if all_affected_articles:
        store.upsert(all_affected_articles, hidden_size=hidden_size)

    print(f"[lineage] LLM split: dissolved {len(split_narrative_ids)} bad narratives into {len(split_summaries)} single-article narratives")
    return {
        "narratives": split_summaries,
        "edges": len(new_derivatives),
    }
