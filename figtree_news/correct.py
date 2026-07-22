"""Correction engine: collect pending corrections, confirm at threshold, apply.

Corrections are figments with ``edge_type="correction"``. The engine groups
unapplied corrections by (correction_type, target_narrative, target_article),
and applies those with ≥ threshold confirmations from separate eval runs.
"""

from __future__ import annotations

import time
from typing import Any

from figtree import Figment, FigmentStore


def _article_images(store: FigmentStore) -> list[Figment]:
    return [
        f for f in store.all()
        if f.meta.get("is_image") and f.meta.get("source_id") and not f.is_edge()
    ]


def collect_pending_corrections(
    store: FigmentStore,
    confirmation_threshold: int = 2,
) -> list[dict[str, Any]]:
    """Group unapplied correction figments and return those above threshold."""
    all_figs = store.all()
    corrections = [f for f in all_figs if f.meta.get("edge_type") == "correction" and not f.meta.get("applied", False)]

    groups: dict[tuple[str, str, str], list[Figment]] = {}
    for c in corrections:
        ctype = c.meta.get("correction_type", "")
        target_narrative = c.meta.get("target_narrative", "")
        target_article = c.meta.get("target_article", "")
        key = (ctype, target_narrative, target_article)
        groups.setdefault(key, []).append(c)

    confirmed = []
    for (ctype, tnid, taid), corrs in groups.items():
        confirmations = max(c.meta.get("confirmation_count", 1) for c in corrs)
        if confirmations >= confirmation_threshold:
            confirmed.append({
                "correction_type": ctype,
                "target_narrative": tnid,
                "target_article": taid,
                "confirmations": confirmations,
                "reason": corrs[0].meta.get("reason", ""),
                "correction_ids": [c.figment_id for c in corrs],
            })
    return sorted(confirmed, key=lambda x: -x["confirmations"])


def apply_correction(
    store: FigmentStore,
    correction: dict[str, Any],
) -> bool:
    """Apply a single confirmed correction. Returns True if successful."""
    ctype = correction["correction_type"]
    target_narrative = correction["target_narrative"]
    target_article = correction["target_article"]

    print(f"[correct] applying {ctype}: narrative={target_narrative[:8]} article={target_article[:8]}")

    all_figs = store.all()
    narrative_fig = None
    for f in all_figs:
        if f.figment_id == target_narrative:
            narrative_fig = f
            break
    if narrative_fig is None:
        print(f"[correct] narrative {target_narrative[:8]} not found, skipping")
        return False

    if ctype == "remove":
        return _remove_article(store, narrative_fig, target_article, all_figs, correction)

    elif ctype == "merge":
        return _merge_article(store, narrative_fig, target_article, all_figs, correction)

    elif ctype == "split":
        return _split_article(store, narrative_fig, target_article, all_figs, correction)

    return False


def _remove_article(
    store: FigmentStore, narrative: Figment, article_id: str,
    all_figs: list[Figment], correction: dict[str, Any],
) -> bool:
    members = list(narrative.meta.get("members", []))
    if article_id not in members:
        return False
    members.remove(article_id)
    narrative.meta["members"] = members
    _rebuild_narrative_meta(narrative, members, all_figs)
    _mark_corrections_applied(store, all_figs, correction["correction_ids"])
    hidden = narrative.boundary.shape[0]
    store.upsert([narrative], hidden_size=hidden)
    return True


def _merge_article(
    store: FigmentStore, narrative: Figment, article_id: str,
    all_figs: list[Figment], correction: dict[str, Any],
) -> bool:
    members = list(narrative.meta.get("members", []))
    if article_id in members:
        return False
    members.append(article_id)
    narrative.meta["members"] = members
    _rebuild_narrative_meta(narrative, members, all_figs)
    _mark_corrections_applied(store, all_figs, correction["correction_ids"])
    hidden = narrative.boundary.shape[0]
    store.upsert([narrative], hidden_size=hidden)
    return True


def _split_article(
    store: FigmentStore, narrative: Figment, article_id: str,
    all_figs: list[Figment], correction: dict[str, Any],
) -> bool:
    members = list(narrative.meta.get("members", []))
    if article_id not in members:
        return False
    members.remove(article_id)

    if len(members) < 2:
        return False

    narrative.meta["members"] = members
    _rebuild_narrative_meta(narrative, members, all_figs)

    # Create a new singleton narrative for the split-off article
    article_fig = None
    for f in all_figs:
        if f.figment_id == article_id:
            article_fig = f
            break
    if article_fig:
        import hashlib
        new_nid = hashlib.sha256(f"narr:{article_id}:{time.time()}".encode()).hexdigest()[:16]
        sources = [article_fig.meta.get("source_id", "")]
        new_narrative = Figment.create(
            text=article_fig.meta.get("title") or article_fig.text[:80],
            boundary=article_fig.boundary.copy(),
            meta={
                "edge_type": "narrative",
                "title": article_fig.meta.get("title") or article_fig.text[:80],
                "members": [article_id],
                "sources": sources,
                "first_reporter": article_id,
                "first_reporter_source": article_fig.meta.get("source_id"),
                "first_reporter_url": article_fig.meta.get("url"),
                "entities": [],
                "frame_shift": False,
                "frame_shift_score": None,
                "frame_shift_note": "",
                "split_from": narrative.figment_id,
            },
            figment_id=new_nid,
        )
        store.upsert([new_narrative], hidden_size=narrative.boundary.shape[0])

    _mark_corrections_applied(store, all_figs, correction["correction_ids"])
    hidden = narrative.boundary.shape[0]
    store.upsert([narrative], hidden_size=hidden)
    return True


def _rebuild_narrative_meta(narrative: Figment, member_ids: list[str], all_figs: list[Figment]):
    """Recompute sources, entities, and title after member changes."""
    by_id = {f.figment_id: f for f in all_figs}
    members = [by_id[mid] for mid in member_ids if mid in by_id]

    sources = sorted(set(
        m.meta.get("source_id", "") for m in members if m.meta.get("source_id")
    ))
    narrative.meta["sources"] = sources
    narrative.meta["members"] = member_ids

    if members:
        narrative.meta["first_reporter"] = members[0].figment_id
        narrative.meta["first_reporter_source"] = members[0].meta.get("source_id")
        narrative.meta["first_reporter_url"] = members[0].meta.get("url")
        narrative.meta["title"] = members[0].meta.get("title") or members[0].text[:80]
        narrative.text = narrative.meta["title"]


def _mark_corrections_applied(store: FigmentStore, all_figs: list[Figment], correction_ids: list[str]):
    """Mark all specified correction figments as applied."""
    to_update = []
    for f in all_figs:
        if f.figment_id in correction_ids:
            f.meta["applied"] = True
            to_update.append(f)
    if to_update:
        hidden = to_update[0].boundary.shape[0]
        store.upsert(to_update, hidden_size=hidden)


def confirm_and_apply(
    store: FigmentStore,
    confirmation_threshold: int = 2,
) -> dict[str, Any]:
    """Collect, confirm, and apply pending corrections. Returns stats."""
    pending = collect_pending_corrections(store, confirmation_threshold)
    applied = 0
    for correction in pending:
        if apply_correction(store, correction):
            applied += 1
    return {"corrections_pending_total": len(pending), "corrections_applied": applied}
