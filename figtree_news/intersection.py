"""Multi-role intersection query: find narratives containing all specified roles.

This module provides the core retrieval primitive for compositional queries
like ``"every narrative involving WHO:Trump AND WHERE:Disney World"``.

Because the association-node worker rewrites all references to point at the
canonical node ID, role IDs are already canonical at query time — no expansion
hop is needed.
"""

from __future__ import annotations

import datetime
from datetime import timezone
from typing import Any

from figtree import Figment, FigmentStore

from .lineage import get_narratives, assign_roles_to_narratives
from .trust import get_source_trusts


def _normalize(text: str) -> str:
    from .normalize import normalize as _norm
    return _norm(text)


def find_role_figments(
    store: FigmentStore,
    role: str,
    text: str,
) -> list[Figment]:
    """Find role figments matching a role + text specification.

    Matches by exact normalized_text, by substring, and by boundary similarity.
    Returns the best matches.
    """
    norm = _normalize(text)
    candidates: list[tuple[Figment, float]] = []

    for f in store.all():
        if f.meta.get("role") != role:
            continue
        fig_norm = f.meta.get("normalized") or _normalize(f.text)

        # Exact match
        if fig_norm == norm:
            candidates.append((f, 1.0))
            continue

        # Substring: the query text is contained in the figment text or vice versa
        if norm in fig_norm or fig_norm in norm:
            candidates.append((f, 0.95))
            continue

        # Word overlap
        norm_tokens = set(norm.split())
        fig_tokens = set(fig_norm.split())
        if norm_tokens and fig_tokens:
            overlap = len(norm_tokens & fig_tokens) / max(len(norm_tokens), len(fig_tokens))
            if overlap >= 0.5:
                candidates.append((f, 0.8 + overlap * 0.15))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in candidates]


def find_narratives(
    store: FigmentStore,
    roles: list[dict[str, str]],
    expand_associations: bool = True,
    require_all: bool = True,
    min_trust: float = 0.0,
    limit: int = 50,
    ranking: str = "trust_recency",
) -> list[dict[str, Any]]:
    """Find narratives containing the specified roles.

    Parameters
    ----------
    store: FigmentStore
        The LanceDB-backed store.
    roles: list of dicts with "role" (str) and "text" (str).
        Example: [{"role": "who", "text": "Donald Trump"}, {"role": "where", "text": "Disney World"}].
    expand_associations: kept for backward compatibility (no-op — role IDs
        are already canonical after association-node dedup).
    require_all: if True, a narrative must contain ALL specified roles;
        if False, it must contain at least one.
    min_trust: minimum adjusted trust score for included narratives.
    limit: max narratives to return.
    ranking: "trust_recency" (default), "trust", "recency", "sources", "frame".

    Returns
    -------
    list of narrative dicts with added ``role_matches`` and ``trust_score`` fields.
    """
    # Ensure role figments are linked to narratives
    all_figs = store.all()
    assign_roles_to_narratives(store, all_figs=all_figs)

    # Collect role figment sets (already canonical — association-node worker
    # rewrites all references to point at the canonical node ID, so the
    # expand_associations flag is now a no-op.)
    expanded_roles: list[set[str]] = []
    for role_spec in roles:
        role = role_spec.get("role", "")
        text = role_spec.get("text", "")
        figments = find_role_figments(store, role, text)
        if not figments:
            if require_all:
                return []
            continue
        expanded_roles.append({f.figment_id for f in figments})

    if not expanded_roles:
        return []

    # Build narrative membership from the store
    narratives = get_narratives(store, all_figs=all_figs)
    source_trusts = get_source_trusts(store)
    by_id = {f.figment_id: f for f in all_figs}

    # For each narrative, check which role sets it satisfies
    matching: list[dict[str, Any]] = []
    for n in narratives:
        member_ids = set(n.get("members", []))
        narrative_roles = all_figs_by_narrative(store, n.get("narrative_id", ""), all_figs)

        satisfied = 0
        role_matches: dict[str, list[str]] = {}

        for i, expanded in enumerate(expanded_roles):
            role = roles[i].get("role", "")
            overlap = expanded & narrative_roles
            if overlap:
                satisfied += 1
                # Build text list for matched role figments
                matched_texts: list[str] = []
                for fid in overlap:
                    fig = by_id.get(fid)
                    if fig:
                        matched_texts.append(fig.text)
                role_matches[role] = matched_texts

        if require_all and satisfied < len(expanded_roles):
            continue
        if not require_all and satisfied == 0:
            continue

        # Trust score: average adjusted trust of sources in this narrative
        sources = n.get("sources", [])
        trusts = [source_trusts.get(s, 0.5) for s in sources]
        avg_trust = sum(trusts) / max(1, len(trusts)) if trusts else 0.5

        if avg_trust < min_trust:
            continue

        n["role_matches"] = role_matches
        n["trust_score"] = round(avg_trust, 2)
        n["source_count"] = len(sources)
        n["satisfied_roles"] = satisfied if require_all else None
        n["total_roles"] = len(expanded_roles)
        n["members"] = list(member_ids)
        matching.append(n)

    # Rank
    if ranking == "trust_recency":
        matching.sort(
            key=lambda n: (
                n.get("trust_score", 0) * 0.6
                + (1.0 / (1 + _days_since(n.get("latest_article_date", "")))) * 0.4
            ),
            reverse=True,
        )
    elif ranking == "trust":
        matching.sort(key=lambda n: n.get("trust_score", 0), reverse=True)
    elif ranking == "recency":
        matching.sort(key=lambda n: n.get("latest_article_date", ""), reverse=True)
    elif ranking == "sources":
        matching.sort(key=lambda n: n.get("source_count", 0), reverse=True)
    elif ranking == "frame":
        matching.sort(key=lambda n: (n.get("frame_shift", False), -n.get("trust_score", 0)))
    else:
        matching.sort(key=lambda n: n.get("trust_score", 0), reverse=True)

    return matching[:limit]


def all_figs_by_narrative(
    store: FigmentStore, narrative_id: str, all_figs: list[Figment] | None = None
) -> set[str]:
    """Return the set of role_figment_ids belonging to a narrative."""
    if all_figs is None:
        all_figs = store.all()
    result: set[str] = set()
    for f in all_figs:
        if f.meta.get("story_id") == narrative_id:
            result.add(f.figment_id)
        # Also check if the article (image) belongs to this narrative
        if f.meta.get("edge_type") == "narrative" and f.figment_id == narrative_id:
            for mid in f.meta.get("members", []):
                article = next((a for a in all_figs if a.figment_id == mid), None)
                if article:
                    for rid in article.meta.get("role_figments", []):
                        result.add(rid)
    return result


def _days_since(date_str: str) -> int:
    if not date_str:
        return 9999
    try:
        dt = datetime.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        now = datetime.datetime.now(timezone.utc)
        return max(0, (now - dt).days)
    except Exception:
        return 9999