"""Association figments: co-reference linking for surface-form variants.

Roles extracted by the decomposition engine are canonicalized by exact
text match (lower + strip punctuation). Surface-form variants such as
``"Donald Trump"``, ``"Trump"``, and ``"DJT"`` therefore become separate
role figments. The association layer links them so downstream queries can
expand through aliases and retrieve every role instance for the same entity.

All associations are first-class figments (edge_type="association") stored
in the same LanceDB table, so they're traversable and queryable like
everything else in the system.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any

import numpy as np

from figtree import Figment, FigmentStore

# Characters / tokens stripped before comparison — mirrors decompose._normalize_text.
_NORMALIZE_RE = re.compile(r"[^\w\s]")


def _normalize(text: str) -> str:
    return _NORMALIZE_RE.sub("", text.lower()).strip()


def _boundary_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    dot = float(np.dot(a_f, b_f))
    n = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    return dot / n if n > 0 else 0.0


def _string_overlap(a: str, b: str) -> float:
    """Fraction of the shorter string's tokens that appear in the other."""
    ta = set(_normalize(a).split())
    tb = set(_normalize(b).split())
    if not ta or not tb:
        return 0.0
    smaller = ta if len(ta) <= len(tb) else tb
    larger = tb if smaller is ta else ta
    return len(smaller & larger) / len(smaller)


def _editsim(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def propose_associations(
    store: FigmentStore,
    role_figments: list[Figment] | None = None,
    boundary_threshold: float = 0.90,
    string_overlap_threshold: float = 0.50,
    editsim_threshold: float = 0.85,
    min_co_occurrence: int = 3,
) -> list[dict[str, Any]]:
    """Propose associations between role figments of the same role.

    Scans the store for role figments and proposes edges where two
    variants share the same ``meta["role"]`` and one or more of:
    boundary cosine similarity >= boundary_threshold,
    string overlap >= string_overlap_threshold,
    edit distance ratio >= editsim_threshold,
    or co-occurrence weight in an existing relationship edge
    >= min_co_occurrence.

    Returns a list of dicts ready to be reviewed or auto-asserted.
    """
    if role_figments is None:
        all_figs = store.all()
        role_figments = [f for f in all_figs if f.meta.get("role")]

    by_role: dict[str, list[Figment]] = defaultdict(list)
    for f in role_figments:
        by_role[f.meta.get("role", "")].append(f)

    proposals = []
    seen_pairs: set[tuple[str, str]] = set()

    for role, figs in by_role.items():
        if len(figs) < 2:
            continue
        for i in range(len(figs)):
            for j in range(i + 1, len(figs)):
                a, b = figs[i], figs[j]
                pair = tuple(sorted([a.figment_id, b.figment_id]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                # Skip if already associated
                if _already_associated(store, a.figment_id, b.figment_id):
                    continue

                reasons: list[str] = []
                scores: list[float] = []

                # 1) Boundary similarity
                sim = _boundary_sim(a.boundary, b.boundary)
                if sim >= boundary_threshold:
                    reasons.append("boundary_similarity")
                    scores.append(sim)

                # 2) String overlap (tokens in common)
                olap = _string_overlap(a.text, b.text)
                if olap >= string_overlap_threshold:
                    reasons.append("string_overlap")
                    scores.append(olap)

                # 3) Edit distance
                ed = _editsim(a.text, b.text)
                if ed >= editsim_threshold:
                    reasons.append("edit_similarity")
                    scores.append(ed)

                if not reasons:
                    continue

                # Check co-occurrence weight from existing relationship edges
                cooccur = _co_occurrence_weight(store, a.figment_id, b.figment_id)
                if cooccur >= min_co_occurrence:
                    reasons.append(f"co_occurrence:{cooccur}")
                    scores.append(min(cooccur / 10.0, 1.0))

                confidence = float(max(scores)) if scores else 0.0

                proposals.append(
                    {
                        "figment_a_id": a.figment_id,
                        "figment_b_id": b.figment_id,
                        "role": role,
                        "text_a": a.text,
                        "text_b": b.text,
                        "confidence": confidence,
                        "reasons": reasons,
                    }
                )

    proposals.sort(key=lambda p: p["confidence"], reverse=True)
    return proposals


def _already_associated(store: FigmentStore, id_a: str, id_b: str) -> bool:
    """Quick check: are these two figments already linked by an association?"""
    for a_id in [id_a, id_b]:
        for f in store.all():
            if f.meta.get("edge_type") != "association":
                continue
            links = f.meta.get("links", [])
            if id_a in links and id_b in links:
                return True
    return False


def _co_occurrence_weight(
    store: FigmentStore, id_a: str, id_b: str
) -> int:
    """Return the weight of the existing relationship edge between two role figments."""
    for f in store.all():
        if f.meta.get("edge_type") != "relationship":
            continue
        fa = f.meta.get("figment_a")
        fb = f.meta.get("figment_b")
        if (fa == id_a and fb == id_b) or (fa == id_b and fb == id_a):
            return int(f.meta.get("weight", 0))
    return 0


def assert_association(
    store: FigmentStore,
    figment_a_id: str,
    figment_b_id: str,
    confidence: float = 1.0,
    evidence: str = "manual",
    hidden_size: int | None = None,
) -> Figment | None:
    """Create a bidirectional association edge between two role figments.

    Returns the association Figment that was upserted, or None if both
    figments are already linked.
    """
    a = store.get(figment_a_id)
    b = store.get(figment_b_id)
    if a is None or b is None:
        return None
    role = a.meta.get("role") or b.meta.get("role", "")

    # Check for existing association linking this pair
    for f in store.all():
        if f.meta.get("edge_type") != "association":
            continue
        links = f.meta.get("links", [])
        if figment_a_id in links and figment_b_id in links:
            return f

    figment_id = hashlib.sha256(
        f"assoc:{role}:{figment_a_id}:{figment_b_id}".encode()
    ).hexdigest()[:16]

    association = Figment.create(
        text=f"Association: {a.text[:40]} <-> {b.text[:40]} ({role})",
        boundary=(a.boundary if a.boundary.shape[0] > 0 else np.zeros(1, dtype=np.float32)),
        meta={
            "edge_type": "association",
            "role": role,
            "links": [figment_a_id, figment_b_id],
            "confidence": confidence,
            "evidence": evidence,
        },
        figment_id=figment_id,
    )

    hs = hidden_size or association.boundary.shape[0]
    store.upsert([association], hidden_size=hs)
    return association


def expand_associations(
    store: FigmentStore,
    role_figment_id: str,
    max_hops: int = 2,
) -> set[str]:
    """Return the full set of variant figment IDs reachable from *role_figment_id*.

    Walks association edges up to ``max_hops`` (default 2). The starting
    figment ID is always included in the result.
    """
    result: set[str] = {role_figment_id}
    frontier: set[str] = {role_figment_id}

    for _ in range(max_hops):
        next_frontier: set[str] = set()
        for fid in frontier:
            for f in store.all():
                if f.meta.get("edge_type") != "association":
                    continue
                links = f.meta.get("links", [])
                if fid in links:
                    for other in links:
                        if other not in result:
                            result.add(other)
                            next_frontier.add(other)
        frontier = next_frontier
        if not frontier:
            break

    return result


def get_association_groups(store: FigmentStore) -> dict[str, list[str]]:
    """Return all association clusters as {canonical_id: [variant_ids]}."""
    associations: list[Figment] = []
    for f in store.all():
        if f.meta.get("edge_type") == "association":
            associations.append(f)

    if not associations:
        return {}

    # Union-Find to build clusters
    all_ids: set[str] = set()
    for a in associations:
        for lid in a.meta.get("links", []):
            all_ids.add(lid)

    parent = {fid: fid for fid in all_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a_id: str, b_id: str) -> None:
        ra, rb = find(a_id), find(b_id)
        if ra != rb:
            parent[rb] = ra

    for a in associations:
        links = a.meta.get("links", [])
        for i in range(len(links)):
            for j in range(i + 1, len(links)):
                union(links[i], links[j])

    groups: dict[str, list[str]] = defaultdict(list)
    for fid in all_ids:
        groups[find(fid)].append(fid)

    return dict(groups)


def integrate_associations(
    store: FigmentStore,
    role: str,
    normalized_text: str,
) -> list[str]:
    """Given a role + normalized text, return all variant figment IDs.

    Looks up an existing role figment by exact match, then expands through
    associations. Returns the full set of variant IDs (including the original).
    """
    for f in store.all():
        if f.meta.get("role") == role and f.meta.get("normalized") == normalized_text:
            return sorted(expand_associations(store, f.figment_id))
    return []