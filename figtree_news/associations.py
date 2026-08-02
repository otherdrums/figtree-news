"""Association figments: co-reference linking for surface-form variants.

Thin wrapper over :mod:`figtree.identity`, which implements the merge engine
(propose -> assert -> merge_role_figments rewrite). All association figments
(edge_type="association") live in the same LanceDB table as everything else.

Roles extracted by the decomposition engine are canonicalized by exact text
match (lower + strip punctuation), so surface-form variants such as
``"Donald Trump"``, ``"Trump"``, and ``"DJT"`` become separate role figments.
The identity layer links them so downstream queries can expand through aliases
and retrieve every role instance for the same entity.
"""

from __future__ import annotations

from typing import Any

from figtree import Figment, FigmentStore
from figtree.identity import (
    assert_identity,
    expand_identities,
    identity_groups,
    integrate_identity,
    merge_role_figments,
    propose_identity_merges,
)

__all__ = [
    "propose_associations",
    "assert_association",
    "expand_associations",
    "get_association_groups",
    "integrate_associations",
    "merge_role_figments",
]


def propose_associations(
    store: FigmentStore,
    role_figments: list[Figment] | None = None,
    boundary_threshold: float = 0.90,
    string_overlap_threshold: float = 0.50,
    editsim_threshold: float = 0.85,
    min_co_occurrence: int = 3,
) -> list[dict[str, Any]]:
    """Propose associations between role figments of the same role."""
    return propose_identity_merges(
        store,
        role_figments=role_figments,
        boundary_threshold=boundary_threshold,
        string_overlap_threshold=string_overlap_threshold,
        editsim_threshold=editsim_threshold,
        min_co_occurrence=min_co_occurrence,
    )


def assert_association(
    store: FigmentStore,
    figment_a_id: str,
    figment_b_id: str,
    confidence: float = 1.0,
    evidence: str = "manual",
    hidden_size: int | None = None,
) -> Figment | None:
    """Create a bidirectional association edge between two role figments."""
    return assert_identity(
        store,
        figment_a_id,
        figment_b_id,
        confidence=confidence,
        evidence=evidence,
        hidden_size=hidden_size,
    )


def expand_associations(
    store: FigmentStore,
    role_figment_id: str,
    max_hops: int = 2,
) -> set[str]:
    """Return the full set of variant figment IDs reachable from *role_figment_id*."""
    return expand_identities(store, role_figment_id, max_hops=max_hops)


def get_association_groups(store: FigmentStore) -> dict[str, list[str]]:
    """Return all association clusters as {canonical_id: [variant_ids]}."""
    return identity_groups(store)


def integrate_associations(
    store: FigmentStore,
    role: str,
    normalized_text: str,
) -> list[str]:
    """Given a role + normalized text, return all variant figment IDs."""
    return integrate_identity(store, role, normalized_text)
