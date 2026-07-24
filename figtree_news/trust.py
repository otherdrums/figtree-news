"""Source-level trust, built on top of Figtree's figment-based trust model.

Figtree already computes ``adjusted_trust`` per source from corroboration and
contradiction edges and persists it as a ``trust:{source_id}`` figment. This
module is a thin, dependency-free wrapper: it builds the graph from the store,
runs edge creation + propagation, and reads the persisted scores back for
display and for filtering queries by credibility.
"""

from __future__ import annotations

from typing import Any

from figtree import FigmentStore, Figtree


def _build_graph(store: FigmentStore, *, all_figs: list | None = None) -> Figtree:
    figs = all_figs if all_figs is not None else store.all()
    return Figtree(figs, store=store)


def update_trust(store: FigmentStore, *, all_figs: list | None = None, dedupe: bool = False) -> dict[str, Any]:
    """Build edges, analyze sources, and persist adjusted trust scores.

    Returns ``{"analysis": {...}, "updates": [...]}`` where ``updates`` is the
    list returned by ``Figtree.propagate_trust`` (one entry per source).
    """
    graph = _build_graph(store, all_figs=all_figs)
    if dedupe:
        graph.deduplicate()
    graph.create_edges()
    analysis = graph.analyze_sources()
    updates = graph.propagate_trust(store=store)
    return {"analysis": analysis, "updates": updates}


def get_source_trusts(store: FigmentStore) -> dict[str, float]:
    """Read persisted ``trust:{source_id}`` scores from the store."""
    out: dict[str, float] = {}
    for f in store.all():
        sid = f.meta.get("source_id")
        if f.meta.get("edge_type") == "trust" and sid:
            out[sid] = f.trust
    return out


def show_source_trust(store: FigmentStore) -> list[dict[str, Any]]:
    """Return a per-source report combining persisted score + rationale."""
    graph = _build_graph(store)
    analysis = graph.analyze_sources()
    rows: list[dict[str, Any]] = []
    for src, info in analysis.items():
        rows.append(
            {
                "source_id": src,
                "adjusted_trust": info["adjusted_trust"],
                "base_trust": info["base_trust"],
                "corroborated_frac": info["corroborated_frac"],
                "related": info["related"],
                "agreeing": info["agreeing"],
                "contradicting": info["contradicting"],
                "rationale": info["rationale"],
            }
        )
    rows.sort(key=lambda r: r["adjusted_trust"], reverse=True)
    return rows
