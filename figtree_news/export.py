"""Export the figment graph (nodes + edges) as JSON for external tooling."""

from __future__ import annotations

import json
from typing import Any

from figtree import FigmentStore


def export_graph(store: FigmentStore, path: str | None = None) -> dict[str, Any]:
    """Dump figments and their edges to a JSON-serialisable dict.

    Each node carries its id, text, source, trust, and edge type; edges are
    derived from ``children`` and ``sources`` so the structure can be loaded
    into networkx, neo4j, or a viewer. No graph library is required here.
    """
    figs = store.all()
    nodes = []
    edges = []
    for f in figs:
        nodes.append(
            {
                "id": f.figment_id,
                "text": f.text,
                "source_id": f.meta.get("source_id"),
                "edge_type": f.meta.get("edge_type"),
                "trust": f.trust,
                "is_image": f.meta.get("is_image", False),
            }
        )
        for child in f.children:
            edges.append({"from": f.figment_id, "to": child, "kind": "child"})
        for src in f.sources:
            edges.append({"from": f.figment_id, "to": src, "kind": "source"})

    graph = {"nodes": nodes, "edges": edges}
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(graph, fh, indent=2)
    return graph
