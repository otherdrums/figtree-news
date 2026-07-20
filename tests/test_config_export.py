"""CPU-only tests for figtree-news (no model / GPU required)."""

from __future__ import annotations


import numpy as np
import pytest

from figtree import Figment, FigmentStore, connect

from figtree_news.config import SourceRegistry
from figtree_news import export as export_mod
from figtree_news import trust as trust_mod


def _fake_figments(source_id="reuters", n=3):
    figs = []
    children = []
    for i in range(n):
        f = Figment.create(
            text=f"Claim number {i} from the report.",
            boundary=np.zeros(8, dtype="float32"),
            meta={"source_id": source_id, "crystal_layer": 0},
            trust=0.9,
        )
        children.append(f.figment_id)
        figs.append(f)
    image = Figment.create(
        text="Image summary of the report.",
        boundary=np.zeros(8, dtype="float32"),
        meta={"source_id": source_id, "is_image": True, "base_trust": 0.9},
        children=children,
        trust=0.9,
    )
    trust_fig = Figment.create(
        text=f"Source {source_id} has trust 0.90",
        boundary=np.zeros(8, dtype="float32"),
        meta={"edge_type": "trust", "source_id": source_id, "score": 0.9, "base_trust": 0.9},
        sources=[image.figment_id],
        trust=0.9,
    )
    return [image] + figs + [trust_fig]


def test_registry_roundtrip(tmp_path):
    path = tmp_path / "sources.json"
    reg = SourceRegistry.load(str(path))
    reg.ensure("reuters", "Reuters", 0.9)
    assert reg.base_trust("reuters") == 0.9
    assert reg.base_trust("unknown", 0.3) == 0.3
    reg.save(str(path))
    reloaded = SourceRegistry.load(str(path))
    assert reloaded.base_trust("reuters") == 0.9


def test_export_graph(tmp_path):
    store: FigmentStore = connect(str(tmp_path / "news.lance"))
    store.upsert(_fake_figments(), hidden_size=8)
    graph = export_mod.export_graph(store)
    assert len(graph["nodes"]) == 5  # image + 3 children + trust
    assert any(e["kind"] == "child" for e in graph["edges"])
    assert any(e["kind"] == "source" for e in graph["edges"])


def test_get_source_trusts(tmp_path):
    store: FigmentStore = connect(str(tmp_path / "news.lance"))
    store.upsert(_fake_figments("ap", n=2), hidden_size=8)
    trusts = trust_mod.get_source_trusts(store)
    assert trusts.get("ap") == pytest.approx(0.9)


def test_update_trust_runs(tmp_path):
    store: FigmentStore = connect(str(tmp_path / "news.lance"))
    store.upsert(_fake_figments("blog", n=2), hidden_size=8)
    out = trust_mod.update_trust(store)
    assert "updates" in out
    assert any(u["source_id"] == "blog" for u in out["updates"])
