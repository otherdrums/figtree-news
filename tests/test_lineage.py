"""CPU tests for the lineage engine (no model)."""

from __future__ import annotations

import numpy as np
from figtree import Figment, FigmentStore, connect

from figtree_news import lineage as lineage_mod


def _article(source_id, text, published, url, fid):
    return Figment.create(
        text=text,
        boundary=np.zeros(8, dtype="float32"),
        meta={
            "source_id": source_id,
            "is_image": True,
            "url": url,
            "published": published,
            "first_seen": published,
        },
        figment_id=fid,
        trust=0.8,
    )


def _seed_store(tmp_path):
    store: FigmentStore = connect(str(tmp_path / "news.lance"))
    a = _article("reuters", "The Election was held on Tuesday.", "Mon, 01 Jan 2024 10:00:00 GMT", "http://reuters.com/1", "a1")
    b = _article("blog", "The Election results were announced Wednesday.", "Tue, 02 Jan 2024 10:00:00 GMT", "http://blog.com/1", "b1")
    store.upsert([a, b], hidden_size=8)
    return store


def test_first_reporter_and_derivative(tmp_path):
    store = _seed_store(tmp_path)
    out = lineage_mod.compute_lineage(store)
    assert len(out["narratives"]) == 1
    n = out["narratives"][0]
    assert n["first_reporter"] == "reuters"
    assert n["first_reporter_url"] == "http://reuters.com/1"

    derivs = lineage_mod.get_derivatives(store)
    assert len(derivs) == 1
    assert derivs[0]["derivative_url"] == "http://blog.com/1"

    # The blog article should be marked derivative_of the reuters article.
    figs = {f.figment_id: f for f in store.all()}
    assert figs["b1"].meta.get("derivative_of") == "a1"
    assert figs["a1"].meta.get("first_reporter") is True


def test_lineage_idempotent(tmp_path):
    store = _seed_store(tmp_path)
    lineage_mod.compute_lineage(store)
    before = len(store.all())
    lineage_mod.compute_lineage(store)  # re-run
    assert len(store.all()) == before  # no duplicate figments created
