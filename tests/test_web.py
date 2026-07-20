"""CPU smoke tests for the FastAPI web app (no model / GPU)."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient
from figtree import Figment, FigmentStore, connect

from figtree_news import lineage as lineage_mod
from figtree_news import trust as trust_mod
from figtree_news.web.serve import create_app


def _seed(tmp_path):
    store: FigmentStore = connect(str(tmp_path / "news.lance"))
    a = Figment.create(
        text="The Election was held on Tuesday.", boundary=np.zeros(8, dtype="float32"),
        meta={"source_id": "reuters", "is_image": True, "url": "http://reuters.com/1",
              "published": "Mon, 01 Jan 2024 10:00:00 GMT", "title": "Election day"},
        figment_id="a1", trust=0.9,
    )
    b = Figment.create(
        text="The Election results were announced.", boundary=np.zeros(8, dtype="float32"),
        meta={"source_id": "blog", "is_image": True, "url": "http://blog.com/1",
              "published": "Tue, 02 Jan 2024 10:00:00 GMT", "title": "Results"},
        figment_id="b1", trust=0.5,
    )
    store.upsert([a, b], hidden_size=8)
    lineage_mod.compute_lineage(store)
    trust_mod.update_trust(store)
    return store, a, b


def test_pages_render(tmp_path):
    store, a, b = _seed(tmp_path)
    app = create_app(db=str(tmp_path / "news.lance"), sources=str(tmp_path / "sources.json"))
    client = TestClient(app)

    assert client.get("/").status_code == 200
    nid = lineage_mod.get_narratives(store)[0]["narrative_id"]
    r = client.get(f"/narrative/{nid}")
    assert r.status_code == 200
    assert "reuters" in r.text

    assert client.get(f"/article/{a.figment_id}").status_code == 200
    assert client.get("/source/reuters").status_code == 200
    assert client.get("/lineage").status_code == 200


def test_api_endpoints(tmp_path):
    store, a, b = _seed(tmp_path)
    app = create_app(db=str(tmp_path / "news.lance"), sources=str(tmp_path / "sources.json"))
    client = TestClient(app)

    assert client.get("/api/narratives").status_code == 200
    assert client.get("/api/sources").status_code == 200
    assert client.get("/api/lineage").status_code == 200
    arts = client.get("/api/articles").json()
    assert len(arts) == 2
    assert arts[0]["url"] == "http://reuters.com/1"
