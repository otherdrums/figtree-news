"""CPU smoke tests for the FastAPI web app (no model / GPU)."""

from __future__ import annotations

import hashlib

import numpy as np
from fastapi.testclient import TestClient
from figtree import Figment, FigmentStore, connect

from figtree_news import lineage as lineage_mod
from figtree_news import trust as trust_mod
from figtree_news.web.serve import create_app


def _role_figment(role: str, text: str, article_id: str) -> Figment:
    normalized = text.lower().strip()
    fid = hashlib.sha256(f"role:{role}:{normalized}".encode()).hexdigest()[:16]
    return Figment.create(
        text=text,
        boundary=np.zeros(8, dtype="float32"),
        meta={
            "role": role,
            "article_id": article_id,
            "normalized": normalized,
            "references": [],
            "reference_count": 0,
        },
        figment_id=fid,
    )


def _seed(tmp_path):
    store: FigmentStore = connect(str(tmp_path / "news.lance"))

    who = _role_figment("who", "Election Commission", "a1")
    what = _role_figment("what", "Election held", "a1")
    when = _role_figment("when", "Tuesday", "a1")

    a = Figment.create(
        text="The Election was held on Tuesday.", boundary=np.zeros(8, dtype="float32"),
        meta={
            "source_id": "reuters", "url": "http://reuters.com/1",
            "published": "Mon, 01 Jan 2024 10:00:00 GMT", "title": "France Election Results",
            "decomposed": True, "role_figments": [who.figment_id, what.figment_id, when.figment_id],
        },
        figment_id="a1", trust=0.9, kind="article",
    )

    who_b = _role_figment("who", "Election Commission", "b1")
    what_b = _role_figment("what", "Election held", "b1")
    when_b = _role_figment("when", "Wednesday", "b1")

    b = Figment.create(
        text="The Election results were announced.", boundary=np.zeros(8, dtype="float32"),
        meta={
            "source_id": "blog", "url": "http://blog.com/1",
            "published": "Tue, 02 Jan 2024 10:00:00 GMT", "title": "France Election Results Announced",
            "decomposed": True, "role_figments": [who_b.figment_id, what_b.figment_id, when_b.figment_id],
        },
        figment_id="b1", trust=0.5, kind="article",
    )

    store.upsert([a, b, who, what, when, who_b, what_b, when_b], hidden_size=8)
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


def test_viewer_mode_device_none(tmp_path):
    """device=none: pages render, but every model-touching path is refused."""
    store, a, b = _seed(tmp_path)
    app = create_app(db=str(tmp_path / "news.lance"), sources=str(tmp_path / "sources.json"), device="none")
    client = TestClient(app)

    assert client.get("/").status_code == 200
    assert client.get("/api/articles").status_code == 200
    assert client.get("/api/query", params={"q": "election"}).status_code == 503
    assert "error" in client.post("/api/crawl/run", json={"feeds": {"x": "http://x"}}).json()
    assert "error" in client.post("/api/pipeline/run", json={}).json()
    assert "error" in client.post("/api/summaries/regenerate", json={}).json()
