"""CPU tests for the lineage engine (no model)."""

from __future__ import annotations

import hashlib

import numpy as np
from figtree import Figment, FigmentStore, connect

from figtree_news import lineage as lineage_mod


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


def _article(source_id, text, published, url, fid, role_figment_ids=None):
    return Figment.create(
        text=text,
        boundary=np.zeros(8, dtype="float32"),
        meta={
            "source_id": source_id,
            "url": url,
            "published": published,
            "first_seen": published,
            "decomposed": True,
            "role_figments": role_figment_ids or [],
        },
        figment_id=fid,
        trust=0.8,
        kind="article",
    )


def _seed_store(tmp_path):
    store: FigmentStore = connect(str(tmp_path / "news.lance"))

    who_election = _role_figment("who", "Election Commission", "a1")
    what_election = _role_figment("what", "Election held", "a1")
    when_election = _role_figment("when", "Tuesday", "a1")

    a = _article(
        "reuters", "The Election was held on Tuesday.",
        "Mon, 01 Jan 2024 10:00:00 GMT", "http://reuters.com/1", "a1",
        role_figment_ids=[who_election.figment_id, what_election.figment_id, when_election.figment_id],
    )

    who_election_b = _role_figment("who", "Election Commission", "b1")
    what_election_b = _role_figment("what", "Election held", "b1")
    when_election_b = _role_figment("when", "Wednesday", "b1")

    b = _article(
        "blog", "The Election results were announced Wednesday.",
        "Tue, 02 Jan 2024 10:00:00 GMT", "http://blog.com/1", "b1",
        role_figment_ids=[who_election_b.figment_id, what_election_b.figment_id, when_election_b.figment_id],
    )

    store.upsert([a, b, who_election, what_election, when_election, who_election_b, what_election_b, when_election_b], hidden_size=8)
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

    figs = {f.figment_id: f for f in store.all()}
    assert figs["b1"].meta.get("derivative_of") == "a1"
    assert figs["a1"].meta.get("first_reporter") is True


def test_lineage_idempotent(tmp_path):
    store = _seed_store(tmp_path)
    lineage_mod.compute_lineage(store)
    before = len(store.all())
    lineage_mod.compute_lineage(store)
    assert len(store.all()) == before


def _seed_disjoint_narratives(tmp_path):
    """Two articles with no shared roles → two separate narratives."""
    store: FigmentStore = connect(str(tmp_path / "news.lance"))

    who_a = _role_figment("who", "Alice", "a1")
    what_a = _role_figment("what", "Won marathon", "a1")
    where_a = _role_figment("where", "Boston", "a1")
    a = _article(
        "reuters", "Alice won the Boston marathon.",
        "Mon, 01 Jan 2024 10:00:00 GMT", "http://reuters.com/a", "a1",
        role_figment_ids=[who_a.figment_id, what_a.figment_id, where_a.figment_id],
    )

    who_b = _role_figment("who", "Bob", "b1")
    what_b = _role_figment("what", "Won marathon", "b1")
    where_b = _role_figment("where", "Chicago", "b1")
    b = _article(
        "bbc", "Bob won the Chicago marathon.",
        "Tue, 02 Jan 2024 10:00:00 GMT", "http://bbc.com/b", "b1",
        role_figment_ids=[who_b.figment_id, what_b.figment_id, where_b.figment_id],
    )

    store.upsert([a, b, who_a, what_a, where_a, who_b, what_b, where_b], hidden_size=8)
    return store


def test_merge_narratives_by_llm_labels(tmp_path):
    store = _seed_disjoint_narratives(tmp_path)
    out = lineage_mod.compute_lineage(store)
    assert len(out["narratives"]) == 2

    reporters_before = {n["first_reporter"] for n in out["narratives"]}
    assert "reuters" in reporters_before
    assert "bbc" in reporters_before

    labels = [
        {"a1": "a1", "a2": "b1", "same_event": True, "reason": "Both are marathon winners"},
    ]
    merge_out = lineage_mod.merge_narratives_by_llm_labels(store, labels)
    assert len(merge_out["narratives"]) == 1

    merged = merge_out["narratives"][0]
    assert "a1" in merged["members"]
    assert "b1" in merged["members"]
    assert merged["first_reporter"] == "reuters"
