"""CPU tests for the association figment system."""

from __future__ import annotations

import hashlib
import numpy as np
from figtree import Figment, FigmentStore, connect

from figtree_news import associations as assoc_mod


def _role_fig(role: str, text: str, article_id: str, references: list | None = None) -> Figment:
    from figtree_news.normalize import normalize as _norm
    normalized = _norm(text)
    fid = hashlib.sha256(f"role:{role}:{normalized}".encode()).hexdigest()[:16]
    refs = references or []
    return Figment.create(
        text=text,
        boundary=np.zeros(8, dtype="float32"),
        meta={
            "role": role,
            "article_id": article_id,
            "normalized": normalized,
            "references": list(refs),
            "reference_count": len(refs),
        },
        figment_id=fid,
        kind="role",
    )


def _seed_store(tmp_path) -> FigmentStore:
    store: FigmentStore = connect(str(tmp_path / "news.lance"))

    trump1 = _role_fig("who", "Donald Trump", "a1")
    trump2 = _role_fig("who", "Trump", "a2")
    djt = _role_fig("who", "DJT", "a3")

    disney1 = _role_fig("where", "Disney World", "a1")
    disney2 = _role_fig("where", "Walt Disney World", "a2")

    a = Figment.create(
        text="Article about Trump at Disney",
        boundary=np.zeros(8, dtype="float32"),
        meta={
            "source_id": "reuters",
            "decomposed": True,
            "role_figments": [trump1.figment_id, disney1.figment_id],
        },
        figment_id="img1",
        trust=0.8,
        kind="article",
    )

    b = Figment.create(
        text="Article about DJT and Walt Disney World",
        boundary=np.zeros(8, dtype="float32"),
        meta={
            "source_id": "cnn",
            "decomposed": True,
            "role_figments": [djt.figment_id, disney2.figment_id],
        },
        figment_id="img2",
        trust=0.7,
        kind="article",
    )

    store.upsert(
        [a, b, trump1, trump2, djt, disney1, disney2],
        hidden_size=8,
    )
    return store


def test_propose_associations(tmp_path):
    store = _seed_store(tmp_path)
    all_figs = store.all()
    role_figs = [f for f in all_figs if f.meta.get("role")]

    proposals = assoc_mod.propose_associations(store, role_figments=role_figs)
    assert len(proposals) > 0, "Should find at least one association proposal"

    # Trump variants should be proposed
    trump_proposals = [p for p in proposals if p["role"] == "who"]
    assert len(trump_proposals) > 0, "Should propose Trump variant associations"


def test_assert_association(tmp_path):
    store = _seed_store(tmp_path)
    all_figs = store.all()
    trump1 = next(f for f in all_figs if f.meta.get("role") == "who" and f.text == "Donald Trump")
    trump2 = next(f for f in all_figs if f.meta.get("role") == "who" and f.text == "Trump")

    result = assoc_mod.assert_association(
        store, trump1.figment_id, trump2.figment_id, confidence=1.0, evidence="test"
    )
    assert result is not None, "Should create an association"

    # Calling again should return the existing association
    result2 = assoc_mod.assert_association(
        store, trump1.figment_id, trump2.figment_id, confidence=1.0, evidence="test"
    )
    assert result2 is not None
    assert result2.figment_id == result.figment_id, "Should return existing association"


def test_expand_associations(tmp_path):
    store = _seed_store(tmp_path)
    all_figs = store.all()
    trump1 = next(f for f in all_figs if f.meta.get("role") == "who" and f.text == "Donald Trump")

    # Before asserting, only the figment itself
    result = assoc_mod.expand_associations(store, trump1.figment_id)
    assert trump1.figment_id in result

    # Assert association with Trump variant
    trump2 = next(f for f in all_figs if f.meta.get("role") == "who" and f.text == "Trump")
    assoc_mod.assert_association(store, trump1.figment_id, trump2.figment_id, confidence=1.0)

    # Now expanded set should include both
    result = assoc_mod.expand_associations(store, trump1.figment_id)
    assert trump1.figment_id in result
    assert trump2.figment_id in result


def test_get_association_groups(tmp_path):
    store = _seed_store(tmp_path)
    all_figs = store.all()
    trump1 = next(f for f in all_figs if f.meta.get("role") == "who" and f.text == "Donald Trump")
    trump2 = next(f for f in all_figs if f.meta.get("role") == "who" and f.text == "Trump")
    djt = next(f for f in all_figs if f.meta.get("role") == "who" and f.text == "DJT")

    assoc_mod.assert_association(store, trump1.figment_id, trump2.figment_id, confidence=1.0)
    assoc_mod.assert_association(store, trump1.figment_id, djt.figment_id, confidence=1.0)

    groups = assoc_mod.get_association_groups(store)
    # Trump and DJT should be in the same group
    found_group = None
    for canonical, members in groups.items():
        if trump1.figment_id in members:
            found_group = members
            break

    assert found_group is not None
    assert trump2.figment_id in found_group
    assert djt.figment_id in found_group


def test_integrate_associations(tmp_path):
    store = _seed_store(tmp_path)
    all_figs = store.all()
    trump1 = next(f for f in all_figs if f.meta.get("role") == "who" and f.text == "Donald Trump")
    trump2 = next(f for f in all_figs if f.meta.get("role") == "who" and f.text == "Trump")

    assoc_mod.assert_association(store, trump1.figment_id, trump2.figment_id, confidence=1.0)

    result = assoc_mod.integrate_associations(store, "who", "donald trump")
    assert trump1.figment_id in result
    assert trump2.figment_id in result


# ── merge_role_figments tests ──────────────────────────────────────────

def _sentence_fig(text: str, parent_id: str, children: list | None = None) -> Figment:
    return Figment.create(
        text=text,
        boundary=np.zeros(8, dtype="float32"),
        meta={},
        figment_id=hashlib.sha256(text.encode()).hexdigest()[:16],
        children=children or [],
        kind="sentence",
    )


def _paragraph_fig(text: str, parent_id: str) -> Figment:
    return Figment.create(
        text=text,
        boundary=np.zeros(8, dtype="float32"),
        meta={},
        figment_id=hashlib.sha256(f"p:{text}".encode()).hexdigest()[:16],
        kind="paragraph",
    )


def _article_fig(article_id: str, role_ids: list[str], children: list | None = None) -> Figment:
    return Figment.create(
        text=f"Article {article_id}",
        boundary=np.zeros(8, dtype="float32"),
        meta={"role_figments": list(role_ids), "source_id": "test"},
        figment_id=article_id,
        kind="article",
    )


def _rel_edge(fa: str, fb: str, weight: int = 1) -> Figment:
    pair = tuple(sorted([fa, fb]))
    eid = hashlib.sha256(f"rel:{pair[0]}:{pair[1]}".encode()).hexdigest()[:16]
    return Figment.create(
        text=f"rel {fa[:8]} {fb[:8]}",
        boundary=np.zeros(8, dtype="float32"),
        meta={"edge_type": "relationship", "figment_a": fa, "figment_b": fb, "weight": weight},
        figment_id=eid,
        kind="edge",
    )


def _assoc_edge(a: str, b: str, role: str = "who") -> Figment:
    sl = sorted([a, b])
    eid = hashlib.sha256(f"assoc:{role}:{sl[0]}:{sl[1]}".encode()).hexdigest()[:16]
    return Figment.create(
        text=f"assoc {a[:8]} {b[:8]}",
        boundary=np.zeros(8, dtype="float32"),
        meta={"edge_type": "association", "role": role, "links": [a, b], "confidence": 1.0},
        figment_id=eid,
        kind="edge",
    )


def _dedup_obs_fig(fa: str, fb: str, role: str = "who") -> Figment:
    eid = hashlib.sha256(f"dedup_obs:{fa}:{fb}:merge".encode()).hexdigest()[:16]
    return Figment.create(
        text=f"dedup {fa[:8]} vs {fb[:8]}",
        boundary=np.zeros(8, dtype="float32"),
        meta={"role_figment_a": fa, "role_figment_b": fb, "role": role, "verdict": "merge"},
        figment_id=eid,
        kind="dedup_obs",
    )


def test_merge_basic(tmp_path):
    """Basic merge: references united, is_association=True, removed rows deleted."""
    store: FigmentStore = connect(str(tmp_path / "news.lance"))

    keep = _role_fig("who", "Donald Trump", "a1", references=["s1", "s2"])
    remove = _role_fig("who", "Trump", "a2", references=["s3"])
    store.upsert([keep, remove], hidden_size=8)

    result = assoc_mod.merge_role_figments(store, keep.figment_id, [remove.figment_id])

    assert result > 0
    stored_keep = store.get(keep.figment_id)
    assert stored_keep is not None
    assert sorted(stored_keep.meta["references"]) == ["s1", "s2", "s3"]
    assert stored_keep.meta["reference_count"] == 3
    assert stored_keep.meta["is_association"] is True
    assert remove.figment_id in stored_keep.meta.get("merged_from", [])

    stored_remove = store.get(remove.figment_id)
    assert stored_remove is None, "Removed figment should be deleted"


def test_merge_article_role_figments(tmp_path):
    """article.meta['role_figments'] rewritten from removed ID to keep ID."""
    store: FigmentStore = connect(str(tmp_path / "news.lance"))

    keep = _role_fig("who", "Donald Trump", "a1")
    remove = _role_fig("who", "Trump", "a2")
    article = _article_fig("art1", [keep.figment_id, remove.figment_id])
    store.upsert([keep, remove, article], hidden_size=8)

    assoc_mod.merge_role_figments(store, keep.figment_id, [remove.figment_id])

    stored = store.get("art1")
    assert stored is not None
    rfs = stored.meta["role_figments"]
    assert keep.figment_id in rfs
    assert remove.figment_id not in rfs


def test_merge_sentence_children(tmp_path):
    """sentence.children rewritten."""
    store: FigmentStore = connect(str(tmp_path / "news.lance"))

    keep = _role_fig("who", "Donald Trump", "a1")
    remove = _role_fig("who", "Trump", "a2")
    sent = _sentence_fig("Trump said...", "p1", children=[remove.figment_id])
    store.upsert([keep, remove, sent], hidden_size=8)

    assoc_mod.merge_role_figments(store, keep.figment_id, [remove.figment_id])

    stored = store.get(sent.figment_id)
    assert stored is not None
    assert keep.figment_id in stored.children
    assert remove.figment_id not in stored.children


def test_merge_relationship_edge(tmp_path):
    """Relationship edge IDs recomputed; weights merged on collision."""
    store: FigmentStore = connect(str(tmp_path / "news.lance"))

    keep = _role_fig("who", "Donald Trump", "a1")
    remove = _role_fig("who", "Trump", "a2")
    third = _role_fig("what", "tariffs", "a1")

    # rel: remove <-> third  → becomes  rel: keep <-> third
    rel1 = _rel_edge(remove.figment_id, third.figment_id, weight=2)
    # rel: keep <-> third already exists with weight 3 → should merge to weight 5
    rel2 = _rel_edge(keep.figment_id, third.figment_id, weight=3)

    store.upsert([keep, remove, third, rel1, rel2], hidden_size=8)

    assoc_mod.merge_role_figments(store, keep.figment_id, [remove.figment_id])

    expected_id = hashlib.sha256(
        f"rel:{tuple(sorted([keep.figment_id, third.figment_id]))[0]}:{tuple(sorted([keep.figment_id, third.figment_id]))[1]}".encode()
    ).hexdigest()[:16]

    merged = store.get(expected_id)
    assert merged is not None, "Merged rel edge should exist"
    assert merged.meta["weight"] == 5, f"Expected weight 5, got {merged.meta['weight']}"

    old = store.get(rel1.figment_id)
    assert old is None, "Old rel edge should be deleted"


def test_merge_association_edge(tmp_path):
    """Association edge IDs recomputed after rewrite."""
    store: FigmentStore = connect(str(tmp_path / "news.lance"))

    keep = _role_fig("who", "Donald Trump", "a1")
    remove = _role_fig("who", "Trump", "a2")
    third = _role_fig("who", "DJT", "a3")

    assoc1 = _assoc_edge(remove.figment_id, third.figment_id)

    store.upsert([keep, remove, third, assoc1], hidden_size=8)

    assoc_mod.merge_role_figments(store, keep.figment_id, [remove.figment_id])

    expected_id = hashlib.sha256(
        f"assoc:who:{tuple(sorted([keep.figment_id, third.figment_id]))[0]}:{tuple(sorted([keep.figment_id, third.figment_id]))[1]}".encode()
    ).hexdigest()[:16]

    merged = store.get(expected_id)
    assert merged is not None, "Merged assoc edge should exist"
    assert merged.meta["edge_type"] == "association"

    old = store.get(assoc1.figment_id)
    assert old is None, "Old assoc edge should be deleted"


def test_merge_dedup_obs(tmp_path):
    """dedup_obs role_figment_a/role_figment_b rewritten."""
    store: FigmentStore = connect(str(tmp_path / "news.lance"))

    keep = _role_fig("who", "Donald Trump", "a1")
    remove = _role_fig("who", "Trump", "a2")
    dedup = _dedup_obs_fig(keep.figment_id, remove.figment_id)

    store.upsert([keep, remove, dedup], hidden_size=8)

    assoc_mod.merge_role_figments(store, keep.figment_id, [remove.figment_id])

    stored = store.get(dedup.figment_id)
    assert stored is not None
    assert stored.meta["role_figment_a"] == keep.figment_id
    assert stored.meta["role_figment_b"] == keep.figment_id


def test_merge_idempotent(tmp_path):
    """Second merge with same keep_id does not duplicate references."""
    store: FigmentStore = connect(str(tmp_path / "news.lance"))

    keep = _role_fig("who", "Donald Trump", "a1", references=["s1"])
    r1 = _role_fig("who", "Trump", "a2", references=["s2"])
    r2 = _role_fig("who", "DJT", "a3", references=["s3"])

    article1 = _article_fig("art1", [keep.figment_id, r1.figment_id])
    article2 = _article_fig("art2", [keep.figment_id, r2.figment_id])

    store.upsert([keep, r1, r2, article1, article2], hidden_size=8)

    assoc_mod.merge_role_figments(store, keep.figment_id, [r1.figment_id])
    assoc_mod.merge_role_figments(store, keep.figment_id, [r2.figment_id])

    stored = store.get(keep.figment_id)
    assert stored is not None
    assert sorted(stored.meta["references"]) == ["s1", "s2", "s3"]
    assert stored.meta["reference_count"] == 3
    assert len(stored.meta["merged_from"]) == 2

    # Each article still points at keep, not a removed ID
    for aid in ["art1", "art2"]:
        a = store.get(aid)
        assert a is not None
        assert keep.figment_id in a.meta["role_figments"]


def test_merge_empty_remove(tmp_path):
    """Empty remove_ids is a no-op."""
    store: FigmentStore = connect(str(tmp_path / "news.lance"))
    keep = _role_fig("who", "Donald Trump", "a1")
    store.upsert([keep], hidden_size=8)
    result = assoc_mod.merge_role_figments(store, keep.figment_id, [])
    assert result == 0


def test_merge_unknown_keep(tmp_path):
    """Unknown keep_id is a no-op."""
    store: FigmentStore = connect(str(tmp_path / "news.lance"))
    result = assoc_mod.merge_role_figments(store, "nonexistent", ["something"])
    assert result == 0