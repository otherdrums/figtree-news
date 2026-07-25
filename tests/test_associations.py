"""CPU tests for the association figment system."""

from __future__ import annotations

import hashlib
import numpy as np
from figtree import Figment, FigmentStore, connect

from figtree_news import associations as assoc_mod


def _role_fig(role: str, text: str, article_id: str) -> Figment:
    normalized = _normalize_text(text)
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


def _normalize_text(text: str) -> str:
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
            "is_image": True,
            "decomposed": True,
            "role_figments": [trump1.figment_id, disney1.figment_id],
        },
        figment_id="img1",
        trust=0.8,
    )

    b = Figment.create(
        text="Article about DJT and Walt Disney World",
        boundary=np.zeros(8, dtype="float32"),
        meta={
            "source_id": "cnn",
            "is_image": True,
            "decomposed": True,
            "role_figments": [djt.figment_id, disney2.figment_id],
        },
        figment_id="img2",
        trust=0.7,
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