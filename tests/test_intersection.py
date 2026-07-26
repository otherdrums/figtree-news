"""CPU tests for the multi-role intersection query."""

from __future__ import annotations

import hashlib
import numpy as np
from figtree import Figment, FigmentStore, connect

from figtree_news import intersection as int_mod
from figtree_news import lineage as lineage_mod


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

    # Article 1: Trump + Disney World (via two role figments)
    trump = _role_fig("who", "Donald Trump", "a1")
    disney = _role_fig("where", "Disney World", "a1")
    what = _role_fig("what", "visit", "a1")

    a = Figment.create(
        text="Trump visits Disney World in Florida.",
        boundary=np.zeros(8, dtype="float32"),
        meta={
            "source_id": "reuters",
            "decomposed": True,
            "role_figments": [trump.figment_id, disney.figment_id, what.figment_id],
            "url": "http://reuters.com/1",
            "published": "Mon, 01 Jan 2024 10:00:00 GMT",
            "first_seen": "2024-01-01T10:00:00+00:00",
        },
        figment_id="img1",
        trust=0.8,
        kind="article",
    )

    # Article 2: Trump + Disney World but different surface form (DJT + Walt Disney World)
    # Without associations, these would NOT share role figment IDs with article 1.
    djt_fig = _role_fig("who", "DJT", "a2")
    wdw_fig = _role_fig("where", "Walt Disney World", "a2")
    what2 = _role_fig("what", "trip", "a2")

    b = Figment.create(
        text="DJT takes a trip to Walt Disney World.",
        boundary=np.zeros(8, dtype="float32"),
        meta={
            "source_id": "cnn",
            "decomposed": True,
            "role_figments": [djt_fig.figment_id, wdw_fig.figment_id, what2.figment_id],
            "url": "http://cnn.com/1",
            "published": "Tue, 02 Jan 2024 10:00:00 GMT",
            "first_seen": "2024-01-02T10:00:00+00:00",
        },
        figment_id="img2",
        trust=0.7,
        kind="article",
    )

    # Article 3: only has Trump (Disney World), should NOT be in the intersection with Disney
    what3 = _role_fig("what", "speech", "a3")
    c = Figment.create(
        text="Trump gives a speech about the economy.",
        boundary=np.zeros(8, dtype="float32"),
        meta={
            "source_id": "foxnews",
            "decomposed": True,
            "role_figments": [trump.figment_id, what3.figment_id],
            "url": "http://foxnews.com/1",
            "published": "Wed, 03 Jan 2024 10:00:00 GMT",
            "first_seen": "2024-01-03T10:00:00+00:00",
        },
        figment_id="img3",
        trust=0.5,
        kind="article",
    )

    store.upsert(
        [a, b, c, trump, disney, what, djt_fig, wdw_fig, what2, what3],
        hidden_size=8,
    )

    # Compute lineage so narratives exist and role figments are linked via story_id
    lineage_mod.compute_lineage(store)

    return store


def test_find_narratives_no_assoc(tmp_path):
    """Without associations, Trump and DJT are separate — only article 1 matches WHO:Trump + WHERE:Disney World."""
    store = _seed_store(tmp_path)

    results = int_mod.find_narratives(
        store,
        roles=[{"role": "who", "text": "Donald Trump"}, {"role": "where", "text": "Disney World"}],
        expand_associations=False,
        require_all=True,
    )
    assert len(results) == 1, f"Expected 1 narrative without associations, got {len(results)}"
    assert results[0]["narrative_id"] is not None


def test_find_narratives_with_assoc(tmp_path):
    """With associations, DJT and Walt Disney World are variants — both articles should match."""
    store = _seed_store(tmp_path)
    all_figs = store.all()

    # Assert associations: Trump <-> DJT and Disney World <-> Walt Disney World
    trump = next(f for f in all_figs if f.text == "Donald Trump" and f.meta.get("role") == "who")
    djt = next(f for f in all_figs if f.text == "DJT" and f.meta.get("role") == "who")
    disney = next(f for f in all_figs if f.text == "Disney World" and f.meta.get("role") == "where")
    wdw = next(f for f in all_figs if f.text == "Walt Disney World" and f.meta.get("role") == "where")

    from figtree_news.associations import assert_association
    assert_association(store, trump.figment_id, djt.figment_id, confidence=1.0, evidence="test")
    assert_association(store, disney.figment_id, wdw.figment_id, confidence=1.0, evidence="test")

    results = int_mod.find_narratives(
        store,
        roles=[{"role": "who", "text": "Donald Trump"}, {"role": "where", "text": "Disney World"}],
        expand_associations=True,
        require_all=True,
    )
    assert len(results) >= 1, f"Expected >=1 narrative with associations, got {len(results)}"

    # Verify role_matches are present
    for r in results:
        assert "role_matches" in r


def test_find_narratives_require_any(tmp_path):
    """require_all=False should return narratives matching at least one role."""
    store = _seed_store(tmp_path)

    results = int_mod.find_narratives(
        store,
        roles=[{"role": "who", "text": "Donald Trump"}, {"role": "where", "text": "Disney World"}],
        require_all=False,
        expand_associations=False,
    )
    # Article 1 and Article 2 match Trump, Article 1 and Article 2 match Disney World
    # Article 3 matches only Trump
    assert len(results) >= 2, f"Expected >=2 narratives for OR query, got {len(results)}"


def test_find_narratives_empty_roles(tmp_path):
    """Empty roles should return empty results."""
    store = _seed_store(tmp_path)

    results = int_mod.find_narratives(store, roles=[])
    assert results == []


def test_find_role_figments(tmp_path):
    """find_role_figments should return matching role figments."""
    store = _seed_store(tmp_path)

    results = int_mod.find_role_figments(store, "who", "Donald Trump")
    assert len(results) >= 1
    assert all(f.meta.get("role") == "who" for f in results)


def test_find_narratives_no_match(tmp_path):
    """Roles with no matching figments should return empty."""
    store = _seed_store(tmp_path)

    results = int_mod.find_narratives(
        store,
        roles=[{"role": "who", "text": "Nobody"}],
        expand_associations=False,
    )
    assert results == []