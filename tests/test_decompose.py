"""CPU tests for paragraph-aware decomposition (parsing helpers only).

The local model generation is exercised in the v2 pipeline test; here we test
prompt construction and output parsing without loading the model.
"""

from __future__ import annotations

import json

import pytest
from figtree import Figment, FigmentStore, connect
import numpy as np

from figtree_news.decompose import (
    _build_decompose_prompt,
    _extract_json,
    _parse_roles,
    _create_role_figments,
    decompose_articles,
)


def test_build_decompose_prompt_includes_paragraph_and_sentences():
    prompt = _build_decompose_prompt("The paragraph.", ["Sentence one.", "Sentence two."])
    assert "The paragraph." in prompt
    assert "Sentence one." in prompt
    assert "Sentence two." in prompt
    assert '"paragraph"' in prompt
    assert '"sentences"' in prompt


def test_extract_json_strips_think_tags():
    raw = "<think>some reasoning</think>{\"paragraph\": {}} extra"
    parsed = _extract_json(raw)
    assert parsed == {"paragraph": {}}


def test_extract_json_handles_markdown_block():
    raw = "```json\n{\"paragraph\": {\"who\": \"x\"}, \"sentences\": []}\n```"
    parsed = _extract_json(raw)
    assert parsed == {"paragraph": {"who": "x"}, "sentences": []}


def test_parse_roles_normalizes_empty_fields():
    parsed = {
        "paragraph": {"who": "Trump", "what": "visited", "where": "Disney World", "when": "", "why": "", "how": ""},
        "sentences": [
            {"who": "Trump", "what": "visited", "where": "Disney World", "when": "", "why": "", "how": ""},
        ],
    }
    para, sents = _parse_roles(parsed, 1)
    assert para["who"] == "Trump"
    assert para["when"] == ""
    assert len(sents) == 1
    assert sents[0]["where"] == "Disney World"


def test_parse_roles_pads_missing_sentences():
    parsed = {"paragraph": {}, "sentences": []}
    para, sents = _parse_roles(parsed, 2)
    assert len(sents) == 2
    assert sents[0]["who"] == ""


def test_create_role_figments(tmp_path):
    store: FigmentStore = connect(str(tmp_path / "roles.lance"))
    parent = Figment.create("parent", np.zeros(8, dtype="float32"), figment_id="parent1")
    article = Figment.create("article", np.zeros(8, dtype="float32"), figment_id="article1", kind="article")
    store.upsert([parent, article], hidden_size=8)
    by_id = {"parent1": parent, "article1": article}
    created: dict[str, Figment] = {}
    ids = _create_role_figments(
        {"who": "Trump", "what": "visit", "where": ""},
        "parent1",
        parent,
        "article1",
        by_id,
        store,
        created,
    )
    assert len(ids) == 2
    assert len(created) == 2
    for fid in ids:
        assert created[fid].kind == "role"
        assert created[fid].meta["role"] in ("who", "what")


def test_create_role_figments_reuses_existing(tmp_path):
    store: FigmentStore = connect(str(tmp_path / "roles.lance"))
    parent = Figment.create("parent", np.zeros(8, dtype="float32"), figment_id="parent1")
    article = Figment.create("article", np.zeros(8, dtype="float32"), figment_id="article1", kind="article")
    store.upsert([parent, article], hidden_size=8)
    by_id = {"parent1": parent, "article1": article}
    created: dict[str, Figment] = {}
    ids1 = _create_role_figments({"who": "Trump"}, "parent1", parent, "article1", by_id, store, created)
    created.clear()
    ids2 = _create_role_figments({"who": "Trump"}, "parent1", parent, "article1", by_id, store, created)
    assert ids1 == ids2
    assert len(created) == 1
    assert "parent1" in created[ids2[0]].meta["references"]


def test_get_or_create_role_figment_uses_boundary_dedup(tmp_path):
    """Boundary similarity (>= 0.90) merges same-role figments from different
    parent sentences about the same event, even when exact text differs."""
    import hashlib
    from figtree_news.decompose import _get_or_create_role_figment, _normalize_text, _boundary_sim

    store: FigmentStore = connect(str(tmp_path / "boundary_dedup.lance"))

    # A paragraph with a distinctive boundary (unit vector along x)
    para_boundary = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    paragraph = Figment.create(
        "Trump announced sweeping tariffs on China", para_boundary,
        figment_id="para1", kind="paragraph",
    )
    article = Figment.create(
        "article", np.zeros(3, dtype=np.float32),
        figment_id="article1", kind="article",
    )
    store.upsert([paragraph, article], hidden_size=3)

    # Pre-existing role figment "Donald Trump" with a nearly-identical boundary
    # (same event, slightly different sentence embedding)
    existing_text = "Donald Trump"
    existing_normalized = _normalize_text(existing_text)
    existing_id = hashlib.sha256(f"role:who:{existing_normalized}".encode()).hexdigest()[:16]
    existing_boundary = np.array([0.95, 0.05, 0.0], dtype=np.float32)
    existing = Figment.create(
        text=existing_text,
        boundary=existing_boundary,
        meta={
            "role": "who",
            "parent_id": "other_para",
            "article_id": "article1",
            "references": ["other_para"],
            "reference_count": 1,
            "normalized": existing_normalized,
        },
        figment_id=existing_id,
        kind="role",
    )
    store.upsert([existing], hidden_size=3)

    # Verify boundaries ARE similar: para_boundary · existing_boundary ≈ 0.999
    assert _boundary_sim(para_boundary, existing_boundary) >= 0.99

    # Load by_id from store
    by_id = {f.figment_id: f for f in store.all()}
    created: dict[str, Figment] = {}

    # Call with "Trump" (different text) — should find existing via boundary dedup
    result = _get_or_create_role_figment(
        "Trump", "who", "para1", "article1", store, by_id, created, boundary_threshold=0.90,
    )
    assert result is not None
    assert result.figment_id == existing_id, "boundary dedup should return existing figment"
    assert "para1" in result.meta["references"], "should track new parent reference"


def test_get_or_create_role_figment_boundary_no_match_different_role(tmp_path):
    """Boundary similarity only matches same-role figments (different roles
    from the same sentence have identical boundaries but should not merge)."""
    import hashlib
    from figtree_news.decompose import _get_or_create_role_figment, _normalize_text

    store: FigmentStore = connect(str(tmp_path / "boundary_no_role.lance"))

    para_boundary = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    paragraph = Figment.create(
        "Trump visited Disney World", para_boundary,
        figment_id="para1", kind="paragraph",
    )
    article = Figment.create(
        "article", np.zeros(3, dtype=np.float32),
        figment_id="article1", kind="article",
    )
    store.upsert([paragraph, article], hidden_size=3)

    # A WHERE role figment with the same boundary as the parent
    where_text = "Disney World"
    where_normalized = _normalize_text(where_text)
    where_id = hashlib.sha256(f"role:where:{where_normalized}".encode()).hexdigest()[:16]
    existing = Figment.create(
        text=where_text,
        boundary=para_boundary.copy(),
        meta={
            "role": "where",
            "parent_id": "para1",
            "article_id": "article1",
            "references": ["para1"],
            "reference_count": 1,
            "normalized": where_normalized,
        },
        figment_id=where_id,
        kind="role",
    )
    store.upsert([existing], hidden_size=3)

    by_id = {f.figment_id: f for f in store.all()}
    created: dict[str, Figment] = {}

    # Calling for WHO="Trump" should NOT match the WHERE figment (different role)
    result = _get_or_create_role_figment(
        "Trump", "who", "para1", "article1", store, by_id, created, boundary_threshold=0.90,
    )
    assert result is not None
    assert result.figment_id != where_id, "different role should not merge"


def test_get_or_create_role_figment_boundary_no_match_orthogonal(tmp_path):
    """Strings with no heuristic overlap do not trigger text-based dedup."""
    import hashlib
    from figtree_news.decompose import _get_or_create_role_figment, _normalize_text

    store: FigmentStore = connect(str(tmp_path) + "/boundary_ortho.lance")

    para_boundary = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    paragraph = Figment.create(
        "Trump at Disney World", para_boundary,
        figment_id="para1", kind="paragraph",
    )
    article = Figment.create(
        "article", np.zeros(3, dtype=np.float32),
        figment_id="article1", kind="article",
    )
    store.upsert([paragraph, article], hidden_size=3)

    # Existing WHO figment with a COMPLETELY UNRELATED text
    other_text = "Hong Kong protests"
    other_normalized = _normalize_text(other_text)
    other_id = hashlib.sha256(f"role:who:{other_normalized}".encode()).hexdigest()[:16]
    other_boundary = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    existing = Figment.create(
        text=other_text,
        boundary=other_boundary,
        meta={
            "role": "who",
            "parent_id": "other_para",
            "article_id": "article1",
            "references": ["other_para"],
            "reference_count": 1,
            "normalized": other_normalized,
        },
        figment_id=other_id,
        kind="role",
    )
    store.upsert([existing], hidden_size=3)

    by_id = {f.figment_id: f for f in store.all()}
    created: dict[str, Figment] = {}

    # Completely unrelated texts → no heuristic overlap → new figment
    result = _get_or_create_role_figment(
        "Trump", "who", "para1", "article1", store, by_id, created, boundary_threshold=0.90,
    )
    assert result is not None
    assert result.figment_id != other_id, "dissimilar texts should not merge"
    assert result.text == "Trump"


def test_get_or_create_role_figment_dedup_within_batch(tmp_path):
    """Within the same batch, two same-role figments with similar parent
    boundaries dedup via the ``created`` dict (step 2a), not the ANN store."""
    import hashlib
    from figtree_news.decompose import _get_or_create_role_figment, _normalize_text

    store: FigmentStore = connect(str(tmp_path / "batch_dedup.lance"))

    para1 = Figment.create("para1", np.array([1.0, 0.0, 0.0], dtype=np.float32),
                           figment_id="para1", kind="paragraph")
    para2 = Figment.create("para2", np.array([0.95, 0.05, 0.0], dtype=np.float32),
                           figment_id="para2", kind="paragraph")
    article = Figment.create("article", np.zeros(3, dtype=np.float32),
                             figment_id="article1", kind="article")
    store.upsert([para1, para2, article], hidden_size=3)

    by_id = {f.figment_id: f for f in store.all()}
    created: dict[str, Figment] = {}

    # First call creates a fresh WHO figment for "Trump".
    # (_create_role_figments would add it to created; we replicate that here.)
    fig1 = _get_or_create_role_figment(
        "Trump", "who", "para1", "article1", store, by_id, created, boundary_threshold=0.90,
    )
    assert fig1 is not None
    if fig1.figment_id not in created:
        created[fig1.figment_id] = fig1
    first_id = fig1.figment_id

    # Second call for "Donald Trump" from para2 — text hash differs, but
    # parent boundaries are very similar → should find fig1 in created.
    fig2 = _get_or_create_role_figment(
        "Donald Trump", "who", "para2", "article1", store, by_id, created, boundary_threshold=0.90,
    )
    assert fig2 is not None
    assert fig2.figment_id == first_id, "within-batch boundary dedup should match"
    assert "para2" in fig2.meta["references"]
    # created should NOT grow — fig2 reused fig1
    assert len(created) == 1


def test_decompose_articles_without_model_skips():
    """When no model is provided, decomposition is a no-op."""
    store: FigmentStore = connect("/tmp/figtree_news_decompose_skip_test.lance")
    result = decompose_articles(None, None, store, ["nonexistent"])
    assert result["completed"] == 0
    assert result["queued"] == 0


@pytest.mark.parametrize("raw", ["not json", "{\"incomplete\":", ""])
def test_extract_json_returns_empty_on_invalid(raw):
    assert _extract_json(raw) == {}


def test_searxng_time_range_mapping():
    from figtree_news.web.serve import _normalize_searx_time_range, _next_time_range

    assert _normalize_searx_time_range("day") == "day"
    assert _normalize_searx_time_range("last_week") == "week"
    assert _normalize_searx_time_range("last_month") == "month"
    assert _normalize_searx_time_range("last_year") == "year"
    assert _normalize_searx_time_range("all") == ""
    assert _next_time_range("day") == "last_week"
    assert _next_time_range("last_week") == "last_month"
    assert _next_time_range("all") == "all"
