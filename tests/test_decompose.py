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
    ids1 = _create_role_figments({"who": "Trump"}, "parent1", "article1", by_id, store, created)
    created.clear()
    ids2 = _create_role_figments({"who": "Trump"}, "parent1", "article1", by_id, store, created)
    assert ids1 == ids2
    assert len(created) == 1
    assert "parent1" in created[ids2[0]].meta["references"]


def test_decompose_articles_without_model_skips():
    """When no model is provided, decomposition is a no-op."""
    store: FigmentStore = connect("/tmp/figtree_news_decompose_skip_test.lance")
    result = decompose_articles(None, None, store, ["nonexistent"])
    assert result["completed"] == 0
    assert result["queued"] == 0


@pytest.mark.parametrize("raw", ["not json", "{\"incomplete\":", ""])
def test_extract_json_returns_empty_on_invalid(raw):
    assert _extract_json(raw) == {}
