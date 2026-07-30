"""CPU tests for the async association worker (heuristic-only path)."""

from __future__ import annotations

import hashlib
import numpy as np
import pytest
from figtree import Figment, FigmentStore, connect

from figtree_news.normalize import normalize as _norm
from figtree_news.association_worker import AssociationWorker, _compute_similarities, _passes_prefilter, _candidate_score


def _role_fig(role: str, text: str, article_id: str, references: list | None = None) -> Figment:
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


def _article_fig(article_id: str, role_ids: list[str]) -> Figment:
    return Figment.create(
        text=f"Article {article_id}",
        boundary=np.zeros(8, dtype="float32"),
        meta={"role_figments": list(role_ids), "source_id": "test"},
        figment_id=article_id,
        kind="article",
    )


@pytest.mark.asyncio
async def test_worker_heuristic_merge(tmp_path):
    """Worker merges two similar role figments via heuristic path."""
    store: FigmentStore = connect(str(tmp_path / "news.lance"))

    keep = _role_fig("who", "Donald Trump", "a1", references=["s1", "s2"])
    remove = _role_fig("who", "Trump", "a2", references=["s3"])
    article = _article_fig("art1", [keep.figment_id, remove.figment_id])

    store.upsert([keep, remove, article], hidden_size=8)

    worker = AssociationWorker(store, llm_config=None, interval=0.1)
    try:
        await worker._tick()
    finally:
        await worker.stop()

    # The worker should have merged remove into keep
    stored_keep = store.get(keep.figment_id)
    assert stored_keep is not None
    assert stored_keep.meta["is_association"] is True
    assert remove.figment_id in stored_keep.meta.get("merged_from", [])

    stored_remove = store.get(remove.figment_id)
    assert stored_remove is None, "Removed figment should be deleted"


@pytest.mark.asyncio
async def test_worker_skips_processed(tmp_path):
    """Worker does not re-process already processed IDs."""
    store: FigmentStore = connect(str(tmp_path / "news.lance"))

    keep = _role_fig("who", "Donald Trump", "a1", references=["s1"])
    remove = _role_fig("who", "Trump", "a2", references=["s2"])
    store.upsert([keep, remove], hidden_size=8)

    worker = AssociationWorker(store, llm_config=None, interval=0.1)
    # Mark both as already processed
    worker._processed.add(keep.figment_id)
    worker._processed.add(remove.figment_id)

    try:
        await worker._tick()
    finally:
        await worker.stop()

    # Neither should be merged (processed set skipped them)
    assert store.get(remove.figment_id) is not None, "Should NOT have been merged"


@pytest.mark.asyncio
async def test_worker_empty_store(tmp_path):
    """Worker handles empty store without error."""
    store: FigmentStore = connect(str(tmp_path / "news.lance"))

    worker = AssociationWorker(store, llm_config=None, interval=0.1)
    try:
        await worker._tick()
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_worker_skips_is_association(tmp_path):
    """Worker skips figments that are already association nodes."""
    store: FigmentStore = connect(str(tmp_path / "news.lance"))

    node = _role_fig("who", "Donald Trump", "a1", references=["s1", "s2"])
    node.meta["is_association"] = True
    other = _role_fig("who", "Trump", "a2", references=["s3"])
    store.upsert([node, other], hidden_size=8)

    worker = AssociationWorker(store, llm_config=None, interval=0.1)
    try:
        await worker._tick()
    finally:
        await worker.stop()

    # The node (is_association=True) should be skipped; "Trump" should be merged into it
    stored_node = store.get(node.figment_id)
    assert stored_node is not None
    assert stored_node.meta["is_association"] is True
    assert other.figment_id in stored_node.meta.get("merged_from", [])

    stored_other = store.get(other.figment_id)
    assert stored_other is None


def test_pick_winner():
    """_pick_winner returns the figment with more references as keep."""
    a = _role_fig("who", "Donald Trump", "a1", references=["s1", "s2"])
    b = _role_fig("who", "Trump", "a2", references=["s1"])

    worker = AssociationWorker.__new__(AssociationWorker)
    # Mock store to avoid the __init__ store.all() call
    worker._llm_client = None
    worker._processed = set()
    worker._semaphore = None

    keep_id, remove_id = worker._pick_winner(a, b)
    assert keep_id == a.figment_id
    assert remove_id == b.figment_id

    # Equal refs: a wins (first arg)
    b.meta["references"] = ["s1", "s2"]
    b.meta["reference_count"] = 2
    keep_id, remove_id = worker._pick_winner(a, b)
    assert keep_id == a.figment_id


def test_compute_similarities():
    """Similarity computation works correctly."""
    b = np.zeros(8, dtype="float32")
    sims = _compute_similarities("Donald Trump", b, "Trump", b)
    assert sims["containment"] >= 0.4
    assert sims["jaccard"] >= 0.25
    # Boundary sim for zero vectors
    assert sims["boundary_sim"] == 0.0


def test_passes_prefilter():
    """Prefilter thresholds match expectations."""
    # boundary >= 0.90
    assert _passes_prefilter({"boundary_sim": 0.95, "containment": 0.0, "edit_sim": 0.0})
    # containment >= 0.4
    assert _passes_prefilter({"boundary_sim": 0.0, "containment": 0.5, "edit_sim": 0.0})
    # edit_sim >= 0.7
    assert _passes_prefilter({"boundary_sim": 0.0, "containment": 0.0, "edit_sim": 0.8})
    # none pass
    assert not _passes_prefilter({"boundary_sim": 0.5, "containment": 0.3, "edit_sim": 0.5})


def test_candidate_score():
    """Returns max of the three scores."""
    score = _candidate_score({"boundary_sim": 0.5, "containment": 0.8, "edit_sim": 0.6})
    assert score == 0.8
