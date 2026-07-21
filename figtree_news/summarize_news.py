"""Pre-generated summaries so the newspaper renders without on-demand GPU.

Runs on the crawler (which holds the model). For each article that lacks a
``summary``, generate a short recap and persist it on the figment's meta. Also
build a single "world brief" across the top stories for the front page. All
steps are idempotent (skips figments that already have a summary / brief).
"""

from __future__ import annotations

from typing import Any

from figtree import Figment, FigmentGenerator, FigmentStore

from .lineage import get_narratives


def _article_images(store: FigmentStore) -> list[Figment]:
    return [
        f
        for f in store.all()
        if f.meta.get("is_image") and f.meta.get("source_id") and not f.is_edge()
    ]


def ensure_article_summaries(
    model, tokenizer, store: FigmentStore, limit: int = 500
) -> dict[str, Any]:
    gen = FigmentGenerator(model, tokenizer)
    done = 0
    updated: list[Figment] = []
    for f in _article_images(store):
        if f.meta.get("summary"):
            continue
        result = gen.generate(
            [f], "Summarize the above article in 2-3 concise sentences.", max_new_tokens=96
        )
        f.meta["summary"] = result.get("text", "").strip()
        updated.append(f)
        done += 1
        if done >= limit:
            break
    if updated:
        hidden = updated[0].boundary.shape[0]
        store.upsert(updated, hidden_size=hidden)
    return {"summarized": done}


def build_world_brief(
    model, tokenizer, store: FigmentStore, top_n: int = 8
) -> dict[str, Any]:
    """Generate a combined brief over the top narratives; persist as a figment."""
    narratives = get_narratives(store)[:top_n]
    members: list[str] = []
    for n in narratives:
        members.extend(n["members"][:2])
    figs = {f.figment_id: f for f in _article_images(store)}
    selected = [figs[mid] for mid in dict.fromkeys(members) if mid in figs][:top_n]
    if not selected:
        return {"brief": "", "used": 0}

    gen = FigmentGenerator(model, tokenizer)
    result = gen.generate(
        selected,
        "Write a brief world news summary covering the following reports.",
        max_new_tokens=300,
    )
    brief = result.get("text", "").strip()
    brief_fig = Figment.create(
        text=brief,
        boundary=selected[0].boundary.copy(),
        meta={"edge_type": "brief", "kind": "world"},
        figment_id="brief:world",
    )
    hidden = selected[0].boundary.shape[0]
    store.upsert([brief_fig], hidden_size=hidden)
    return {"brief": brief, "used": len(selected)}


def get_world_brief(store: FigmentStore, *, all_figs: list | None = None) -> str:
    for f in all_figs if all_figs is not None else store.all():
        if f.figment_id == "brief:world":
            return f.text
    return ""
