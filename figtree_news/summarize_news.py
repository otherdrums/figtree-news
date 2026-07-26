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


def _article_images(store: FigmentStore, *, all_figs: list | None = None) -> list[Figment]:
    return [
        f
        for f in (all_figs if all_figs is not None else store.all())
        if f.kind == "article" and f.meta.get("source_id")
    ]


def ensure_article_summaries(
    model, tokenizer, store: FigmentStore, *, all_figs: list | None = None, limit: int = 500
) -> dict[str, Any]:
    gen = FigmentGenerator(model, tokenizer)
    done = 0
    updated: list[Figment] = []
    for f in _article_images(store, all_figs=all_figs):
        if f.meta.get("summary"):
            continue
        result = gen.generate(
            [f], "Summarize the above article in 2-3 concise sentences.", max_new_tokens=96
        )
        f.meta["summary"] = result.get("generated_text", "").strip()
        updated.append(f)
        done += 1
        
        # Clear GPU cache every 5 summaries to prevent OOM on low-VRAM GPUs
        if done % 5 == 0:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if done >= limit:
            break
    if updated:
        hidden = updated[0].boundary.shape[0]
        store.upsert(updated, hidden_size=hidden)
    return {"summarized": done}


def build_world_brief(
    model, tokenizer, store: FigmentStore, *, all_figs: list | None = None, top_n: int = 2
) -> dict[str, Any]:
    """Generate a combined brief over the top narratives; persist as a figment.

    Falls back to the top articles directly when no narratives exist yet.
    """
    # Clear VRAM before brief generation to prevent OOM
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    narratives = get_narratives(store)[:top_n]
    members: list[str] = []
    for n in narratives:
        members.extend(n["members"][:1])  # Only 1 article per narrative to reduce context
    figs = {f.figment_id: f for f in _article_images(store, all_figs=all_figs)}
    selected = [figs[mid] for mid in dict.fromkeys(members) if mid in figs][:top_n]

    if not selected:
        # Fallback: use the top articles directly when no narratives exist.
        articles = sorted(
            _article_images(store, all_figs=all_figs),
            key=lambda f: f.meta.get("first_seen", "") or f.figment_id,
            reverse=True,
        )[:top_n]
        selected = [a for a in articles if a.figment_id]
        if not selected:
            print("[brief] no articles selected for brief generation")
            return {"brief": "", "used": 0}

    print(f"[brief] generating from {len(selected)} articles:")
    for f in selected:
        src = f.meta.get("source_id", "?")
        title = f.meta.get("title", "")[:50]
        print(f"[brief]   - {src}: {title}")

    gen = FigmentGenerator(model, tokenizer)
    result = gen.generate(
        selected,
        "Write a brief world news summary covering the following reports.",
        max_new_tokens=150,
    )
    brief = result.get("generated_text", "").strip()
    print(f"[brief] generated {len(brief)} chars: {brief[:100]}...")
    brief_fig = Figment.create(
        text=brief,
        boundary=selected[0].boundary.copy(),
        meta={"edge_type": "brief", "brief_kind": "world"},
        figment_id="brief:world",
        kind="edge",
    )
    hidden = selected[0].boundary.shape[0]
    store.upsert([brief_fig], hidden_size=hidden)
    return {"brief": brief, "used": len(selected)}


def get_world_brief(store: FigmentStore, *, all_figs: list | None = None) -> str:
    for f in all_figs if all_figs is not None else store.all():
        if f.figment_id == "brief:world":
            return f.text
    return ""
