"""Post-crawl pipeline: trust -> lineage -> summaries -> world brief.

Runs on the crawler (which holds the model). Keeps the store coherent so the
web viewer can render everything without touching the GPU.
"""

from __future__ import annotations

from typing import Any

from figtree import FigmentStore

from . import summarize_news
from . import trust as trust_mod
from . import lineage as lineage_mod


def run_pipeline(
    model,
    tokenizer,
    store: FigmentStore,
    do_summaries: bool = True,
    do_brief: bool = True,
    max_stories: int = 0,
) -> dict[str, Any]:
    trust_out = trust_mod.update_trust(store)
    lineage_out = lineage_mod.compute_lineage(store, max_stories=max_stories)
    summaries = {"summarized": 0}
    brief = {"used": 0, "brief": ""}
    if do_summaries:
        summaries = summarize_news.ensure_article_summaries(model, tokenizer, store)
    if do_brief:
        brief = summarize_news.build_world_brief(model, tokenizer, store)
    return {
        "trust_updates": len(trust_out["updates"]),
        "narratives": len(lineage_out["narratives"]),
        "lineage_edges": lineage_out["edges"],
        "summarized": summaries["summarized"],
        "brief_used": brief["used"],
    }
