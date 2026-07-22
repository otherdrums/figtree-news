"""Post-crawl pipeline: trust -> lineage -> (eval) -> (correct) -> summaries -> brief.

Runs on the crawler (which holds the model). Keeps the store coherent so the
web viewer can render everything without touching the GPU.
"""

from __future__ import annotations

from typing import Any

from figtree import FigmentStore

from . import summarize_news
from . import trust as trust_mod
from . import lineage as lineage_mod
from .llm_config import LLMConfig


def run_pipeline(
    model,
    tokenizer,
    store: FigmentStore,
    do_summaries: bool = True,
    do_brief: bool = True,
    max_stories: int = 0,
    llm_config: LLMConfig | None = None,
) -> dict[str, Any]:
    trust_out = trust_mod.update_trust(store)
    lineage_out = lineage_mod.compute_lineage(store, max_stories=max_stories)

    eval_out = {"evaluated": 0, "corrections_suggested": 0, "corrections_applied": 0}
    if llm_config and llm_config.enabled:
        eval_out = _run_evaluation(store, llm_config)
        if llm_config.auto_correct:
            from . import correct
            corr_out = correct.confirm_and_apply(store, llm_config.confirmation_threshold)
            eval_out["corrections_applied"] = corr_out.get("corrections_applied", 0)

    brief_out = {"used": 0, "brief": ""}
    summaries_out = {"summarized": 0}
    if do_summaries:
        summaries_out = summarize_news.ensure_article_summaries(model, tokenizer, store)
    if do_brief:
        brief_out = summarize_news.build_world_brief(model, tokenizer, store)

    # Review brief with external LLM if available
    brief_eval = {}
    if llm_config and llm_config.enabled and llm_config.review_brief and brief_out.get("brief"):
        from . import evaluate
        brief_eval = evaluate.review_brief(store, brief_out["brief"],
                                            evaluate.LLMClient(llm_config), llm_config)

    return {
        "trust_updates": len(trust_out["updates"]),
        "narratives": len(lineage_out["narratives"]),
        "lineage_edges": lineage_out["edges"],
        "summarized": summaries_out["summarized"],
        "brief_used": brief_out["used"],
        **eval_out,
        **brief_eval,
    }


def _run_evaluation(store: FigmentStore, llm_config: LLMConfig) -> dict[str, Any]:
    try:
        from . import evaluate
        client = evaluate.LLMClient(llm_config)
        result = evaluate.evaluate_narratives(store, client, llm_config)
        return result
    except Exception as exc:
        print(f"[pipeline] evaluation skipped: {exc}")
        return {"evaluated": 0, "corrections_suggested": 0,
                "evaluation_error": str(exc)}
