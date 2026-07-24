"""Post-crawl pipeline: trust -> lineage -> (eval) -> (correct) -> summaries -> brief.

Runs on the crawler (which holds the model). Keeps the store coherent so the
web viewer can render everything without touching the GPU.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from figtree import FigmentStore

from . import summarize_news
from . import trust as trust_mod
from . import lineage as lineage_mod
from .llm_config import LLMConfig

log = logging.getLogger(__name__)


def run_pipeline(
    model,
    tokenizer,
    store: FigmentStore,
    do_summaries: bool = True,
    do_brief: bool = True,
    max_stories: int = 0,
    llm_config: LLMConfig | None = None,
) -> dict[str, Any]:
    log.info("START — llm_enabled=%s", llm_config.enabled if llm_config else False)
    t_start = time.time()

    # Load all figments once — threaded through every phase to avoid redundant store.all()
    all_figs = store.all()
    log.info("loaded %d figments from store", len(all_figs))

    # Phase 1: Trust
    trust_out = {"analysis": {}, "updates": []}
    try:
        t0 = time.time()
        log.info("Phase 1: Trust propagation")
        trust_out = trust_mod.update_trust(store, all_figs=all_figs)
        log.info("  trust_updates=%d  (%.1fs)", len(trust_out["updates"]), time.time() - t0)
    except Exception as exc:
        log.error("Phase 1 FAILED: %s", exc, exc_info=True)

    # Phase 2: Lineage
    lineage_out = {"narratives": [], "edges": 0}
    try:
        t0 = time.time()
        log.info("Phase 2: Lineage (narrative clustering)")
        lineage_out = lineage_mod.compute_lineage(store, max_stories=max_stories)
        log.info("  narratives=%d  edges=%d  (%.1fs)", len(lineage_out["narratives"]), lineage_out["edges"], time.time() - t0)
        for n in lineage_out.get("narratives", [])[:5]:
            log.info("    - %s: %s", n["narrative_id"][:8], n.get("title", "")[:60])
            log.info("      sources=%s  members=%d  frame_shift=%s", n.get("sources", []), len(n.get("members", [])), n.get("frame_shift", False))
    except Exception as exc:
        log.error("Phase 2 FAILED: %s", exc, exc_info=True)

    # Phase 2.5: LLM-based clustering evaluation (ground truth)
    if llm_config and llm_config.enabled:
        try:
            t0 = time.time()
            log.info("Phase 2.5: LLM-based clustering evaluation (ground truth)")
            from . import evaluate
            articles = [f for f in all_figs if f.meta.get("is_image") and f.meta.get("source_id") and not f.is_edge()]
            log.info("  found %d articles for labeling", len(articles))

            if len(articles) >= 2:
                log.info("  starting LLM labeling...")
                client = evaluate.LLMClient(llm_config)
                labels = evaluate.label_article_pairs(articles, client, max_pairs=20)
                log.info("  got %d labels", len(labels))
                log.info("  clustering evaluation complete (%.1fs)", time.time() - t0)
            else:
                log.info("  not enough articles for evaluation")
        except Exception as exc:
            log.error("Phase 2.5 FAILED: %s", exc, exc_info=True)

    # Phase 3: LLM Evaluation
    eval_out = {"evaluated": 0, "corrections_suggested": 0, "corrections_applied": 0}
    if llm_config and llm_config.enabled:
        try:
            t0 = time.time()
            log.info("Phase 3: LLM evaluation")
            eval_out = _run_evaluation(store, llm_config)
            log.info("  evaluated=%d  corrections_suggested=%d  (%.1fs)", eval_out.get("evaluated", 0), eval_out.get("corrections_suggested", 0), time.time() - t0)

            if llm_config.auto_correct:
                from . import correct
                log.info("Phase 4: Apply corrections (threshold=%s)", llm_config.confirmation_threshold)
                corr_out = correct.confirm_and_apply(store, llm_config.confirmation_threshold)
                eval_out["corrections_applied"] = corr_out.get("corrections_applied", 0)
                log.info("  corrections_applied=%d", eval_out["corrections_applied"])
        except Exception as exc:
            log.error("Phase 3-4 FAILED: %s", exc, exc_info=True)
    else:
        log.info("Phase 3-4: SKIPPED (LLM not enabled)")

    # Phase 5: Summaries
    brief_out = {"used": 0, "brief": ""}
    summaries_out = {"summarized": 0}
    if do_summaries:
        try:
            t0 = time.time()
            log.info("Phase 5: Article summaries")
            summaries_out = summarize_news.ensure_article_summaries(model, tokenizer, store, all_figs=all_figs)
            log.info("  summarized=%d  (%.1fs)", summaries_out["summarized"], time.time() - t0)
        except Exception as exc:
            log.error("Phase 5 FAILED: %s", exc, exc_info=True)
    else:
        log.info("Phase 5: Summaries SKIPPED")

    # Phase 6: Brief
    if do_brief:
        try:
            t0 = time.time()
            log.info("Phase 6: World brief")
            brief_out = summarize_news.build_world_brief(model, tokenizer, store, all_figs=all_figs)
            log.info("  brief_used=%d articles  (%.1fs)", brief_out["used"], time.time() - t0)
            if brief_out.get("brief"):
                log.info("  brief_text: %s...", brief_out["brief"][:150])
        except Exception as exc:
            log.error("Phase 6 FAILED: %s", exc, exc_info=True)
    else:
        log.info("Phase 6: Brief SKIPPED")

    # Phase 7: Brief review
    brief_eval = {}
    if llm_config and llm_config.enabled and llm_config.review_brief and brief_out.get("brief"):
        try:
            t0 = time.time()
            log.info("Phase 7: Brief review (LLM)")
            from . import evaluate
            brief_eval = evaluate.review_brief(store, brief_out["brief"],
                                                evaluate.LLMClient(llm_config), llm_config)
            log.info("  brief_acceptable=%s  issues=%d  (%.1fs)", brief_eval.get("brief_acceptable"), brief_eval.get("brief_issues", 0), time.time() - t0)
        except Exception as exc:
            log.error("Phase 7 FAILED: %s", exc, exc_info=True)
    else:
        log.info("Phase 7: Brief review SKIPPED")

    # Phase 8: Queue articles for decomposition (background processing)
    decompose_out = {"queued": 0}
    if llm_config and llm_config.url:
        try:
            t0 = time.time()
            log.info("Phase 8: Queue articles for decomposition (background)")
            articles = [f for f in all_figs if f.meta.get("is_image") and f.meta.get("source_id") and not f.is_edge()]
            needs_decomp = sum(1 for a in articles if not a.meta.get("decomposed"))
            log.info("  %d articles will be decomposed in background (engine handles queueing)", needs_decomp)
            decompose_out["queued"] = needs_decomp
        except Exception as exc:
            log.error("Phase 8 FAILED: %s", exc, exc_info=True)
    else:
        log.info("Phase 8: Decomposition SKIPPED (LLM not enabled)")

    total_time = time.time() - t_start
    log.info("COMPLETE — total_time=%.1fs", total_time)

    return {
        "trust_updates": len(trust_out["updates"]),
        "narratives": len(lineage_out["narratives"]),
        "lineage_edges": lineage_out["edges"],
        "summarized": summaries_out["summarized"],
        "brief_used": brief_out["used"],
        "decomposed": decompose_out["queued"],
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
        log.error("EVALUATION ERROR: %s", exc, exc_info=True)
        return {"evaluated": 0, "corrections_suggested": 0,
                "evaluation_error": str(exc)}
