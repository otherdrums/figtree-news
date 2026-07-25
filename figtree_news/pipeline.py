"""Post-crawl pipeline: decompose -> trust -> lineage -> eval -> summaries -> brief.

Runs on the crawler (which holds the model). Keeps the store coherent so the
web viewer can render everything without touching the GPU.

Phase order:
  1. Decomposition (external LLM — wait for new articles)
  2. Trust + Lineage (parallel, CPU) — lineage uses role figments from phase 1
  3. Summaries (GPU, sequential)
  4. Brief (GPU) + Eval (I/O) — parallel
  5. Corrections + Brief review
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from figtree import FigmentStore

from . import summarize_news
from . import trust as trust_mod
from . import lineage as lineage_mod
from .llm_config import LLMConfig

log = logging.getLogger(__name__)

_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pipeline")

_DECOMPOSE_POLL_INTERVAL = 2.0
_DECOMPOSE_TIMEOUT = 600.0


def run_pipeline(
    model,
    tokenizer,
    store: FigmentStore,
    do_summaries: bool = True,
    do_brief: bool = True,
    max_stories: int = 0,
    max_summaries: int = 10,
    llm_config: LLMConfig | None = None,
    decompose_engine=None,
) -> dict[str, Any]:
    log.info("START — llm_enabled=%s", llm_config.enabled if llm_config else False)
    t_start = time.time()

    all_figs = store.all()
    log.info("loaded %d figments from store", len(all_figs))

    # ── Phase 1: Decomposition (external LLM, wait for new articles) ──────
    decompose_out = {"queued": 0, "completed": 0}
    if llm_config and llm_config.url and decompose_engine:
        try:
            t0 = time.time()
            articles = [f for f in all_figs if f.meta.get("is_image") and f.meta.get("source_id") and not f.is_edge()]
            needs_decomp = [a for a in articles if not a.meta.get("decomposed")]
            decompose_out["queued"] = len(needs_decomp)
            
            if needs_decomp:
                log.info("Phase 1: Decomposition — %d articles queued, waiting...", len(needs_decomp))
                for a in needs_decomp:
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            import concurrent.futures
                            future = concurrent.futures.Future()
                            asyncio.run_coroutine_threadsafe(
                                decompose_engine.queue_article(a.figment_id), loop
                            )
                        else:
                            asyncio.run(decompose_engine.queue_article(a.figment_id))
                    except RuntimeError:
                        pass
                
                completed = _wait_for_decomposition(store, [a.figment_id for a in needs_decomp])
                decompose_out["completed"] = completed
                log.info("  decomposed=%d/%d  (%.1fs)", completed, len(needs_decomp), time.time() - t0)
            else:
                log.info("Phase 1: Decomposition — all articles already decomposed")
        except Exception as exc:
            log.error("Phase 1 FAILED: %s", exc, exc_info=True)
    else:
        log.info("Phase 1: Decomposition SKIPPED (LLM not enabled)")

    # ── Phase 2: Trust + Lineage (parallel, CPU) ──────────────────────────
    all_figs = store.all()
    trust_future = _POOL.submit(_phase_trust, store, all_figs)
    lineage_future = _POOL.submit(_phase_lineage, store, max_stories)

    llm_label_future = None
    if llm_config and llm_config.enabled:
        llm_label_future = _POOL.submit(
            _phase_llm_labeling, store, all_figs, llm_config
        )

    trust_out = trust_future.result()
    lineage_out = lineage_future.result()
    if llm_label_future:
        llm_label_future.result()

    # ── VRAM check before GPU phases ──────────────────────────────────────
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        free_mb = free // (1024 * 1024)
        log.info("VRAM: %dMB free / %dMB total", free_mb, total // (1024 * 1024))
        if free_mb < 200:
            log.warning("Low VRAM (%dMB free) — skipping summaries and brief", free_mb)
            do_summaries = False
            do_brief = False

    # ── Phase 3: Summaries (GPU, sequential) ──────────────────────────────
    summaries_out = {"summarized": 0}
    if do_summaries:
        try:
            t0 = time.time()
            log.info("Phase 3: Article summaries (max=%d)", max_summaries)
            summaries_out = summarize_news.ensure_article_summaries(
                model, tokenizer, store, all_figs=all_figs, limit=max_summaries
            )
            log.info("  summarized=%d  (%.1fs)", summaries_out["summarized"], time.time() - t0)
        except Exception as exc:
            log.error("Phase 3 FAILED: %s", exc, exc_info=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        log.info("Phase 3: Summaries SKIPPED")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Phase 4: Brief + Eval (parallel) ──────────────────────────────────
    brief_future = None
    if do_brief:
        brief_future = _POOL.submit(_phase_brief, model, tokenizer, store, all_figs)

    eval_future = None
    if llm_config and llm_config.enabled:
        eval_future = _POOL.submit(_phase_eval, store, llm_config)

    brief_out = {"used": 0, "brief": ""}
    if brief_future:
        brief_out = brief_future.result()

    eval_out = {"evaluated": 0, "corrections_suggested": 0, "corrections_applied": 0}
    if eval_future:
        eval_out = eval_future.result()
        if llm_config and llm_config.auto_correct:
            try:
                from . import correct
                log.info("Phase 5: Apply corrections (threshold=%s)", llm_config.confirmation_threshold)
                corr_out = correct.confirm_and_apply(store, llm_config.confirmation_threshold)
                eval_out["corrections_applied"] = corr_out.get("corrections_applied", 0)
                log.info("  corrections_applied=%d", eval_out["corrections_applied"])
            except Exception as exc:
                log.error("Phase 5 FAILED: %s", exc, exc_info=True)

    # ── Phase 6: Brief review ─────────────────────────────────────────────
    brief_eval = {}
    if llm_config and llm_config.enabled and llm_config.review_brief and brief_out.get("brief"):
        try:
            t0 = time.time()
            log.info("Phase 6: Brief review (LLM)")
            from . import evaluate
            brief_eval = evaluate.review_brief(
                store, brief_out["brief"],
                evaluate.LLMClient(llm_config), llm_config
            )
            log.info("  brief_acceptable=%s  issues=%d  (%.1fs)",
                     brief_eval.get("brief_acceptable"), brief_eval.get("brief_issues", 0),
                     time.time() - t0)
        except Exception as exc:
            log.error("Phase 6 FAILED: %s", exc, exc_info=True)
    else:
        log.info("Phase 6: Brief review SKIPPED")

    total_time = time.time() - t_start
    log.info("COMPLETE — total_time=%.1fs", total_time)

    return {
        "trust_updates": len(trust_out.get("updates", [])),
        "narratives": len(lineage_out.get("narratives", [])),
        "lineage_edges": lineage_out.get("edges", 0),
        "summarized": summaries_out.get("summarized", 0),
        "brief_used": brief_out.get("used", 0),
        "decomposed": decompose_out.get("completed", 0),
        **eval_out,
        **brief_eval,
    }


def _wait_for_decomposition(store: FigmentStore, article_ids: list[str]) -> int:
    """Poll store until articles have role_figments (decomposed=True).

    Returns count of successfully decomposed articles.
    """
    completed = 0
    remaining = set(article_ids)
    deadline = time.time() + _DECOMPOSE_TIMEOUT
    while remaining and time.time() < deadline:
        newly_done = set()
        for aid in remaining:
            fig = store.get(aid)
            if fig and fig.meta.get("decomposed"):
                newly_done.add(aid)
        completed += len(newly_done)
        remaining -= newly_done
        if remaining:
            time.sleep(_DECOMPOSE_POLL_INTERVAL)
    if remaining:
        log.warning("Decomposition timeout — %d articles still pending", len(remaining))
    return completed


def _phase_trust(store: FigmentStore, all_figs: list) -> dict[str, Any]:
    try:
        t0 = time.time()
        log.info("Phase 2a: Trust propagation")
        out = trust_mod.update_trust(store, all_figs=all_figs)
        log.info("  trust_updates=%d  (%.1fs)", len(out.get("updates", [])), time.time() - t0)
        return out
    except Exception as exc:
        log.error("Phase 2a FAILED: %s", exc, exc_info=True)
        return {"analysis": {}, "updates": []}


def _phase_lineage(store: FigmentStore, max_stories: int) -> dict[str, Any]:
    try:
        t0 = time.time()
        log.info("Phase 2b: Lineage (role figment clustering)")
        out = lineage_mod.compute_lineage(store, max_stories=max_stories)
        log.info("  narratives=%d  edges=%d  (%.1fs)",
                 len(out.get("narratives", [])), out.get("edges", 0), time.time() - t0)
        for n in out.get("narratives", [])[:5]:
            log.info("    - %s: %s", n["narrative_id"][:8], n.get("title", "")[:60])
            log.info("      sources=%s  members=%d  frame_shift=%s",
                     n.get("sources", []), len(n.get("members", [])), n.get("frame_shift", False))
        all_figs = store.all()
        role_linked = lineage_mod.assign_roles_to_narratives(store, all_figs=all_figs)
        log.info("  linked %d role figments to narratives", role_linked)
        return out
    except Exception as exc:
        log.error("Phase 2b FAILED: %s", exc, exc_info=True)
        return {"narratives": [], "edges": 0}


def _phase_llm_labeling(store: FigmentStore, all_figs: list, llm_config: LLMConfig) -> None:
    try:
        t0 = time.time()
        log.info("Phase 2c: LLM-based clustering evaluation")
        from . import evaluate
        articles = [f for f in all_figs if f.meta.get("is_image") and f.meta.get("source_id") and not f.is_edge()]
        if len(articles) >= 2:
            client = evaluate.LLMClient(llm_config)
            labels = evaluate.label_article_pairs(articles, client, max_pairs=20)
            log.info("  got %d labels (%.1fs)", len(labels), time.time() - t0)
        else:
            log.info("  not enough articles for evaluation")
    except Exception as exc:
        log.error("Phase 2c FAILED: %s", exc, exc_info=True)


def _phase_brief(model, tokenizer, store: FigmentStore, all_figs: list) -> dict[str, Any]:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        t0 = time.time()
        log.info("Phase 4a: World brief")
        out = summarize_news.build_world_brief(model, tokenizer, store, all_figs=all_figs)
        log.info("  brief_used=%d articles  (%.1fs)", out.get("used", 0), time.time() - t0)
        if out.get("brief"):
            log.info("  brief_text: %s...", out["brief"][:150])
        return out
    except Exception as exc:
        log.error("Phase 4a FAILED: %s", exc, exc_info=True)
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"used": 0, "brief": ""}


def _phase_eval(store: FigmentStore, llm_config: LLMConfig) -> dict[str, Any]:
    try:
        t0 = time.time()
        log.info("Phase 4b: LLM evaluation")
        from . import evaluate
        client = evaluate.LLMClient(llm_config)
        result = evaluate.evaluate_narratives(store, client, llm_config)
        log.info("  evaluated=%d  corrections_suggested=%d  (%.1fs)",
                 result.get("evaluated", 0), result.get("corrections_suggested", 0),
                 time.time() - t0)
        return result
    except Exception as exc:
        log.error("Phase 4b FAILED: %s", exc, exc_info=True)
        return {"evaluated": 0, "corrections_suggested": 0, "evaluation_error": str(exc)}
