"""Post-crawl pipeline: trust -> lineage -> (eval) -> (correct) -> summaries -> brief.

Runs on the crawler (which holds the model). Keeps the store coherent so the
web viewer can render everything without touching the GPU.
"""

from __future__ import annotations

import time
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
    print(f"\n{'='*60}")
    print(f"[pipeline] START — llm_enabled={llm_config.enabled if llm_config else False}")
    print(f"{'='*60}")
    t_start = time.time()

    # Phase 1: Trust
    t0 = time.time()
    print(f"\n[pipeline] Phase 1: Trust propagation")
    trust_out = trust_mod.update_trust(store)
    print(f"[pipeline]   trust_updates={len(trust_out['updates'])}  ({time.time()-t0:.1f}s)")

    # Phase 2: Lineage
    t0 = time.time()
    print(f"\n[pipeline] Phase 2: Lineage (narrative clustering)")
    lineage_out = lineage_mod.compute_lineage(store, max_stories=max_stories)
    print(f"[pipeline]   narratives={len(lineage_out['narratives'])}  edges={lineage_out['edges']}  ({time.time()-t0:.1f}s)")
    for n in lineage_out.get("narratives", [])[:5]:
        print(f"[pipeline]     - {n['narrative_id'][:8]}: {n.get('title', '')[:60]}")
        print(f"[pipeline]       sources={n.get('sources', [])}  members={len(n.get('members', []))}  frame_shift={n.get('frame_shift', False)}")

    # Phase 2.5: LLM-based clustering evaluation (ground truth)
    if llm_config and llm_config.enabled:
        t0 = time.time()
        print(f"\n[pipeline] Phase 2.5: LLM-based clustering evaluation (ground truth)")
        from . import evaluate
        # Get articles for labeling
        all_figs = store.all()
        articles = [f for f in all_figs if f.meta.get("is_image") and f.meta.get("source_id") and not f.is_edge()]
        print(f"[pipeline]   found {len(articles)} articles for labeling")
        
        if len(articles) >= 2:
            print(f"[pipeline]   starting LLM labeling...")
            client = evaluate.LLMClient(llm_config)
            # Label 20 random pairs
            labels = evaluate.label_article_pairs(articles, client, max_pairs=20)
            print(f"[pipeline]   got {len(labels)} labels")
            
            print(f"[pipeline]   clustering evaluation complete ({time.time()-t0:.1f}s)")
        else:
            print(f"[pipeline]   not enough articles for evaluation")

    # Phase 3: LLM Evaluation
    eval_out = {"evaluated": 0, "corrections_suggested": 0, "corrections_applied": 0}
    if llm_config and llm_config.enabled:
        t0 = time.time()
        print(f"\n[pipeline] Phase 3: LLM evaluation")
        eval_out = _run_evaluation(store, llm_config)
        print(f"[pipeline]   evaluated={eval_out.get('evaluated', 0)}  corrections_suggested={eval_out.get('corrections_suggested', 0)}  ({time.time()-t0:.1f}s)")

        if llm_config.auto_correct:
            from . import correct
            print(f"\n[pipeline] Phase 4: Apply corrections (threshold={llm_config.confirmation_threshold})")
            corr_out = correct.confirm_and_apply(store, llm_config.confirmation_threshold)
            eval_out["corrections_applied"] = corr_out.get("corrections_applied", 0)
            print(f"[pipeline]   corrections_applied={eval_out['corrections_applied']}")
    else:
        print(f"\n[pipeline] Phase 3-4: SKIPPED (LLM not enabled)")

    # Phase 5: Summaries
    brief_out = {"used": 0, "brief": ""}
    summaries_out = {"summarized": 0}
    if do_summaries:
        t0 = time.time()
        print(f"\n[pipeline] Phase 5: Article summaries")
        summaries_out = summarize_news.ensure_article_summaries(model, tokenizer, store)
        print(f"[pipeline]   summarized={summaries_out['summarized']}  ({time.time()-t0:.1f}s)")
    else:
        print(f"\n[pipeline] Phase 5: Summaries SKIPPED")

    # Phase 6: Brief
    if do_brief:
        t0 = time.time()
        print(f"\n[pipeline] Phase 6: World brief")
        brief_out = summarize_news.build_world_brief(model, tokenizer, store)
        print(f"[pipeline]   brief_used={brief_out['used']} articles  ({time.time()-t0:.1f}s)")
        if brief_out.get("brief"):
            print(f"[pipeline]   brief_text: {brief_out['brief'][:150]}...")
    else:
        print(f"\n[pipeline] Phase 6: Brief SKIPPED")

    # Phase 7: Brief review
    brief_eval = {}
    if llm_config and llm_config.enabled and llm_config.review_brief and brief_out.get("brief"):
        t0 = time.time()
        print(f"\n[pipeline] Phase 7: Brief review (LLM)")
        from . import evaluate
        brief_eval = evaluate.review_brief(store, brief_out["brief"],
                                            evaluate.LLMClient(llm_config), llm_config)
        print(f"[pipeline]   brief_acceptable={brief_eval.get('brief_acceptable')}  issues={brief_eval.get('brief_issues', 0)}  ({time.time()-t0:.1f}s)")
    else:
        print(f"\n[pipeline] Phase 7: Brief review SKIPPED")

    # Phase 8: Queue articles for decomposition (background processing)
    decompose_out = {"queued": 0}
    if llm_config and llm_config.url:
        t0 = time.time()
        print(f"\n[pipeline] Phase 8: Queue articles for decomposition (background)")
        
        # The decomposition engine is started by the server and will queue existing articles
        # Here we just report how many need processing
        all_figs = store.all()
        articles = [f for f in all_figs if f.meta.get("is_image") and f.meta.get("source_id") and not f.is_edge()]
        
        # Count articles that need decomposition
        needs_decomp = sum(1 for a in articles if not a.meta.get("decomposed"))
        print(f"[pipeline]   {needs_decomp} articles will be decomposed in background (engine handles queueing)")
        decompose_out["queued"] = needs_decomp
    else:
        print(f"\n[pipeline] Phase 8: Decomposition SKIPPED (LLM not enabled)")

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"[pipeline] COMPLETE — total_time={total_time:.1f}s")
    print(f"{'='*60}\n")

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
        print(f"[pipeline] EVALUATION ERROR: {exc}")
        import traceback
        traceback.print_exc()
        return {"evaluated": 0, "corrections_suggested": 0,
                "evaluation_error": str(exc)}
