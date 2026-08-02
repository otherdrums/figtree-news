"""Post-crawl pipeline: trust -> lineage -> summaries -> brief.

Runs on the crawler (which holds the model). Keeps the store coherent so the
web viewer can render everything without touching the GPU.

Phase order:
  1. Trust (CPU) + Lineage (CPU, parallel) — lineage clusters articles by
     shared role figments into narratives. Lineage rebuild is atomic: new
     narrative figments are persisted before stale ones are deleted, so a
     crash can never leave the store with zero narratives.
  2. Role->narrative linking on the FRESH lineage output (single snapshot —
     no stale store re-reads).
  3. Summaries (GPU, sequential, max 10)
  4. World brief (GPU)
  5. Store compaction

No external-LLM phases: role figments arrive at ingest (single-pass decode),
trust/lineage are CPU, summaries/brief use the local model.
"""

from __future__ import annotations

import ctypes
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from figtree import FigmentStore

from . import summarize_news
from . import trust as trust_mod
from . import lineage as lineage_mod

log = logging.getLogger(__name__)


def _force_free():
    try:
        import gc
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect(2)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _log_mem(tag: str = ""):
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    log.info("mem %s: RSS=%dMB", tag, rss_kb // 1024)
                    return
    except Exception:
        pass


_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pipeline")


def run_pipeline(
    model,
    tokenizer,
    store: FigmentStore,
    do_summaries: bool = True,
    do_brief: bool = True,
    max_stories: int = 0,
    max_summaries: int = 10,
    post_lineage_callback=None,
) -> dict[str, Any]:
    t_start = time.time()

    # ── Phase 1: Trust + Lineage (parallel, CPU, ONE store snapshot) ──────
    all_figs = store.all()
    log.info("loaded %d figments from store", len(all_figs))

    trust_future = _POOL.submit(_phase_trust, store, all_figs)
    lineage_future = _POOL.submit(_phase_lineage, store, max_stories, all_figs)

    trust_out = trust_future.result()
    lineage_out = lineage_future.result()
    if post_lineage_callback is not None:
        try:
            post_lineage_callback(store, trust_out, lineage_out)
        except Exception as exc:
            log.warning("post_lineage_callback failed: %s", exc, exc_info=True)

    # ── Phase 2: Link roles to the FRESH narratives (stale-snapshot fix) ──
    try:
        role_linked = lineage_mod.assign_roles_to_narratives(
            store, all_figs=all_figs, narratives=lineage_out.get("narratives", [])
        )
        log.info("  linked %d role figments to narratives", role_linked)
    except Exception as exc:
        log.error("role->narrative linking FAILED: %s", exc, exc_info=True)

    # ── VRAM check before GPU phases ──────────────────────────────────────
    try:
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
    except Exception:
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
    else:
        log.info("Phase 3: Summaries SKIPPED")

    # ── Phase 4: World brief (GPU) ────────────────────────────────────────
    brief_out = {"used": 0, "brief": ""}
    if do_brief:
        try:
            t0 = time.time()
            log.info("Phase 4: World brief")
            brief_out = summarize_news.build_world_brief(
                model, tokenizer, store, all_figs=all_figs, top_n=8
            )
            log.info("  brief_used=%d articles  (%.1fs)", brief_out.get("used", 0), time.time() - t0)
        except Exception as exc:
            log.error("Phase 4 FAILED: %s", exc, exc_info=True)
    else:
        log.info("Phase 4: Brief SKIPPED")

    _force_free()
    _log_mem("pipeline end")

    # Compact LanceDB fragments + prune old versions to keep query latency
    # bounded (each upsert/merge creates a version; without periodic cleanup
    # the store fragments and every store.all() becomes slower).
    try:
        from datetime import timedelta as _td
        store.table.optimize(cleanup_older_than=_td(0), delete_unverified=True)
        log.info("  store compacted")
    except Exception as exc:
        log.warning("  store compaction failed: %s", exc)

    total_time = time.time() - t_start
    log.info("COMPLETE — total_time=%.1fs", total_time)

    return {
        "trust_updates": len(trust_out.get("updates", [])),
        "narratives": len(lineage_out.get("narratives", [])),
        "lineage_edges": lineage_out.get("edges", 0),
        "summarized": summaries_out.get("summarized", 0),
        "brief_used": brief_out.get("used", 0),
    }


def _phase_trust(store: FigmentStore, all_figs: list) -> dict[str, Any]:
    try:
        t0 = time.time()
        log.info("Phase 1a: Trust propagation")
        out = trust_mod.update_trust(store, all_figs=all_figs)
        log.info("  trust_updates=%d  (%.1fs)", len(out.get("updates", [])), time.time() - t0)
        return out
    except Exception as exc:
        log.error("Phase 1a FAILED: %s", exc, exc_info=True)
        return {"analysis": {}, "updates": []}


def _phase_lineage(store: FigmentStore, max_stories: int, all_figs: list) -> dict[str, Any]:
    try:
        t0 = time.time()
        log.info("Phase 1b: Lineage (role figment clustering)")
        out = lineage_mod.compute_lineage(store, max_stories=max_stories, all_figs=all_figs)
        log.info("  narratives=%d  edges=%d  (%.1fs)",
                 len(out.get("narratives", [])), out.get("edges", 0), time.time() - t0)
        for n in out.get("narratives", [])[:5]:
            log.info("    - %s: %s", n["narrative_id"][:8], n.get("title", "")[:60])
            log.info("      sources=%s  members=%d  frame_shift=%s",
                     n.get("sources", []), len(n.get("members", [])), n.get("frame_shift", False))
        return out
    except Exception as exc:
        log.error("Phase 1b FAILED: %s", exc, exc_info=True)
        return {"narratives": [], "edges": 0}
