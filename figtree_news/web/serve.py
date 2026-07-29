"""FastAPI web app: the local, source-aware newspaper.

Reads the shared LanceDB store and renders: a front page (world brief, top
stories, source-trust board, agenda map), per-article, per-source, and
per-narrative pages (each linking back to the original outlet), and a lineage
view (who broke it first / who echoed whom). JSON API mirrors the pages, plus
on-demand generation endpoints that lazily load the model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from figtree import FigmentStore, connect, load_model, FigmentGenerator
from figtree.kv_cache_manager import KVCacheManager

from .. import summarize_news
from ..config import SourceRegistry
from ..crawler import Crawler
from ..lineage import get_narratives, get_derivatives, source_agenda, _normalize_source
from ..llm_config import LLMConfig
from ..pipeline import run_pipeline
from ..query import query as run_query
from ..search_index import get_index

warnings.filterwarnings("ignore", message=".*_check_is_size.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="bitsandbytes")

# Configure logging AFTER torch/imports; levels are applied to existing loggers.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
# Reduce noise from third-party libraries
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

def _e(s: str) -> str:
    """Escape HTML special characters."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

# Date range helpers

def _parse_range_date(range_str: str, article_date: str) -> bool:
    """Return True if article_date falls within range_str."""
    from datetime import datetime, timezone, timedelta

    try:
        article_dt = datetime.fromisoformat(article_date.replace("Z", "+00:00"))
        if article_dt.tzinfo is None:
            article_dt = article_dt.replace(tzinfo=timezone.utc)
    except Exception:
        return True

    now = datetime.now(timezone.utc)
    if range_str == "today":
        since = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return article_dt >= since
    elif range_str == "yesterday":
        since = now - timedelta(days=1)
        since = since.replace(hour=0, minute=0, second=0, microsecond=0)
        until = since + timedelta(days=1)
        return since <= article_dt < until
    elif range_str == "last_week":
        since = now - timedelta(days=7)
        return article_dt >= since
    elif range_str == "last_month":
        since = now - timedelta(days=30)
        return article_dt >= since
    elif range_str == "last_year":
        since = now - timedelta(days=365)
        return article_dt >= since
    return True

# Let uvicorn/asyncio handle SIGINT natively — the crawl's stop_requested
# flag provides graceful shutdown, and sys.exit(0) from a signal handler
# does not reliably kill a running asyncio event loop.

_HERE = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(_HERE, "templates")
STATIC_DIR = os.path.join(_HERE, "static")

_gen_cache: dict[str, Any] = {}
_model_cache: dict[str, Any] = {}
_model_load_lock: asyncio.Lock = asyncio.Lock()
# Dedicated executor for pipeline/crawl work so it does not contend with the
# background decompose workers for the default executor's limited workers.
_pipeline_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pipeline")
_crawl_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="crawl")
_data_cache: dict[str, Any] = {"data": None, "ts": 0.0}
_decompose_engine: Any = None
_cogitate_engine: Any = None
_pipeline_task: asyncio.Task | None = None
_first_crawl_done: asyncio.Event = asyncio.Event()
_pipeline_running = False
_crawl_state: dict[str, Any] = {
    "running": False,
    "task": None,
    "continuous": False,
    "stop_requested": False,
    "current_step": "idle",
    "progress": 0,
    "total": 0,
    "message": "",
    "stats": {},
    "start_time": None,
    "feeds": [],
    "seeds": [],
    "max_articles": 15,
    "interval": 300,
    "compute_kv": False,
    "summarize": True,
}
_ws_connections: list[WebSocket] = []
_crawl_mode: str = "forward"
_consecutive_empty_ticks: int = 0
_backward_time_range = "last_month"

_TIME_RANGE_PROGRESSION = ["day", "last_week", "last_month", "last_year", "all"]

# Map UI-style ranges to SearXNG's supported time_range parameter values.
_UI_TO_SEARXNG_TIME_RANGE = {
    "day": "day",
    "last_week": "week",
    "last_month": "month",
    "last_year": "year",
    "all": "",
}


def _normalize_searx_time_range(r: str) -> str:
    """Return the SearXNG time_range value for a UI/internal range name."""
    return _UI_TO_SEARXNG_TIME_RANGE.get(r, r)


def _next_time_range(current: str) -> str:
    try:
        idx = _TIME_RANGE_PROGRESSION.index(current)
        return _TIME_RANGE_PROGRESSION[min(idx + 1, len(_TIME_RANGE_PROGRESSION) - 1)]
    except ValueError:
        return "last_month"


async def _drain_decompose(crawler):
    """Drain article IDs queued for decomposition during a sync thread call and submit them."""
    if not _decompose_engine:
        return
    for fid in crawler.drain_pending_decompose():
        asyncio.create_task(_decompose_engine.queue_article(fid))


def _get_generator():
    if "gen" not in _gen_cache:
        model, tokenizer = load_model("unsloth/Qwen3-4B-bnb-4bit")
        _gen_cache["gen"] = FigmentGenerator(model, tokenizer)
    return _gen_cache["gen"]


async def _load_model_cached(model_id: str = "unsloth/Qwen3-4B-bnb-4bit") -> tuple[Any, Any]:
    if model_id in _model_cache:
        return _model_cache[model_id]
    async with _model_load_lock:
        if model_id in _model_cache:
            return _model_cache[model_id]
        model, tokenizer = await asyncio.to_thread(load_model, model_id)
        _model_cache[model_id] = (model, tokenizer)
        if "gen" not in _gen_cache:
            _gen_cache["gen"] = FigmentGenerator(model, tokenizer)
        return model, tokenizer


async def _run_pipeline_loop(
    store: FigmentStore,
    sources: str,
    interval: int = 300,
):
    """Background task that keeps the UI data cache fresh by running the pipeline.

    Waits for the first crawl tick to finish before running (pipeline needs
    articles to decompose). Runs full pipeline (decompose, lineage, summaries,
    brief, eval) every ``interval`` seconds.
    """
    # Wait for the first crawl tick to finish so articles are ingested
    # before the pipeline tries to decompose and cluster them.
    try:
        await asyncio.wait_for(_first_crawl_done.wait(), timeout=3600)
    except asyncio.TimeoutError:
        print("[pipeline-loop] first crawl not done yet, proceeding (waited 1h)")

    while True:
        try:
            model, tokenizer = await _load_model_cached()
            llm_config = LLMConfig.from_sources_json(sources)
            print("[pipeline-loop] running pipeline...")

            def _on_lineage(store, _trust_out, lineage_out):
                _data_cache["data"] = None
                _warm_cache(store)
                print(
                    f"[pipeline-lineage] {len(lineage_out.get('narratives', []))} narratives ready"
                )

            # Bind model and pause background workers so the pipeline
            # has exclusive GPU access (same pattern as the crawl tick).
            if _decompose_engine:
                _decompose_engine.model = model
                _decompose_engine.tokenizer = tokenizer
                _decompose_engine.stop()
                print("[pipeline-loop] paused background decompose workers")
            if _cogitate_engine:
                _cogitate_engine.stop()
                print("[pipeline-loop] paused background cogitate engine")

            _pipeline_running = True
            loop = asyncio.get_running_loop()
            stats = await loop.run_in_executor(
                _pipeline_executor,
                run_pipeline, model, tokenizer, store,
                True,  # do_summaries
                True,  # do_brief
                0,
                10,
                llm_config,
                _decompose_engine,
                _on_lineage,
            )

            # Don't resume background workers — pipeline handles decomposition
            if _cogitate_engine:
                _cogitate_engine.start()
                print("[pipeline-loop] resumed background cogitate engine")

            _data_cache["data"] = None
            _warm_cache(store)
            print(
                f"[pipeline-loop] done: {stats.get('narratives', 0)} narratives, "
                f"{stats.get('brief_used', 0)} brief articles, "
                f"{stats.get('summarized', 0)} summaries"
            )
            _pipeline_running = False
        except asyncio.CancelledError:
            break
        except Exception as exc:
            print(f"[pipeline-loop] error: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            _pipeline_running = False
            if _decompose_engine:
                try:
                    _decompose_engine.start()
                except Exception:
                    pass
            if _cogitate_engine:
                try:
                    _cogitate_engine.start()
                except Exception:
                    pass
        # Poll frequently until narratives appear, then relax.
        current_stats = _data_cache.get("data")
        narr_val = current_stats.get("narratives", 0) if current_stats else 0
        narratives_count = len(narr_val) if isinstance(narr_val, list) else int(narr_val)
        await asyncio.sleep(30 if narratives_count == 0 else interval)


def _build(store: FigmentStore, *, force: bool = False) -> dict[str, Any]:
    if not force and _data_cache["data"] is not None:
        return _data_cache["data"]
    all_figs = store.all()
    by_id = {f.figment_id: f for f in all_figs}
    articles = [
        f
        for f in all_figs
        if f.kind == "article" and f.meta.get("source_id")
    ]
    roles = [f for f in all_figs if f.kind == "role"]
    narratives = get_narratives(store, all_figs=all_figs)
    derivatives = get_derivatives(store, all_figs=all_figs)
    agenda = source_agenda(store, all_figs=all_figs)
    brief = summarize_news.get_world_brief(store, all_figs=all_figs)
    result = {
        "articles": articles,
        "by_id": by_id,
        "roles": roles,
        "narratives": narratives,
        "derivatives": derivatives,
        "agenda": agenda,
        "brief": brief,
    }
    _data_cache["data"] = result
    _data_cache["ts"] = time.time()
    return result


def _warm_cache(store: FigmentStore):
    """Pre-load the data cache at startup so first request is instant."""
    try:
        _build(store, force=True)
    except Exception as exc:
        print(f"[warm_cache] failed: {exc}")


def _get_stats(store: FigmentStore) -> dict[str, Any]:
    d = _build(store)
    return {
        "articles": len(d["articles"]),
        "narratives": len(d["narratives"]),
        "derivatives": len(d["derivatives"]),
        "sources": len(d["agenda"]),
        "has_brief": bool(d["brief"]),
        "last_updated": max((a.meta.get("first_seen", "") for a in d["articles"]), default=""),
    }


async def _broadcast(msg: dict[str, Any]):
    """Broadcast message to all connected WebSocket clients."""
    # Strip non-JSON fields from crawl_status messages
    if msg.get("type") == "crawl_status" and "data" in msg:
        msg["data"] = {k: v for k, v in msg["data"].items() if k != "task"}
    dead = []
    for ws in _ws_connections:
        try:
            await ws.send_text(json.dumps(msg))
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_connections.remove(ws)


async def _run_crawl_tick(
    store: FigmentStore,
    sources_path: str,
    feeds: dict[str, str],
    seeds: list[str],
    max_articles: int,
    summarize: bool,
    compute_kv: bool,
    model_id: str,
    max_stories: int = 0,
    since: str = "",
    before: str = "",
    llm_enabled: bool = False,
):
    """Single crawl tick. Feeds+seeds run in parallel; SearXNG queries run in parallel."""
    global _crawl_state, _consecutive_empty_ticks, _crawl_mode, _backward_time_range
    _crawl_state["running"] = True
    _crawl_state["start_time"] = time.time()
    _crawl_state["current_step"] = "loading_model"
    _crawl_state["message"] = "Loading model (~5 min on first use)..."
    _crawl_state["progress"] = 0
    _crawl_state["total"] = 1
    await _broadcast({"type": "crawl_status", "data": _crawl_state})

    try:
        already_cached = model_id in _model_cache
        model, tokenizer = await _load_model_cached(model_id)
        print(f"[crawl] model {'reused' if already_cached else 'loaded'} ({model_id.rsplit('/',1)[-1]})")

        # Bind the local model to the decomposition engine so it can decompose
        # articles using the local model rather than the external LLM.
        if _decompose_engine:
            _decompose_engine.model = model
            _decompose_engine.tokenizer = tokenizer

        registry = SourceRegistry.load(sources_path)

        # Pause background engines so the crawl has exclusive GPU access
        if _decompose_engine:
            _decompose_engine.stop()
            print("[crawl] paused background decompose workers")
        if _cogitate_engine:
            _cogitate_engine.stop()
            print("[crawl] paused background cogitate engine")

        kv_manager = None
        if compute_kv:
            kv_manager = KVCacheManager(model, tokenizer, kv_root="./figtree_kv", mode="eager")
            print("[crawl] KV cache manager created (mode=eager)")

        crawler = Crawler(
            model, tokenizer, store, registry,
            seen_path="./seen_urls.json",
            compute_kv=compute_kv, summarize_images=summarize,
            kv_manager=kv_manager,
            decompose_engine=_decompose_engine,
        )

        # ── Phase 1: Parallel feeds + seeds ────────────────────────────────
        _crawl_state["current_step"] = "crawling"
        _crawl_state["message"] = f"Crawling {len(feeds)} feeds + {len(seeds)} seeds in parallel..."
        _crawl_state["feeds"] = list(feeds.keys())
        per_feed = max(1, max_articles // max(len(feeds), 1))
        _crawl_state["total"] = len(feeds) + (1 if seeds else 0)
        await _broadcast({"type": "crawl_status", "data": _crawl_state})

        async def _crawl_one_feed(sid: str, uri: str) -> tuple[str, int]:
            loop = asyncio.get_running_loop()
            added = await loop.run_in_executor(
                _crawl_executor,
                crawler.crawl_feed, sid, uri, per_feed, since, before,
            )
            return sid, added

        async def _crawl_seeds_task() -> int:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(_crawl_executor, crawler.crawl_seeds, seeds)

        ASYNC_TIMEOUT = 300  # seconds per feed/seed task (matches tick interval)
        feed_tasks = [
            asyncio.wait_for(_crawl_one_feed(sid, uri), timeout=ASYNC_TIMEOUT)
            for sid, uri in feeds.items()
        ]
        seed_task = asyncio.wait_for(_crawl_seeds_task(), timeout=ASYNC_TIMEOUT) if seeds else None
        all_tasks = feed_tasks + ([seed_task] if seed_task else [])

        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        feeds_added = 0
        seeds_added = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                sid = list(feeds.keys())[i] if i < len(feeds) else "seeds"
                print(f"[crawl] {sid} timed out or failed: {result}")
                continue
            if i < len(feed_tasks):
                sid, added = result
                feeds_added += added
                if added > 0:
                    print(f"[crawl] {sid}: +{added} articles")
            else:
                seeds_added = result

        await _drain_decompose(crawler)

        if feeds_added > 0 or seeds_added > 0:
            _data_cache["data"] = None
            _warm_cache(store)
            await _broadcast({"type": "content_update", "data": {
                "feeds_added": feeds_added, "seeds_added": seeds_added,
                "total_articles": get_index().article_count(),
            }})

        if _crawl_state.get("stop_requested"):
            raise asyncio.CancelledError("Stop requested")

        # ── Phase 2: Parallel SearXNG queries ──────────────────────────────
        search_added = 0
        cfg = registry.searxng
        if cfg and cfg.enabled:
            _crawl_state["current_step"] = "searching"

            from figtree_news.crawler import _extract_keywords
            with crawler._ingest_lock:
                new_articles_snapshot = list(crawler._new_articles)
            queries = _extract_keywords(new_articles_snapshot, top_n=10)
            if not queries:
                queries = [
                    "breaking news",
                    "world news today",
                    "technology news",
                    "business markets",
                    "politics government",
                ]

            srch_time_range = cfg.time_range
            if _crawl_mode == "backward":
                srch_time_range = _backward_time_range

            _crawl_state["message"] = f"Searching {len(queries)} queries sequentially ({srch_time_range})..."
            _crawl_state["total"] = len(queries)
            _crawl_state["progress"] = 0
            await _broadcast({"type": "crawl_status", "data": _crawl_state})

            for qi, q in enumerate(queries):
                if _crawl_state.get("stop_requested"):
                    break
                _crawl_state["progress"] = qi
                _crawl_state["message"] = f"Searching ({srch_time_range}): {q}"
                await _broadcast({"type": "crawl_status", "data": _crawl_state})

                # VRAM guard: stop SearXNG queries if GPU is running low
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    free, total = torch.cuda.mem_get_info()
                    free_mb = free // (1024 * 1024)
                    if free_mb < 500:
                        print(f"[crawl] VRAM low ({free_mb}MB free) — stopping SearXNG queries to avoid OOM")
                        _crawl_state["message"] = f"VRAM low ({free_mb}MB) — SearXNG queries stopped"
                        await _broadcast({"type": "crawl_status", "data": _crawl_state})
                        break

                try:
                    loop = asyncio.get_running_loop()
                    got = await loop.run_in_executor(
                        _crawl_executor,
                        crawler.search_searxng, q,
                        cfg.categories,
                        _normalize_searx_time_range(srch_time_range),
                        cfg.max_results, cfg.pages,
                    )
                    search_added += got
                except Exception as exc:
                    print(f"[crawl] search '{q}' failed: {exc}")
                # Brief pause between queries to let CUDA reclaim fragmented memory
                await asyncio.sleep(1)
                await _drain_decompose(crawler)
            print(f"[crawl] SearXNG: +{search_added} articles across {len(queries)} queries (mode: {_crawl_mode})")

        total_new = feeds_added + seeds_added + search_added

        # ── Backward/forward mode auto-switch ──────────────────────────────
        if total_new > 0:
            _consecutive_empty_ticks = 0
            if _crawl_mode == "backward":
                _crawl_mode = "forward"
                _backward_time_range = "last_month"
                print(f"[crawl] mode → forward (found {total_new} new articles)")
        else:
            _consecutive_empty_ticks += 1
            if _crawl_mode == "forward" and _consecutive_empty_ticks >= 2:
                _crawl_mode = "backward"
                _backward_time_range = "last_month"
                print("[crawl] no new articles → mode → backward")
            elif _crawl_mode == "backward" and total_new == 0:
                _backward_time_range = _next_time_range(_backward_time_range)
                if _backward_time_range == "all":
                    _backward_time_range = "last_month"
                    _consecutive_empty_ticks = 0
                    print("[crawl] backward range exhausted, resetting to forward")
                else:
                    print(f"[crawl] backward range → {_backward_time_range}")

        _crawl_state["mode"] = _crawl_mode
        _crawl_state["consecutive_empty"] = _consecutive_empty_ticks

        if _crawl_state.get("stop_requested"):
            raise asyncio.CancelledError("Stop requested")

        # Phase 3: Pipeline is handled by the background pipeline loop
        # so the crawl tick returns quickly.  Just drain remaining decompose
        # entries, warm the cache, and signal the pipeline loop.
        _data_cache["data"] = None
        _warm_cache(store)
        n_narr = len(_build(store)["narratives"])
        _crawl_state["stats"] = {
            "feeds_added": feeds_added,
            "seeds_added": seeds_added,
            "search_added": search_added,
            "narratives": n_narr,
        }
        _crawl_state["current_step"] = "done"
        _crawl_state["message"] = f"Done: {feeds_added + seeds_added + search_added} new articles, {n_narr} narratives"
        _crawl_state["running"] = False
        print(f"[crawl] tick complete — {feeds_added + seeds_added + search_added} new articles, {n_narr} narratives")
        await _broadcast({"type": "crawl_status", "data": _crawl_state})
        _first_crawl_done.set()

        if not _pipeline_running:
            # Background workers would decompose articles WITHOUT cross-article
            # dedup (each worker has its own local `created` dict). Since the
            # pipeline runs Phase 1 immediately after the crawl, skip workers.
            print("[crawl] pipeline handles decomposition — not starting workers")
        else:
            print("[crawl] pipeline still running — will not resume workers yet")
    except asyncio.CancelledError:
        _crawl_state["running"] = False
        _crawl_state["continuous"] = False
        _crawl_state["current_step"] = "idle"
        _crawl_state["message"] = "Crawl stopped"
        print("[crawl] stopped by user")
        await _broadcast({"type": "crawl_status", "data": _crawl_state})
    except Exception as exc:
        _crawl_state["running"] = False
        _crawl_state["current_step"] = "error"
        _crawl_state["message"] = f"Error: {exc}"
        print(f"[crawl] ERROR: {exc}")
        await _broadcast({"type": "crawl_status", "data": _crawl_state})
        raise
    finally:
        # Ensure background workers are always restarted
        if _decompose_engine:
            try:
                _decompose_engine.start()
            except Exception:
                pass
        if _cogitate_engine:
            try:
                _cogitate_engine.start()
            except Exception:
                pass


async def _run_continuous_crawl(
    store: FigmentStore,
    sources_path: str,
    feeds: dict[str, str],
    seeds: list[str],
    max_articles: int,
    summarize: bool,
    compute_kv: bool,
    model_id: str,
    interval: int,
    max_stories: int = 0,
    since: str = "",
    before: str = "",
    llm_enabled: bool = False,
):
    """Loop crawl ticks until stop_requested."""
    global _crawl_state
    _crawl_state["continuous"] = True
    _crawl_state["stop_requested"] = False
    tick_num = 0
    print(f"[crawl] continuous mode started (interval={interval}s, feeds={len(feeds)})")

    while not _crawl_state.get("stop_requested"):
        if _crawl_state.get("stop_requested"):
            break
        tick_num += 1
        print(f"[crawl] tick #{tick_num} starting")
        try:
            await _run_crawl_tick(
                store, sources_path, feeds, seeds, max_articles, summarize, compute_kv, model_id,
                max_stories=max_stories, since=since, before=before, llm_enabled=llm_enabled,
            )
        except asyncio.CancelledError:
            break
        except Exception as exc:
            _crawl_state["message"] = f"Tick failed: {exc}; retrying in {interval}s"
            await _broadcast({"type": "crawl_status", "data": _crawl_state})

        if _crawl_state.get("stop_requested"):
            break

        if interval > 0:
            _crawl_state["current_step"] = "sleeping"
            _crawl_state["message"] = f"Sleeping {interval}s until next tick..."
            _crawl_state["running"] = False
            await _broadcast({"type": "crawl_status", "data": _crawl_state})

            # Sleep in small chunks so we can respond to stop quickly
            for _ in range(interval):
                if _crawl_state.get("stop_requested"):
                    break
                await asyncio.sleep(1)
        else:
            _crawl_state["current_step"] = "next_tick"
            _crawl_state["message"] = "Starting next tick immediately..."
            _crawl_state["running"] = False
            await _broadcast({"type": "crawl_status", "data": _crawl_state})
            # Yield control so the event loop can process cancellation
            await asyncio.sleep(0)

    _crawl_state["continuous"] = False
    _crawl_state["running"] = False
    _crawl_state["current_step"] = "idle"
    _crawl_state["message"] = "Continuous crawl stopped"
    print(f"[crawl] stopped after {tick_num} ticks")
    await _broadcast({"type": "crawl_status", "data": _crawl_state})


def create_app(db: str = "./news.lance", sources: str = "./sources.json") -> FastAPI:
    app = FastAPI(title="figtree-news", description="Source-aware web newspaper")
    templates = Jinja2Templates(directory=TEMPLATES_DIR)
    registry = SourceRegistry.load(sources)
    store: FigmentStore = connect(db)
    search_idx = get_index(db.replace(".lance", "_fts.db"))
    source_logos = {s.source_id: s.logo_url for s in registry.all() if s.logo_url}

    # Set boundary data log path alongside the store database
    boundary_log_path = db.replace(".lance", "_boundary_data.jsonl")
    from ..evaluate import set_boundary_log_path as _set_bp
    _set_bp(boundary_log_path)

    app.state.store = store
    app.state.registry = registry
    
    # Initialize background engines
    llm_config = LLMConfig.from_sources_json(sources)
    global _decompose_engine, _cogitate_engine
    
    if llm_config.url and llm_config.enabled:
        from ..decompose import DecompositionEngine, set_llm_client
        from ..cogitate import CogitationEngine
        from ..evaluate import LLMClient
        
        llm_client = LLMClient(llm_config)
        set_llm_client(llm_client)
        
        _decompose_engine = DecompositionEngine(llm_config, store)
        _cogitate_engine = CogitationEngine(llm_config, store, interval_hours=0.5)
        
        app.state.decompose_engine = _decompose_engine
        app.state.cogitate_engine = _cogitate_engine

    if os.path.isdir(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ---- Startup/Shutdown Events ----------------------------------------- #
    @app.on_event("startup")
    async def startup_event():
        global _decompose_engine, _cogitate_engine, _pipeline_task
        # Don't start background decompose workers yet — the first crawl
        # tick stops them anyway, and by then zombie threads may already hold
        # model_lock.  Workers are started after the first tick completes.
        if _cogitate_engine:
            _cogitate_engine.start()

        # Independent pipeline loop: keeps the UI cache fresh even when a crawl
        # tick is blocked by slow SearXNG/page fetches.
        _pipeline_task = asyncio.create_task(
            _run_pipeline_loop(store, sources, interval=300)
        )

        cs = registry.crawler_state
        auto_start = cs.continuous or (search_idx.article_count() < 50)
        if auto_start and not _crawl_state.get("running"):
            interval = max(int(cs.interval) if cs.interval else 300, 60)
            max_arts = cs.max_articles or 100
            feeds = registry.feeds
            seeds = registry.seeds
            if feeds or seeds or (registry.searxng and registry.searxng.enabled):
                print(f"[startup] auto-starting continuous crawl (interval={interval}s, max_articles={max_arts})")
                task = asyncio.create_task(
                    _run_continuous_crawl(
                        store, sources, feeds, seeds, max_arts,
                        summarize=True, compute_kv=False,
                        model_id="unsloth/Qwen3-4B-bnb-4bit",
                        interval=interval,
                        max_stories=cs.max_stories,
                        llm_enabled=cs.llm_enabled,
                    )
                )
                _crawl_state["task"] = task

    @app.on_event("shutdown")
    async def shutdown_event():
        global _decompose_engine, _cogitate_engine, _pipeline_task
        _crawl_state["stop_requested"] = True
        _crawl_state["running"] = False
        if _crawl_state.get("task") and not _crawl_state["task"].done():
            _crawl_state["task"].cancel()
            try:
                await asyncio.wait_for(_crawl_state["task"], timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
        if _pipeline_task and not _pipeline_task.done():
            _pipeline_task.cancel()
            try:
                await asyncio.wait_for(_pipeline_task, timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
        if _decompose_engine:
            _decompose_engine.stop()
            for worker in getattr(_decompose_engine, '_workers', []):
                worker.cancel()
        if _cogitate_engine:
            _cogitate_engine.stop()

    # Let uvicorn handle SIGINT/SIGTERM natively. The shutdown_event above
    # is called by FastAPI's lifespan management when uvicorn stops. The
    # crawl's stop_requested flag provides graceful crawl cancellation.

    def _render(request: Request, name: str, context: dict[str, Any]) -> HTMLResponse:
        template = templates.get_template(name)
        return HTMLResponse(template.render(**context))

    def data() -> dict[str, Any]:
        return _build(store)

    # ---- HTML Pages ------------------------------------------------------ #
    @app.get("/")
    def index(request: Request):
        d = data()
        feeds_html = ""
        for sid, url in registry.feeds.items():
            feeds_html += (
                '<div class="feed-row">'
                '<input type="text" class="feed-source" placeholder="source" value="' + _e(sid) + '">'
                '<input type="url" class="feed-url" placeholder="Feed URL" value="' + _e(url) + '">'
                '<button class="btn btn-sm btn-danger" onclick="this.parentElement.remove()">&times;</button>'
                '</div>'
            )
        seeds_html = ""
        for url in registry.seeds:
            seeds_html += (
                '<div class="seed-row">'
                '<input type="url" class="seed-url" placeholder="Seed URL" value="' + _e(url) + '">'
                '<button class="btn btn-sm btn-danger" onclick="this.parentElement.remove()">&times;</button>'
                '</div>'
            )
        return _render(request,
            "index.html",
            {
                "request": request,
                "brief": d["brief"],
                "narratives": d["narratives"],
                "agenda": d["agenda"],
                "by_id": d["by_id"],
                "source_logos": source_logos,
                "articles": sorted(
                    d["articles"],
                    key=lambda f: f.meta.get("first_seen", ""),
                    reverse=True,
                )[:30],
                "feeds_html": feeds_html,
                "seeds_html": seeds_html,
            },
        )

    @app.get("/article/{fid}")
    def article(request: Request, fid: str):
        d = data()
        f = d["by_id"].get(fid)
        if not f:
            raise HTTPException(404, "article not found")
        related = [n for n in d["narratives"] if fid in n.get("members", [])]
        return _render(request,
            "article.html",
            {"request": request, "article": f, "related": related, "agenda": d["agenda"], "source_logos": source_logos},
        )

    @app.get("/source/{sid}")
    def source(request: Request, sid: str):
        d = data()
        src_articles = [a for a in d["articles"] if a.meta.get("source_id") == sid]
        src_narratives = [n for n in d["narratives"] if sid in n.get("sources", [])]
        info = d["agenda"].get(sid, {})
        return _render(request,
            "source.html",
            {
                "request": request,
                "sid": sid,
                "articles": src_articles,
                "narratives": src_narratives,
                "info": info,
                "source_logos": source_logos,
            },
        )

    @app.get("/narrative/{nid}")
    def narrative(request: Request, nid: str):
        d = data()
        n = next((x for x in d["narratives"] if x["narrative_id"] == nid), None)
        if not n:
            raise HTTPException(404, "narrative not found")
        members = [d["by_id"].get(m) for m in n.get("members", []) if m in d["by_id"]]
        return _render(request,
            "narrative.html",
            {"request": request, "narrative": n, "members": members, "agenda": d["agenda"], "source_logos": source_logos},
        )

    @app.get("/lineage")
    def lineage(request: Request):
        d = data()
        return _render(request,
            "lineage.html",
            {"request": request, "derivatives": d["derivatives"], "narratives": d["narratives"], "source_logos": source_logos},
        )

    # ---- JSON API -------------------------------------------------------- #
    @app.get("/api/articles")
    def api_articles():
        return [
            {
                "id": a.figment_id,
                "title": a.meta.get("title") or a.text[:80],
                "source": a.meta.get("source_id"),
                "url": a.meta.get("url"),
                "published": a.meta.get("published"),
                "first_seen": a.meta.get("first_seen"),
                "summary": a.meta.get("summary", ""),
            }
            for a in data()["articles"]
        ]

    @app.get("/api/narratives")
    def api_narratives(page: int = 1, per_page: int = 20, sort: str = "newest"):
        """Get narratives with pagination and sorting."""
        narrs = data()["narratives"]
        
        # Apply sorting
        if sort == "newest":
            narrs.sort(key=lambda n: n.get("first_seen", ""), reverse=True)
        elif sort == "updated":
            narrs.sort(key=lambda n: n.get("last_updated", ""), reverse=True)
        elif sort == "oldest":
            narrs.sort(key=lambda n: n.get("first_seen", ""))
        elif sort == "sources":
            narrs.sort(key=lambda n: len(n.get("sources", [])), reverse=True)
        else:
            narrs.sort(key=lambda n: n.get("latest_article_date", ""), reverse=True)
        
        # Pagination
        total = len(narrs)
        per_page = min(max(per_page, 1), 100)
        page = max(page, 1)
        start = (page - 1) * per_page
        end = start + per_page
        
        return {
            "narratives": narrs[start:end],
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": (total + per_page - 1) // per_page
            },
            "sort": sort
        }

    @app.get("/api/roles")
    def api_roles(role: str = "", range: str = "all"):
        """Return role figments grouped by text, with associated narrative stories.

        `role` filter: who|what|where|why|how (empty = all roles).
        `range` filter: today|yesterday|last_week|last_month|last_year|all (date range for WHEN).
        """
        d = data()
        narrs = d["narratives"]

        # Build narrative_id → member_article_ids mapping
        narrative_members: dict[str, set[str]] = {}
        narrative_by_id: dict[str, dict] = {}
        for n in narrs:
            narrative_members[n["narrative_id"]] = set(n["members"])
            narrative_by_id[n["narrative_id"]] = n

        # Date-range filter for WHEN tab: filter narratives by latest_article_date
        def _in_range(n: dict) -> bool:
            if range == "all" or not n.get("latest_article_date"):
                return True
            nd = _parse_range_date(range, n["latest_article_date"])
            return nd is not False  # False means out of range

        filtered_narrs = [n for n in narrs if _in_range(n)]
        filtered_narr_ids = {n["narrative_id"] for n in filtered_narrs}

        # Group role figments by (role, normalized_text)
        groups: dict[tuple[str, str], dict[str, Any]] = {}
        for f in d["roles"]:
            r = f.meta.get("role")
            if not r:
                continue
            if role and r != role:
                continue
            # Skip non-role figments (narratives, edges, etc.)
            norm = f.meta.get("normalized", "")
            key = (r, norm)
            if key not in groups:
                # Find which narratives this role figment belongs to
                article_id = f.meta.get("article_id")
                story_ids: list[str] = []
                if article_id:
                    for nid, members in narrative_members.items():
                        if article_id in members and nid in filtered_narr_ids:
                            story_ids.append(nid)
                groups[key] = {
                    "role": r,
                    "text": norm or f.text,
                    "count": 0,
                    "story_ids": story_ids,
                }
            # Use reference_count to capture cross-article role sharing
            ref_count = f.meta.get("reference_count", 1)
            groups[key]["count"] += ref_count
            if article_id and story_ids:
                narrative_members.setdefault(story_ids[0], set())

        out = sorted(groups.values(), key=lambda g: g["count"], reverse=True)
        return {"roles": out, "range": range, "role_filter": role, "narrative_count": len(filtered_narrs)}

    @app.post("/api/narratives/intersect")
    async def api_intersect(request: Request):
        """Multi-role intersection query.

        Body: {
            "roles": [{"role": "who", "text": "Donald Trump"}, ...],
            "expand_associations": true,
            "require_all": true,
            "min_trust": 0.0,
            "limit": 50,
            "ranking": "trust_recency"
        }

        Returns ranked narratives that contain all (or any) specified roles,
        expanded through association edges for co-reference.
        """
        try:
            body = await request.json()
        except Exception:
            return {"error": "Invalid JSON body", "narratives": []}

        roles = body.get("roles", [])
        if not roles:
            return {"error": "No roles specified", "narratives": []}

        from ..intersection import find_narratives

        results = find_narratives(
            store,
            roles=roles,
            expand_associations=body.get("expand_associations", True),
            require_all=body.get("require_all", True),
            min_trust=float(body.get("min_trust", 0.0)),
            limit=int(body.get("limit", 50)),
            ranking=body.get("ranking", "trust_recency"),
        )

        # Strip heavy fields for API response
        lightweight = []
        for n in results:
            lightweight.append(
                {
                    "narrative_id": n.get("narrative_id"),
                    "title": n.get("title", ""),
                    "sources": n.get("sources", []),
                    "members": n.get("members", []),
                    "trust_score": n.get("trust_score", 0.5),
                    "source_count": n.get("source_count", 0),
                    "role_matches": n.get("role_matches", {}),
                    "latest_article_date": n.get("latest_article_date", ""),
                    "first_seen": n.get("first_seen", ""),
                    "frame_shift": n.get("frame_shift", False),
                }
            )

        return {"narratives": lightweight, "total": len(lightweight)}

    @app.post("/api/context/materialize")
    async def api_materialize(request: Request):
        """Materialize narratives into a structured context package.

        Body: {
            "narrative_ids": ["narrative:abc123", ...],
            "include_text": true,
            "max_articles_per_narrative": 10
        }

        Returns a provenance-preserving context package with
        source attribution, trust scores, and chronological ordering.
        """
        try:
            body = await request.json()
        except Exception:
            return {"error": "Invalid JSON body"}

        narrative_ids = body.get("narrative_ids", [])
        if not narrative_ids:
            return {"error": "No narrative IDs provided", "context": {}}

        from ..context import materialize_context

        result = materialize_context(
            store,
            narrative_ids,
            include_text=body.get("include_text", True),
            max_articles_per_narrative=int(body.get("max_articles_per_narrative", 10)),
        )

        return {"context": result}

    @app.get("/api/sources")
    def api_sources():
        return data()["agenda"]

    @app.get("/api/lineage")
    def api_lineage():
        return data()["derivatives"]

    @app.get("/api/query")
    def api_query(q: str, k: int = 8, min_trust: float = 0.0):
        gen = _get_generator()
        res = run_query(
            gen.model, gen.tokenizer, store, q, k=k, min_trust=min_trust, faithful=True
        )
        return {"query": q, "answer": res.get("generated_text", ""), "figments_used": res.get("figments_used", 0)}

    # ---- Crawl Control API ---------------------------------------------- #
    @app.get("/api/crawl/status")
    def crawl_status():
        # Filter out non-serializable fields (task is an asyncio.Task)
        return {k: v for k, v in _crawl_state.items() if k != "task"}

    @app.post("/api/crawl/run")
    async def crawl_run(request: Request):
        """Trigger a single crawl tick or start continuous mode."""
        global _crawl_state
        if _crawl_state["running"]:
            return {"error": "Crawl already running", "state": {k: v for k, v in _crawl_state.items() if k != "task"}}

        try:
            body = await request.json()
        except Exception as e:
            return {"error": f"Bad JSON body: {e}"}

        feeds = body.get("feeds", {})
        seeds = body.get("seeds", [])
        max_articles = body.get("max_articles", 40)
        # Auto-backfill: if store is nearly empty, do a deep initial crawl
        if search_idx.article_count() < 10:
            max_articles = max(max_articles, 200)
        compute_kv = body.get("compute_kv", False)
        summarize = body.get("summarize", True)
        model_id = body.get("model_id", "unsloth/Qwen3-4B-bnb-4bit")
        continuous = body.get("continuous", False)
        interval = body.get("interval", 3600)
        max_stories = body.get("max_stories", 0)
        since = body.get("since", "")
        before = body.get("before", "")
        llm_enabled = body.get("llm_enabled", False)

        # Apply SearXNG overrides from control panel
        if registry.searxng:
            sx = registry.searxng
            if "searxng_enabled" in body:
                sx.enabled = bool(body["searxng_enabled"])
            if "searxng_queries" in body:
                sx.queries = [q.strip() for q in body["searxng_queries"].split("\n") if q.strip()]
            if "searxng_time_range" in body:
                sx.time_range = body["searxng_time_range"]
            if "searxng_categories" in body:
                sx.categories = body["searxng_categories"]

        # Load feeds/seeds from sources.json if not provided
        if not feeds and not seeds:
            try:
                feeds = getattr(registry, "feeds", {})
                seeds = getattr(registry, "seeds", [])
            except Exception:
                try:
                    import json as _json
                    with open(sources, "r", encoding="utf-8") as fh:
                        raw = _json.load(fh)
                    feeds = raw.get("feeds", {})
                    seeds = raw.get("seeds", [])
                except Exception:
                    return {"error": "No feeds configured and could not read sources.json"}

        if not feeds and not seeds:
            # Also check if SearXNG search has queries
            has_search = (registry.searxng and registry.searxng.enabled
                          and registry.searxng.queries)
            if not has_search:
                return {"error": "No feeds, seeds, or search queries configured"}

        _crawl_state["stop_requested"] = False

        if continuous:
            task = asyncio.create_task(
                _run_continuous_crawl(
                    store, sources, feeds, seeds, max_articles, summarize, compute_kv, model_id, interval,
                    max_stories=max_stories, since=since, before=before, llm_enabled=llm_enabled,
                )
            )
        else:
            task = asyncio.create_task(
                _run_crawl_tick(
                    store, sources, feeds, seeds, max_articles, summarize, compute_kv, model_id,
                    max_stories=max_stories, since=since, before=before, llm_enabled=llm_enabled,
                )
            )
        _crawl_state["task"] = task
        return_state = {k: v for k, v in _crawl_state.items() if k != "task"}
        return {"started": True, "continuous": continuous, "state": return_state}

    @app.post("/api/crawl/stop")
    async def crawl_stop():
        global _crawl_state
        _crawl_state["stop_requested"] = True
        _crawl_state["running"] = False
        _crawl_state["continuous"] = False
        _crawl_state["current_step"] = "stopping"
        _crawl_state["message"] = "Stopping..."
        await _broadcast({"type": "crawl_status", "data": _crawl_state})
        if _crawl_state.get("task") and not _crawl_state["task"].done():
            _crawl_state["task"].cancel()
        return {"stopped": True}

    @app.post("/api/pipeline/run")
    async def pipeline_run(request: Request):
        """Run just the pipeline (trust, lineage, summaries, brief)."""
        try:
            body = await request.json()
        except Exception as e:
            return {"error": f"Bad JSON: {e}"}
        do_summaries = body.get("summarize", True)
        do_brief = body.get("brief", True)
        max_stories = body.get("max_stories", 0)
        try:
            mid = "unsloth/Qwen3-4B-bnb-4bit"
            model, tokenizer = await _load_model_cached(mid)
            loop = asyncio.get_running_loop()
            stats = await loop.run_in_executor(
                _pipeline_executor,
                run_pipeline, model, tokenizer, store,
                do_summaries, do_brief, max_stories,
                3,
                LLMConfig.from_sources_json(sources),
            )
            _data_cache["data"] = None
            _warm_cache(store)
            return stats
        except Exception as e:
            return {"error": str(e)}

    @app.post("/api/summaries/regenerate")
    async def summaries_regenerate(request: Request):
        """Regenerate article summaries and world brief."""
        try:
            body = await request.json()
        except Exception as e:
            return {"error": f"Bad JSON: {e}"}
        limit = body.get("limit", 500)
        top_n = body.get("top_n", 8)
        try:
            mid = "unsloth/Qwen3-4B-bnb-4bit"
            model, tokenizer = await _load_model_cached(mid)
            loop = asyncio.get_running_loop()
            s1 = await loop.run_in_executor(
                _pipeline_executor,
                summarize_news.ensure_article_summaries, model, tokenizer, store, limit
            )
            s2 = await loop.run_in_executor(
                _pipeline_executor,
                summarize_news.build_world_brief, model, tokenizer, store, top_n
            )
            _data_cache["data"] = None
            _warm_cache(store)
            return {"summaries": s1, "brief": s2}
        except Exception as e:
            return {"error": str(e)}

    _stats_cache: dict[str, Any] = {}

    @app.get("/api/stats")
    def api_stats():
        """Quick store stats for dashboard."""
        try:
            s = _get_stats(store)
            _stats_cache.clear()
            _stats_cache.update(s)
            return s
        except Exception as exc:
            logging.getLogger(__name__).warning("api_stats failed, returning cache: %s", exc)
            return dict(_stats_cache) if _stats_cache else {"articles": 0, "narratives": 0, "derivatives": 0, "sources": 0, "has_brief": False, "last_updated": ""}

    @app.get("/api/config")
    def api_config():
        """Return feeds/seeds/llm/searxng from sources.json for the control panel."""
        llm_config = LLMConfig.from_sources_json(sources)
        cfg = {"feeds": registry.feeds, "seeds": registry.seeds, "llm": {
            "url": llm_config.url, "model": llm_config.model, "enabled": llm_config.enabled,
        }}
        if registry.searxng:
            cfg["searxng"] = {
                "url": registry.searxng.url,
                "enabled": registry.searxng.enabled,
                "categories": registry.searxng.categories,
                "time_range": registry.searxng.time_range,
                "max_results": registry.searxng.max_results,
                "pages": registry.searxng.pages,
            }
        return cfg

    @app.get("/api/crawl/state")
    def api_crawl_state():
        """Return persisted crawler state from sources.json."""
        return registry.crawler_state.__dict__ if hasattr(registry.crawler_state, '__dict__') else {}

    @app.post("/api/crawl/state")
    async def api_crawl_state_save(request: Request):
        """Update persisted crawler state in sources.json."""
        body = await request.json()
        # Update crawler_state fields
        for key, value in body.items():
            if hasattr(registry.crawler_state, key):
                setattr(registry.crawler_state, key, value)
        # Persist to sources.json
        registry.save(sources)
        return {"status": "ok"}

    @app.get("/api/search")
    def api_search(q: str = "", range: str = "all", sort: str = "date_desc", page: int = 1, limit: int = 20):
        """Full-text search with date range filter."""
        result = search_idx.search(q=q, range=range, sort=sort, page=page, limit=limit)
        # Resolve article IDs to article metadata from the data cache
        d = data()
        articles = []
        for aid in result.get("article_ids", []):
            fig = d["by_id"].get(aid)
            if fig:
                articles.append({
                    "id": fig.figment_id,
                    "title": fig.meta.get("title") or fig.text[:80],
                    "source": fig.meta.get("source_id"),
                    "url": fig.meta.get("url"),
                    "published": fig.meta.get("published"),
                    "author": fig.meta.get("author", ""),
                    "summary": fig.meta.get("summary", ""),
                })
        result["articles"] = articles
        return result

    @app.post("/api/story/{nid}/find-more")
    async def story_find_more(nid: str, request: Request):
        """Search for more articles related to a narrative's entities."""
        from urllib.parse import urlparse

        try:
            body = await request.json()
        except Exception:
            body = {}
        query = body.get("query", "")
        story_sources = body.get("sources", [])
        max_results = body.get("max_results", 10)

        if not query:
            return {"error": "No query provided", "found": 0, "results": []}

        results_list = []
        found = 0

        try:
            cfg = SourceRegistry.load(sources)
            if cfg.searxng and cfg.searxng.enabled:
                from ..searxng import search as searxng_search, results_to_articles
                sresults = searxng_search(cfg.searxng, query, max_results=max_results)
                articles = results_to_articles(sresults[:max_results])
                story_source_set = {_normalize_source(s) for s in story_sources}
                for art in articles:
                    sid = art.get("source_id") or urlparse(art.get("url", "")).netloc
                    if story_sources and _normalize_source(sid) not in story_source_set:
                        continue
                    results_list.append({
                        "title": art.get("title") or art.get("url", ""),
                        "url": art.get("url", ""),
                        "source": sid,
                        "published": art.get("published", ""),
                    })
                    found += 1
        except Exception as exc:
            logging.getLogger(__name__).warning("find-more search error: %s", exc)

        return {"found": found, "results": results_list[:max_results]}

    @app.get("/search")
    def search_page(request: Request):
        return _render(request, "search.html", {"request": request, "source_logos": source_logos})

    # ---- WebSocket for live updates ------------------------------------- #
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        _ws_connections.append(websocket)
        # Send initial state + stats (strip non-JSON fields)
        state = {k: v for k, v in _crawl_state.items() if k != "task"}
        await websocket.send_text(json.dumps({"type": "crawl_status", "data": state}))
        await websocket.send_text(json.dumps({"type": "stats", "data": _get_stats(store)}))
        try:
            while True:
                await websocket.receive_text()  # Keep alive / handle ping
        except WebSocketDisconnect:
            _ws_connections.remove(websocket)

    _warm_cache(store)
    return app