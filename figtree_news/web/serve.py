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
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect

warnings.filterwarnings("ignore", message=".*_check_is_size.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="bitsandbytes")
logging.getLogger("transformers").setLevel(logging.ERROR)
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from figtree import FigmentStore, connect, load_model, FigmentGenerator
from figtree.kv_cache_manager import KVCacheManager

from .. import summarize_news
from ..config import SourceRegistry
from ..crawler import Crawler
from ..lineage import get_narratives, get_derivatives, source_agenda
from ..llm_config import LLMConfig
from ..pipeline import run_pipeline
from ..query import query as run_query
from ..search_index import get_index

# Let uvicorn/asyncio handle SIGINT natively — the crawl's stop_requested
# flag provides graceful shutdown, and sys.exit(0) from a signal handler
# does not reliably kill a running asyncio event loop.

_HERE = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(_HERE, "templates")
STATIC_DIR = os.path.join(_HERE, "static")

_gen_cache: dict[str, Any] = {}
_model_cache: dict[str, Any] = {}
_data_cache: dict[str, Any] = {"data": None, "ts": 0.0}
_decompose_engine: Any = None
_cogitate_engine: Any = None
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
    "max_articles": 40,
    "interval": 3600,
    "compute_kv": False,
    "summarize": True,
}
_ws_connections: list[WebSocket] = []


def _get_generator():
    if "gen" not in _gen_cache:
        model, tokenizer = load_model("unsloth/Qwen3-4B-bnb-4bit")
        _gen_cache["gen"] = FigmentGenerator(model, tokenizer)
    return _gen_cache["gen"]


def _build(store: FigmentStore, *, force: bool = False) -> dict[str, Any]:
    if not force and _data_cache["data"] is not None:
        return _data_cache["data"]
    all_figs = store.all()
    by_id = {f.figment_id: f for f in all_figs}
    articles = [
        f
        for f in all_figs
        if f.meta.get("is_image") and f.meta.get("source_id") and not f.is_edge()
    ]
    narratives = get_narratives(store, all_figs=all_figs)
    derivatives = get_derivatives(store, all_figs=all_figs)
    agenda = source_agenda(store, all_figs=all_figs)
    brief = summarize_news.get_world_brief(store, all_figs=all_figs)
    result = {
        "articles": articles,
        "by_id": by_id,
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
    db: str,
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
    """Single crawl tick. Heavy (model) work runs in threads so the event loop stays free."""
    global _crawl_state
    _crawl_state["running"] = True
    _crawl_state["start_time"] = time.time()
    _crawl_state["current_step"] = "loading_model"
    _crawl_state["message"] = "Loading model (~5 min on first use)..."
    _crawl_state["progress"] = 0
    _crawl_state["total"] = 1
    await _broadcast({"type": "crawl_status", "data": _crawl_state})

    try:
        # Load model once and reuse across ticks to avoid GPU OOM
        cache_key = model_id
        if cache_key in _model_cache:
            model, tokenizer = _model_cache[cache_key]
            print(f"[crawl] model reused ({model_id.rsplit('/',1)[-1]})")
        else:
            model, tokenizer = await asyncio.to_thread(load_model, model_id)
            _model_cache[cache_key] = (model, tokenizer)
            _gen_cache["gen"] = FigmentGenerator(model, tokenizer)
            print(f"[crawl] model loaded ({model_id.rsplit('/',1)[-1]})")

        store: FigmentStore = connect(db)
        registry = SourceRegistry.load(sources_path)

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

        _crawl_state["current_step"] = "crawling_feeds"
        _crawl_state["message"] = f"Crawling {len(feeds)} feeds..."
        _crawl_state["feeds"] = list(feeds.keys())
        _crawl_state["total"] = len(feeds)
        await _broadcast({"type": "crawl_status", "data": _crawl_state})

        stats = {"feeds_added": 0, "seeds_added": 0, "sources": set()}
        per_feed = max(1, max_articles // len(feeds)) if feeds else max_articles
        budget = max_articles

        for i, (sid, uri) in enumerate(feeds.items()):
            if budget <= 0 or _crawl_state.get("stop_requested"):
                break
            _crawl_state["current_step"] = f"crawling_feed:{sid}"
            _crawl_state["progress"] = i
            _crawl_state["message"] = f"Crawling {sid} ({i+1}/{len(feeds)})"
            await _broadcast({"type": "crawl_status", "data": _crawl_state})

            # Run feed crawl in thread so loop stays free
            added = await asyncio.to_thread(
                crawler.crawl_feed, sid, uri, min(per_feed, budget),
                since=since, before=before,
            )
            print(f"[crawl] {sid}: +{added} articles")
            stats["feeds_added"] += added
            stats["sources"].add(sid)
            budget -= added

            # Broadcast content update so the page can refresh live
            if added > 0:
                _data_cache["data"] = None
                _warm_cache(store)
                await _broadcast({"type": "content_update", "data": {
                    "source": sid, "added": added,
                    "total_articles": get_index(db.replace(".lance", "_fts.db")).article_count(),
                }})

        if _crawl_state.get("stop_requested"):
            raise asyncio.CancelledError("Stop requested")

        if seeds and budget > 0:
            _crawl_state["current_step"] = "crawling_seeds"
            _crawl_state["message"] = f"Crawling {len(seeds)} seed URLs..."
            _crawl_state["total"] = len(seeds)
            _crawl_state["progress"] = 0
            await _broadcast({"type": "crawl_status", "data": _crawl_state})
            added = await asyncio.to_thread(crawler.crawl_seeds, seeds)
            stats["seeds_added"] += added

        if _crawl_state.get("stop_requested"):
            raise asyncio.CancelledError("Stop requested")

        _crawl_state["current_step"] = "pipeline"
        _crawl_state["message"] = "Running pipeline (trust, lineage, summaries, brief)..."
        _crawl_state["progress"] = 0
        _crawl_state["total"] = 4
        await _broadcast({"type": "crawl_status", "data": _crawl_state})

        # Run pipeline in thread
        llm_config = LLMConfig.from_sources_json(sources_path)
        if llm_enabled and llm_config.url:
            llm_config.enabled = True
            _crawl_state["current_step"] = "pipeline"
            _crawl_state["message"] = "Running pipeline (trust, lineage, eval, summaries, brief)..."
            _crawl_state["total"] = 6
        await _broadcast({"type": "crawl_status", "data": _crawl_state})

        pipe_stats = await asyncio.to_thread(
            run_pipeline, model, tokenizer, store,
            do_summaries=summarize, do_brief=True, max_stories=max_stories,
            llm_config=llm_config if llm_enabled and llm_config.url else None,
        )
        _crawl_state["progress"] = 4
        await _broadcast({"type": "crawl_status", "data": _crawl_state})

        stats["sources"] = sorted(stats["sources"])
        stats.update(pipe_stats)
        n_narr = pipe_stats.get("narratives", [])
        n_narr = len(n_narr) if isinstance(n_narr, list) else n_narr
        print(f"[crawl] pipeline done — {n_narr} narratives, {pipe_stats.get('edges','')} edges")
        _crawl_state["stats"] = stats
        _crawl_state["running"] = False
        _crawl_state["current_step"] = "done"
        _crawl_state["message"] = f"Done: {stats.get('feeds_added',0)} new articles, {stats.get('narratives',0)} narratives"
        _data_cache["data"] = None
        _warm_cache(store)
        print(f"[crawl] tick complete — {stats.get('feeds_added',0)} new articles, {n_narr} narratives")
        await _broadcast({"type": "crawl_status", "data": _crawl_state})

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


async def _run_continuous_crawl(
    db: str,
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
                db, sources_path, feeds, seeds, max_articles, summarize, compute_kv, model_id,
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

    app.state.store = store
    app.state.registry = registry
    
    # Initialize background engines
    llm_config = LLMConfig.from_sources_json(sources)
    global _decompose_engine, _cogitate_engine
    
    # Always create engines if LLM URL is configured (enabled/disabled is a UI toggle)
    if llm_config.url:
        from ..decompose import DecompositionEngine
        from ..cogitate import CogitationEngine
        
        _decompose_engine = DecompositionEngine(llm_config, store)
        _cogitate_engine = CogitationEngine(llm_config, store, interval_hours=6)
        
        app.state.decompose_engine = _decompose_engine
        app.state.cogitate_engine = _cogitate_engine

    if os.path.isdir(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ---- Startup/Shutdown Events ----------------------------------------- #
    @app.on_event("startup")
    async def startup_event():
        global _decompose_engine, _cogitate_engine
        if _decompose_engine:
            _decompose_engine.start()
        if _cogitate_engine:
            _cogitate_engine.start()

    @app.on_event("shutdown")
    async def shutdown_event():
        global _decompose_engine, _cogitate_engine
        if _decompose_engine:
            _decompose_engine.stop()
        if _cogitate_engine:
            _cogitate_engine.stop()

    def _render(request: Request, name: str, context: dict[str, Any]) -> HTMLResponse:
        template = templates.get_template(name)
        return HTMLResponse(template.render(**context))

    def data() -> dict[str, Any]:
        return _build(store)

    # ---- HTML Pages ------------------------------------------------------ #
    @app.get("/")
    def index(request: Request):
        d = data()
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
    def api_narratives():
        return data()["narratives"]

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
                    db, sources, feeds, seeds, max_articles, summarize, compute_kv, model_id, interval,
                    max_stories=max_stories, since=since, before=before, llm_enabled=llm_enabled,
                )
            )
        else:
            task = asyncio.create_task(
                _run_crawl_tick(
                    db, sources, feeds, seeds, max_articles, summarize, compute_kv, model_id,
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
            if mid in _model_cache:
                model, tokenizer = _model_cache[mid]
            else:
                model, tokenizer = await asyncio.to_thread(load_model, mid)
                _model_cache[mid] = (model, tokenizer)
            stats = await asyncio.to_thread(
                run_pipeline, model, tokenizer, store,
                do_summaries=do_summaries, do_brief=do_brief, max_stories=max_stories,
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
            if mid in _model_cache:
                model, tokenizer = _model_cache[mid]
            else:
                model, tokenizer = await asyncio.to_thread(load_model, mid)
                _model_cache[mid] = (model, tokenizer)
            s1 = await asyncio.to_thread(
                summarize_news.ensure_article_summaries, model, tokenizer, store, limit
            )
            s2 = await asyncio.to_thread(
                summarize_news.build_world_brief, model, tokenizer, store, top_n
            )
            _data_cache["data"] = None
            _warm_cache(store)
            return {"summaries": s1, "brief": s2}
        except Exception as e:
            return {"error": str(e)}

    @app.get("/api/stats")
    def api_stats():
        """Quick store stats for dashboard."""
        return _get_stats(store)

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
                "queries": registry.searxng.queries,
                "categories": registry.searxng.categories,
                "time_range": registry.searxng.time_range,
                "max_results": registry.searxng.max_results,
                "pages": registry.searxng.pages,
            }
        return cfg

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