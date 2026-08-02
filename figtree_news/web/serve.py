"""FastAPI web app: the local, source-aware newspaper.

Single-process design: the server owns the (GPU) model and runs the crawl +
pipeline loop as one background asyncio task. ``--device none`` gives a
viewer-only mode that reads the shared store without ever touching a model
(suitable for low-VRAM boxes or when a separate crawler process owns the GPU).

Renders: front page (world brief, top stories, source-trust board, agenda),
per-article, per-source, per-narrative pages, lineage view (who broke it
first / who echoed whom), and a JSON API mirror.
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

from figtree import FigmentStore, FigmentGenerator, connect, load_model

from .. import summarize_news
from ..config import SourceRegistry
from ..crawler import Crawler
from ..lineage import get_narratives, get_derivatives, source_agenda, _normalize_source
from ..pipeline import run_pipeline
from ..query import query as run_query
from ..search_index import get_index

warnings.filterwarnings("ignore", message=".*_check_is_size.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="bitsandbytes")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

MODEL_ID = "unsloth/Qwen3-4B-bnb-4bit"


def _e(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


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
        return article_dt >= now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif range_str == "yesterday":
        since = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return since <= article_dt < since + timedelta(days=1)
    elif range_str == "last_week":
        return article_dt >= now - timedelta(days=7)
    elif range_str == "last_month":
        return article_dt >= now - timedelta(days=30)
    elif range_str == "last_year":
        return article_dt >= now - timedelta(days=365)
    return True


_HERE = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(_HERE, "templates")
STATIC_DIR = os.path.join(_HERE, "static")

_model_cache: dict[str, Any] = {}
_model_load_lock: asyncio.Lock = asyncio.Lock()
_pipeline_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pipeline")
_crawl_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="crawl")
_data_cache: dict[str, Any] = {"data": None, "ts": 0.0}
_loop_task: asyncio.Task | None = None
_FTS_DB: str = "demo/news_fts.db"
_loop_running = False
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


async def _load_model_cached(device: str) -> tuple[Any, Any]:
    """Load the model once, on the executor. ``device`` is auto|cpu|none."""
    if "model" in _model_cache:
        return _model_cache["model"]
    async with _model_load_lock:
        if "model" in _model_cache:
            return _model_cache["model"]
        dev = None if device == "auto" else ("cpu" if device == "cpu" else None)
        model, tokenizer = await asyncio.to_thread(load_model, MODEL_ID, dev)
        _model_cache["model"] = (model, tokenizer)
        _model_cache["gen"] = FigmentGenerator(model, tokenizer)
        return model, tokenizer


def _get_generator(device: str):
    if "gen" not in _model_cache:
        raise RuntimeError("model not loaded (viewer-only mode)")
    return _model_cache["gen"]


def _build(store: FigmentStore, *, force: bool = False) -> dict[str, Any]:
    if not force and _data_cache["data"] is not None:
        return _data_cache["data"]
    all_figs = store.all()
    by_id = {f.figment_id: f for f in all_figs}
    articles = [f for f in all_figs if f.kind == "article" and f.meta.get("source_id")]
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


async def _run_tick(
    store: FigmentStore,
    sources_path: str,
    feeds: dict[str, str],
    seeds: list[str],
    max_articles: int,
    summarize: bool,
    compute_kv: bool,
    max_stories: int,
    since: str,
    before: str,
    device: str,
):
    """One crawl tick: ingest feeds/seeds, then run the full pipeline in-place.

    Crawling and pipeline share the same process and the same model — there is
    no second pipeline loop to contend with (the old dual-loop design OOM-killed
    the box every ~20 minutes).
    """
    global _loop_running
    _loop_running = True
    _crawl_state["running"] = True
    _crawl_state["start_time"] = time.time()
    _crawl_state["current_step"] = "loading_model"
    _crawl_state["message"] = "Loading model (~5 min on first use)..."
    _crawl_state["progress"] = 0
    _crawl_state["total"] = 1
    await _broadcast({"type": "crawl_status", "data": _crawl_state})

    try:
        already_cached = "model" in _model_cache
        model, tokenizer = await _load_model_cached(device)
        print(f"[crawl] model {'reused' if already_cached else 'loaded'} ({MODEL_ID.rsplit('/',1)[-1]})")

        registry = SourceRegistry.load(sources_path)

        kv_manager = None
        if compute_kv:
            from figtree.kv_cache_manager import KVCacheManager
            kv_manager = KVCacheManager(model, tokenizer, kv_root="./figtree_kv", mode="eager")

        crawler = Crawler(
            model, tokenizer, store, registry,
            seen_path="./seen_urls.json",
            compute_kv=compute_kv, summarize_images=summarize,
            kv_manager=kv_manager,
            fts_path=_FTS_DB,
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
                _crawl_executor, crawler.crawl_feed, sid, uri, per_feed, since, before,
            )
            return sid, added

        async def _crawl_seeds_task() -> int:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(_crawl_executor, crawler.crawl_seeds, seeds)

        ASYNC_TIMEOUT = 300
        feed_tasks = [asyncio.wait_for(_crawl_one_feed(sid, uri), timeout=ASYNC_TIMEOUT) for sid, uri in feeds.items()]
        seed_task = asyncio.wait_for(_crawl_seeds_task(), timeout=ASYNC_TIMEOUT) if seeds else None
        results = await asyncio.gather(*feed_tasks + ([seed_task] if seed_task else []), return_exceptions=True)

        feeds_added = seeds_added = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                sid = list(feeds.keys())[i] if i < len(feeds) else "seeds"
                print(f"[crawl] {sid} timed out or failed: {result}")
                continue
            if i < len(feed_tasks):
                _, added = result
                feeds_added += added
            else:
                seeds_added = result

        # ── Phase 2: SearXNG (sequential; each query triggers GPU ingest) ──
        search_added = 0
        cfg = registry.searxng
        if cfg and cfg.enabled:
            _crawl_state["current_step"] = "searching"
            from figtree_news.crawler import _extract_keywords
            with crawler._ingest_lock:
                new_articles_snapshot = list(crawler._new_articles)
            queries = _extract_keywords(new_articles_snapshot, top_n=10)
            if not queries:
                queries = ["breaking news", "world news today", "technology news", "business markets", "politics government"]

            _crawl_state["message"] = f"Searching {len(queries)} queries sequentially..."
            _crawl_state["total"] = len(queries)
            _crawl_state["progress"] = 0
            await _broadcast({"type": "crawl_status", "data": _crawl_state})

            for qi, q in enumerate(queries):
                if _crawl_state.get("stop_requested"):
                    break
                _crawl_state["progress"] = qi
                _crawl_state["message"] = f"Searching: {q}"
                await _broadcast({"type": "crawl_status", "data": _crawl_state})

                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    free_mb = torch.cuda.mem_get_info()[0] // (1024 * 1024)
                    if free_mb < 500:
                        print(f"[crawl] VRAM low ({free_mb}MB free) — stopping SearXNG queries")
                        _crawl_state["message"] = f"VRAM low ({free_mb}MB) — SearXNG queries stopped"
                        await _broadcast({"type": "crawl_status", "data": _crawl_state})
                        break

                try:
                    loop = asyncio.get_running_loop()
                    got = await loop.run_in_executor(
                        _crawl_executor, crawler.search_searxng, q, cfg.categories, cfg.time_range,
                        cfg.max_results, cfg.pages,
                    )
                    search_added += got
                except Exception as exc:
                    print(f"[crawl] search '{q}' failed: {exc}")
                await asyncio.sleep(1)

            print(f"[crawl] SearXNG: +{search_added} articles across {len(queries)} queries")

        total_new = feeds_added + seeds_added + search_added
        _crawl_state["stats"] = {"feeds_added": feeds_added, "seeds_added": seeds_added, "search_added": search_added}
        if total_new:
            _data_cache["data"] = None
            _warm_cache(store)
            await _broadcast({"type": "content_update", "data": {
                "feeds_added": feeds_added, "seeds_added": seeds_added,
                "total_articles": get_index(_FTS_DB).article_count(),
            }})

        # ── Phase 3: Full pipeline, in-process (trust → lineage → brief) ──
        _crawl_state["current_step"] = "pipeline"
        _crawl_state["message"] = "Running trust + lineage + summaries..."
        await _broadcast({"type": "crawl_status", "data": _crawl_state})

        def _on_lineage(_store, _trust_out, lineage_out):
            _data_cache["data"] = None
            _warm_cache(store)
            print(f"[lineage] {len(lineage_out.get('narratives', []))} narratives ready")

        loop = asyncio.get_running_loop()
        stats = await loop.run_in_executor(
            _pipeline_executor,
            run_pipeline, model, tokenizer, store,
            True,  # do_summaries
            True,  # do_brief
            max_stories,
            10,
            _on_lineage,
        )
        print(f"[pipeline] done: {stats.get('narratives', 0)} narratives, "
              f"{stats.get('brief_used', 0)} brief articles, {stats.get('summarized', 0)} summaries")

        _crawl_state["stats"].update({k: stats.get(k, 0) for k in ("narratives", "summarized", "brief_used")})
        _crawl_state["current_step"] = "done"
        _crawl_state["message"] = f"Done: {total_new} new articles, {stats.get('narratives', 0)} narratives"
        print(f"[crawl] tick complete — {total_new} new articles, {stats.get('narratives', 0)} narratives")
        await _broadcast({"type": "crawl_status", "data": _crawl_state})

    except asyncio.CancelledError:
        _crawl_state["running"] = False
        _crawl_state["current_step"] = "idle"
        _crawl_state["message"] = "Crawl stopped"
        print("[crawl] stopped")
        await _broadcast({"type": "crawl_status", "data": _crawl_state})
    except Exception as exc:
        _crawl_state["running"] = False
        _crawl_state["current_step"] = "error"
        _crawl_state["message"] = f"Error: {exc}"
        print(f"[crawl] ERROR: {exc}")
        await _broadcast({"type": "crawl_status", "data": _crawl_state})
        import traceback
        traceback.print_exc()
    finally:
        _loop_running = False


async def _run_loop(
    store: FigmentStore,
    sources_path: str,
    feeds: dict[str, str],
    seeds: list[str],
    max_articles: int,
    interval: int,
    max_stories: int,
    device: str,
):
    """Continuous crawl + pipeline loop. One task owns the GPU model."""
    global _crawl_state
    _crawl_state["continuous"] = True
    _crawl_state["stop_requested"] = False
    tick_num = 0
    print(f"[crawl] continuous mode started (interval={interval}s, feeds={len(feeds)})")

    while not _crawl_state.get("stop_requested"):
        tick_num += 1
        print(f"[crawl] tick #{tick_num} starting")
        try:
            await _run_tick(
                store, sources_path, feeds, seeds, max_articles,
                summarize=True, compute_kv=False, max_stories=max_stories,
                since="", before="", device=device,
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
            for _ in range(interval):
                if _crawl_state.get("stop_requested"):
                    break
                await asyncio.sleep(1)
        else:
            _crawl_state["current_step"] = "next_tick"
            _crawl_state["running"] = False
            await _broadcast({"type": "crawl_status", "data": _crawl_state})
            await asyncio.sleep(0)

    _crawl_state["continuous"] = False
    _crawl_state["running"] = False
    _crawl_state["current_step"] = "idle"
    _crawl_state["message"] = "Continuous crawl stopped"
    print(f"[crawl] stopped after {tick_num} ticks")
    await _broadcast({"type": "crawl_status", "data": _crawl_state})


def create_app(
    db: str = "./news.lance",
    sources: str = "./sources.json",
    device: str = "auto",
) -> FastAPI:
    """Build the FastAPI app.

    device:
        "auto" — load the model on CUDA if available, else CPU (default).
        "cpu"  — force CPU inference.
        "none" — viewer-only: no model, no crawling (read the store only).
    """
    global _loop_task, _FTS_DB
    app = FastAPI(title="figtree-news", description="Source-aware web newspaper")
    templates = Jinja2Templates(directory=TEMPLATES_DIR)
    registry = SourceRegistry.load(sources)
    store: FigmentStore = connect(db)
    _FTS_DB = db.replace(".lance", "_fts.db")
    search_idx = get_index(_FTS_DB)
    source_logos = {s.source_id: s.logo_url for s in registry.all() if s.logo_url}

    app.state.store = store
    app.state.registry = registry
    app.state.device = device

    if os.path.isdir(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ---- Startup / Shutdown ---------------------------------------------- #
    @app.on_event("startup")
    async def startup_event():
        global _loop_task
        if device == "none":
            print("[startup] viewer-only mode (device=none) — no crawling, no model")
            return
        cs = registry.crawler_state
        ac = search_idx.article_count()
        auto_start = cs.continuous or (ac < 50)
        if auto_start and not _crawl_state.get("running"):
            interval = max(int(cs.interval) if cs.interval else 300, 60)
            max_arts = cs.max_articles or 100
            feeds = registry.feeds
            seeds = registry.seeds
            if feeds or seeds or (registry.searxng and registry.searxng.enabled):
                print(f"[startup] auto-starting continuous crawl (interval={interval}s, max_articles={max_arts})")
                _loop_task = asyncio.create_task(
                    _run_loop(
                        store, sources, feeds, seeds, max_arts, interval,
                        cs.max_stories, device,
                    )
                )
                _crawl_state["task"] = _loop_task

    @app.on_event("shutdown")
    async def shutdown_event():
        global _loop_task
        _crawl_state["stop_requested"] = True
        _crawl_state["running"] = False
        if _loop_task and not _loop_task.done():
            _loop_task.cancel()
            try:
                await asyncio.wait_for(_loop_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

    def _render(request: Request, name: str, context: dict[str, Any]) -> HTMLResponse:
        template = templates.get_template(name)
        return HTMLResponse(template.render(**context))

    def data() -> dict[str, Any]:
        return _build(store)

    def _require_model():
        if "gen" not in _model_cache:
            raise HTTPException(503, "model not loaded (device=none viewer mode)")

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
        return _render(request, "index.html", {
            "request": request,
            "brief": d["brief"],
            "narratives": d["narratives"],
            "agenda": d["agenda"],
            "by_id": d["by_id"],
            "source_logos": source_logos,
            "articles": sorted(d["articles"], key=lambda f: f.meta.get("first_seen", ""), reverse=True)[:30],
            "feeds_html": feeds_html,
            "seeds_html": seeds_html,
        })

    @app.get("/article/{fid}")
    def article(request: Request, fid: str):
        d = data()
        f = d["by_id"].get(fid)
        if not f:
            raise HTTPException(404, "article not found")
        related = [n for n in d["narratives"] if fid in n.get("members", [])]
        return _render(request, "article.html",
                       {"request": request, "article": f, "related": related,
                        "agenda": d["agenda"], "source_logos": source_logos})

    @app.get("/source/{sid}")
    def source(request: Request, sid: str):
        d = data()
        src_articles = [a for a in d["articles"] if a.meta.get("source_id") == sid]
        src_narratives = [n for n in d["narratives"] if sid in n.get("sources", [])]
        info = d["agenda"].get(sid, {})
        return _render(request, "source.html",
                       {"request": request, "sid": sid, "articles": src_articles,
                        "narratives": src_narratives, "info": info,
                        "source_logos": source_logos})

    @app.get("/narrative/{nid}")
    def narrative(request: Request, nid: str):
        d = data()
        n = next((x for x in d["narratives"] if x["narrative_id"] == nid), None)
        if not n:
            raise HTTPException(404, "narrative not found")
        members = [d["by_id"].get(m) for m in n.get("members", []) if m in d["by_id"]]
        return _render(request, "narrative.html",
                       {"request": request, "narrative": n, "members": members,
                        "agenda": d["agenda"], "source_logos": source_logos})

    @app.get("/lineage")
    def lineage(request: Request):
        d = data()
        return _render(request, "lineage.html",
                       {"request": request, "derivatives": d["derivatives"],
                        "narratives": d["narratives"], "source_logos": source_logos})

    # ---- JSON API -------------------------------------------------------- #
    @app.get("/api/articles")
    def api_articles():
        return [{
            "id": a.figment_id,
            "title": a.meta.get("title") or a.text[:80],
            "source": a.meta.get("source_id"),
            "url": a.meta.get("url"),
            "published": a.meta.get("published"),
            "first_seen": a.meta.get("first_seen"),
            "summary": a.meta.get("summary", ""),
        } for a in data()["articles"]]

    @app.get("/api/narratives")
    def api_narratives(page: int = 1, per_page: int = 20, sort: str = "newest"):
        narrs = data()["narratives"]
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
        total = len(narrs)
        per_page = min(max(per_page, 1), 100)
        page = max(page, 1)
        start, end = (page - 1) * per_page, page * per_page
        return {"narratives": narrs[start:end],
                "pagination": {"page": page, "per_page": per_page, "total": total,
                               "total_pages": (total + per_page - 1) // per_page},
                "sort": sort}

    @app.get("/api/roles")
    def api_roles(role: str = "", range: str = "all"):
        d = data()
        narrs = d["narratives"]
        narrative_members = {n["narrative_id"]: set(n["members"]) for n in narrs}

        def _in_range(n: dict) -> bool:
            if range == "all" or not n.get("latest_article_date"):
                return True
            return _parse_range_date(range, n["latest_article_date"])

        filtered_narr_ids = {n["narrative_id"] for n in narrs if _in_range(n)}
        groups: dict[tuple[str, str], dict[str, Any]] = {}
        for f in d["roles"]:
            r = f.meta.get("role")
            if not r or (role and r != role):
                continue
            norm = f.meta.get("normalized", "")
            key = (r, norm)
            if key not in groups:
                article_id = f.meta.get("article_id")
                story_ids = []
                if article_id:
                    story_ids = [nid for nid, members in narrative_members.items()
                                 if article_id in members and nid in filtered_narr_ids]
                groups[key] = {"role": r, "text": norm or f.text, "count": 0, "story_ids": story_ids}
            groups[key]["count"] += f.meta.get("reference_count", 1)
        out = sorted(groups.values(), key=lambda g: g["count"], reverse=True)
        return {"roles": out, "range": range, "role_filter": role, "narrative_count": len(filtered_narr_ids)}

    @app.post("/api/narratives/intersect")
    async def api_intersect(request: Request):
        try:
            body = await request.json()
        except Exception:
            return {"error": "Invalid JSON body", "narratives": []}
        roles = body.get("roles", [])
        if not roles:
            return {"error": "No roles specified", "narratives": []}
        from ..intersection import find_narratives
        results = await asyncio.to_thread(
            find_narratives, store,
            roles=roles,
            expand_associations=body.get("expand_associations", True),
            require_all=body.get("require_all", True),
            min_trust=float(body.get("min_trust", 0.0)),
            limit=int(body.get("limit", 50)),
            ranking=body.get("ranking", "trust_recency"),
        )
        lightweight = [{
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
        } for n in results]
        return {"narratives": lightweight, "total": len(lightweight)}

    @app.post("/api/context/materialize")
    async def api_materialize(request: Request):
        try:
            body = await request.json()
        except Exception:
            return {"error": "Invalid JSON body"}
        narrative_ids = body.get("narrative_ids", [])
        if not narrative_ids:
            return {"error": "No narrative IDs provided", "context": {}}
        from ..context import materialize_context
        result = await asyncio.to_thread(
            materialize_context, store, narrative_ids,
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
        _require_model()
        gen = _model_cache["gen"]
        res = run_query(gen.model, gen.tokenizer, store, q, k=k, min_trust=min_trust, faithful=True)
        return {"query": q, "answer": res.get("generated_text", ""), "figments_used": res.get("figments_used", 0)}

    # ---- Crawl Control API ---------------------------------------------- #
    @app.get("/api/crawl/status")
    def crawl_status():
        return {k: v for k, v in _crawl_state.items() if k != "task"}

    @app.post("/api/crawl/run")
    async def crawl_run(request: Request):
        global _crawl_state
        if _crawl_state["running"]:
            return {"error": "Crawl already running", "state": {k: v for k, v in _crawl_state.items() if k != "task"}}
        if device == "none":
            return {"error": "viewer-only mode (device=none): crawling disabled"}
        try:
            body = await request.json()
        except Exception as e:
            return {"error": f"Bad JSON body: {e}"}
        feeds = body.get("feeds", {})
        seeds = body.get("seeds", [])
        max_articles = body.get("max_articles", 40)
        if search_idx.article_count() < 10:
            max_articles = max(max_articles, 200)
        continuous = body.get("continuous", False)
        interval = body.get("interval", 3600)
        max_stories = body.get("max_stories", 0)
        if not feeds and not seeds:
            feeds = getattr(registry, "feeds", {})
            seeds = getattr(registry, "seeds", [])
        if not feeds and not seeds:
            return {"error": "No feeds, seeds, or search queries configured"}
        _crawl_state["stop_requested"] = False
        task = asyncio.create_task(
            _run_loop(store, sources, feeds, seeds, max_articles, interval, max_stories, device)
            if continuous else
            _run_tick(store, sources, feeds, seeds, max_articles,
                      body.get("summarize", True), body.get("compute_kv", False),
                      max_stories, body.get("since", ""), body.get("before", ""), device)
        )
        _crawl_state["task"] = task
        return {"started": True, "continuous": continuous,
                "state": {k: v for k, v in _crawl_state.items() if k != "task"}}

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
        if device == "none":
            return {"error": "viewer-only mode (device=none): pipeline disabled"}
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            model, tokenizer = await _load_model_cached(device)
            loop = asyncio.get_running_loop()
            stats = await loop.run_in_executor(
                _pipeline_executor, run_pipeline, model, tokenizer, store,
                body.get("summarize", True), body.get("brief", True),
                body.get("max_stories", 0), 3,
            )
            _data_cache["data"] = None
            _warm_cache(store)
            return stats
        except Exception as e:
            return {"error": str(e)}

    @app.post("/api/summaries/regenerate")
    async def summaries_regenerate(request: Request):
        if device == "none":
            return {"error": "viewer-only mode (device=none): no model"}
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            model, tokenizer = await _load_model_cached(device)
            loop = asyncio.get_running_loop()
            s1 = await loop.run_in_executor(
                _pipeline_executor, summarize_news.ensure_article_summaries,
                model, tokenizer, store, None, body.get("limit", 500),
            )
            s2 = await loop.run_in_executor(
                _pipeline_executor, summarize_news.build_world_brief,
                model, tokenizer, store, None, body.get("top_n", 8),
            )
            _data_cache["data"] = None
            _warm_cache(store)
            return {"summaries": s1, "brief": s2}
        except Exception as e:
            return {"error": str(e)}

    _stats_cache: dict[str, Any] = {}

    @app.get("/api/stats")
    def api_stats():
        try:
            s = _get_stats(store)
            _stats_cache.clear()
            _stats_cache.update(s)
            return s
        except Exception as exc:
            logging.getLogger(__name__).warning("api_stats failed, returning cache: %s", exc)
            return dict(_stats_cache) if _stats_cache else {
                "articles": 0, "narratives": 0, "derivatives": 0, "sources": 0,
                "has_brief": False, "last_updated": "",
            }

    @app.get("/api/config")
    def api_config():
        cfg = {"feeds": registry.feeds, "seeds": registry.seeds,
               "llm": {"url": "", "model": MODEL_ID, "enabled": False},
               "device": device}
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
        return registry.crawler_state.__dict__ if hasattr(registry.crawler_state, '__dict__') else {}

    @app.post("/api/crawl/state")
    async def api_crawl_state_save(request: Request):
        body = await request.json()
        for key, value in body.items():
            if hasattr(registry.crawler_state, key):
                setattr(registry.crawler_state, key, value)
        registry.save(sources)
        return {"status": "ok"}

    @app.get("/api/search")
    def api_search(q: str = "", range: str = "all", sort: str = "date_desc", page: int = 1, limit: int = 20):
        result = search_idx.search(q=q, range=range, sort=sort, page=page, limit=limit)
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
                sresults = searxng_search(cfg.searxng, query, categories=cfg.categories, time_range=cfg.time_range)
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
        state = {k: v for k, v in _crawl_state.items() if k != "task"}
        await websocket.send_text(json.dumps({"type": "crawl_status", "data": state}))
        await websocket.send_text(json.dumps({"type": "stats", "data": _get_stats(store)}))
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            _ws_connections.remove(websocket)

    _warm_cache(store)
    return app
