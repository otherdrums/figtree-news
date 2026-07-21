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
import os
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from figtree import FigmentStore, connect, load_model, FigmentGenerator

from .. import summarize_news
from ..config import SourceRegistry
from ..crawler import Crawler
from ..lineage import get_narratives, get_derivatives, source_agenda
from ..pipeline import run_pipeline
from ..query import query as run_query

_HERE = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(_HERE, "templates")
STATIC_DIR = os.path.join(_HERE, "static")

_gen_cache: dict[str, Any] = {}
_data_cache: dict[str, Any] = {"data": None, "ts": 0.0}
_crawl_state: dict[str, Any] = {
    "running": False,
    "task": None,
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
    now = time.time()
    if not force and _data_cache["data"] and (now - _data_cache["ts"] < 30):
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
    _data_cache["ts"] = now
    return result


async def _broadcast(msg: dict[str, Any]):
    """Broadcast message to all connected WebSocket clients."""
    dead = []
    for ws in _ws_connections:
        try:
            await ws.send_text(json.dumps(msg))
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_connections.remove(ws)


async def _run_crawl_task(
    db: str,
    sources_path: str,
    feeds: dict[str, str],
    seeds: list[str],
    max_articles: int,
    compute_kv: bool,
    summarize: bool,
    model_id: str,
):
    """Background task that runs a single crawl tick + pipeline."""
    global _crawl_state
    _crawl_state["running"] = True
    _crawl_state["start_time"] = time.time()
    _crawl_state["current_step"] = "loading_model"
    _crawl_state["message"] = "Loading model..."
    _crawl_state["progress"] = 0
    _crawl_state["total"] = 1
    await _broadcast({"type": "crawl_status", "data": _crawl_state})

    model, tokenizer = load_model(model_id)
    _gen_cache["gen"] = FigmentGenerator(model, tokenizer)

    store: FigmentStore = connect(db)
    registry = SourceRegistry.load(sources_path)

    crawler = Crawler(
        model, tokenizer, store, registry,
        seen_path="./seen_urls.json",
        compute_kv=compute_kv, summarize_images=summarize,
    )

    _crawl_state["current_step"] = "crawling_feeds"
    _crawl_state["message"] = f"Crawling {len(feeds)} feeds..."
    _crawl_state["feeds"] = list(feeds.keys())
    _crawl_state["total"] = len(feeds)
    await _broadcast({"type": "crawl_status", "data": _crawl_state})

    # Custom crawl with progress
    stats = {"feeds_added": 0, "seeds_added": 0, "sources": set()}
    per_feed = max(1, max_articles // len(feeds)) if feeds else max_articles
    budget = max_articles

    for i, (sid, uri) in enumerate(feeds.items()):
        if budget <= 0:
            break
        _crawl_state["current_step"] = f"crawling_feed:{sid}"
        _crawl_state["progress"] = i
        _crawl_state["message"] = f"Crawling {sid} ({i+1}/{len(feeds)})"
        await _broadcast({"type": "crawl_status", "data": _crawl_state})

        added = crawler.crawl_feed(sid, uri, max_articles=min(per_feed, budget))
        stats["feeds_added"] += added
        stats["sources"].add(sid)
        budget -= added

    if seeds and budget > 0:
        _crawl_state["current_step"] = "crawling_seeds"
        _crawl_state["message"] = f"Crawling {len(seeds)} seed URLs..."
        _crawl_state["total"] = len(seeds)
        _crawl_state["progress"] = 0
        await _broadcast({"type": "crawl_status", "data": _crawl_state})
        added = crawler.crawl_seeds(seeds)
        stats["seeds_added"] += added

    _crawl_state["current_step"] = "pipeline"
    _crawl_state["message"] = "Running pipeline (trust, lineage, summaries, brief)..."
    _crawl_state["progress"] = 0
    _crawl_state["total"] = 4
    await _broadcast({"type": "crawl_status", "data": _crawl_state})

    pipe_stats = run_pipeline(
        model, tokenizer, store,
        do_summaries=summarize, do_brief=True,
    )
    _crawl_state["progress"] = 4
    await _broadcast({"type": "crawl_status", "data": _crawl_state})

    stats["sources"] = sorted(stats["sources"])
    stats.update(pipe_stats)
    _crawl_state["stats"] = stats
    _crawl_state["running"] = False
    _crawl_state["current_step"] = "done"
    _crawl_state["message"] = f"Done: {stats.get('feeds_added',0)} new articles, {stats.get('narratives',0)} narratives"
    _data_cache["data"] = None
    await _broadcast({"type": "crawl_status", "data": _crawl_state})


def create_app(db: str = "./news.lance", sources: str = "./sources.json") -> FastAPI:
    app = FastAPI(title="figtree-news", description="Source-aware web newspaper")
    templates = Jinja2Templates(directory=TEMPLATES_DIR)
    registry = SourceRegistry.load(sources)
    store: FigmentStore = connect(db)

    app.state.store = store
    app.state.registry = registry

    if os.path.isdir(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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
            {"request": request, "article": f, "related": related, "agenda": d["agenda"]},
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
            {"request": request, "narrative": n, "members": members, "agenda": d["agenda"]},
        )

    @app.get("/lineage")
    def lineage(request: Request):
        d = data()
        return _render(request,
            "lineage.html",
            {"request": request, "derivatives": d["derivatives"], "narratives": d["narratives"]},
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
        return {"query": q, "answer": res.get("text", ""), "figments_used": res.get("figments_used", 0)}

    # ---- Crawl Control API ---------------------------------------------- #
    @app.get("/api/crawl/status")
    def crawl_status():
        return _crawl_state

    @app.post("/api/crawl/run")
    async def crawl_run(request: Request):
        """Trigger a single crawl tick + pipeline."""
        global _crawl_state
        if _crawl_state["running"]:
            return {"error": "Crawl already running", "state": _crawl_state}

        body = await request.json()
        feeds = body.get("feeds", {})
        seeds = body.get("seeds", [])
        max_articles = body.get("max_articles", 40)
        compute_kv = body.get("compute_kv", False)
        summarize = body.get("summarize", True)
        model_id = body.get("model_id", "unsloth/Qwen3-4B-bnb-4bit")

        # Load feeds/seeds from sources.json if not provided
        if not feeds and not seeds:
            feeds, seeds = registry.feeds, registry.seeds

        task = asyncio.create_task(
            _run_crawl_task(db, sources, feeds, seeds, max_articles, compute_kv, summarize, model_id)
        )
        _crawl_state["task"] = task
        return {"started": True, "state": _crawl_state}

    @app.post("/api/crawl/stop")
    def crawl_stop():
        global _crawl_state
        if _crawl_state["task"] and not _crawl_state["task"].done():
            _crawl_state["task"].cancel()
            _crawl_state["running"] = False
            _crawl_state["message"] = "Cancelled"
            return {"stopped": True}
        return {"error": "No running crawl"}

    @app.post("/api/pipeline/run")
    async def pipeline_run(request: Request):
        """Run just the pipeline (trust, lineage, summaries, brief)."""
        body = await request.json()
        do_summaries = body.get("summarize", True)
        do_brief = body.get("brief", True)
        model, tokenizer = load_model("unsloth/Qwen3-4B-bnb-4bit")
        stats = run_pipeline(model, tokenizer, store, do_summaries=do_summaries, do_brief=do_brief)
        _data_cache["data"] = None
        return stats

    @app.post("/api/summaries/regenerate")
    async def summaries_regenerate(request: Request):
        """Regenerate article summaries and world brief."""
        body = await request.json()
        limit = body.get("limit", 500)
        top_n = body.get("top_n", 8)
        model, tokenizer = load_model("unsloth/Qwen3-4B-bnb-4bit")
        s1 = summarize_news.ensure_article_summaries(model, tokenizer, store, limit=limit)
        s2 = summarize_news.build_world_brief(model, tokenizer, store, top_n=top_n)
        _data_cache["data"] = None
        return {"summaries": s1, "brief": s2}

    @app.get("/api/stats")
    def api_stats():
        """Quick store stats for dashboard."""
        d = data()
        return {
            "articles": len(d["articles"]),
            "narratives": len(d["narratives"]),
            "derivatives": len(d["derivatives"]),
            "sources": len(d["agenda"]),
            "has_brief": bool(d["brief"]),
            "last_updated": max((a.meta.get("first_seen", "") for a in d["articles"]), default=""),
        }

    @app.get("/api/config")
    def api_config():
        """Return feeds/seeds from sources.json for the control panel."""
        return {"feeds": registry.feeds, "seeds": registry.seeds}

    # ---- WebSocket for live updates ------------------------------------- #
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        _ws_connections.append(websocket)
        # Send initial state
        await websocket.send_text(json.dumps({"type": "crawl_status", "data": _crawl_state}))
        try:
            while True:
                await websocket.receive_text()  # Keep alive / handle ping
        except WebSocketDisconnect:
            _ws_connections.remove(websocket)

    return app