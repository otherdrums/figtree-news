"""FastAPI web app: the local, source-aware newspaper.

Reads the shared LanceDB store and renders: a front page (world brief, top
stories, source-trust board, agenda map), per-article, per-source, and
per-narrative pages (each linking back to the original outlet), and a lineage
view (who broke it first / who echoed whom). JSON API mirrors the pages, plus
on-demand generation endpoints that lazily load the model.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from figtree import FigmentStore, connect, load_model, FigmentGenerator

from .. import summarize_news
from ..config import SourceRegistry
from ..lineage import get_narratives, get_derivatives, source_agenda
from ..query import query as run_query

_HERE = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(_HERE, "templates")
STATIC_DIR = os.path.join(_HERE, "static")

_gen_cache: dict[str, Any] = {}


def _get_generator():
    if "gen" not in _gen_cache:
        model, tokenizer = load_model("unsloth/Qwen3-4B-bnb-4bit")
        _gen_cache["gen"] = FigmentGenerator(model, tokenizer)
    return _gen_cache["gen"]


def _build(store: FigmentStore) -> dict[str, Any]:
    all_figs = store.all()
    by_id = {f.figment_id: f for f in all_figs}
    articles = [
        f
        for f in all_figs
        if f.meta.get("is_image") and f.meta.get("source_id") and not f.is_edge()
    ]
    narratives = get_narratives(store)
    derivatives = get_derivatives(store)
    agenda = source_agenda(store)
    brief = summarize_news.get_world_brief(store)
    return {
        "articles": articles,
        "by_id": by_id,
        "narratives": narratives,
        "derivatives": derivatives,
        "agenda": agenda,
        "brief": brief,
    }


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
        # Render manually: Starlette's TemplateResponse passes the whole context
        # as Jinja globals, which breaks the template cache for dict values.
        template = templates.get_template(name)
        return HTMLResponse(template.render(**context))

    def data() -> dict[str, Any]:
        return _build(store)

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

    # ---- JSON API ------------------------------------------------------ #
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

    return app
