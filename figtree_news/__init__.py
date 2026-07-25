"""figtree-news: a source-aware news aggregator built on Figtree figments."""

from .config import SourceConfig, SourceRegistry
from . import ingest, trust, query, export, eval
from . import crawler, lineage, pipeline, summarize_news, associations, intersection, context
from .web import serve as web

__all__ = [
    "SourceConfig",
    "SourceRegistry",
    "ingest",
    "trust",
    "query",
    "export",
    "eval",
    "crawler",
    "lineage",
    "pipeline",
    "summarize_news",
    "associations",
    "intersection",
    "context",
    "web",
]
