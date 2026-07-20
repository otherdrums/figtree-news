"""figtree-news: a source-aware news aggregator built on Figtree figments."""

from .config import SourceConfig, SourceRegistry
from . import ingest, trust, query, export, eval

__all__ = ["SourceConfig", "SourceRegistry", "ingest", "trust", "query", "export", "eval"]
