"""Source registry for the news aggregator.

Keeps per-source identity and an *initial* trust that feeds Figtree's
source-based trust model. This is application state, deliberately kept out of
the core library: the library treats trust as figments; this repo decides what
a source's starting credibility is (e.g. Reuters 0.9, a random blog 0.5).
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class SourceConfig:
    source_id: str
    name: str
    base_trust: float = 0.5
    url: str | None = None
    kind: str = "news"  # news | blog | social | official


class SourceRegistry:
    """Loads/saves a small JSON map of source_id -> SourceConfig."""

    def __init__(self, sources: dict[str, SourceConfig], feeds: dict[str, str] = None, seeds: list[str] = None):
        self.sources = sources
        self.feeds = feeds or {}
        self.seeds = seeds or []

    @classmethod
    def load(cls, path: str) -> "SourceRegistry":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except FileNotFoundError:
            return cls({}, {}, [])
        out: dict[str, SourceConfig] = {}
        # Top-level "feeds"/"seeds" keys describe crawling, not sources.
        for sid, spec in raw.items():
            if sid in ("feeds", "seeds") or not isinstance(spec, dict):
                continue
            out[sid] = SourceConfig(
                source_id=sid,
                name=spec.get("name", sid),
                base_trust=float(spec.get("base_trust", 0.5)),
                url=spec.get("url"),
                kind=spec.get("kind", "news"),
            )
        feeds = raw.get("feeds", {}) if isinstance(raw.get("feeds"), dict) else {}
        seeds = raw.get("seeds", []) if isinstance(raw.get("seeds"), list) else []
        return cls(out, feeds, seeds)

    def save(self, path: str) -> None:
        raw = {
            sid: {
                "name": s.name,
                "base_trust": s.base_trust,
                "url": s.url,
                "kind": s.kind,
            }
            for sid, s in self.sources.items()
        }
        if self.feeds:
            raw["feeds"] = self.feeds
        if self.seeds:
            raw["seeds"] = self.seeds
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(raw, fh, indent=2)

    def ensure(self, source_id: str, name: str | None = None, base_trust: float = 0.5) -> SourceConfig:
        if source_id not in self.sources:
            self.sources[source_id] = SourceConfig(
                source_id=source_id, name=name or source_id, base_trust=base_trust
            )
        return self.sources[source_id]

    def base_trust(self, source_id: str, default: float = 0.5) -> float:
        return self.sources.get(source_id, SourceConfig(source_id, source_id, default)).base_trust

    def all(self) -> list[SourceConfig]:
        return list(self.sources.values())
