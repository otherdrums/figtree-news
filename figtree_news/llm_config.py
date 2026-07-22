"""External LLM configuration for evaluation and self-tuning.

Loaded from the ``"llm"`` key in ``sources.json``. When ``enabled`` is False
(or the key is absent), all evaluation is skipped — the pipeline runs as before.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    url: str = ""
    model: str = "default"
    timeout: int = 120
    enabled: bool = False
    evaluate_clusters: bool = True
    verify_frame_shifts: bool = True
    review_brief: bool = True
    auto_correct: bool = True
    confirmation_threshold: int = 2
    find_missed_merges: bool = True
    missed_merge_interval: int = 10

    @classmethod
    def from_sources_json(cls, path: str) -> "LLMConfig":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()
        llm = raw.get("llm")
        if not isinstance(llm, dict):
            return cls()
        return cls(
            url=str(llm.get("url", "")),
            model=str(llm.get("model", "default")),
            timeout=int(llm.get("timeout", 120)),
            enabled=bool(llm.get("enabled", False)),
            evaluate_clusters=bool(llm.get("evaluate_clusters", True)),
            verify_frame_shifts=bool(llm.get("verify_frame_shifts", True)),
            review_brief=bool(llm.get("review_brief", True)),
            auto_correct=bool(llm.get("auto_correct", True)),
            confirmation_threshold=int(llm.get("confirmation_threshold", 2)),
            find_missed_merges=bool(llm.get("find_missed_merges", True)),
            missed_merge_interval=int(llm.get("missed_merge_interval", 10)),
        )
