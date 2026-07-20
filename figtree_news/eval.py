"""Evaluation: per-source factual recall, contradictions, and trust shifts.

Re-runs faithful generation over each source's own figments and measures how
completely the source's key claims are re-verbalised (``recall_score``, which
the library guarantees ~1.0 for well-bounded single-pass generation). Also
records the cross-source contradictions and the before/after trust shift
produced by propagation. Output is a JSON report suitable for a dashboard.
"""

from __future__ import annotations

import json
from typing import Any

from figtree import FigmentGenerator, FigmentStore, recall_score
from figtree.recall import extract_atoms

from .trust import get_source_trusts, show_source_trust, update_trust


def evaluate(
    model,
    tokenizer,
    store: FigmentStore,
    source_id: str | None = None,
    max_new_tokens: int = 400,
) -> dict[str, Any]:
    """Evaluate recall per source and report trust/contradiction state."""
    figs = store.all()
    gen = FigmentGenerator(model, tokenizer)

    by_source: dict[str, list] = {}
    for f in figs:
        sid = f.meta.get("source_id")
        if not sid or f.is_edge() or f.is_trust_assertion():
            continue
        by_source.setdefault(sid, []).append(f)

    prior_trust = get_source_trusts(store)
    update_trust(store)
    post_trust = get_source_trusts(store)

    per_source: list[dict[str, Any]] = []
    for sid, sfigs in by_source.items():
        if source_id and sid != source_id:
            continue
        article = next((f for f in sfigs if f.meta.get("is_image")), None)
        source_text = article.text if article else "\n\n".join(f.text for f in sfigs)
        source_atom_count = len(extract_atoms(source_text))
        result = gen.generate_faithful(
            sfigs, "Restate the above verbatim.", max_new_tokens=max_new_tokens
        )
        generated = result.get("text", "")
        per_source.append(
            {
                "source_id": sid,
                "figments": len(sfigs),
                "source_atoms": source_atom_count,
                "recall_score": recall_score(source_text, generated),
                "trust_before": prior_trust.get(sid),
                "trust_after": post_trust.get(sid),
            }
        )

    report = {
        "per_source": per_source,
        "trust_report": show_source_trust(store),
    }
    return report


def write_report(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
