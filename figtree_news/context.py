"""Context materialization: assemble structured context from narrative sets.

Takes the output of an intersection query and builds a provenance-preserving
context package that can be fed directly into a FigmentGenerator.

Preserves source attribution, trust scores, chronological ordering, frame
information, and optional boundary/KV materialization for faithful context."""

from __future__ import annotations

from typing import Any

from figtree import FigmentStore
from figtree.generate import FigmentGenerator


def materialize_context(
    store: FigmentStore,
    narrative_ids: list[str],
    generator: FigmentGenerator | None = None,
    model=None,
    tokenizer=None,
    kv_manager=None,
    include_text: bool = True,
    include_boundaries: bool = False,
    include_kv: bool = False,
    include_paragraphs: bool = False,
    max_articles_per_narrative: int = 10,
    max_paragraphs_per_article: int = 10,
) -> dict[str, Any]:
    """Assemble a structured context package from narrative IDs.

    Parameters
    ----------
    store: FigmentStore
        The LanceDB-backed store.
    narrative_ids: list of narrative figment IDs.
    generator: optional FigmentGenerator for KV materialization.
    model/tokenizer: required if include_kv=True.
    kv_manager: optional KVCacheManager for cached K/V loading.
    include_text: include article text in context.
    include_boundaries: include boundary vectors per article.
    include_kv: include per-article K/V blobs (requires model/tokenizer/kv_manager).
    include_paragraphs: drill down to paragraphs within each article.
    max_articles_per_narrative: cap articles per narrative.
    max_paragraphs_per_article: cap paragraphs per article when include_paragraphs=True.

    Returns
    -------
    dict with:
        - narratives: list of enriched narrative dicts
        - total_articles: int
        - total_sources: int
        - trust_profile: {source_id: adjusted_trust}
        - chronological_order: [article_id, ...]
        - context_text: concatenated article texts for generation
    """
    all_figs = store.all()
    by_id = {f.figment_id: f for f in all_figs}
    narratives_data: list[dict[str, Any]] = []
    all_article_ids: list[str] = []
    source_trusts: dict[str, float] = {}

    for nid in narrative_ids:
        narrative_fig = None
        for f in all_figs:
            if f.figment_id == nid:
                narrative_fig = f
                break
        if narrative_fig is None:
            continue

        members = narrative_fig.meta.get("members", [])
        sources = narrative_fig.meta.get("sources", [])

        # Load articles
        articles_text: list[str] = []
        articles_meta: list[dict[str, Any]] = []

        for mid in members:
            article_fig = None
            for f in all_figs:
                if f.figment_id == mid:
                    article_fig = f
                    break
            if article_fig is None:
                continue

            entry: dict[str, Any] = {
                "article_id": mid,
                "source_id": article_fig.meta.get("source_id", ""),
                "url": article_fig.meta.get("url", ""),
                "published": article_fig.meta.get("published", ""),
                "title": article_fig.meta.get("title", ""),
            }

            if include_text:
                entry["text"] = article_fig.text
                articles_text.append(article_fig.text)

            if include_boundaries:
                entry["boundary"] = article_fig.boundary.tolist()
                entry["boundary_emb"] = (
                    article_fig.boundary_emb.tolist()
                    if article_fig.boundary_emb is not None
                    else None
                )

            # Paragraph drill-down
            if include_paragraphs:
                paragraphs_meta: list[dict[str, Any]] = []
                for pid in article_fig.children:
                    para = by_id.get(pid)
                    if para is None or para.kind != "paragraph":
                        continue
                    para_entry: dict[str, Any] = {
                        "paragraph_id": pid,
                        "text": para.text,
                        "role_figments": para.meta.get("role_figments", []),
                    }
                    if include_boundaries:
                        para_entry["boundary"] = para.boundary.tolist()
                        para_entry["boundary_emb"] = (
                            para.boundary_emb.tolist()
                            if para.boundary_emb is not None
                            else None
                        )
                    paragraphs_meta.append(para_entry)
                entry["paragraphs"] = paragraphs_meta[:max_paragraphs_per_article]

            all_article_ids.append(mid)
            articles_meta.append(entry)

        # Trust info
        source_trusts_local: dict[str, float] = {}
        for src in sources:
            if src not in source_trusts:
                for f in all_figs:
                    if f.meta.get("edge_type") == "trust" and f.meta.get("source_id") == src:
                        source_trusts[src] = f.trust
                        source_trusts_local[src] = f.trust
                        break
            if src in source_trusts:
                source_trusts_local[src] = source_trusts[src]

        # Sort articles chronologically
        articles_meta.sort(
            key=lambda a: a.get("published") or a.get("first_seen") or "",
        )

        narratives_data.append(
            {
                "narrative_id": nid,
                "title": narrative_fig.meta.get("title", "")
                or narrative_fig.text.split(".")[0].strip(),
                "sources": sources,
                "source_trusts": source_trusts_local,
                "articles": articles_meta[:max_articles_per_narrative],
                "first_reporter": narrative_fig.meta.get("first_reporter_source"),
                "first_reporter_url": narrative_fig.meta.get("first_reporter_url"),
                "frame_shift": narrative_fig.meta.get("frame_shift", False),
                "frame_shift_score": narrative_fig.meta.get("frame_shift_score"),
                "entities": narrative_fig.meta.get("entities", []),
                "member_count": len(members),
            }
        )

    # Concatenate context text for generation
    context_text_parts: list[str] = []
    for n in narratives_data:
        if n.get("articles"):
            context_text_parts.append(
                f"--- Narrative: {n['title']} (sources: {', '.join(n['sources'])}) ---"
            )
            for a in n["articles"]:
                if include_text:
                    context_text_parts.append(f"[{a.get('source_id', '?')}] {a.get('text', '')}")

    context_text = "\n\n".join(context_text_parts)

    return {
        "narratives": narratives_data,
        "total_articles": len(all_article_ids),
        "total_sources": len(set(s["source_id"] for n in narratives_data for s in n.get("articles", []))),
        "trust_profile": source_trusts,
        "chronological_order": all_article_ids,
        "context_text": context_text,
    }