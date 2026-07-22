"""Lineage: who broke it first, who is derivative, and narrative/agenda maps.

Pure CPU analysis over stored figments — no model required. Runs after
ingestion (e.g. at the end of each crawler tick) and persists its findings as
figments so the web UI can query them directly:

* ``narrative:{key}``  — one figment per cluster of articles about the same
  entities (the "story"). Carries the member article ids, the sources, the
  first reporter, and the stance lean.
* ``derivative:{orig}:{der}`` — an edge figment marking ``der`` as published
  later than ``orig`` while covering the same story (an echo / derivative).

Deterministic ids make the whole step idempotent: re-running overwrites the
previous lineage rather than duplicating it.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from figtree import Figment, FigmentStore, Figtree

_ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z]{2,})\b")
_STOP = {
    "The", "This", "That", "These", "Those", "We", "They", "He", "She", "It",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December", "Reuters", "AP", "BBC",
    "Guardian", "NPR", "Aljazeera", "CNN", "NYT", "New York Times",
    "United States", "US", "USA", "U.S.", "U.S.A.", "America", "American",
    "United Kingdom", "UK", "U.K.", "Britain", "British", "England",
    "Canada", "Australian", "China", "Chinese",
    "Russia", "Russian", "Ukraine", "Ukrainian", "Iran", "Iranian",
    "Israel", "Israeli", "Palestinian", "Palestine", "Gaza", "Hamas",
    "President", "Prime", "Minister", "Government", "Official",
    "World", "News", "Report", "Story", "Article", "People", "Some", "Amid",
    "After", "Before", "During", "Following", "Because", "Despite",
    "About", "Over", "Under", "Between", "Against", "Through", "Into",
    "According", "Said", "Says", "Told", "New", "Old", "First", "Last",
    "More", "Most", "Many", "Much", "Several", "One", "Two", "Three",
    "South", "North", "East", "West",
    "Watch", "Video", "Live", "Updated", "Breaking",
}


def _entities(text: str) -> set[str]:
    if not text:
        return set()
    toks = {t.strip(".") for t in _ENTITY_RE.findall(text)}
    return {t for t in toks if t not in _STOP and len(t) > 2}


def _parse_time(fig: Figment) -> datetime | None:
    for key in ("published", "first_seen"):
        raw = fig.meta.get(key)
        if not raw:
            continue
        try:
            if key == "first_seen":
                return datetime.fromisoformat(raw)
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            continue
    return None


def _articles(store: FigmentStore, *, all_figs: list | None = None) -> list[Figment]:
    return [
        f
        for f in (all_figs if all_figs is not None else store.all())
        if f.meta.get("is_image") and f.meta.get("source_id") and not f.is_edge()
    ]


def _cluster(articles: list[Figment]) -> list[list[Figment]]:
    """Group articles by Jaccard entity overlap (>= 0.25) instead of single shared entity."""
    by_id = {f.figment_id: f for f in articles}
    ent_sets = {f.figment_id: _entities(f.text) for f in articles}
    parent = {f.figment_id: f.figment_id for f in articles}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def jaccard(a, b):
        sa, sb = ent_sets[a], ent_sets[b]
        inter = len(sa & sb)
        union_len = len(sa | sb)
        return inter / union_len if union_len else 0.0

    fids = list(ent_sets.keys())
    for i in range(len(fids)):
        for j in range(i + 1, len(fids)):
            if jaccard(fids[i], fids[j]) >= 0.25:
                union(fids[i], fids[j])

    groups: dict[str, list[Figment]] = {}
    for fid in parent:
        groups.setdefault(find(fid), []).append(by_id[fid])
    return [g for g in groups.values() if len(g) >= 2]


def _cluster_by_boundary(store: FigmentStore, articles: list[Figment], threshold: float = 0.80) -> list[list[Figment]]:
    """Group articles by boundary vector similarity (cosine >= threshold).
    
    Uses the store's ANN search to find similar articles for each article,
    then groups them using union-find on similarity edges.
    """
    import numpy as np
    
    by_id = {f.figment_id: f for f in articles}
    parent = {f.figment_id: f.figment_id for f in articles}
    similarity_edges = []

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # For each article, search for similar articles using boundary vector
    total_hits = 0
    max_similarity = 0.0
    for article in articles:
        try:
            # Search store for similar articles (returns list of (figment, distance))
            hits = store.search(article.boundary, k=20)
            total_hits += len(hits)
            for hit_fig, distance in hits:
                if hit_fig.figment_id == article.figment_id:
                    continue
                if hit_fig.figment_id not in by_id:
                    continue
                # LanceDB returns cosine distance, convert to similarity
                # distance = 1 - similarity, so similarity = 1 - distance
                similarity = 1.0 - distance
                max_similarity = max(max_similarity, similarity)
                if similarity >= threshold:
                    union(article.figment_id, hit_fig.figment_id)
                    similarity_edges.append((article.figment_id[:8], hit_fig.figment_id[:8], similarity))
        except Exception as e:
            print(f"[boundary_cluster] search failed for {article.figment_id[:8]}: {e}")
            continue
    
    print(f"[boundary_cluster] total search hits: {total_hits}, max similarity: {max_similarity:.3f}, threshold: {threshold}")

    # Group by connected components
    groups: dict[str, list[Figment]] = {}
    for fid in parent:
        if fid in by_id:
            groups.setdefault(find(fid), []).append(by_id[fid])
    
    clusters = [g for g in groups.values() if len(g) >= 2]
    
    # Log clustering details
    if similarity_edges:
        avg_sim = sum(s for _, _, s in similarity_edges) / len(similarity_edges)
        print(f"[boundary_cluster] found {len(similarity_edges)} similarity edges (avg={avg_sim:.3f})")
        print(f"[boundary_cluster] formed {len(clusters)} clusters from {len(articles)} articles")
        for i, cluster in enumerate(clusters[:5]):
            sources = sorted({f.meta.get('source_id', '?') for f in cluster})
            print(f"[boundary_cluster]   cluster[{i}]: {len(cluster)} articles, sources={sources}")
    
    return clusters


def _cluster_hybrid(store: FigmentStore, articles: list[Figment], 
                   entity_threshold: float = 0.20, boundary_threshold: float = 0.60) -> list[list[Figment]]:
    """Hybrid clustering: combine entity overlap AND boundary similarity.
    
    Two articles are clustered if they have BOTH:
    - Entity Jaccard overlap >= entity_threshold
    - Boundary cosine similarity >= boundary_threshold
    
    This combines the strengths of both approaches.
    """
    import numpy as np
    
    by_id = {f.figment_id: f for f in articles}
    ent_sets = {f.figment_id: _entities(f.text) for f in articles}
    parent = {f.figment_id: f.figment_id for f in articles}
    similarity_edges = []

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def jaccard(a, b):
        sa, sb = ent_sets[a], ent_sets[b]
        inter = len(sa & sb)
        union_len = len(sa | sb)
        return inter / union_len if union_len else 0.0

    # Calculate pairwise similarities
    total_pairs = 0
    hybrid_matches = 0
    max_boundary_sim = 0.0
    
    for i, article in enumerate(articles):
        # Search for similar articles by boundary
        try:
            hits = store.search(article.boundary, k=20)
            for hit_fig, distance in hits:
                if hit_fig.figment_id == article.figment_id:
                    continue
                if hit_fig.figment_id not in by_id:
                    continue
                
                total_pairs += 1
                boundary_sim = 1.0 - distance
                max_boundary_sim = max(max_boundary_sim, boundary_sim)
                
                # Check entity overlap
                entity_overlap = jaccard(article.figment_id, hit_fig.figment_id)
                
                # Hybrid condition: BOTH entity overlap AND boundary similarity must pass
                if entity_overlap >= entity_threshold and boundary_sim >= boundary_threshold:
                    union(article.figment_id, hit_fig.figment_id)
                    similarity_edges.append((article.figment_id[:8], hit_fig.figment_id[:8], 
                                           entity_overlap, boundary_sim))
                    hybrid_matches += 1
        except Exception as e:
            print(f"[hybrid_cluster] search failed for {article.figment_id[:8]}: {e}")
            continue
    
    print(f"[hybrid_cluster] total pairs checked: {total_pairs}, hybrid matches: {hybrid_matches}")
    print(f"[hybrid_cluster] max boundary sim: {max_boundary_sim:.3f}, thresholds: entity={entity_threshold}, boundary={boundary_threshold}")
    
    # Group by connected components
    groups: dict[str, list[Figment]] = {}
    for fid in parent:
        if fid in by_id:
            groups.setdefault(find(fid), []).append(by_id[fid])
    
    clusters = [g for g in groups.values() if len(g) >= 2]
    
    if similarity_edges:
        avg_entity = sum(e for _, _, e, _ in similarity_edges) / len(similarity_edges)
        avg_boundary = sum(b for _, _, _, b in similarity_edges) / len(similarity_edges)
        print(f"[hybrid_cluster] found {len(similarity_edges)} matches (avg entity={avg_entity:.3f}, avg boundary={avg_boundary:.3f})")
        print(f"[hybrid_cluster] formed {len(clusters)} clusters from {len(articles)} articles")
        for i, cluster in enumerate(clusters[:5]):
            sources = sorted({f.meta.get('source_id', '?') for f in cluster})
            print(f"[hybrid_cluster]   cluster[{i}]: {len(cluster)} articles, sources={sources}")
    
    return clusters


def _compute_sentence_boundary_similarity(article1: Figment, article2: Figment) -> float:
    """Compute max sentence-level boundary similarity between two articles.
    
    Returns the maximum cosine similarity between any pair of sentence boundaries.
    This is more granular than article-level comparison.
    """
    import numpy as np
    
    # Extract sentence boundaries (children of article image)
    # For now, we'll use the article's atomic figments if available
    # This is a placeholder - we'd need to store sentence boundaries separately
    # or compute them on-the-fly
    
    # For now, fall back to article-level comparison
    # TODO: Implement proper sentence-level comparison
    a = article1.boundary.astype(np.float64)
    b = article2.boundary.astype(np.float64)
    cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
    return cos_sim


def compute_lineage(store: FigmentStore, max_stories: int = 0) -> dict[str, Any]:
    """Recompute lineage figments from the current store. Idempotent.
    
    Runs entity-based, boundary-based, and hybrid clustering in parallel and compares results.
    Uses hybrid clustering as the primary approach (best of both worlds).
    
    max_stories: if > 0, keep only the top N narratives by member count.
    """
    all_figs = store.all()
    # Purge old narrative/derivative figments so stale clusters don't linger.
    for f in all_figs:
        if f.meta.get("edge_type") in ("narrative", "derivative"):
            store.delete(f.figment_id)

    articles = _articles(store, all_figs=all_figs)
    
    # Run all clustering approaches
    print(f"\n[lineage] Clustering {len(articles)} articles...")
    entity_clusters = _cluster(articles)
    boundary_clusters = _cluster_by_boundary(store, articles, threshold=0.80)
    hybrid_clusters = _cluster_hybrid(store, articles, entity_threshold=0.15, boundary_threshold=0.50)
    
    # Compare results
    print(f"\n[lineage] Comparison:")
    print(f"[lineage]   Entity-based:    {len(entity_clusters)} clusters")
    print(f"[lineage]   Boundary-based:  {len(boundary_clusters)} clusters (threshold=0.80)")
    print(f"[lineage]   Hybrid:          {len(hybrid_clusters)} clusters (entity>=0.20, boundary>=0.60)")
    
    # Use hybrid clustering as primary
    clusters = hybrid_clusters if hybrid_clusters else entity_clusters
    
    figments: list[Figment] = []
    summaries: list[dict[str, Any]] = []

    # Sort clusters by size (largest first) so max_stories keeps the most-covered stories
    clusters.sort(key=lambda g: len(g), reverse=True)
    if max_stories > 0:
        clusters = clusters[:max_stories]

    for group in clusters:
        group = sorted(group, key=lambda f: _parse_time(f) or datetime.max.replace(tzinfo=timezone.utc))
        times = [(f, _parse_time(f)) for f in group]
        first = min(times, key=lambda ft: ft[1] or datetime.max.replace(tzinfo=timezone.utc))[0]
        members = [f.figment_id for f in group]
        sources = sorted({f.meta.get("source_id") for f in group})
        key = hashlib.sha1("|".join(members).encode()).hexdigest()[:12]
        narrative_id = f"narrative:{key}"

        # Mark first reporter + derivatives on the article figments themselves.
        updated_articles: list[Figment] = []
        for f in group:
            if f.figment_id == first.figment_id:
                f.meta["first_reporter"] = True
            else:
                f.meta["derivative_of"] = first.figment_id
                deriv_id = f"deriv:{first.figment_id}:{f.figment_id}"
                figments.append(
                    Figment.create(
                        text=f"{f.meta.get('source_id')} echoed a story first reported by "
                             f"{first.meta.get('source_id')}",
                        boundary=first.boundary.copy(),
                        meta={
                            "edge_type": "derivative",
                            "original": first.figment_id,
                            "original_url": first.meta.get("url"),
                            "derivative": f.figment_id,
                            "derivative_url": f.meta.get("url"),
                        },
                        figment_id=deriv_id,
                        sources=[first.figment_id],
                        children=[f.figment_id],
                    )
                )
            updated_articles.append(f)

        # Use first reporter's headline as the narrative title; fall back to first sentence.
        narrative_title = first.meta.get("title") or first.text.split(".")[0].strip()
        narrative_text = narrative_title

        # Detect frame shift: compare newest article's boundary to first reporter's
        newest = group[-1]
        frame_shift = False
        frame_shift_score = None
        if len(group) >= 2 and newest.figment_id != first.figment_id:
            import numpy as np
            a = first.boundary.astype(np.float64)
            b = newest.boundary.astype(np.float64)
            cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
            frame_shift = cos_sim < 0.85
            frame_shift_score = cos_sim

        narrative = Figment.create(
            text=narrative_text,
            boundary=first.boundary.copy(),
            meta={
                "edge_type": "narrative",
                "title": narrative_title,
                "members": members,
                "sources": sources,
                "first_reporter": first.figment_id,
                "first_reporter_source": first.meta.get("source_id"),
                "first_reporter_url": first.meta.get("url"),
                "entities": sorted(set().union(*[_entities(f.text) for f in group]))[:12],
                "frame_shift": frame_shift,
                "frame_shift_score": frame_shift_score,
                "frame_shift_note": (
                    f"Boundary similarity {frame_shift_score:.2f} < 0.85 threshold "
                    f"(first: {first.meta.get('source_id')}, latest: {newest.meta.get('source_id')})"
                    if frame_shift else ""
                ),
            },
            figment_id=narrative_id,
        )
        figments.append(narrative)
        summaries.append(
            {
                "narrative_id": narrative_id,
                "sources": sources,
                "members": members,
                "first_reporter": first.meta.get("source_id"),
                "first_reporter_url": first.meta.get("url"),
                "size": len(group),
            }
        )

        # Persist article meta updates (first_reporter / derivative_of).
        hidden = group[0].boundary.shape[0]
        store.upsert(updated_articles, hidden_size=hidden)

    if figments:
        hidden = figments[0].boundary.shape[0]
        store.upsert(figments, hidden_size=hidden)

    return {
        "narratives": summaries, 
        "edges": len(figments) - len(summaries),
        "entity_clusters": entity_clusters,
        "hybrid_clusters": hybrid_clusters,
    }


def get_narratives(store: FigmentStore, *, all_figs: list | None = None) -> list[dict[str, Any]]:
    """Read persisted narrative figments for display."""
    out = []
    for f in all_figs if all_figs is not None else store.all():
        if f.meta.get("edge_type") == "narrative":
            out.append(
                {
                    "narrative_id": f.figment_id,
                    "title": f.meta.get("title", ""),
                    "sources": f.meta.get("sources", []),
                    "members": f.meta.get("members", []),
                    "first_reporter": f.meta.get("first_reporter_source"),
                    "first_reporter_url": f.meta.get("first_reporter_url"),
                    "entities": f.meta.get("entities", []),
                    "text": f.text,
                    "frame_shift": f.meta.get("frame_shift", False),
                    "frame_shift_score": f.meta.get("frame_shift_score"),
                    "frame_shift_note": f.meta.get("frame_shift_note", ""),
                }
            )
    return out


def get_derivatives(store: FigmentStore, *, all_figs: list | None = None) -> list[dict[str, Any]]:
    out = []
    for f in all_figs if all_figs is not None else store.all():
        if f.meta.get("edge_type") == "derivative":
            out.append(
                {
                    "original_url": f.meta.get("original_url"),
                    "derivative_url": f.meta.get("derivative_url"),
                    "derivative": f.meta.get("derivative"),
                    "original": f.meta.get("original"),
                }
            )
    return out


def source_agenda(store: FigmentStore, *, all_figs: list | None = None) -> dict[str, dict[str, Any]]:
    """Per-source agenda lean: stories led vs echoed, and trust."""
    figs = all_figs if all_figs is not None else store.all()
    graph = Figtree(figs, store=store)
    analysis = graph.analyze_sources()
    narrs = get_narratives(store, all_figs=figs)
    led = {}
    echoed = {}
    for n in narrs:
        fr = n["first_reporter"]
        led.setdefault(fr, 0)
        led[fr] += 1
        for s in n["sources"]:
            if s != fr:
                echoed.setdefault(s, 0)
                echoed[s] += 1
    agenda = {}
    for src, info in analysis.items():
        agenda[src] = {
            "adjusted_trust": info["adjusted_trust"],
            "base_trust": info["base_trust"],
            "led": led.get(src, 0),
            "echoed": echoed.get(src, 0),
            "contradicting": info["contradicting"],
            "agreeing": info["agreeing"],
        }
    return agenda
