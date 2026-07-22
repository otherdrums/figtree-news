"""Cogitation engine: background consolidation and link discovery.

Periodically reviews figments to:
1. Find and merge duplicate figments (semantic deduplication)
2. Discover new relationships through co-occurrence patterns
3. Strengthen relationship weights
4. Prune unused or weak connections
5. Generate insights about patterns and trends

This is the system's "dreaming" phase - it processes what it learned
during ingestion and consolidates knowledge.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

import numpy as np

from figtree import Figment, FigmentStore

from .decompose import ROLES
from .llm_config import LLMConfig


class CogitationEngine:
    """Background engine for periodic consolidation and insight generation."""
    
    def __init__(self, llm_config: LLMConfig, store: FigmentStore, interval_hours: int = 6):
        self.llm_config = llm_config
        self.store = store
        self.interval_hours = interval_hours
        self._running = False
        self._task: asyncio.Task | None = None
    
    def start(self):
        """Start background cogitation worker."""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._consolidation_loop())
            print(f"[cogitate] Background worker started (interval={self.interval_hours}h)")
    
    def stop(self):
        """Stop background cogitation worker."""
        self._running = False
        if self._task:
            self._task.cancel()
            print(f"[cogitate] Background worker stopped")
    
    async def _consolidation_loop(self):
        """Run consolidation periodically."""
        while self._running:
            try:
                await asyncio.sleep(self.interval_hours * 3600)
                await self.consolidate()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                print(f"[cogitate] Error in consolidation: {exc}")
                import traceback
                traceback.print_exc()
    
    async def consolidate(self):
        """Main consolidation routine."""
        print(f"\n[cogitate] Starting consolidation phase...")
        
        # Step 1: Find and merge duplicate figments
        merged = await self.merge_duplicates()
        print(f"[cogitate]   Merged {merged} duplicate figments")
        
        # Step 2: Discover new relationships
        relationships = await self.discover_links()
        print(f"[cogitate]   Discovered {relationships} new relationships")
        
        # Step 3: Strengthen co-occurrence patterns
        strengthened = await self.strengthen_patterns()
        print(f"[cogitate]   Strengthened {strengthened} patterns")
        
        # Step 4: Prune unused figments
        pruned = await self.prune_weak_links()
        print(f"[cogitate]   Pruned {pruned} weak links")
        
        # Step 5: Generate insights
        insights = await self.generate_insights()
        print(f"[cogitate]   Generated {insights} insights")
        
        print(f"[cogitate] Consolidation complete\n")
    
    async def merge_duplicates(self) -> int:
        """Find figments with similar boundaries and merge them."""
        all_figments = self.store.all()
        role_groups = {}
        
        # Group figments by role
        for fig in all_figments:
            role = fig.meta.get('role')
            if role:
                role_groups.setdefault(role, []).append(fig)
        
        merged_count = 0
        
        for role, figments in role_groups.items():
            # Find clusters of similar figments
            clusters = self._cluster_by_boundary(figments, threshold=0.95)
            
            for cluster in clusters:
                if len(cluster) > 1:
                    # Merge cluster into canonical figment (first one)
                    canonical = cluster[0]
                    for duplicate in cluster[1:]:
                        await self._merge_figments(canonical, duplicate)
                        merged_count += 1
        
        return merged_count
    
    def _cluster_by_boundary(self, figments: list[Figment], threshold: float) -> list[list[Figment]]:
        """Cluster figments by boundary similarity using union-find."""
        if not figments:
            return []
        
        parent = {f.figment_id: f.figment_id for f in figments}
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        
        # Compare all pairs
        for i in range(len(figments)):
            for j in range(i + 1, len(figments)):
                sim = self._boundary_similarity(figments[i].boundary, figments[j].boundary)
                if sim >= threshold:
                    union(figments[i].figment_id, figments[j].figment_id)
        
        # Group by root
        clusters = {}
        for fig in figments:
            root = find(fig.figment_id)
            clusters.setdefault(root, []).append(fig)
        
        return list(clusters.values())
    
    def _boundary_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between boundaries."""
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
    
    async def _merge_figments(self, canonical: Figment, duplicate: Figment):
        """Merge duplicate figment into canonical."""
        # Combine references
        refs = canonical.meta.get('references', [])
        dup_refs = duplicate.meta.get('references', [])
        combined_refs = list(set(refs + dup_refs))
        
        # Update canonical
        canonical.meta['references'] = combined_refs
        canonical.meta['reference_count'] = len(combined_refs)
        canonical.meta['merged_from'] = canonical.meta.get('merged_from', []) + [duplicate.figment_id]
        
        # Update all articles that reference the duplicate
        for ref in dup_refs:
            # Find parent sentence and update its children
            sentence = self.store.get(ref)
            if sentence:
                children = sentence.children
                if duplicate.figment_id in children:
                    children.remove(duplicate.figment_id)
                    if canonical.figment_id not in children:
                        children.append(canonical.figment_id)
                    sentence.children = children
                    self.store.upsert([sentence], hidden_size=sentence.boundary.shape[0])
            
            # Find parent article and update role_figments
            article = self.store.get(ref)
            if article and article.meta.get('role_figments'):
                role_figs = article.meta['role_figments']
                if duplicate.figment_id in role_figs:
                    role_figs.remove(duplicate.figment_id)
                    if canonical.figment_id not in role_figs:
                        role_figs.append(canonical.figment_id)
                    article.meta['role_figments'] = role_figs
                    self.store.upsert([article], hidden_size=article.boundary.shape[0])
        
        # Upsert canonical
        self.store.upsert([canonical], hidden_size=canonical.boundary.shape[0])
        
        # Delete duplicate
        self.store.delete(duplicate.figment_id)
        
        print(f"[cogitate]   Merged {duplicate.figment_id[:8]} into {canonical.figment_id[:8]}")
    
    async def discover_links(self) -> int:
        """Find figments that co-occur and create relationships."""
        cooccurrence = {}
        
        for article in self.store.all():
            if not article.meta.get('is_image'):
                continue
            
            role_figment_ids = article.meta.get('role_figments', [])
            if len(role_figment_ids) < 2:
                continue
            
            # For each pair of figments in this article
            for i, fig1_id in enumerate(role_figment_ids):
                for fig2_id in role_figment_ids[i+1:]:
                    pair = tuple(sorted([fig1_id, fig2_id]))
                    cooccurrence[pair] = cooccurrence.get(pair, 0) + 1
        
        # Create relationships for frequent co-occurrences
        relationship_count = 0
        for (fig1_id, fig2_id), count in cooccurrence.items():
            if count >= 3:  # Threshold for relationship
                # Check if relationship already exists
                rel_id = hashlib.sha256(f"rel:{fig1_id}:{fig2_id}".encode()).hexdigest()[:16]
                existing = self.store.get(rel_id)
                
                if existing:
                    # Strengthen existing relationship
                    existing.meta['weight'] = existing.meta.get('weight', 1) + count
                    self.store.upsert([existing], hidden_size=existing.boundary.shape[0])
                else:
                    # Create new relationship
                    fig1 = self.store.get(fig1_id)
                    if fig1:
                        rel = Figment.create(
                            text=f"Relationship: {fig1_id[:8]} <-> {fig2_id[:8]}",
                            boundary=fig1.boundary.copy(),
                            meta={
                                'edge_type': 'relationship',
                                'figment_a': fig1_id,
                                'figment_b': fig2_id,
                                'weight': count
                            },
                            figment_id=rel_id
                        )
                        self.store.upsert([rel], hidden_size=fig1.boundary.shape[0])
                        relationship_count += 1
        
        return relationship_count
    
    async def strengthen_patterns(self) -> int:
        """Strengthen co-occurrence patterns."""
        # This is a placeholder for more sophisticated pattern detection
        # For now, we just count co-occurrences in discover_links
        return 0
    
    async def prune_weak_links(self) -> int:
        """Remove unused or weak figments."""
        pruned = 0
        
        for fig in self.store.all():
            if not fig.meta.get('role'):
                continue
            
            ref_count = fig.meta.get('reference_count', 0)
            
            # Prune figments with no references
            if ref_count == 0:
                self.store.delete(fig.figment_id)
                pruned += 1
        
        return pruned
    
    async def generate_insights(self) -> int:
        """Use LLM to analyze patterns and generate insights."""
        if not self.llm_config.url:
            return 0
        
        from .evaluate import LLMClient
        client = LLMClient(self.llm_config)
        
        # Get high-level statistics
        all_figments = self.store.all()
        role_figments = [f for f in all_figments if f.meta.get('role')]
        
        stats = {
            'total_role_figments': len(role_figments),
            'most_referenced': self._get_most_referenced(role_figments, limit=10),
            'cooccurrence_patterns': self._get_top_cooccurrences(limit=5)
        }
        
        # Ask LLM for insights
        prompt = f"""Analyze these patterns and generate 3-5 insights about the news landscape:

Statistics:
{stats}

Respond with ONLY valid JSON:
{{
  "insights": [
    {{"text": "Insight 1", "confidence": 0.85}},
    {{"text": "Insight 2", "confidence": 0.75}}
  ]
}}"""
        
        messages = [
            {"role": "system", "content": "You are a news analyst."},
            {"role": "user", "content": prompt}
        ]
        
        result = client.chat_json(messages, max_tokens=1024)
        parsed = result.get('parsed', {})
        
        insights = parsed.get('insights', [])
        insight_count = 0
        
        for insight in insights:
            if isinstance(insight, dict) and 'text' in insight:
                insight_id = hashlib.sha256(f"insight:{insight['text'][:50]}".encode()).hexdigest()[:16]
                
                # Check if insight already exists
                existing = self.store.get(insight_id)
                if not existing:
                    # Create insight figment
                    if role_figments:
                        boundary = role_figments[0].boundary.copy()
                    else:
                        boundary = np.zeros(2560, dtype=np.float32)
                    
                    insight_fig = Figment.create(
                        text=insight['text'],
                        boundary=boundary,
                        meta={
                            'edge_type': 'insight',
                            'confidence': insight.get('confidence', 0.5),
                            'generated_at': str(asyncio.get_event_loop().time())
                        },
                        figment_id=insight_id
                    )
                    self.store.upsert([insight_fig], hidden_size=boundary.shape[0])
                    insight_count += 1
        
        return insight_count
    
    def _get_most_referenced(self, role_figments: list[Figment], limit: int) -> list[dict]:
        """Get most referenced role figments."""
        sorted_figs = sorted(
            role_figments,
            key=lambda f: f.meta.get('reference_count', 0),
            reverse=True
        )
        
        return [
            {
                'figment_id': f.figment_id[:8],
                'role': f.meta.get('role'),
                'text': f.text[:50],
                'reference_count': f.meta.get('reference_count', 0)
            }
            for f in sorted_figs[:limit]
        ]
    
    def _get_top_cooccurrences(self, limit: int) -> list[dict]:
        """Get top co-occurrence relationships."""
        all_figs = self.store.all()
        relationships = [f for f in all_figs if f.meta.get('edge_type') == 'relationship']
        
        sorted_rels = sorted(
            relationships,
            key=lambda f: f.meta.get('weight', 0),
            reverse=True
        )
        
        return [
            {
                'figment_a': r.meta.get('figment_a', '')[:8],
                'figment_b': r.meta.get('figment_b', '')[:8],
                'weight': r.meta.get('weight', 0)
            }
            for r in sorted_rels[:limit]
        ]
