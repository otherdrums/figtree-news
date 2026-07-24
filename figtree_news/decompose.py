"""Figment decomposition: break sentences into structured semantic roles.

Each sentence is decomposed into WHO/WHAT/WHERE/WHEN/WHY/HOW figments,
enabling structured search and natural narrative emergence through figment reuse.
"""

from __future__ import annotations

import asyncio
import json
import re

from figtree import Figment, FigmentStore

from .llm_config import LLMConfig

ROLES = ['who', 'what', 'where', 'when', 'why', 'how']

ROLE_EXTRACTION_PROMPT = """You are a journalistic fact extractor. Break down this sentence into its core components.

Sentence: {sentence}

Extract the following roles (return empty string if not present):
- WHO: People, organizations, or entities involved (max 3 entities, comma-separated)
- WHAT: The action or event that occurred (concise, max 10 words)
- WHERE: Location where the event happened (place name only)
- WHEN: Time/date of the event (as mentioned in text)
- WHY: Reason or cause (if stated)
- HOW: Method or process (if described)

Respond with ONLY valid JSON:
{{
  "who": "",
  "what": "",
  "where": "",
  "when": "",
  "why": "",
  "how": ""
}}"""


class DecompositionEngine:
    """Background engine that decomposes sentences into role figments."""
    
    def __init__(self, llm_config: LLMConfig, store: FigmentStore, num_workers: int = 3):
        self.llm_config = llm_config
        self.store = store
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self._running = False
        self._workers: list[asyncio.Task] = []
        self.num_workers = num_workers  # Number of parallel workers
    
    def start(self):
        """Start background decomposition workers."""
        if not self._running:
            self._running = True
            # Start multiple workers for parallel processing
            for i in range(self.num_workers):
                worker = asyncio.create_task(self._worker_loop(worker_id=i))
                self._workers.append(worker)
            print(f"[decompose] Started {self.num_workers} background workers")
            
            # Queue existing articles that need decomposition
            asyncio.create_task(self._queue_existing_articles())
    
    async def _queue_existing_articles(self):
        """Find and queue existing articles that haven't been decomposed."""
        try:
            all_figs = self.store.all()
            articles = [f for f in all_figs if f.meta.get("is_image") and f.meta.get("source_id") and not f.is_edge()]
            
            needs_decomp = [a for a in articles if not a.meta.get("decomposed")]
            
            if needs_decomp:
                print(f"[decompose] Found {len(needs_decomp)} existing articles needing decomposition")
                for article in needs_decomp:
                    await self.queue_article(article.figment_id)
                print(f"[decompose] Queued {len(needs_decomp)} existing articles")
            else:
                print(f"[decompose] All existing articles already decomposed")
        except Exception as exc:
            print(f"[decompose] Error queueing existing articles: {exc}")
            import traceback
            traceback.print_exc()
    
    def stop(self):
        """Stop background decomposition workers."""
        self._running = False
        for worker in self._workers:
            worker.cancel()
        self._workers.clear()
        print(f"[decompose] All background workers stopped")
    
    async def queue_article(self, article_id: str):
        """Queue an article for decomposition."""
        await self.queue.put(article_id)
    
    async def _worker_loop(self, worker_id: int = 0):
        """Background worker that processes decomposition queue."""
        from .evaluate import LLMClient
        
        if not self.llm_config.url:
            print(f"[decompose-{worker_id}] No LLM URL configured, skipping decomposition")
            return
        
        client = LLMClient(self.llm_config)
        processed_count = 0
        
        while self._running:
            try:
                # Get next item from queue (blocks until available)
                article_id = await self.queue.get()
                print(f"[decompose-{worker_id}] Picked up article {article_id[:8]}, queue={self.queue.qsize()}")
                
                # Process immediately - no artificial delays
                await self._decompose_article(article_id, client)
                processed_count += 1
                
                # Log progress every 10 articles
                if processed_count % 10 == 0:
                    queue_size = self.queue.qsize()
                    print(f"[decompose-{worker_id}] Progress: {processed_count} processed, {queue_size} remaining")
                    
            except asyncio.CancelledError:
                print(f"[decompose-{worker_id}] Worker cancelled after processing {processed_count} articles")
                break
            except Exception as exc:
                print(f"[decompose-{worker_id}] Error processing article: {exc}")
                import traceback
                traceback.print_exc()
    
    async def _decompose_article(self, article_id: str, client):
        """Extract WHO/WHAT/WHERE/WHEN/WHY/HOW from article sentences."""
        article = self.store.get(article_id)
        if not article:
            print(f"[decompose] Article {article_id[:8]} not found in store, skipping")
            return
        
        # Get sentence children
        sentences = [self.store.get(fid) for fid in article.children]
        sentences = [s for s in sentences if s and not s.is_edge()]
        
        if not sentences:
            print(f"[decompose] Article {article_id[:8]} has no sentences, skipping")
            return
        
        role_figment_ids = []
        for sentence in sentences:
            # Skip if already decomposed
            if sentence.meta.get('decomposed'):
                role_figment_ids.extend(sentence.children)
                continue
            
            # Extract roles using LLM
            roles = await self._extract_roles(sentence.text, client)
            if not roles:
                continue
            
            # Create or reuse role figments
            sentence_role_ids = []
            for role in ROLES:
                text = roles.get(role, '').strip()
                if not text:
                    continue
                
                figment = await self._get_or_create_role_figment(
                    text=text,
                    role=role,
                    parent_sentence=sentence.figment_id,
                    article_id=article_id
                )
                sentence_role_ids.append(figment.figment_id)
            
            # Only mark decomposed if we actually created role figments
            if sentence_role_ids:
                sentence.meta['decomposed'] = True
                sentence.children.extend([fid for fid in sentence_role_ids if fid not in sentence.children])
                self.store.upsert([sentence], hidden_size=sentence.boundary.shape[0])
                role_figment_ids.extend(sentence_role_ids)
        
        # Update article with role figment references
        if role_figment_ids:
            article.meta['role_figments'] = list(set(article.meta.get('role_figments', []) + role_figment_ids))
            self.store.upsert([article], hidden_size=article.boundary.shape[0])
            print(f"[decompose] Article {article_id[:8]}: {len(role_figment_ids)} role figments")
    
    async def _extract_roles(self, sentence: str, client) -> dict[str, str]:
        """Use LLM to extract roles from sentence."""
        prompt = ROLE_EXTRACTION_PROMPT.format(sentence=sentence[:500])
        messages = [
            {"role": "system", "content": "You are a journalistic fact extractor. Reply ONLY with a JSON object, no thinking tags."},
            {"role": "user", "content": prompt}
        ]
        
        # Run synchronous LLM call in thread to avoid blocking event loop
        result = await asyncio.to_thread(client.chat_json, messages, 512)
        
        # Try parsed field first (from chat_json)
        parsed = result.get('parsed')
        if parsed is None and 'content' in result:
            # Fallback: try to parse content ourselves, stripping <think>...</think> tags
            import re
            text = result['content']
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                # Try extracting first JSON object
                brace_start = text.find("{")
                brace_end = text.rfind("}")
                if brace_start >= 0 and brace_end > brace_start:
                    try:
                        parsed = json.loads(text[brace_start:brace_end + 1])
                    except json.JSONDecodeError:
                        print(f"[decompose] Failed to parse LLM output: {text[:200]}")
                        return {}
        
        if not isinstance(parsed, dict):
            return {}
        
        return {role: str(parsed.get(role, '')).strip() for role in ROLES}
    
    async def _get_or_create_role_figment(
        self,
        text: str,
        role: str,
        parent_sentence: str,
        article_id: str
    ) -> Figment:
        """Reuse existing figment if semantically similar, else create new."""
        # Generate deterministic ID based on role and normalized text
        import hashlib
        normalized = self._normalize_text(text)
        figment_id = hashlib.sha256(f"role:{role}:{normalized}".encode()).hexdigest()[:16]
        
        # Check if figment already exists
        existing = self.store.get(figment_id)
        if existing:
            # Reuse existing figment, just add reference
            refs = existing.meta.get('references', [])
            if parent_sentence not in refs:
                refs.append(parent_sentence)
                existing.meta['references'] = refs
                existing.meta['reference_count'] = len(refs)
                self.store.upsert([existing], hidden_size=existing.boundary.shape[0])
            return existing
        
        # Create new role figment
        # Use parent sentence boundary (will be refined later)
        parent = self.store.get(parent_sentence)
        boundary = parent.boundary.copy() if parent else None
        
        if boundary is None:
            # Fallback: zero boundary
            import numpy as np
            boundary = np.zeros(2560, dtype=np.float32)
        
        figment = Figment.create(
            text=text,
            boundary=boundary,
            meta={
                'role': role,
                'parent_sentence': parent_sentence,
                'article_id': article_id,
                'references': [parent_sentence],
                'reference_count': 1,
                'normalized': normalized
            },
            figment_id=figment_id
        )
        
        self.store.upsert([figment], hidden_size=boundary.shape[0])
        return figment
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for deduplication."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
