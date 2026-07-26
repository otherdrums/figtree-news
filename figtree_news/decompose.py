"""Figment decomposition: break paragraphs and sentences into structured semantic roles.

Uses the local model for extraction. Paragraphs get their own paragraph-level
roles (stored in paragraph.meta["role_figments"]), and sentences get sentence-level
roles as children. This creates the intermediate paragraph figment layer that the
rest of the app uses for coarse-to-fine search and narrative clustering.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import Any

import numpy as np
from figtree import Figment, FigmentStore, FigmentGenerator

from .associations import assert_association
from .llm_config import LLMConfig
from .model_lock import model_lock

ROLES = ['who', 'what', 'where', 'when', 'why', 'how']

DECOMPOSE_PROMPT = """You are a precise journalistic fact extractor. Extract the WHO, WHAT, WHERE, WHEN, WHY, and HOW roles from the paragraph and each sentence.

Return ONLY a valid JSON object with this exact structure and no other text:
{
  "paragraph": {"who": "", "what": "", "where": "", "when": "", "why": "", "how": ""},
  "sentences": [
    {"who": "", "what": "", "where": "", "when": "", "why": "", "how": ""},
    ...
  ]
}
Use empty strings for missing roles. Do not include markdown or explanations.

Paragraph:
{paragraph}

Sentences:
{sentences}
"""


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _build_decompose_prompt(paragraph_text: str, sentence_texts: list[str]) -> str:
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentence_texts))
    return DECOMPOSE_PROMPT.replace("{paragraph}", paragraph_text).replace("{sentences}", numbered)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model output, stripping think tags."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass
    return {}


def _parse_roles(parsed: Any, expected_sentences: int) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Normalize the model's JSON output into paragraph and sentence roles."""
    if not isinstance(parsed, dict):
        return {}, []
    para = parsed.get("paragraph") or {}
    para_roles = {role: str(para.get(role, "")).strip() for role in ROLES}
    sents = parsed.get("sentences") or []
    sent_roles = []
    if isinstance(sents, list):
        for s in sents:
            if not isinstance(s, dict):
                s = {}
            sent_roles.append({role: str(s.get(role, "")).strip() for role in ROLES})
    # Pad or truncate to match expected number of sentences
    while len(sent_roles) < expected_sentences:
        sent_roles.append({role: "" for role in ROLES})
    sent_roles = sent_roles[:expected_sentences]
    return para_roles, sent_roles


def _get_or_create_role_figment(
    text: str,
    role: str,
    parent_id: str,
    article_id: str,
    store: FigmentStore,
    by_id: dict[str, Figment],
) -> Figment | None:
    """Reuse existing role figment if semantically identical, else create new."""
    normalized = _normalize_text(text)
    if not normalized:
        return None
    figment_id = hashlib.sha256(f"role:{role}:{normalized}".encode()).hexdigest()[:16]
    existing = by_id.get(figment_id)
    if existing is None:
        existing = store.get(figment_id)
    if existing:
        refs = existing.meta.get('references', [])
        if parent_id not in refs:
            refs.append(parent_id)
            existing.meta['references'] = refs
            existing.meta['reference_count'] = len(refs)
        return existing

    parent = by_id.get(parent_id)
    boundary = parent.boundary.copy() if parent else None
    if boundary is None:
        boundary = np.zeros(2560, dtype=np.float32)

    figment = Figment.create(
        text=text,
        boundary=boundary,
        meta={
            'role': role,
            'parent_id': parent_id,
            'article_id': article_id,
            'references': [parent_id],
            'reference_count': 1,
            'normalized': normalized,
        },
        figment_id=figment_id,
        kind="role",
    )
    return figment


def _create_role_figments(
    role_texts: dict[str, str],
    parent_id: str,
    article_id: str,
    by_id: dict[str, Figment],
    store: FigmentStore,
    created: dict[str, Figment],
) -> list[str]:
    """Create or reuse role figments for the given role->text mapping."""
    ids: list[str] = []
    for role, text in role_texts.items():
        if not text:
            continue
        fig = _get_or_create_role_figment(text, role, parent_id, article_id, store, by_id)
        if fig is None:
            continue
        if fig.figment_id not in created:
            created[fig.figment_id] = fig
        else:
            # Reused existing: merge references
            existing = created[fig.figment_id]
            refs = existing.meta.get('references', [])
            if parent_id not in refs:
                refs.append(parent_id)
                existing.meta['references'] = refs
                existing.meta['reference_count'] = len(refs)
        ids.append(fig.figment_id)
    return ids


def _decompose_paragraph(
    paragraph: Figment,
    sentences: list[Figment],
    gen: FigmentGenerator,
    by_id: dict[str, Figment],
    store: FigmentStore,
    created: dict[str, Figment],
    article_id: str,
) -> tuple[list[str], list[list[str]]]:
    """Use the local model to extract paragraph and sentence roles.

    Returns (paragraph_role_ids, [sentence_role_ids, ...]).
    """
    sentence_texts = [s.text for s in sentences]
    prompt = _build_decompose_prompt(paragraph.text, sentence_texts)
    try:
        with model_lock:
            result = gen.generate(
                figments=[paragraph],
                prompt=prompt,
                max_new_tokens=1024,
                temperature=0.0,
                top_k=1,
                top_p=1.0,
                repetition_penalty=1.02,
            )
    except Exception as exc:
        print(f"[decompose] local model generation failed for paragraph {paragraph.figment_id[:8]}: {exc}")
        return [], [[] for _ in sentences]

    text = result.get("generated_text", "")
    parsed = _extract_json(text)
    para_roles, sent_roles = _parse_roles(parsed, len(sentences))

    para_ids = _create_role_figments(para_roles, paragraph.figment_id, article_id, by_id, store, created)
    sent_ids_list: list[list[str]] = []
    for sentence, roles in zip(sentences, sent_roles):
        ids = _create_role_figments(roles, sentence.figment_id, article_id, by_id, store, created)
        sent_ids_list.append(ids)

    return para_ids, sent_ids_list


def _cooccurrence_relationships(
    role_ids: list[str],
    by_id: dict[str, Figment],
    store: FigmentStore,
) -> list[Figment]:
    """Create or strengthen relationship edges between co-occurring role figments."""
    unique_ids = list(set(role_ids))
    if len(unique_ids) < 2:
        return []
    to_upsert: list[Figment] = []
    for i, fig1_id in enumerate(unique_ids):
        for fig2_id in unique_ids[i + 1:]:
            pair = tuple(sorted([fig1_id, fig2_id]))
            rel_id = hashlib.sha256(f"rel:{pair[0]}:{pair[1]}".encode()).hexdigest()[:16]
            existing = by_id.get(rel_id) or store.get(rel_id)
            if existing:
                existing.meta['weight'] = existing.meta.get('weight', 0) + 1
                to_upsert.append(existing)
            else:
                fig1 = by_id.get(fig1_id) or store.get(fig1_id)
                if fig1:
                    rel = Figment.create(
                        text=f"Relationship: {pair[0][:8]} <-> {pair[1][:8]}",
                        boundary=fig1.boundary.copy(),
                        meta={
                            'edge_type': 'relationship',
                            'figment_a': pair[0],
                            'figment_b': pair[1],
                            'weight': 1,
                        },
                        figment_id=rel_id,
                        kind="edge",
                    )
                    to_upsert.append(rel)
    return to_upsert


def _try_auto_associate(figment: Figment, store: FigmentStore, by_id: dict[str, Figment]) -> None:
    """Propose associations between a new role figment and existing variants."""
    role = figment.meta.get("role", "")
    if not role:
        return
    try:
        same_role = [
            f for f in by_id.values()
            if f.meta.get("role") == role and f.figment_id != figment.figment_id
        ]
        if not same_role:
            return
        from .associations import propose_associations
        proposals = propose_associations(
            store,
            role_figments=[figment] + same_role,
            boundary_threshold=0.90,
            string_overlap_threshold=0.50,
            editsim_threshold=0.85,
            min_co_occurrence=0,
        )
        for p in proposals:
            if p["figment_a_id"] == figment.figment_id or p["figment_b_id"] == figment.figment_id:
                assert_association(
                    store,
                    p["figment_a_id"],
                    p["figment_b_id"],
                    confidence=p["confidence"],
                    evidence="auto_decompose",
                    hidden_size=figment.boundary.shape[0],
                )
    except Exception:
        pass


def decompose_articles(
    model,
    tokenizer,
    store: FigmentStore,
    article_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Decompose articles into paragraph-level and sentence-level roles.

    Uses the local model. If ``article_ids`` is None, decomposes all articles that
    are not already marked decomposed.
    """
    if model is None or tokenizer is None:
        return {"completed": 0, "queued": 0}

    all_figs = store.all()
    by_id = {f.figment_id: f for f in all_figs}

    if article_ids is None:
        article_ids = [
            f.figment_id for f in all_figs
            if f.kind == "article" and f.meta.get("source_id") and not f.meta.get("decomposed")
        ]

    gen = FigmentGenerator(model, tokenizer)

    created: dict[str, Figment] = {}
    to_upsert: dict[str, Figment] = {}
    completed = 0

    for article_id in article_ids:
        article = by_id.get(article_id)
        if article is None or article.kind != "article":
            continue
        if article.meta.get("decomposed"):
            continue

        paragraphs = [
            by_id.get(pid) for pid in article.children
            if by_id.get(pid) and by_id.get(pid).kind == "paragraph"
        ]
        if not paragraphs:
            # Legacy fallback: article children are sentences
            paragraphs = [
                by_id.get(sid) for sid in article.children
                if by_id.get(sid) and by_id.get(sid).kind == "sentence"
            ]

        article_role_ids: list[str] = []
        for paragraph in paragraphs:
            if paragraph is None:
                continue
            sentences = [
                by_id.get(sid) for sid in paragraph.children
                if by_id.get(sid) and by_id.get(sid).kind == "sentence"
            ]
            if not sentences:
                continue

            para_role_ids, sent_role_ids_list = _decompose_paragraph(
                paragraph, sentences, gen, by_id, store, created, article_id
            )

            # Paragraph-level roles
            paragraph.meta["role_figments"] = para_role_ids
            paragraph.meta["decomposed"] = True
            to_upsert[paragraph.figment_id] = paragraph
            article_role_ids.extend(para_role_ids)

            # Sentence-level roles
            for sentence, sent_role_ids in zip(sentences, sent_role_ids_list):
                if sentence is None:
                    continue
                sentence.children = list(set(sentence.children + sent_role_ids))
                sentence.meta["decomposed"] = True
                to_upsert[sentence.figment_id] = sentence
                article_role_ids.extend(sent_role_ids)

        # Mark article decomposed and aggregate all role ids
        article.meta["decomposed"] = True
        article.meta["role_figments"] = list(set(
            article.meta.get("role_figments", []) + article_role_ids
        ))
        to_upsert[article.figment_id] = article
        completed += 1
        print(f"[decompose] Article {article_id[:8]}: {len(article_role_ids)} role figments")

    # Upsert all created role figments
    if created:
        for fig in created.values():
            _try_auto_associate(fig, store, by_id)
        hidden = next(iter(created.values())).boundary.shape[0]
        store.upsert(list(created.values()), hidden_size=hidden)

    # Upsert updated paragraph/sentence/article figments
    if to_upsert:
        hidden = next(iter(to_upsert.values())).boundary.shape[0]
        store.upsert(list(to_upsert.values()), hidden_size=hidden)

    # Co-occurrence relationships across all new roles
    all_new_role_ids = [fid for fid in created]
    if all_new_role_ids:
        rels = _cooccurrence_relationships(all_new_role_ids, by_id, store)
        if rels:
            hidden = rels[0].boundary.shape[0]
            store.upsert(rels, hidden_size=hidden)

    return {"completed": completed, "queued": len(article_ids)}


class DecompositionEngine:
    """Background + synchronous decomposition engine using the local model."""

    def __init__(
        self,
        llm_config: LLMConfig,
        store: FigmentStore,
        model=None,
        tokenizer=None,
        num_workers: int = 3,
    ):
        self.llm_config = llm_config
        self.store = store
        self.model = model
        self.tokenizer = tokenizer
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self._running = False
        self._workers: list[asyncio.Task] = []
        self.num_workers = num_workers
        self._queued: set[str] = set()

    def start(self):
        """Start background decomposition workers."""
        if not self._running:
            self._running = True
            self._queued.clear()
            for i in range(self.num_workers):
                worker = asyncio.create_task(self._worker_loop(worker_id=i))
                self._workers.append(worker)
            print(f"[decompose] Started {self.num_workers} background workers")
            asyncio.create_task(self._queue_existing_articles())

    def stop(self):
        """Stop background decomposition workers."""
        self._running = False
        self._queued.clear()
        for worker in self._workers:
            worker.cancel()
        self._workers.clear()
        print("[decompose] All background workers stopped")

    async def queue_article(self, article_id: str):
        """Queue an article for background decomposition."""
        if article_id in self._queued:
            return
        self._queued.add(article_id)
        await self.queue.put(article_id)

    async def _queue_existing_articles(self):
        """Queue existing articles that need decomposition."""
        try:
            await asyncio.sleep(2)
            all_figs = self.store.all()
            needs = [
                f.figment_id for f in all_figs
                if f.kind == "article" and f.meta.get("source_id") and not f.meta.get("decomposed")
            ]
            if needs:
                print(f"[decompose] Found {len(needs)} existing articles needing decomposition")
                for aid in needs:
                    await self.queue_article(aid)
                print(f"[decompose] Queued {len(needs)} existing articles for background processing")
            else:
                print("[decompose] All existing articles already decomposed")
        except Exception as exc:
            print(f"[decompose] Error queueing existing articles: {exc}")
            import traceback
            traceback.print_exc()

    async def _worker_loop(self, worker_id: int = 0):
        """Background worker that processes decomposition queue.

        Waits for the local model to be bound before processing items.
        """
        processed_count = 0
        while self._running:
            try:
                if self.model is None or self.tokenizer is None:
                    await asyncio.sleep(1)
                    continue
                article_id = await self.queue.get()
                print(f"[decompose-{worker_id}] Picked up article {article_id[:8]}, queue={self.queue.qsize()}")
                await asyncio.to_thread(
                    decompose_articles,
                    self.model,
                    self.tokenizer,
                    self.store,
                    [article_id],
                )
                self._queued.discard(article_id)
                processed_count += 1
                await asyncio.sleep(0.01)
                if processed_count % 10 == 0:
                    print(f"[decompose-{worker_id}] Progress: {processed_count} processed")
            except asyncio.CancelledError:
                print(f"[decompose-{worker_id}] Worker cancelled after processing {processed_count} articles")
                break
            except Exception as exc:
                print(f"[decompose-{worker_id}] Error processing article: {exc}")
                import traceback
                traceback.print_exc()

    def decompose_articles_sync(self, article_ids: list[str] | None = None) -> dict[str, Any]:
        """Synchronous entry point for the pipeline thread."""
        if self.model is None or self.tokenizer is None:
            print("[decompose] No local model configured, skipping decomposition")
            return {"completed": 0, "queued": 0}
        return decompose_articles(self.model, self.tokenizer, self.store, article_ids)
