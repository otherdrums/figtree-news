"""Async association worker: background dedup of role figments into canonical nodes.

Runs continuously as an asyncio task.  Each tick scans for unprocessed role
figments, picks same-role candidates by heuristic, and (when an external LLM
is configured) confirms equivalence via the arbiter before merging.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np

from figtree import Figment, FigmentStore

from .associations import merge_role_figments
from .evaluate import LLMClient
from .llm_config import LLMConfig
from .normalize import normalize as _norm

ARBITER_PROMPT = """You are an entity-resolution assistant. Determine whether two extracted {role} mentions refer to the same real-world entity or concept.

Role: {role}
A: "{text_a}"
B: "{text_b}"

Answer ONLY with "yes" or "no"."""


def _tokenize(s: str) -> set[str]:
    import re
    return set(re.findall(r'\w+', s.lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _containment_ratio(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    shorter = ta if len(ta) <= len(tb) else tb
    longer = tb if len(ta) <= len(tb) else ta
    if not shorter:
        return 0.0
    return len(shorter & longer) / len(shorter)


def _edit_sim(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _boundary_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    dot = float(np.dot(a_f, b_f))
    n = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    return dot / n if n > 0 else 0.0


def _compute_similarities(
    text_a: str, boundary_a: np.ndarray,
    text_b: str, boundary_b: np.ndarray,
) -> dict[str, float]:
    na, nb = _norm(text_a), _norm(text_b)
    return {
        "containment": _containment_ratio(na, nb),
        "jaccard": _jaccard(na, nb),
        "edit_sim": _edit_sim(na, nb),
        "boundary_sim": _boundary_sim(boundary_a, boundary_b),
    }


def _candidate_score(sims: dict[str, float]) -> float:
    return max(
        sims.get("boundary_sim", 0),
        sims.get("containment", 0),
        sims.get("edit_sim", 0),
    )


def _passes_prefilter(sims: dict[str, float]) -> bool:
    return (
        sims.get("boundary_sim", 0) >= 0.90
        or sims.get("containment", 0) >= 0.4
        or sims.get("edit_sim", 0) >= 0.7
    )


class AssociationWorker:
    """Background worker that merges confirmed-equivalent role figments.

    Runs an async loop.  Each tick scans for role figments that have not yet
    been processed, finds same-role heuristic candidates, and (when an LLM
    is configured) confirms equivalence via the arbiter before merging.
    """

    def __init__(
        self,
        store: FigmentStore,
        llm_config: LLMConfig | None = None,
        interval: float = 10.0,
        max_concurrent_llm: int = 2,
    ):
        self.store = store
        self._llm_client: LLMClient | None = (
            LLMClient(llm_config) if llm_config and llm_config.enabled else None
        )
        self.interval = interval
        self._semaphore = asyncio.Semaphore(max_concurrent_llm)
        self._processed: set[str] = set()
        self._running = False
        self._task: asyncio.Task | None = None

    def start(self) -> asyncio.Task:
        self._running = True
        self._task = asyncio.create_task(self._run())
        return self._task

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self):
        while self._running:
            try:
                await self._tick()
            except Exception as exc:
                print(f"[assoc_worker] Tick error: {exc}")
            await asyncio.sleep(self.interval)

    async def _tick(self):
        all_f = self.store.all()
        role_figs = [
            f for f in all_f
            if f.kind == "role" and not f.meta.get("is_association")
        ]
        for fig in role_figs:
            if fig.figment_id in self._processed:
                continue
            try:
                await self._process_one(fig)
            except Exception as exc:
                print(f"[assoc_worker] Error processing {fig.figment_id[:8]}: {exc}")
            self._processed.add(fig.figment_id)

    async def _process_one(self, fig: Figment):
        all_f = self.store.all()
        same_role = [
            f for f in all_f
            if f.kind == "role"
            and f.meta.get("role") == fig.meta.get("role")
            and f.figment_id != fig.figment_id
        ]
        if not same_role:
            return

        candidates: list[tuple[Figment, dict[str, float]]] = []
        for other in same_role:
            sims = _compute_similarities(
                fig.text, fig.boundary,
                other.text, other.boundary,
            )
            if _passes_prefilter(sims):
                candidates.append((other, sims))

        if not candidates:
            return

        candidates.sort(key=lambda t: _candidate_score(t[1]), reverse=True)
        best, best_sims = candidates[0]

        # If no LLM, use heuristic-only: merge if max sim >= 0.75
        if self._llm_client is None:
            max_sim = _candidate_score(best_sims)
            if max_sim < 0.75:
                return
            keep_id, remove_id = self._pick_winner(fig, best)
            merge_role_figments(self.store, keep_id, [remove_id])
            self._processed.add(keep_id)
            self._processed.discard(remove_id)
            print(f"[assoc_worker] Heuristic merge: {keep_id[:8]} <- {remove_id[:8]} (sim={max_sim:.2f})")
            return

        # LLM arbiter
        verdict = await self._ask_llm(fig, best)
        if verdict != "yes":
            return

        keep_id, remove_id = self._pick_winner(fig, best)
        merge_role_figments(self.store, keep_id, [remove_id])
        self._processed.add(keep_id)
        self._processed.discard(remove_id)

        role = fig.meta.get("role", "?")
        print(f"[assoc_worker] LLM-confirmed merge: {keep_id[:8]} <- {remove_id[:8]} ({role})")

    async def _ask_llm(self, a: Figment, b: Figment) -> str:
        role = a.meta.get("role", "")
        prompt = (
            ARBITER_PROMPT
            .replace("{role}", role)
            .replace("{text_a}", a.text)
            .replace("{text_b}", b.text)
        )
        messages = [
            {"role": "system", "content": "You are an entity-resolution assistant."},
            {"role": "user", "content": prompt},
        ]

        async def _call():
            return self._llm_client.chat(messages, max_tokens=10, temperature=0.0)

        async with self._semaphore:
            result = await asyncio.to_thread(_call)
        reply = result.get("content", "").strip().lower() if result else ""
        return reply

    def _pick_winner(self, a: Figment, b: Figment) -> tuple[str, str]:
        """Return (keep_id, remove_id) — the more established figment is kept."""
        a_refs = len(a.meta.get("references", []))
        b_refs = len(b.meta.get("references", []))
        if a_refs >= b_refs:
            return a.figment_id, b.figment_id
        return b.figment_id, a.figment_id
