"""Figment decomposition via external LLM with LLM-arbiter role resolution.

Paragraphs are decomposed into WHO/WHAT/WHERE/WHEN/WHY/HOW role figments
using the external 35B LLM with a canonicalization prompt.  A second LLM
call (the arbiter) resolves whether extracted strings refer to the same
real-world entity, using heuristic prefilter scores (containment, Jaccard,
edit similarity, boundary similarity) that are stored as dedup_obs figments
alongside the LLM verdict so thresholds can later be tuned.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import ctypes
import torch
import httpx

from figtree import Figment, FigmentStore, FigmentGenerator

from . import evaluate
from .llm_config import LLMConfig
from .model_lock import model_lock

# ── Constants ────────────────────────────────────────────────────────────

ROLES = ['who', 'what', 'where', 'when', 'why', 'how']

# Concurrency limit for external LLM calls (2 parallel = best throughput)
_DECOMPOSE_SEMAPHORE = threading.Semaphore(2)

# Shared LLM client reference — set by callers so both the pipeline thread
# and background workers share the same semaphore-limited pool.
_llm_client: evaluate.LLMClient | None = None


def set_llm_client(client: evaluate.LLMClient | None) -> None:
    global _llm_client
    _llm_client = client


# ── Prompt (canonicalization + few-shot) ─────────────────────────────────

DECOMPOSE_PROMPT = """You are a precise journalistic entity extractor. Extract the WHO, WHAT, WHERE, WHEN, WHY, and HOW from the paragraph and each sentence below.

Rules:
- **Canonical names**: Use the SHORTEST unambiguous name. "Lindsey Graham" not "the late Sen. Lindsey Graham" or "Sen. Lindsey Graham". "Trump" not "President Donald Trump".
- **No titles**: Strip honorifics (Sen., President, Rep., Dr., Mr., Mrs., Ms., Gov.).
- **No appositives**: "Emillie Boggs Roberts" not "Emillie Boggs Roberts, niece of the late Sen. Lindsey Graham".
- **SINGLE entity only**: NEVER comma-separated lists. Pick ONE primary entity. NOT "Barack Obama, Joe Biden". Just "Barack Obama".
- **WHAT**: Short verb phrase (3-8 words), no grammatical subject. "Delivered eulogy at funeral" not "Trump delivered eulogy".
- **WHERE**: Short canonical place name. "Washington National Cathedral" not "the Washington National Cathedral where the service was held".
- **WHEN**: Specific time reference. "Tuesday" or "July 28, 2026" not "earlier today".
- **WHY/HOW**: Short reason/means (3-8 words). NEVER comma-separated clauses.

Good Example:
Paragraph: "President Donald Trump delivered a moving eulogy for the late Sen. Lindsey Graham at Washington National Cathedral on Tuesday, praising him as a dear friend and mentor."
Sentences:
1. President Donald Trump delivered a moving eulogy for the late Sen. Lindsey Graham.
2. The service was held at Washington National Cathedral on Tuesday.
{
  "paragraph": {
    "who": "Donald Trump",
    "what": "delivered eulogy for Lindsey Graham",
    "where": "Washington National Cathedral",
    "when": "Tuesday",
    "why": "to honor Lindsey Graham",
    "how": "by speaking at the memorial"
  },
  "sentences": [
    {
      "who": "Donald Trump",
      "what": "delivered eulogy for Lindsey Graham",
      "where": "",
      "when": "",
      "why": "",
      "how": ""
    },
    {
      "who": "",
      "what": "service was held",
      "where": "Washington National Cathedral",
      "when": "Tuesday",
      "why": "",
      "how": ""
    }
  ]
}

Paragraph:
{paragraph}

Sentences:
{sentences}

BAD Example (WRONG — comma list in who):
Paragraph: "Sen. Lindsey Graham, President Donald Trump, and Vice President Mike Pence all spoke at the event."
{
  "paragraph": {
    "who": "Lindsey Graham, Donald Trump, Mike Pence",
    "what": "spoke at event",
    "where": "",
    "when": "",
    "why": "",
    "how": ""
  }
}
CORRECTED:
{
  "paragraph": {
    "who": "Lindsey Graham",
    "what": "spoke at event",
    "where": "",
    "when": "",
    "why": "",
    "how": ""
  }
}

Return ONLY a valid JSON object with that exact structure and no other text. Use empty strings for missing roles. Do not include markdown or explanations."""


# ── Arbiter prompt ───────────────────────────────────────────────────────

ARBITER_PROMPT = """You are an entity-resolution assistant. Determine whether two extracted {role} mentions refer to the same real-world entity or concept.

Role: {role}
A: "{text_a}"
B: "{text_b}"

Answer ONLY with "yes" or "no"."""


# ── Normalization ────────────────────────────────────────────────────────

_HONORIFICS = re.compile(
    r'\b(?:sen(?:ator)?s?\b'
    r'|presidents?\b'
    r'|reps?\b|representative\b'
    r'|drs?\b|doctor\b'
    r'|mrs?\b|ms\.?\b|mr\.?\b'
    r'|gov(?:ernor)?s?\b'
    r'|the\s+late\b'
    r'|former\b'
    r'|saint\b|st\.?\b'
    r'|prof(?:essor)?\b'
    r'|capt(?:ain)?\b|gen(?:eral)?\b|lt\.?\b'
    r'|chief\b|deputy\b'
    r'|acting\b|interim\b'
    r'|ambassador\b'
    r'|judge\b'
    r'|attorney\b'
    r'|sheriff\b'
    r'|officer\b'
    r'|detective\b'
    r')\.?\s*', re.I
)


def _normalize_text(text: str) -> str:
    from .normalize import normalize as _norm
    return _norm(text)


# ── Helpers ──────────────────────────────────────────────────────────────

def _force_free():
    try:
        import gc
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect(2)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _log_mem(tag: str = ""):
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    print(f"[mem] {tag}: RSS={rss_kb // 1024}MB", flush=True)
                    return
    except Exception:
        pass


def _rss_mb() -> int:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 0


def _boundary_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    a_f = a.ravel().astype(np.float64)
    b_f = b.ravel().astype(np.float64)
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    return float(np.dot(a_f, b_f) / max(denom, 1e-10))


def _build_decompose_prompt(paragraph_text: str, sentence_texts: list[str]) -> str:
    paragraph_short = paragraph_text[:2000] if len(paragraph_text) > 2000 else paragraph_text
    sentences_short = [s[:400] for s in sentence_texts]
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences_short))
    return DECOMPOSE_PROMPT.replace("{paragraph}", paragraph_short).replace("{sentences}", numbered)


def _extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    def _match_opener(t: str, close_pos: int) -> int:
        depth = 1
        p = close_pos - 1
        while p >= 0 and depth > 0:
            if t[p] == '}':
                depth += 1
            elif t[p] == '{':
                depth -= 1
            p -= 1
        return p + 1 if depth == 0 else -1

    pos = len(text)
    while True:
        brace_end = text.rfind("}", 0, pos)
        if brace_end < 0:
            break
        brace_start = _match_opener(text, brace_end)
        if brace_start < 0:
            break
        candidate = text[brace_start:brace_end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pos = brace_start
    return {}


def _parse_roles(parsed: Any, expected_sentences: int) -> tuple[dict[str, str], list[dict[str, str]]]:
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
    while len(sent_roles) < expected_sentences:
        sent_roles.append({role: "" for role in ROLES})
    sent_roles = sent_roles[:expected_sentences]
    return para_roles, sent_roles


# ── External-LLM extraction ──────────────────────────────────────────────

def _call_llm_retry(client: evaluate.LLMClient, messages: list[dict],
                    max_tokens: int = 1024, temperature: float = 0.0) -> dict[str, Any]:
    """Call the external LLM with repeating retries until success (no fallback)."""
    for attempt in range(30):
        try:
            with _DECOMPOSE_SEMAPHORE:
                result = client.chat(messages, max_tokens=max_tokens, temperature=temperature)
            if result.get("content"):
                return result
            if result.get("error"):
                print(f"[decompose] LLM retry {attempt + 1}: {result['error']}")
        except Exception as exc:
            print(f"[decompose] LLM retry {attempt + 1}: {exc}")
        time.sleep(min(2 ** attempt, 60))
    raise RuntimeError("External LLM unreachable after 30 retries")


def _extract_roles_external(paragraph_text: str, sentence_texts: list[str],
                            llm_client: evaluate.LLMClient) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Extract roles by calling the external LLM with the canonicalization prompt."""
    prompt = _build_decompose_prompt(paragraph_text, sentence_texts)
    messages = [
        {"role": "system", "content": "You extract canonical journalistic roles from text."},
        {"role": "user", "content": prompt},
    ]
    result = _call_llm_retry(llm_client, messages, max_tokens=1024, temperature=0.0)
    parsed = _extract_json(result["content"])
    return _parse_roles(parsed, len(sentence_texts))


# ── Similarity heuristics ────────────────────────────────────────────────

def _tokenize(s: str) -> set[str]:
    return set(re.findall(r'\w+', s.lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _containment_ratio(a: str, b: str) -> float:
    """Fraction of shorter string's tokens contained in the longer."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    shorter = ta if len(ta) <= len(tb) else tb
    longer = tb if len(ta) <= len(tb) else ta
    if not shorter:
        return 0.0
    return len(shorter & longer) / len(shorter)


def _edit_sim(a: str, b: str) -> float:
    """Normalized edit similarity (Levenshtein-based)."""
    a_low, b_low = a.lower(), b.lower()
    n, m = len(a_low), len(b_low)
    if n == 0 and m == 0:
        return 1.0
    if n == 0 or m == 0:
        return 0.0
    # simple DP for short strings
    if n * m > 2500:
        return 0.0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a_low[i - 1] == b_low[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    dist = prev[m]
    return 1.0 - dist / max(n, m)


def _compute_similarities(text_a: str, text_b: str, boundary_a: np.ndarray | None,
                         boundary_b: np.ndarray | None) -> dict[str, float]:
    """Return heuristic similarity scores between two text strings."""
    na, nb = _normalize_text(text_a), _normalize_text(text_b)
    return {
        "containment": _containment_ratio(na, nb),
        "jaccard": _jaccard(na, nb),
        "edit_sim": _edit_sim(na, nb),
        "boundary_sim": _boundary_sim(boundary_a, boundary_b),
    }


# ── LLM-arbiter role resolution ──────────────────────────────────────────

def _resolve_role_figment(
    text: str,
    role: str,
    parent_id: str,
    parent_figment: Figment | None,
    article_id: str,
    store: FigmentStore,
    by_id: dict[str, Figment],
    created: dict[str, Figment],
    llm_client: evaluate.LLMClient | None,
) -> Figment | None:
    """Resolve a role text to an existing or new role figment.

    1. Exact normalized-text match → reuse (no arbiter call).
    2. Gather same-role candidates; prefilter by any heuristic > 0.2.
    3. Call LLM arbiter (yes/no) to decide merge vs create.
    4. Store dedup_obs figment with heuristic scores + verdict.
    """
    global _llm_client
    client = llm_client or _llm_client

    normalized = _normalize_text(text)
    if not normalized:
        return None
    figment_id = hashlib.sha256(f"role:{role}:{normalized}".encode()).hexdigest()[:16]

    # ── 1. Exact text match ────────────────────────────────────────────
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

    parent_boundary = parent_figment.boundary if parent_figment else None

    # ── 2. Gather same-role candidates ─────────────────────────────────
    candidates: list[tuple[Figment, dict[str, float]]] = []
    for existing_candidate in list(created.values()) + list(by_id.values()):
        if (existing_candidate.kind == "role"
                and existing_candidate.meta.get("role") == role
                and existing_candidate.figment_id != figment_id):
            sims = _compute_similarities(
                text, existing_candidate.text,
                parent_boundary, existing_candidate.boundary,
            )
            cand_art = existing_candidate.meta.get('article_id', '')
            is_same_article = cand_art == article_id
            if is_same_article:
                # Intra-article: loose threshold — containment catches entity variants
                passes = any(v > 0.2 for v in sims.values())
            else:
                # Cross-article: require textual overlap (boundary sim alone is too noisy)
                contain = sims.get("containment", 0)
                jac = sims.get("jaccard", 0)
                edit = sims.get("edit_sim", 0)
                passes = contain >= 0.4 or (jac >= 0.25 and edit >= 0.25)
            if passes:
                candidates.append((existing_candidate, sims))
    # Sort by best heuristic score descending
    candidates.sort(key=lambda t: max(t[1].values()), reverse=True)

    # ── 3. LLM arbiter for best candidate ──────────────────────────────
    dedup_obs_list: list[Figment] = []
    best_candidate = candidates[0] if candidates else None

    if best_candidate is not None and client is not None:
        cand, sims = best_candidate
        cand_article = cand.meta.get('article_id', '')
        is_intra_article = cand_article == article_id

        # Skip LLM arbiter for intra-article pairs — use heuristic-only
        # containment >= 0.66 catches "PBS News" in "PBS News Hour" (2/3=1.0)
        if is_intra_article:
            contain = sims.get("containment", 0)
            jac = sims.get("jaccard", 0)
            edit = sims.get("edit_sim", 0)
            merge_intra = contain >= 0.66 or (contain >= 0.45 and jac >= 0.45) or edit >= 0.70
            verdict = "merge" if merge_intra else "keep_separate"
            source = "intra_heuristic"
            reply = ""
        else:
            arbiter_prompt = ARBITER_PROMPT.replace("{role}", role).replace("{text_a}", text).replace("{text_b}", cand.text)
            messages = [
                {"role": "system", "content": "You are an entity-resolution assistant."},
                {"role": "user", "content": arbiter_prompt},
            ]
            result = _call_llm_retry(client, messages, max_tokens=10, temperature=0.0)
            reply = result.get("content", "").strip().lower() if result else "error"
            verdict = "merge" if reply.startswith("yes") else "keep_separate"
            source = "llm"
        if source == "llm":
            print(f"[dedup] {role:4s} \"{text[:45]:45s}\" vs \"{cand.text[:45]:45s}\" → {verdict}")

        hidden = parent_boundary.shape[0] if parent_boundary is not None else 2560

        # ── 4. Store dedup_obs ─────────────────────────────────────────
        obs = _make_dedup_obs(
            kept_id=cand.figment_id,
            candidate_id=figment_id,
            role=role,
            verdict=verdict,
            source=source,
            sims=sims,
            llm_response=reply,
            article_id=article_id,
            hidden_size=hidden,
        )
        dedup_obs_list.append(obs)

        if verdict == "merge":
            refs = cand.meta.get('references', [])
            if parent_id not in refs:
                refs.append(parent_id)
                cand.meta['references'] = refs
                cand.meta['reference_count'] = len(refs)
            for obs in dedup_obs_list:
                _store_dedup_obs(store, obs)
            return cand

    elif best_candidate is not None and client is None:
        # No LLM available — fallback based on heuristics
        cand, sims = best_candidate
        max_sim = max(sims.values())
        verdict = "merge" if max_sim >= 0.75 else "keep_separate"
        hidden = parent_boundary.shape[0] if parent_boundary is not None else 2560
        obs = _make_dedup_obs(
            kept_id=cand.figment_id,
            candidate_id=figment_id,
            role=role,
            verdict=verdict,
            source="no_llm",
            sims=sims,
            llm_response="",
            article_id=article_id,
            hidden_size=hidden,
        )
        dedup_obs_list.append(obs)
        if verdict == "merge":
            refs = cand.meta.get('references', [])
            if parent_id not in refs:
                refs.append(parent_id)
                cand.meta['references'] = refs
                cand.meta['reference_count'] = len(refs)
            for obs in dedup_obs_list:
                _store_dedup_obs(store, obs)
            return cand

    # ── Not merging: create fresh figment ──────────────────────────────
    hidden = parent_boundary.shape[0] if parent_boundary is not None else 2560
    boundary = parent_boundary.copy() if parent_boundary is not None else np.zeros(hidden, dtype=np.float32)

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

    for obs in dedup_obs_list:
        _store_dedup_obs(store, obs)
    return figment


def _make_dedup_obs(
    kept_id: str,
    candidate_id: str,
    role: str,
    verdict: str,
    source: str,
    sims: dict[str, float],
    llm_response: str,
    article_id: str,
    hidden_size: int = 2560,
) -> Figment:
    obs_id = hashlib.sha256(f"dedup_obs:{kept_id}:{candidate_id}:{verdict}".encode()).hexdigest()[:16]
    return Figment.create(
        text=f"dedup {role}: {kept_id[:8]} vs {candidate_id[:8]} -> {verdict}",
        boundary=np.zeros(hidden_size, dtype=np.float32),
        meta={
            "role_figment_a": kept_id,
            "role_figment_b": candidate_id,
            "role": role,
            "verdict": verdict,
            "source": source,
            "containment": sims.get("containment", 0.0),
            "jaccard": sims.get("jaccard", 0.0),
            "edit_sim": sims.get("edit_sim", 0.0),
            "boundary_sim": sims.get("boundary_sim", 0.0),
            "llm_response": llm_response,
            "article_id": article_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        figment_id=obs_id,
        kind="dedup_obs",
    )


def _store_dedup_obs(store: FigmentStore, obs: Figment) -> None:
    """Upsert a single dedup_obs figment."""
    try:
        hidden = obs.boundary.shape[0]
        store.upsert([obs], hidden_size=hidden)
    except Exception as exc:
        print(f"[decompose] Failed to store dedup_obs: {exc}")


# Backward-compat: old name -> new resolver
def _get_or_create_role_figment(
    text: str,
    role: str,
    parent_id: str,
    article_id: str,
    store: FigmentStore,
    by_id: dict[str, Figment],
    created: dict[str, Figment] | None = None,
    boundary_threshold: float = 0.90,
) -> Figment | None:
    """Legacy wrapper — delegates to _resolve_role_figment."""
    parent = by_id.get(parent_id)
    return _resolve_role_figment(
        text, role, parent_id, parent, article_id,
        store, by_id, created or {}, None,
    )


def _create_role_figments(
    role_texts: dict[str, str],
    parent_id: str,
    parent_figment: Figment | None,
    article_id: str,
    by_id: dict[str, Figment],
    store: FigmentStore,
    created: dict[str, Figment],
) -> list[str]:
    """Create or resolve role figments via the LLM arbiter."""
    ids: list[str] = []
    for role, text in role_texts.items():
        if not text:
            continue
        fig = _resolve_role_figment(
            text, role, parent_id, parent_figment,
            article_id, store, by_id, created, None,
        )
        if fig is None:
            continue
        if fig.figment_id not in created:
            created[fig.figment_id] = fig
        else:
            existing = created[fig.figment_id]
            refs = existing.meta.get('references', [])
            if parent_id not in refs:
                refs.append(parent_id)
                existing.meta['references'] = refs
                existing.meta['reference_count'] = len(refs)
        ids.append(fig.figment_id)
    return ids


# ── Decompose one paragraph via external LLM ─────────────────────────────

def _decompose_paragraph_external(
    paragraph: Figment,
    sentences: list[Figment],
    store: FigmentStore,
    created: dict[str, Figment],
    article_id: str,
    by_id: dict[str, Figment],
) -> tuple[list[str], list[list[str]]]:
    """Extract roles from one paragraph via the external LLM."""
    sentence_texts = [s.text for s in sentences if s]
    if not sentence_texts:
        return [], []

    global _llm_client
    client = _llm_client

    para_roles, sent_roles = _extract_roles_external(paragraph.text, sentence_texts, client)

    para_ids = _create_role_figments(para_roles, paragraph.figment_id, paragraph,
                                     article_id, by_id, store, created)
    sent_ids_list: list[list[str]] = []
    for sentence, roles in zip(sentences, sent_roles):
        if sentence is None:
            sent_ids_list.append([])
            continue
        ids = _create_role_figments(roles, sentence.figment_id, sentence,
                                    article_id, by_id, store, created)
        sent_ids_list.append(ids)

    for si, sentence in enumerate(sentences):
        if sentence and not sent_ids_list[si]:
            print(f"[decompose] sentence {sentence.figment_id[:8]} yielded 0 role figments")

    if not any(sent_ids_list):
        print(f"[decompose] WARNING: paragraph {paragraph.figment_id[:8]} has 0 sentence role figments")

    return para_ids, sent_ids_list


# ── Boilerplate filter ───────────────────────────────────────────────────

_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    re.compile(r'subscribe\s+(to|now)', re.I),
    re.compile(r'download\s+the\s.*app', re.I),
    re.compile(r'become\s+a\s.*patriot', re.I),
    re.compile(r'click\s+here', re.I),
    re.compile(r'watch\s+(more|24/7|247)', re.I),
    re.compile(r'sign\s+up', re.I),
    re.compile(r'newsletter', re.I),
    re.compile(r'licensing@', re.I),
    re.compile(r'all rights reserved', re.I),
    re.compile(r'fox news channel \(fnc\)|fnc is', re.I),
    re.compile(r'ms\s+now', re.I),
    re.compile(r'my\s+source\s+for\s+news', re.I),
    re.compile(r'cbs news 24.?7', re.I),
    re.compile(r'available for archive', re.I),
    re.compile(r'by emailing', re.I),
    re.compile(r'©', re.I),
    re.compile(r'be part of it', re.I),
    re.compile(r'touch to listen to cbs news', re.I),
    re.compile(r'follow\s+us', re.I),
    re.compile(r'follow on', re.I),
    re.compile(r'watch full episodes', re.I),
    re.compile(r'read the latest', re.I),
    re.compile(r'(your\s+)?favorite\s+shows?', re.I),
    re.compile(r'available in the', re.I),
    re.compile(r'by texting', re.I),
    re.compile(r'for more information', re.I),
    re.compile(r'visit\s+our\s+website', re.I),
]


def _is_boilerplate(text: str) -> bool:
    lower = text.lower()
    for pat in _BOILERPLATE_PATTERNS:
        if pat.search(lower):
            return True
    url_chars = sum(1 for c in text if c in ':/?#[]@!$&()*+,;=')
    if len(text) > 20 and url_chars / max(len(text), 1) > 0.15:
        return True
    return False


# ── Main decompose_articles ──────────────────────────────────────────────

def decompose_articles(
    model,
    tokenizer,
    store: FigmentStore,
    article_ids: list[str] | None = None,
    llm_client: evaluate.LLMClient | None = None,
) -> dict[str, Any]:
    """Decompose articles into paragraph-level and sentence-level roles.

    When ``llm_client`` is provided, extraction uses the external LLM.
    Otherwise falls back to the local model path.
    """
    global _llm_client
    if llm_client is not None:
        _llm_client = llm_client

    if _llm_client is None and (model is None or tokenizer is None):
        return {"completed": 0, "queued": 0}

    if _llm_client is None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    _log_mem("decompose_articles start")
    all_figs = store.all()
    _log_mem(f"after store.all ({len(all_figs)} figments)")
    by_id = {f.figment_id: f for f in all_figs}
    _log_mem(f"after by_id ({len(by_id)} entries)")

    if article_ids is None:
        article_ids = [
            f.figment_id for f in all_figs
            if f.kind == "article" and f.meta.get("source_id")
            and (not f.meta.get("decomposed") or not f.meta.get("role_figments"))
        ]

    local_gen: FigmentGenerator | None = None
    if _llm_client is None:
        local_gen = FigmentGenerator(model, tokenizer)

    created: dict[str, Figment] = {}
    to_upsert: dict[str, Figment] = {}
    completed = 0

    for article_id in article_ids:
        try:
            article = by_id.get(article_id)
            if article is None or article.kind != "article":
                continue
            if article.meta.get("decomposed") and article.meta.get("role_figments"):
                continue

            _log_mem(f"  before article {article_id[:8]}")
            article_role_ids: list[str] = []

            def _decompose_one_local(pfig: Figment, sfigs: list[Figment]) -> None:
                """Local-model path (kept for backward compat/tests)."""
                pid_roles, sid_roles_list = _decompose_paragraph(
                    pfig, sfigs, local_gen, by_id, store, created, article_id,
                )
                pfig.meta["role_figments"] = pid_roles
                pfig.meta["decomposed"] = True
                to_upsert[pfig.figment_id] = pfig
                article_role_ids.extend(pid_roles)
                for sentence, sid_roles in zip(sfigs, sid_roles_list):
                    if sentence is None:
                        continue
                    sentence.children = list(set(sentence.children + sid_roles))
                    sentence.meta["decomposed"] = True
                    to_upsert[sentence.figment_id] = sentence
                    article_role_ids.extend(sid_roles)

            def _decompose_one_external(pfig: Figment, sfigs: list[Figment]) -> None:
                """External-LLM path."""
                pid_roles, sid_roles_list = _decompose_paragraph_external(
                    pfig, sfigs, store, created, article_id, by_id,
                )
                pfig.meta["role_figments"] = pid_roles
                pfig.meta["decomposed"] = True
                to_upsert[pfig.figment_id] = pfig
                article_role_ids.extend(pid_roles)
                for sentence, sid_roles in zip(sfigs, sid_roles_list):
                    if sentence is None:
                        continue
                    sentence.children = list(set(sentence.children + sid_roles))
                    sentence.meta["decomposed"] = True
                    to_upsert[sentence.figment_id] = sentence
                    article_role_ids.extend(sid_roles)

            decompose_fn = _decompose_one_external if _llm_client is not None else _decompose_one_local

            paragraphs = [
                by_id.get(pid) for pid in article.children
                if by_id.get(pid) and by_id.get(pid).kind == "paragraph"
            ]
            if paragraphs:
                for paragraph in paragraphs:
                    if paragraph is None:
                        continue
                    if _is_boilerplate(paragraph.text):
                        continue
                    sentences = [
                        by_id.get(sid) for sid in paragraph.children
                        if by_id.get(sid) and by_id.get(sid).kind == "sentence"
                    ]
                    if not sentences:
                        continue
                    decompose_fn(paragraph, sentences)
            else:
                for sent in (
                    by_id.get(sid) for sid in article.children
                    if by_id.get(sid) and by_id.get(sid).kind == "sentence"
                ):
                    decompose_fn(sent, [sent])

            article.meta["decomposed"] = True
            article.meta["role_figments"] = list(set(article_role_ids))
            to_upsert[article.figment_id] = article
            completed += 1
            _log_mem(f"  after article {article_id[:8]}")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            _log_mem(f"  after empty_cache {article_id[:8]}")
            print(f"[decompose] Article {article_id[:8]}: {len(article_role_ids)} role figments")
        except Exception as exc:
            print(f"[decompose] ERROR decomposing article {article_id[:8]}: {exc}")
            import traceback
            traceback.print_exc()
            article = by_id.get(article_id)
            if article is not None and article.kind == "article":
                article.meta["decomposed"] = True
                to_upsert[article.figment_id] = article

    # Upsert all created role figments
    if created:
        hidden = next(iter(created.values())).boundary.shape[0]
        store.upsert(list(created.values()), hidden_size=hidden)

    # Upsert updated paragraph/sentence/article figments
    if to_upsert:
        hidden = next(iter(to_upsert.values())).boundary.shape[0]
        store.upsert(list(to_upsert.values()), hidden_size=hidden)

    # Co-occurrence relationships (external LLM path only — creates edges between co-occuring roles)
    all_new_role_ids = [fid for fid in created]
    if all_new_role_ids:
        rels = _cooccurrence_relationships(all_new_role_ids, by_id, store)
        if rels:
            hidden = rels[0].boundary.shape[0]
            store.upsert(rels, hidden_size=hidden)

    _force_free()

    return {"completed": completed, "queued": len(article_ids)}


# ── Co-occurrence relationships ──────────────────────────────────────────

def _cooccurrence_relationships(
    role_ids: list[str],
    by_id: dict[str, Figment],
    store: FigmentStore,
) -> list[Figment]:
    unique_ids = list(set(role_ids))
    if len(unique_ids) < 2:
        return []
    to_upsert: list[Figment] = []
    # by_id already contains ALL figments from the store at the time of
    # snapshot (loaded at the start of decompose_articles), so store.get
    # is redundant here and would be O(n² * k) with k = store round-trip.
    for i, fig1_id in enumerate(unique_ids):
        for fig2_id in unique_ids[i + 1:]:
            pair = tuple(sorted([fig1_id, fig2_id]))
            rel_id = hashlib.sha256(f"rel:{pair[0]}:{pair[1]}".encode()).hexdigest()[:16]
            existing = by_id.get(rel_id)
            if existing:
                existing.meta['weight'] = existing.meta.get('weight', 0) + 1
                to_upsert.append(existing)
            else:
                fig1 = by_id.get(fig1_id)
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
                from .associations import assert_association
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


# ── Legacy local-model path (kept for tests / backward compat) ───────────

def _decompose_paragraph(
    paragraph: Figment,
    sentences: list[Figment],
    gen: FigmentGenerator,
    by_id: dict[str, Figment],
    store: FigmentStore,
    created: dict[str, Figment],
    article_id: str,
) -> tuple[list[str], list[list[str]]]:
    """Use the local model to extract paragraph and sentence roles (legacy)."""
    sentence_texts = [s.text for s in sentences]
    prompt = _build_decompose_prompt(paragraph.text, sentence_texts)
    try:
        with model_lock:
            result = gen.generate(
                figments=[paragraph],
                prompt=prompt,
                max_new_tokens=256,
                temperature=0.0,
                top_k=1,
                top_p=1.0,
                repetition_penalty=1.02,
            )
    except Exception as exc:
        print(f"[decompose] local model generation failed for paragraph {paragraph.figment_id[:8]}: {exc}")
        return [], [[] for _ in sentences]

    try:
        torch.cuda.synchronize()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    text = result.get("generated_text", "")
    parsed = _extract_json(text)
    para_roles, sent_roles = _parse_roles(parsed, len(sentences))

    para_ids = _create_role_figments(para_roles, paragraph.figment_id, paragraph,
                                     article_id, by_id, store, created)
    sent_ids_list: list[list[str]] = []
    for sentence, roles in zip(sentences, sent_roles):
        ids = _create_role_figments(roles, sentence.figment_id, sentence,
                                    article_id, by_id, store, created)
        sent_ids_list.append(ids)

    for si, sentence in enumerate(sentences):
        if not sent_ids_list[si]:
            print(f"[decompose] sentence {sentence.figment_id[:8]} yielded 0 role figments")

    if not any(sent_ids_list):
        print(f"[decompose] WARNING: paragraph {paragraph.figment_id[:8]} has 0 sentence role figments")

    return para_ids, sent_ids_list


# ── DecompositionEngine (background workers) ─────────────────────────────

class DecompositionEngine:
    """Background + synchronous decomposition engine.

    Uses the external LLM when configured, falling back to local model
    for texts that don't need cross-article dedup arbitration.
    """

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
        """Background worker that processes decomposition queue."""
        processed_count = 0
        while self._running:
            try:
                if self.model is None or self.tokenizer is None:
                    await asyncio.sleep(1)
                    continue
                article_id = await self.queue.get()
                print(f"[decompose-{worker_id}] Picked up article {article_id[:8]}, queue={self.queue.qsize()}")
                # VRAM guard
                try:
                    if torch.cuda.is_available():
                        free, total = torch.cuda.mem_get_info()
                        free_mb = free // (1024 * 1024)
                        if free_mb < 200:
                            print(f"[decompose-{worker_id}] VRAM too low ({free_mb}MB free), re-queuing")
                            await self.queue.put(article_id)
                            await asyncio.sleep(5)
                            continue
                except Exception:
                    pass
                rss = _rss_mb()
                if rss > 22500:
                    print(f"[decompose-{worker_id}] RSS too high ({rss}MB), sleeping 30s")
                    await asyncio.sleep(30)
                    await self.queue.put(article_id)
                    continue
                if rss > 0:
                    _log_mem(f"worker-{worker_id} pre")

                def _decompose_with_lock():
                    with model_lock:
                        return decompose_articles(
                            self.model, self.tokenizer, self.store, [article_id],
                        )
                await asyncio.to_thread(_decompose_with_lock)
                _force_free()
                if rss > 0:
                    _log_mem(f"worker-{worker_id} post")
                self._queued.discard(article_id)
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"[decompose-{worker_id}] Progress: {processed_count} processed, sleeping 5s")
                    _force_free()
                    await asyncio.sleep(5)
                else:
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                print(f"[decompose-{worker_id}] Worker cancelled after processing {processed_count} articles")
                break
            except Exception as exc:
                print(f"[decompose-{worker_id}] Error processing article: {exc}")
                import traceback
                traceback.print_exc()

    def decompose_articles_sync(self, article_ids: list[str] | None = None) -> dict[str, Any]:
        """Synchronous entry point (pipeline thread)."""
        if self.model is None or self.tokenizer is None:
            print("[decompose] No local model configured, skipping decomposition")
            return {"completed": 0, "queued": 0}
        with model_lock:
            return decompose_articles(self.model, self.tokenizer, self.store, article_ids)
