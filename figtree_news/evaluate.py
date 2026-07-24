"""External LLM evaluation engine for the self-correcting pipeline.

Runs evaluations synchronously (the pipeline already runs in an ``asyncio.to_thread``
worker, so blocking HTTP calls are fine). Every verdict and correction is persisted
as a first-class figment in the LanceDB store.

The evaluation loop is resumable: it tracks ``evaluation_run`` figments with
``completed`` flags and resumes from the last partial run on restart.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

from figtree import Figment, FigmentStore

from .llm_config import LLMConfig


class LLMClient:
    def __init__(self, config: LLMConfig):
        self._url = config.url.rstrip("/")
        if not self._url.endswith("/v1/chat/completions"):
            if "/v1" not in self._url:
                self._url += "/v1/chat/completions"
        self._model = config.model
        self._timeout = config.timeout

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 512,
             temperature: float = 0.0) -> dict[str, Any]:
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for attempt in range(3):
            try:
                with httpx.Client(timeout=httpx.Timeout(self._timeout)) as client:
                    resp = client.post(self._url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    # Qwen3 puts response in reasoning_content when thinking is enabled
                    if not content:
                        content = data["choices"][0]["message"].get("reasoning_content", "")
                    return {"content": content, "raw": data}
            except Exception as exc:
                if attempt == 2:
                    return {"content": "", "error": str(exc)}
                time.sleep(2 ** attempt)
        return {"content": "", "error": "max retries exceeded"}

    def chat_json(self, messages: list[dict[str, str]], max_tokens: int = 512) -> dict[str, Any]:
        result = self.chat(messages, max_tokens=max_tokens, temperature=0.0)
        if "error" in result:
            return result
        text = result["content"].strip()
        # Strip <think>...</think> tags from Qwen3 models
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract the first JSON object
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start >= 0 and brace_end > brace_start:
                try:
                    parsed = json.loads(text[brace_start:brace_end + 1])
                except json.JSONDecodeError:
                    return {"content": text, "parse_error": str(text[:200])}
            else:
                return {"content": text, "parse_error": str(text[:200])}
        return {"content": text, "parsed": parsed}


# ── Prompt Builders ──────────────────────────────────────────────────────────

def _truncate(text: str, max_chars: int = 2000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."

def _article_blob(article: Figment, index: int) -> str:
    sid = article.meta.get("source_id", "unknown")
    title = article.meta.get("title", "") or article.text[:80]
    published = article.meta.get("published", "") or ""
    url = article.meta.get("url", "") or ""
    return (
        f"[Article {index}] source={sid} title=\"{title}\" published=\"{published}\" url=\"{url}\"\n"
        f"---\n{_truncate(article.text, 2000)}\n"
    )


CLUSTER_SYSTEM = (
    "You are a news narrative analyst. Your job is to determine whether every "
    "article in a group truly belongs to the same news story. Some articles may "
    "have been incorrectly grouped together by an automated system.\n\n"
    "For EACH article that does NOT belong, explain precisely why — is it a "
    "different event? Different location? Different people involved? Different date?\n\n"
    "Respond with ONLY valid JSON. No explanation outside the JSON."
)

def build_cluster_prompt(narrative: dict[str, Any], articles: list[Figment]) -> list[dict]:
    title = narrative.get("title", "Untitled")
    first_reporter = narrative.get("first_reporter", "unknown")
    parts = [
        f"Narrative: \"{title}\"  (first reported by {first_reporter})\n",
        f"Total articles: {len(articles)}\n",
        "--- ARTICLES ---",
    ]
    for i, a in enumerate(articles):
        parts.append(_article_blob(a, i + 1))
    parts.append(
        "\n---\n"
        "Respond with JSON:\n"
        "{\n"
        '  "valid": true/false,\n'
        '  "issues": [\n'
        '    {"article_index": <N>, "source": "<source_id>", "reason": "<why it does not belong>"},\n'
        '    ...\n'
        '  ],\n'
        '  "suggested_boundary_threshold": <optional float, if you think the current similarity threshold needs adjustment>,\n'
        '  "note": "<any additional observation>"\n'
        "}"
    )
    return [
        {"role": "system", "content": CLUSTER_SYSTEM},
        {"role": "user", "content": "\n".join(parts)},
    ]


FRAME_SHIFT_SYSTEM = (
    "You are a media framing analyst. Compare two articles about the same event "
    "and determine whether the framing (angle, emphasis, narrative spin, or "
    "factual emphasis) has genuinely shifted between them. A shift means the "
    "later coverage tells the story differently — different emphasis, different "
    "moral framing, different facts highlighted, or a clear editorial stance "
    "change.\n\n"
    "Respond with ONLY valid JSON."
)

def build_frame_shift_prompt(
    first: Figment, newest: Figment, cos_sim: float, all_sources: list[str]
) -> list[dict]:
    user = (
        f"FIRST COVERAGE (source={first.meta.get('source_id')}, "
        f"published={first.meta.get('published', 'unknown')}):\n"
        f"---\n{_truncate(first.text, 2500)}\n\n"
        f"LATEST COVERAGE (source={newest.meta.get('source_id')}, "
        f"published={newest.meta.get('published', 'unknown')}):\n"
        f"---\n{_truncate(newest.text, 2500)}\n\n"
        f"Boundary cosine similarity (automated): {cos_sim:.3f}\n"
        f"Auto-detection threshold: 0.85\n"
        f"All sources covering this story: {', '.join(all_sources)}\n\n"
        "Respond with JSON:\n"
        "{\n"
        '  "frame_shift": true/false,\n'
        '  "explanation": "<1-2 sentence diagnosis>",\n'
        '  "severity": "minor"|"moderate"|"major",\n'
        '  "suggested_threshold": <optional float if you disagree with 0.85>\n'
        "}"
    )
    return [
        {"role": "system", "content": FRAME_SHIFT_SYSTEM},
        {"role": "user", "content": user},
    ]


BRIEF_SYSTEM = (
    "You are a copy editor reviewing a world news brief. It was generated by "
    "a small language model. Judge its quality.\n\n"
    "Respond with ONLY valid JSON."
)


MERGE_SYSTEM = (
    "You are a news analyst. Check whether any of the following singleton "
    "articles (not currently in any story cluster) actually belong to one of "
    "the existing narrative clusters. Respond with JSON only."
)

def build_merge_prompt(
    singletons: list[tuple[str, Figment]],
    narratives: list[dict[str, Any]],
) -> list[dict]:
    parts = ["EXISTING NARRATIVES:"]
    for n in narratives:
        parts.append(
            f"  [{n['narrative_id'][:8]}] \"{n.get('title', '')[:100]}\" — "
            f"sources: {', '.join(n.get('sources', []))}"
        )

    parts.append("\nSINGLETON ARTICLES (not in any narrative):")
    for i, (fid, f) in enumerate(singletons):
        sid = f.meta.get("source_id", "?")
        title = f.meta.get("title", "") or f.text[:80]
        parts.append(f"  [{i}] {fid[:8]} source={sid} \"{title}\"")

    parts.append(
        "\nRespond with JSON:\n"
        "{\n"
        '  "merges": [\n'
        '    {"singleton_index": <N>, "narrative_id": "<id>", "reason": "<why>"},\n'
        '    ...\n'
        '  ],\n'
        '  "note": ""\n'
        "}"
    )
    return [
        {"role": "system", "content": MERGE_SYSTEM},
        {"role": "user", "content": "\n".join(parts)},
    ]


# ── Evaluation Loop ──────────────────────────────────────────────────────────

def _make_fig_id(seed: str) -> str:
    import hashlib
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def _upsert_correction(
    store: FigmentStore,
    all_figs: list,
    corr_id: str,
    corr_text: str,
    corr_type: str,
    target_narrative: str,
    target_article: str,
    run_id: str,
    reason: str,
) -> Figment:
    """Find existing correction for same target, increment confirmation_count, or create new."""
    for f in all_figs:
        if f.figment_id == corr_id:
            f.meta["confirmation_count"] = f.meta.get("confirmation_count", 1) + 1
            f.meta["eval_run_id"] = run_id
            return f
    return Figment.create(
        text=corr_text,
        boundary=all_figs[0].boundary.copy() if all_figs else None,
        meta={
            "edge_type": "correction",
            "correction_type": corr_type,
            "target_narrative": target_narrative,
            "target_article": target_article,
            "eval_run_id": run_id,
            "reason": reason,
            "applied": False,
            "confirmation_count": 1,
        },
        figment_id=corr_id,
    )


def _find_partial_run(store: FigmentStore) -> Figment | None:
    for f in store.all():
        if f.meta.get("edge_type") == "evaluation_run" and not f.meta.get("completed", False):
            return f
    return None


def _evaluated_narrative_ids(store: FigmentStore, run: Figment | None) -> set[str]:
    if run is None:
        return set()
    evaluated = set()
    for child_id in run.children:
        try:
            child = store.get(child_id)
            if child and child.meta.get("edge_type") == "verdict":
                target = child.meta.get("target_id")
                if target:
                    evaluated.add(target)
        except Exception:
            continue
    return evaluated


def _get_narrative_figment(store: FigmentStore, narrative: dict[str, Any]) -> Figment | None:
    nid = narrative.get("narrative_id")
    if not nid:
        return None
    for f in store.all():
        if f.figment_id == nid:
            return f
    return None


def _article_images(store: FigmentStore) -> list[Figment]:
    return [
        f for f in store.all()
        if f.meta.get("is_image") and f.meta.get("source_id") and not f.is_edge()
    ]


def evaluate_narratives(
    store: FigmentStore,
    client: LLMClient,
    config: LLMConfig,
) -> dict[str, Any]:
    from .lineage import get_narratives

    all_figs = store.all()
    narratives = get_narratives(store, all_figs=all_figs)
    articles_by_id = {f.figment_id: f for f in _article_images(store)}

    # Resumability
    partial_run = _find_partial_run(store)
    if partial_run:
        print(f"[eval] resuming partial eval run {partial_run.figment_id[:8]}")
        run = partial_run
    else:
        run_id = _make_fig_id(f"eval:{time.strftime('%Y-%m-%dT%H:%M:%S')}")
        run = Figment.create(
            text=f"Evaluation run at {time.strftime('%Y-%m-%d %H:%M:%S')}",
            boundary=all_figs[0].boundary.copy() if all_figs else None,
            meta={"edge_type": "evaluation_run", "cycle": 0, "completed": False,
                  "started": time.strftime("%Y-%m-%dT%H:%M:%S")},
            figment_id=run_id,
        )
        hidden = all_figs[0].boundary.shape[0] if all_figs else 2560
        store.upsert([run], hidden_size=hidden)

    already = _evaluated_narrative_ids(store, run)
    pending = [n for n in narratives if n["narrative_id"] not in already]

    print(f"[eval] total_narratives={len(narratives)}  already_evaluated={len(already)}  pending={len(pending)}")

    verdicts: list[Figment] = []
    corrections_suggested = 0

    for narrative in pending:
        nid = narrative["narrative_id"]
        title = narrative.get('title', '')[:60]
        sources = narrative.get('sources', [])
        members = narrative.get('members', [])
        print(f"\n[eval] ── Narrative {nid[:8]}: \"{title}\"")
        print(f"[eval]    sources={sources}  members={len(members)}")

        member_ids = narrative.get("members", [])
        member_articles = [articles_by_id[mid] for mid in member_ids if mid in articles_by_id]

        if len(member_articles) < 2:
            verdict = Figment.create(
                text=f"Narrative {nid[:8]} has <2 members, skipping.",
                boundary=all_figs[0].boundary.copy() if all_figs else None,
                meta={"edge_type": "verdict", "target_id": nid, "verdict": "pass",
                      "verdict_type": "cluster"},
            )
            verdicts.append(verdict)
            continue

        # ── Cluster evaluation ──
        if config.evaluate_clusters:
            messages = build_cluster_prompt(narrative, member_articles)
            result = client.chat_json(messages, max_tokens=1024)

            parsed = result.get("parsed", {})
            valid = parsed.get("valid", True) if isinstance(parsed, dict) else True
            issues = parsed.get("issues", []) if isinstance(parsed, dict) else []

            verdict_text = f"Narrative {nid[:8]}: {'PASS' if valid else 'FAIL'} — "
            verdict_text += f"{len(issues)} issues" if issues else "all articles belong"

            print(f"[eval]    LLM verdict: {'PASS' if valid else 'FAIL'}  issues={len(issues)}")
            if issues:
                for issue in issues[:3]:
                    if isinstance(issue, dict):
                        idx = issue.get("article_index", "?")
                        reason = issue.get("reason", "")[:80]
                        print(f"[eval]      issue[{idx}]: {reason}")

            verdict = Figment.create(
                text=verdict_text,
                boundary=all_figs[0].boundary.copy() if all_figs else None,
                meta={
                    "edge_type": "verdict",
                    "target_id": nid,
                    "verdict": "pass" if valid else "fail",
                    "verdict_type": "cluster",
                    "issues": json.dumps(issues),
                    "llm_response": result.get("content", ""),
                },
            )
            verdicts.append(verdict)

            # Create correction figments for each issue
            for issue in issues:
                if not isinstance(issue, dict):
                    continue
                article_idx = issue.get("article_index")
                if article_idx is None or article_idx < 1 or article_idx > len(member_articles):
                    continue
                target_article = member_articles[article_idx - 1]
                reason = issue.get("reason", "does not belong to this narrative")
                corr_text = (
                    f"Remove article {target_article.figment_id[:8]} from narrative {nid[:8]}: {reason}"
                )
                corr_id = _make_fig_id(f"corr:{nid}:remove:{target_article.figment_id}")
                existing_count = next((f.meta.get("confirmation_count", 1) for f in all_figs if f.figment_id == corr_id), 0)
                correction = _upsert_correction(
                    store, all_figs, corr_id, corr_text, "remove",
                    nid, target_article.figment_id, run.figment_id, reason,
                )
                verdicts.append(correction)
                corrections_suggested += 1
                if existing_count > 0:
                    print(f"[eval]      correction INCREMENTED: article {target_article.figment_id[:8]} (count={existing_count+1})")
                else:
                    print(f"[eval]      correction NEW: article {target_article.figment_id[:8]} (count=1)")

            # Check for LLM-suggested parameter change
            suggested = parsed.get("suggested_boundary_threshold") if isinstance(parsed, dict) else None
            if suggested is not None and isinstance(suggested, (int, float)):
                param_id = _make_fig_id(f"param:boundary_threshold:{suggested:.3f}")
                param = Figment.create(
                    text=f"LLM suggests boundary_threshold={suggested:.3f}",
                    boundary=all_figs[0].boundary.copy() if all_figs else None,
                    meta={
                        "edge_type": "parameter_change",
                        "parameter": "boundary_threshold",
                        "new_value": float(suggested),
                        "eval_run_id": run.figment_id,
                        "source_narrative": nid,
                    },
                    figment_id=param_id,
                )
                verdicts.append(param)

        # ── Frame shift verification ──
        if config.verify_frame_shifts and narrative.get("frame_shift"):
            narrative_fig = _get_narrative_figment(store, narrative)
            if narrative_fig and len(member_articles) >= 2:
                first = member_articles[0]
                newest = member_articles[-1]
                cos_sim = narrative.get("frame_shift_score")
                if cos_sim is not None and first.figment_id != newest.figment_id:
                    messages = build_frame_shift_prompt(
                        first, newest, cos_sim,
                        narrative.get("sources", []),
                    )
                    result = client.chat_json(messages, max_tokens=512)
                    parsed = result.get("parsed", {})

                    is_shift = parsed.get("frame_shift", True) if isinstance(parsed, dict) else True
                    explanation = parsed.get("explanation", "") if isinstance(parsed, dict) else ""
                    severity = parsed.get("severity", "moderate") if isinstance(parsed, dict) else "moderate"

                    shift_verdict = Figment.create(
                        text=f"Frame shift for {nid[:8]}: {'SHIFT' if is_shift else 'NO SHIFT'} ({severity}) — {explanation[:150]}",
                        boundary=all_figs[0].boundary.copy() if all_figs else None,
                        meta={
                            "edge_type": "verdict",
                            "target_id": nid,
                            "verdict": "fail" if is_shift else "pass",
                            "verdict_type": "frame_shift",
                            "is_shift": is_shift,
                            "explanation": explanation,
                            "severity": severity,
                            "cosine_similarity": cos_sim,
                        },
                    )
                    verdicts.append(shift_verdict)

        # ── Persist verdicts for this narrative ──
        if verdicts:
            hidden = all_figs[0].boundary.shape[0] if all_figs else 2560
            store.upsert(verdicts, hidden_size=hidden)
            for v in verdicts:
                run.children.append(v.figment_id)

    # ── Missed merge detection (periodic) ──
    if config.find_missed_merges:
        eval_cycle = run.meta.get("cycle", 0)
        if eval_cycle % max(1, config.missed_merge_interval) == 0:
            # Find singletons
            all_narrative_members = set()
            for n in narratives:
                for mid in n.get("members", []):
                    all_narrative_members.add(mid)

            singleton_ids = [
                fid for fid in articles_by_id
                if fid not in all_narrative_members
            ]
            if len(singleton_ids) >= 2 and len(narratives) >= 1:
                singletons = [(fid, articles_by_id[fid]) for fid in singleton_ids[:40]]
                messages = build_merge_prompt(singletons, narratives[:10])
                result = client.chat_json(messages, max_tokens=1024)
                parsed = result.get("parsed", {})

                merges = parsed.get("merges", []) if isinstance(parsed, dict) else []
                for m in merges:
                    if not isinstance(m, dict):
                        continue
                    si = m.get("singleton_index")
                    target_nid = m.get("narrative_id", "")
                    reason = m.get("reason", "")
                    if si is None or si >= len(singletons) or not target_nid:
                        continue
                    _, singleton_fig = singletons[si]
                    corr_id = _make_fig_id(f"corr:{target_nid}:merge:{singleton_fig.figment_id}")
                    corr_text = f"Merge singleton {singleton_fig.figment_id[:8]} into narrative {target_nid[:8]}: {reason}"
                    correction = _upsert_correction(
                        store, all_figs, corr_id, corr_text, "merge",
                        target_nid, singleton_fig.figment_id, run.figment_id, reason,
                    )
                    verdicts.append(correction)
                    corrections_suggested += 1

    # ── Finalize eval run ──
    run.meta["completed"] = True
    run.meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    hidden = all_figs[0].boundary.shape[0] if all_figs else 2560
    store.upsert([run], hidden_size=hidden)

    print(f"[eval] run complete: {len(verdicts)} verdicts, {corrections_suggested} corrections suggested")
    return {"evaluated": len(pending), "corrections_suggested": corrections_suggested,
            "verdicts_total": len(verdicts)}


def review_brief(
    store: FigmentStore,
    brief_text: str,
    client: LLMClient,
    config: LLMConfig,
) -> dict[str, Any]:
    """Review the world brief for accuracy and conciseness."""
    if not brief_text:
        return {"brief_acceptable": True, "brief_issues": 0}

    # Get the articles used in the brief
    all_figs = store.all()
    articles = [f for f in all_figs if f.meta.get("is_image") and not f.is_edge()]

    prompt = (
        f"You are a news editor reviewing a world brief.\n\n"
        f"BRIEF:\n{brief_text}\n\n"
        f"SOURCES:\n{len(articles)} articles from multiple outlets.\n\n"
        f"Is this brief accurate, concise (2-3 sentences), and representative of the sources?\n"
        f"Respond with JSON: {{\"acceptable\": true/false, \"issues\": [\"issue1\", ...]}}"
    )
    messages = [{"role": "system", "content": "You are a news editor."}, {"role": "user", "content": prompt}]
    result = client.chat_json(messages, max_tokens=512)
    parsed = result.get("parsed", {})

    acceptable = parsed.get("acceptable", True) if isinstance(parsed, dict) else True
    issues = parsed.get("issues", []) if isinstance(parsed, dict) else []

    verdict_text = f"World brief review: {'PASS' if acceptable else 'FAIL'} — {len(issues)} issues"
    verdict = Figment.create(
        text=verdict_text,
        boundary=all_figs[0].boundary.copy() if all_figs else None,
        meta={
            "edge_type": "verdict",
            "target_id": "brief:world",
            "verdict": "pass" if acceptable else "fail",
            "verdict_type": "brief_review",
            "issues": json.dumps(issues),
            "llm_response": result.get("content", ""),
        },
    )
    hidden = all_figs[0].boundary.shape[0] if all_figs else 2560
    store.upsert([verdict], hidden_size=hidden)

    print(f"[eval] brief review: {'PASS' if acceptable else 'FAIL'} — {len(issues)} issues")
    return {"brief_acceptable": acceptable, "brief_issues": len(issues)}


def label_article_pairs(
    articles: list[Figment],
    client: LLMClient,
    max_pairs: int = 20,
) -> list[dict[str, Any]]:
    """Use LLM to label article pairs as same-event or different-event.
    
    Returns a list of labels: [{article1_id, article2_id, same_event: bool, reason: str}]
    """
    import random
    
    if len(articles) < 2:
        return []
    
    # Sample random pairs
    pairs = []
    attempts = 0
    while len(pairs) < max_pairs and attempts < max_pairs * 2:
        attempts += 1
        a1, a2 = random.sample(articles, 2)
        pair_key = tuple(sorted([a1.figment_id, a2.figment_id]))
        if pair_key not in [(p["a1"], p["a2"]) for p in pairs]:
            pairs.append({"a1": pair_key[0], "a2": pair_key[1], "article1": a1, "article2": a2})
    
    labels = []
    for pair in pairs:
        a1, a2 = pair["article1"], pair["article2"]
        
        # Build prompt
        text1 = a1.text[:500] if len(a1.text) > 500 else a1.text
        text2 = a2.text[:500] if len(a2.text) > 500 else a2.text
        src1 = a1.meta.get("source_id", "unknown")
        src2 = a2.meta.get("source_id", "unknown")
        
        prompt = (
            f"Are these two articles about the SAME news event?\n\n"
            f"ARTICLE 1 (source: {src1}):\n{text1}\n\n"
            f"ARTICLE 2 (source: {src2}):\n{text2}\n\n"
            f"Respond with JSON: {{\"same_event\": true/false, \"reason\": \"brief explanation\"}}"
        )
        
        messages = [{"role": "system", "content": "You are a news analyst."}, {"role": "user", "content": prompt}]
        result = client.chat_json(messages, max_tokens=1024)
        parsed = result.get("parsed", {})
        
        if "error" in result:
            print(f"[eval] LLM error labeling pair {a1.figment_id[:8]} vs {a2.figment_id[:8]}: {result['error']}")
            continue
        
        same_event = parsed.get("same_event", False) if isinstance(parsed, dict) else False
        reason = parsed.get("reason", "") if isinstance(parsed, dict) else ""
        
        labels.append({
            "a1": pair["a1"],
            "a2": pair["a2"],
            "same_event": same_event,
            "reason": reason,
        })
    
    print(f"[eval] labeled {len(labels)} article pairs with LLM")
    same_count = sum(1 for l in labels if l["same_event"])
    print(f"[eval]   same-event: {same_count}, different-event: {len(labels) - same_count}")
    
    return labels
