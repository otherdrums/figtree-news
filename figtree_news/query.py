"""Query the ingested corpus with credibility-aware retrieval.

Retrieval embeds the query with the same model/layer used during ingestion
(so it lives in the same boundary space as stored figments), pulls the nearest
figments from the LanceDB store, filters them by source trust, then generates
an answer with the FigmentGenerator. ``faithful`` mode uses greedy decoding
for factual recall.
"""

from __future__ import annotations

import torch
from figtree import FigmentGenerator, FigmentStore
from figtree.ingest import detect_crystal_layer

from .trust import get_source_trusts


def embed_query(model, tokenizer, text: str) -> "torch.Tensor":
    """Return a (hidden_size,) boundary vector for ``text`` at the crystal layer."""
    device = model.device
    cl = detect_crystal_layer(model, tokenizer)
    ids = tokenizer.encode(text, add_special_tokens=False) or tokenizer.encode(
        text, add_special_tokens=True
    )
    t = torch.tensor([ids], device=device)
    captured: dict[str, torch.Tensor] = {}

    def hook(_mod, _inp, out):
        o = out[0] if isinstance(out, tuple) else out
        captured["h"] = o.detach()

    handle = model.model.layers[cl].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(t)
    finally:
        handle.remove()
    vec = captured["h"][0, -1, :].float().cpu().numpy().astype("float32")
    return vec


def build_query_figments(
    store: FigmentStore,
    model,
    tokenizer,
    query: str,
    k: int = 8,
    min_trust: float = 0.0,
    trust_lookup: dict[str, float] | None = None,
) -> list:
    """Retrieve the k nearest content figments, filtered by source trust."""
    vec = embed_query(model, tokenizer, query)
    hits = store.search(vec, k=k)
    figs = [f for f, _ in hits]
    if min_trust > 0:
        t = trust_lookup or get_source_trusts(store)
        figs = [
            f
            for f in figs
            if not f.is_edge()
            and not f.is_trust_assertion()
            and t.get(f.meta.get("source_id", ""), 0.0) >= min_trust
        ]
    return figs


def query(
    model,
    tokenizer,
    store: FigmentStore,
    prompt: str,
    k: int = 8,
    min_trust: float = 0.0,
    faithful: bool = False,
    max_new_tokens: int = 200,
    source_tokens: int | None = None,
) -> dict:
    """Retrieve trusted figments and generate an answer."""
    figs = build_query_figments(store, model, tokenizer, prompt, k=k, min_trust=min_trust)
    if not figs:
        return {"text": "", "figments_used": 0, "note": "no figments met the trust/retrieval criteria"}
    gen = FigmentGenerator(model, tokenizer)
    if faithful:
        result = gen.generate_faithful(
            figs, prompt, max_new_tokens=max_new_tokens, source_tokens=source_tokens
        )
    else:
        result = gen.generate(figs, prompt, max_new_tokens=max_new_tokens)
    result["figments_used"] = len(figs)
    return result
