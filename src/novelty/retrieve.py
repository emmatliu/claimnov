from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import config, extract, llm


def _facet_text(f: dict) -> str:
    return f"{f['attribute']}: {f['value']}" if f.get("value") else f["attribute"]


def build_pool(corpus: list[dict], max_workers: int = 8) -> list[dict]:
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        per_paper = list(ex.map(extract.extract_claims, corpus))
    facets = []
    for paper, claims in zip(corpus, per_paper):
        for c in claims:
            facets.append({**c, "paper_id": paper.get("id") or paper.get("title", "?"),
                           "paper_title": paper.get("title", ""),
                           "paper_abstract": paper.get("abstract", "")})
    if not facets:
        return []
    embs = np.array(llm.embed([_facet_text(f) for f in facets]))
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    for f, e in zip(facets, embs):
        f["_emb"] = e
    return facets


def top_k_works(target_facet: dict, pool: list[dict], k: int = config.TOP_K) -> list[dict]:
    cands = [f for f in pool if config.ANY_DIMENSION or f["dimension"] == target_facet["dimension"]]
    if not cands:
        return []
    if k == config.TOP_K:
        k = config.top_k_for(len({f["paper_id"] for f in cands}))
    q = np.array(llm.embed([_facet_text(target_facet)])[0])
    q /= np.linalg.norm(q) + 1e-9
    for f in cands:
        f_score = float(q @ f["_emb"])
        f["_cos"] = f_score
    best_per_paper = {}
    for f in cands:
        prev = best_per_paper.get(f["paper_id"])
        if prev is None or f["_cos"] > prev["_cos"]:
            best_per_paper[f["paper_id"]] = f
    ranked = sorted(best_per_paper.values(), key=lambda f: -f["_cos"])
    return ranked[:k]
