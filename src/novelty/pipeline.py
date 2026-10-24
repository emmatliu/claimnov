from . import aggregate, config, entail, extract, llm, retrieve


def score_paper(target: dict, pool: list[dict], k: int = config.TOP_K,
                alpha: float = config.ALPHA) -> dict:
    claims = extract.extract_claims(target)
    supports = extract.count_supports(target, claims)
    paras = target.get("paragraphs") or []

    claim_reports = []
    for i, c in enumerate(claims):
        c = {**c, "paper_abstract": target.get("abstract", "")}
        if supports:
            c["support_passages"] = [paras[j] for j in
                                     supports[i]["idx"][:config.SUPPORT_CTX_N]]
        cands = retrieve.top_k_works(c, pool, k=k)
        judged = entail.judge_all(c, cands)
        n_retrieved = len(cands)
        sig = aggregate.claim_signals([j["e"] for j in judged], alpha=alpha, k=k)
        claim_reports.append({
            "claim": c["claim"],
            "dimension": c["dimension"],
            "attribute": c["attribute"],
            "value": c["value"],
            "n_supports": supports[i]["n"] if supports else None,
            "anticipation": round(sig["anticipation"], 4),
            "max_e": round(sig["max_e"], 4),
            "max_entailment_level": judged[0]["level"] if judged else None,
            "breadth": round(sig["breadth"], 4),
            "candidates_retrieved": n_retrieved,
            "candidates_judged": len(judged),
            "top_matches": [{
                "paper_id": j["paper_id"],
                "paper_title": j["paper_title"],
                "claim": j["claim"],
                "cosine": round(j.get("_cos", 0.0), 4),
                "entailment_level": j["level"],
                "e": j["e"],
            } for j in judged[:3]],
            "judged": [{
                "paper_id": j["paper_id"],
                "paper_title": j["paper_title"],
                "claim": j["claim"],
                "entailment_level": j["level"],
                "e": j["e"],
            } for j in judged],
        })

    scores = aggregate.paper_novelty(claim_reports)
    n_with = sum(1 for c in claim_reports if c["candidates_judged"] > 0)
    search_status = {
        "claims_extracted": len(claims),
        "claims_with_candidates": n_with,
        "resolved": bool(claims) and n_with > 0,
    }
    return {
        "search_status": search_status,
        "target_id": target.get("id") or target.get("title", "?"),
        "target_title": target.get("title", ""),
        "n_claims": len(claims),
        **scores,
        "claims": claim_reports,
    }


def score_corpus(targets: list[dict], corpus: list[dict],
                 k: int = config.TOP_K, alpha: float = config.ALPHA) -> dict:
    pool = retrieve.build_pool(corpus)
    reports = []
    for t in targets:
        tid = t.get("id") or t.get("title", "?")
        own = [f for f in pool if f["paper_id"] != tid]
        reports.append(score_paper(t, own, k=k, alpha=alpha))
    return {
        "pool_papers": len({f["paper_id"] for f in pool}),
        "pool_facets": len(pool),
        "cost_usd": round(llm.cost_usd(), 4),
        "reports": reports,
    }
