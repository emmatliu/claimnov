import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

def _ensure_key():
    if not os.environ.get("OPENAI_API_KEY"):
        for line in open(".env"):
            if line.startswith("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()

def module_gate(idea, pool_records, threshold=0.5, knockout=4):
    _ensure_key()
    from novelty import pipeline, retrieve
    target = {"id": "idea", "title": idea["title"], "abstract": idea["description"]}
    pool = retrieve.build_pool([{"id": r["id"], "title": r["title"], "abstract": r["abstract"]}
                                for r in pool_records])
    rep = pipeline.score_paper(target, pool)
    if rep["novelty"] is None:
        return {"gate": "module", "novel": None, "unresolved": True,
                "trace": {"search_status": rep.get("search_status"), "n_claims": rep["n_claims"]}}
    knocked, ev = False, None
    for c in rep["claims"]:
        lvl = c.get("max_entailment_level")
        if lvl is not None and lvl >= knockout:
            knocked = True
            if c["top_matches"]: ev = c["top_matches"][0]["paper_title"]
            break
    novel = (rep["novelty"] >= threshold) and not knocked
    return {"gate": "module", "novel": novel, "evidence": ev,
            "trace": {"novelty": rep["novelty"], "novelty_max": rep["novelty_max"],
                      "novelty_breadth": rep["novelty_breadth"], "knockout": knocked,
                      "n_claims": rep["n_claims"]}}
