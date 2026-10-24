import json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gate_sources import panel_prompts as pp
from harness import llm


def _idea_text(idea):
    return f"{idea['title']}: {idea['description']}"


def ra_gate(idea, pool, tag="", model="gpt-4o", k=10, threshold=4):
    hits = pool.search(_idea_text(idea), k=k)
    ctx = "Target paper title: " + idea["title"] + "\nTarget paper abstract: " + idea["description"][:1200]
    for i, p in enumerate(hits):
        ctx += f"\nRelated paper #{i + 1} title: {p['title']}"
    user = pp.RA_ORIGINALITY.format(context_block=ctx, problem=idea["title"],
                                    rationale=idea["description"][:1500])
    rating, attempts = None, 0
    for attempts in range(1, 4):
        out = llm.chat(model, [{"role": "system", "content": pp.RA_SYSTEM},
                               {"role": "user", "content": user}],
                       max_tokens=1000, temperature=1.25,
                       cache_tag=f"ra{tag}_s{attempts}")
        m = re.search(r"Rating(?:\s*\(1-5\))?:\s*\**\s*([1-5])", out, re.I | re.S)
        if m:
            rating = int(m.group(1))
            break
    return {"gate": "ra", "novel": (rating or 0) >= threshold, "rating": rating,
            "trace": {"n_context": len(hits), "attempts": attempts}}


def scholareval_gate(idea, pool, tag="", model="gpt-4.1", k=8):
    plan = _idea_text(idea)
    dims = llm.chat(model, [{"role": "user", "content":
                             pp.SE_DIMENSIONS.format(research_plan=plan[:2000])}],
                    max_tokens=100, temperature=0.3, cache_tag=f"sedim{tag}").strip()
    scores, rows = [], []
    for p in pool.search(plan, k=k):
        out = llm.chat(model, [{"role": "system", "content": pp.SE_PAIRWISE_SYSTEM},
                               {"role": "user", "content": pp.SE_PAIRWISE_USER.format(
                                   research_plan=plan[:2500], title=p["title"],
                                   abstract=p["abstract"][:1500], dimensions=dims)}],
                       max_tokens=900, temperature=0.3,
                       cache_tag=f"sepw{tag}_{p['id']}")
        try:
            j = json.loads(re.search(r"\{.*\}", out, re.S).group(0))
            ds = [int(v.get("score", 0)) for v in j.get("dimension_comparisons", {}).values()]
            s = sum(ds) * 10 / max(1, len(ds))
        except (AttributeError, ValueError, json.JSONDecodeError):
            s = 0.0
        scores.append(s)
        rows.append({"paper": p["title"][:60], "score": s})
    final = sum(scores) / max(1, len(scores))
    return {"gate": "scholareval", "novel": final > 0, "score": round(final, 2),
            "trace": {"dimensions": dims, "per_paper": rows}}


def v2_reflection_gate(idea, pool, tag="", model="gpt-4o", k=10):
    hits = pool.search(_idea_text(idea), k=k)
    results = "\n\n".join(f"{i + 1}: {p['title']}\nAbstract: {p['abstract'][:400]}"
                          for i, p in enumerate(hits))
    user = (f"Here is the proposal under review:\n\nTitle: {idea['title']}\n"
            f"Proposal: {idea['description'][:1500]}\n\n"
            + pp.V2_REFLECTION.format(current_round=1, num_reflections=1,
                                      last_tool_results=results)
            + pp.V2_VERDICT_SUFFIX)
    out = llm.chat(model, [{"role": "user", "content": user}],
                   max_tokens=1200, cache_tag=f"v2r{tag}")
    m = re.search(r"NOVELTY:\s*(pass|fail)", out, re.I)
    verdict = m.group(1).lower() if m else None
    return {"gate": "v2_reflection", "novel": verdict != "fail",
            "verdict_slot_imposed": True,
            "trace": {"parsed": verdict, "n_results": len(hits)}}


PANEL = {"ra": ra_gate, "scholareval": scholareval_gate, "v2": v2_reflection_gate}
