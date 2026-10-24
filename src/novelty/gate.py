import math

from . import config, llm

_VETO_PROMPT = """Does the following PAPER do (substantially) what the following IDEA proposes? Answer at the level of research contribution: same core problem AND same core approach = yes; superficial topic overlap = no.

IDEA:
{idea}

PAPER:
Title: {title}
Abstract: {abstract}

Answer with one word, "yes" or "no", then a one-sentence justification."""


def contrastive_veto(target: dict, corpus: list[dict], k: int = None, model: str = None):
    k = k or config.GATE_TOP_K
    model = model or config.GATE_JUDGE_MODEL
    idea = f"{target.get('title', '')}: {target.get('abstract', '')}"
    cands = [c for c in corpus if c.get("abstract")]
    if not cands:
        return {"veto": False, "evidence": None, "checked": 0, "unresolved": True}
    vecs = llm.embed([f"{c['title']}. {c['abstract']}" for c in cands] + [idea])
    qv = vecs[-1]

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b)) / (
            math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b)) + 1e-9)

    ranked = sorted(zip(vecs[:-1], cands), key=lambda p: -cos(qv, p[0]))[:k]
    for _, c in ranked:
        out = llm.gate_chat(
            _VETO_PROMPT.format(idea=idea, title=c["title"], abstract=c["abstract"]),
            model=model)
        if out.strip().lower().startswith("yes"):
            return {"veto": True, "evidence": {"id": c.get("id"), "title": c["title"]},
                    "checked": len(ranked)}
    return {"veto": False, "evidence": None, "checked": len(ranked)}


def rank_candidates(report: dict, k: int = None) -> list[dict]:
    k = k or config.GATE_TOP_K
    agg = {}
    for c in report.get("claims", []):
        for j in c.get("judged", []):
            a = agg.setdefault(j["paper_id"], {"paper_id": j["paper_id"],
                                                "paper_title": j["paper_title"],
                                                "max_level": 0, "n_claims": 0,
                                                "sum_e": 0.0, "per_claim": []})
            lvl = j["entailment_level"] or 0
            a["max_level"] = max(a["max_level"], lvl)
            a["n_claims"] += 1
            a["sum_e"] += j["e"] or 0.0
            a["per_claim"].append({"claim": c["claim"], "dimension": c["dimension"],
                                   "level": lvl})
    ranked = sorted(agg.values(), key=lambda a: (-a["max_level"], -a["n_claims"], -a["sum_e"]))
    return ranked[:k]


def coupled_veto(target: dict, corpus: list[dict], report: dict, k: int = None,
                 model: str = None, chat=None):
    k = k or config.GATE_TOP_K
    model = model or config.GATE_JUDGE_MODEL
    idea = f"{target.get('title', '')}: {target.get('abstract', '')}"
    by_id = {c.get("id"): c for c in corpus}
    cands = rank_candidates(report, k=k)
    checked = []
    for a in cands:
        c = by_id.get(a["paper_id"])
        if not c or not c.get("abstract"):
            continue
        prompt = _VETO_PROMPT.format(idea=idea, title=c["title"], abstract=c["abstract"])
        out = chat(prompt) if chat else llm.gate_chat(prompt, model=model)
        checked.append(a["paper_id"])
        if out.strip().lower().startswith("yes"):
            return {"veto": True, "checked": len(checked),
                    "evidence": {"id": c.get("id"), "title": c["title"],
                                 "max_level": a["max_level"],
                                 "per_claim": a["per_claim"],
                                 "justification": out.strip()},
                    "candidates": [x["paper_id"] for x in cands]}
    return {"veto": False, "checked": len(checked), "evidence": None,
            "candidates": [x["paper_id"] for x in cands],
            "unresolved": not cands}


_INFORMED_PROMPT = """Does the following PAPER do (substantially) what the following IDEA proposes? Answer at the level of research contribution: same core problem AND same core approach = yes; superficial topic overlap = no.

IDEA:
{idea}

PAPER:
Title: {title}
Abstract: {abstract}

CLAIM-LEVEL COMPARISON (automatic; each of the IDEA's contribution claims against the PAPER's closest claim, with a coverage level from 1 = unrelated to 5 = the paper fully covers the claim):
{evidence}

Answer with one word, "yes" or "no", then a one-sentence justification."""


def _evidence_table(report: dict, paper_id: str) -> str:
    lines = []
    for c in report.get("claims", []):
        hit = next((j for j in c.get("judged", []) if j["paper_id"] == paper_id), None)
        if hit:
            lines.append(f"- [{c['dimension']}] {c['claim']}\n    paper's closest claim: {hit.get('claim', '')[:300] if hit.get('claim') else '(see abstract)'}\n    coverage level: {hit['entailment_level']}/5")
        else:
            lines.append(f"- [{c['dimension']}] {c['claim']}\n    no matching claim retrieved from this paper (level 1/5)")
    return "\n".join(lines) if lines else "(no claims extracted)"


def claim_informed_veto(target: dict, corpus: list[dict], report: dict, k: int = None,
                        model: str = None, chat=None, fallback: list[dict] = None):
    k = k or config.GATE_TOP_K
    model = model or config.GATE_JUDGE_MODEL
    idea = f"{target.get('title', '')}: {target.get('abstract', '')}"
    by_id = {c.get("id"): c for c in corpus}
    order = [a["paper_id"] for a in rank_candidates(report, k=k)]
    for f in (fallback or []):
        if f.get("id") not in order:
            order.append(f.get("id"))
    checked = []
    for pid in order:
        c = by_id.get(pid)
        if not c or not c.get("abstract"):
            continue
        prompt = _INFORMED_PROMPT.format(idea=idea, title=c["title"], abstract=c["abstract"],
                                         evidence=_evidence_table(report, pid))
        out = chat(prompt) if chat else llm.gate_chat(prompt, model=model, max_tokens=300)
        checked.append(pid)
        if out.strip().lower().startswith("yes"):
            a = next((x for x in rank_candidates(report, k=10**6) if x["paper_id"] == pid), None)
            return {"veto": True, "checked": len(checked),
                    "evidence": {"id": pid, "title": c["title"],
                                 "max_level": a["max_level"] if a else None,
                                 "per_claim": a["per_claim"] if a else [],
                                 "justification": out.strip()},
                    "candidates": order}
    return {"veto": False, "checked": len(checked), "evidence": None, "candidates": order}


_CLAIM_PROFILE_PROMPT = """You are comparing a proposed research IDEA against an existing PAPER to decide whether the paper already establishes what the idea proposes.

IDEA:
{idea}

IDEA'S CONTRIBUTION CLAIMS (extracted automatically):
{idea_claims}

PAPER:
Title: {title}
Abstract: {abstract}

PAPER'S CONTRIBUTION CLAIMS (extracted automatically):
{paper_claims}

For EACH of the idea's claims, decide:
- established: does the PAPER already establish this claim (same core problem AND same core approach for this claim), yes or no? Superficial topic overlap or merely related prior work = no.
- central: is this claim central to the IDEA (removing it would change what the idea is), yes or no?

Return JSON only:
{{"claims": [{{"i": <claim index>, "established": "yes"|"no", "central": "yes"|"no", "why": "<one sentence>"}}]}}"""


def _claims_block(claims):
    return "\n".join(f"{i + 1}. [{c.get('dimension', '?')}] {c.get('claim', '')}"
                      for i, c in enumerate(claims)) or "(none extracted)"


def claim_profile_veto(target: dict, corpus: list[dict], report: dict, pool_facets: list[dict],
                       k: int = None, model: str = None, chat=None, fallback: list[dict] = None):
    import json as _json, re as _re
    k = k or config.GATE_TOP_K
    model = model or config.GATE_JUDGE_MODEL
    idea = f"{target.get('title', '')}: {target.get('abstract', '')}"
    by_id = {c.get("id"): c for c in corpus}
    facets_by_paper = {}
    for f in pool_facets:
        facets_by_paper.setdefault(f.get("paper_id"), []).append(f)
    order = [a["paper_id"] for a in rank_candidates(report, k=k)]
    for f in (fallback or []):
        if f.get("id") not in order:
            order.append(f.get("id"))
    idea_claims = report.get("claims", [])
    checked = []
    for pid in order:
        c = by_id.get(pid)
        if not c or not c.get("abstract"):
            continue
        prompt = _CLAIM_PROFILE_PROMPT.format(
            idea=idea, idea_claims=_claims_block(idea_claims), title=c["title"],
            abstract=c["abstract"], paper_claims=_claims_block(facets_by_paper.get(pid, [])))
        out = chat(prompt) if chat else llm.gate_chat(prompt, model=model, max_tokens=800)
        checked.append(pid)
        try:
            j = _json.loads(_re.search(r"\{.*\}", out, _re.S).group(0))
            verdicts = j.get("claims", [])
        except Exception:
            verdicts = []
        hit = [v for v in verdicts if str(v.get("established", "")).lower().startswith("y")
               and str(v.get("central", "")).lower().startswith("y")]
        if hit:
            return {"veto": True, "checked": len(checked),
                    "evidence": {"id": pid, "title": c["title"], "per_claim": verdicts},
                    "candidates": order}
    return {"veto": False, "checked": len(checked), "evidence": None, "candidates": order}


_PROFILE_PROMPT = """You are the final novelty judge for a proposed research IDEA. A claim-level analysis has already been run: the idea's contribution claims were extracted, and for each claim the most similar existing claims were retrieved from a pool of prior papers and rated for how far they cover the idea's claim (1 = unrelated, 2 = weak topical overlap, 3 = partial coverage with important details differing, 4 = strong coverage with minor differences, 5 = full coverage of the claim's substantive content).

Decide whether the IDEA is already done: is there a paper in the profile that does substantially what the idea proposes, same core problem AND same core approach? Judge the idea as a whole: an anticipated peripheral claim does not make the idea non-novel, but an anticipated core contribution does, even if side claims are new. Superficial topic overlap or related prior work that the idea builds on = novel.

IDEA:
{idea}

NOVELTY PROFILE:
{profile}

PAPERS IN THE PROFILE:
{papers}

Return JSON only:
{{"decision": "novel" | "not_novel", "anticipating_paper": "<exact paper title or null>", "established_claims": [<claim indices the paper already establishes>], "why": "<one or two sentences>"}}"""


def profile_veto(target: dict, corpus: list[dict], report: dict, k: int = 5,
                 model: str = None, chat=None, max_papers: int = 8, abstract_chars: int = 700):
    import json as _json, re as _re
    model = model or config.GATE_JUDGE_MODEL
    idea = f"{target.get('title', '')}: {target.get('abstract', '')}"
    by_id = {c.get("id"): c for c in corpus}
    lines, seen = [], []
    for i, c in enumerate(report.get("claims", []), 1):
        lines.append(f"Claim {i} [{c.get('dimension', '?')}]: {c.get('claim', '')}")
        js = c.get("judged", [])[:k]
        if not js:
            lines.append("    (no related existing claim retrieved)")
        for j in js:
            t = j.get("paper_title", "?")
            if j["paper_id"] not in seen:
                seen.append(j["paper_id"])
            lines.append(f"    - [{t}] {j.get('claim', '')[:300]}  -> coverage {j['entailment_level']}/5")
    papers = []
    for pid in seen[:max_papers]:
        c = by_id.get(pid)
        if c:
            papers.append(f"- {c['title']}: {(c.get('abstract') or '')[:abstract_chars]}")
    prompt = _PROFILE_PROMPT.format(idea=idea, profile="\n".join(lines) or "(no claims extracted)",
                                    papers="\n".join(papers) or "(none)")
    out = chat(prompt) if chat else llm.gate_chat(prompt, model=model, max_tokens=600)
    try:
        j = _json.loads(_re.search(r"\{.*\}", out, _re.S).group(0))
    except Exception:
        j = {"decision": "novel", "anticipating_paper": None, "established_claims": [], "why": out[:300]}
    veto = str(j.get("decision", "")).lower().startswith("not")
    ev = None
    if veto:
        title = j.get("anticipating_paper") or ""
        c = next((x for x in corpus if x["title"].lower()[:60] == str(title).lower()[:60]), None)
        ev = {"id": c.get("id") if c else None, "title": c["title"] if c else title,
              "established_claims": j.get("established_claims", []), "justification": j.get("why", "")}
    return {"veto": veto, "checked": 1, "evidence": ev, "candidates": seen,
            "unresolved": not report.get("claims")}


_CAND_PROMPT = """Does the following PAPER do (substantially) what the following IDEA proposes? Answer at the level of research contribution: same core problem AND same core approach = yes; superficial topic overlap = no.

IDEA:
{idea}

PAPER:
Title: {title}
Abstract: {abstract}

CLAIM-LEVEL COMPARISON (each contribution claim extracted from the IDEA, beside the closest claim extracted from the PAPER{legend}):
{comparison}

Answer with one word, "yes" or "no", then a one-sentence justification."""


def candidate_claim_veto(target: dict, corpus: list[dict], report: dict, pool_facets: list[dict],
                         candidates: list[str], model: str = None, chat=None,
                         show_levels: bool = True):
    from . import entail
    model = model or config.GATE_JUDGE_MODEL
    idea = f"{target.get('title', '')}: {target.get('abstract', '')}"
    by_id = {c.get("id"): c for c in corpus}
    facets_by_paper = {}
    for f in pool_facets:
        facets_by_paper.setdefault(f.get("paper_id"), []).append(f)
    idea_claims = report.get("claims", [])
    legend = ", with our coverage level 1 = unrelated, 3 = partial coverage with important details differing, 5 = full coverage" if show_levels else ""
    if show_levels and config.CALIBRATED_LEGEND:
        legend += ". Calibration note: on prior work that expert reviewers confirmed as anticipating an idea, this level averages 3; read a level of 3 or higher as evidence that the paper has likely already done this claim in substance, not as mere related work"
    checked = []
    for pid in candidates:
        c = by_id.get(pid)
        if not c or not c.get("abstract"):
            continue
        facets = facets_by_paper.get(pid, [])
        rows, per_claim = [], []
        for i, ic in enumerate(idea_claims, 1):
            tgt = {**ic, "paper_abstract": target.get("abstract", "")}
            best = None
            if facets:
                judged = entail.judge_all(tgt, [{**f, "paper_abstract": c.get("abstract", "")} for f in facets])
                best = judged[0] if judged else None
            if best:
                lvl = best["level"]
                line = f"- Idea claim {i} [{ic.get('dimension')}]: {ic.get('claim', '')}\n    paper's closest claim: {best.get('claim', '')[:300]}"
                if show_levels:
                    line += f"\n    coverage level: {lvl}/5"
                per_claim.append({"claim": ic.get("claim"), "dimension": ic.get("dimension"), "level": lvl, "paper_claim": best.get("claim")})
            else:
                line = f"- Idea claim {i} [{ic.get('dimension')}]: {ic.get('claim', '')}\n    paper's closest claim: (no contribution claim extracted from this paper)"
                per_claim.append({"claim": ic.get("claim"), "dimension": ic.get("dimension"), "level": None, "paper_claim": None})
            rows.append(line)
        prompt = _CAND_PROMPT.format(idea=idea, title=c["title"], abstract=c["abstract"],
                                     legend=legend, comparison="\n".join(rows) or "(no claims extracted)")
        out = chat(prompt) if chat else llm.gate_chat(prompt, model=model, max_tokens=300)
        checked.append(pid)
        if out.strip().lower().startswith("yes"):
            return {"veto": True, "checked": len(checked), "candidates": candidates,
                    "evidence": {"id": pid, "title": c["title"], "per_claim": per_claim,
                                 "justification": out.strip()}}
    return {"veto": False, "checked": len(checked), "evidence": None, "candidates": candidates}


_CENTRAL_PROMPT = """You are comparing a proposed research IDEA against an existing PAPER.

IDEA:
{idea}

PAPER:
Title: {title}
Abstract: {abstract}

CLAIM-LEVEL COMPARISON (each contribution claim extracted from the IDEA, beside the closest claim extracted from the PAPER):
{comparison}

For EACH idea claim decide two things:
- established: does the PAPER already establish this claim in substance (the same thing is done or shown, even if framed or worded differently)? yes/no. Merely related work, or work the idea would build on, = no.
- central: is this claim central to what the IDEA is (removing it would change the idea's core contribution)? yes/no.

Return JSON only:
{{"claims": [{{"i": <claim index>, "established": "yes"|"no", "central": "yes"|"no", "why": "<one sentence>"}}]}}"""


def candidate_central_veto(target: dict, corpus: list[dict], report: dict, pool_facets: list[dict],
                           candidates: list[str], model: str = None, chat=None):
    import json as _json, re as _re
    from . import entail
    model = model or config.GATE_JUDGE_MODEL
    idea = f"{target.get('title', '')}: {target.get('abstract', '')}"
    by_id = {c.get("id"): c for c in corpus}
    facets_by_paper = {}
    for f in pool_facets:
        facets_by_paper.setdefault(f.get("paper_id"), []).append(f)
    idea_claims = report.get("claims", [])
    checked = []
    for pid in candidates:
        c = by_id.get(pid)
        if not c or not c.get("abstract"):
            continue
        facets = facets_by_paper.get(pid, [])
        rows, per_claim = [], []
        for i, ic in enumerate(idea_claims, 1):
            tgt = {**ic, "paper_abstract": target.get("abstract", "")}
            best = None
            if facets:
                judged = entail.judge_all(tgt, [{**f, "paper_abstract": c.get("abstract", "")} for f in facets])
                best = judged[0] if judged else None
            pc = best.get("claim", "")[:300] if best else "(no contribution claim extracted from this paper)"
            rows.append(f"- Idea claim {i} [{ic.get('dimension')}]: {ic.get('claim', '')}\n    paper's closest claim: {pc}")
            per_claim.append({"i": i, "claim": ic.get("claim"), "dimension": ic.get("dimension"),
                              "level": best["level"] if best else None, "paper_claim": best.get("claim") if best else None})
        prompt = _CENTRAL_PROMPT.format(idea=idea, title=c["title"], abstract=c["abstract"],
                                        comparison="\n".join(rows) or "(no claims extracted)")
        out = chat(prompt) if chat else llm.gate_chat(prompt, model=model, max_tokens=800)
        checked.append(pid)
        try:
            verdicts = _json.loads(_re.search(r"\{.*\}", out, _re.S).group(0)).get("claims", [])
        except Exception:
            verdicts = []
        for v in verdicts:
            for pcl in per_claim:
                if pcl["i"] == v.get("i"):
                    pcl["established"] = v.get("established"); pcl["central"] = v.get("central"); pcl["why"] = v.get("why")
        hit = [v for v in verdicts if str(v.get("established", "")).lower().startswith("y")
               and str(v.get("central", "")).lower().startswith("y")]
        if hit:
            return {"veto": True, "checked": len(checked), "candidates": candidates,
                    "evidence": {"id": pid, "title": c["title"], "per_claim": per_claim,
                                 "justification": "; ".join(v.get("why", "") for v in hit)[:400]}}
    return {"veto": False, "checked": len(checked), "evidence": None, "candidates": candidates}
