import json, os, re, hashlib, collections
import numpy as np
from r0_common import OUT, PILOT, PCACHE, mcnemar_exact, wilson
L = lambda f: [json.loads(l) for l in open(f"{OUT}/{f}")] if os.path.exists(f"{OUT}/{f}") else []
I = json.load(open(f"{OUT}/r0_items.json"))["items"]
A = json.load(open(f"{OUT}/r4_x1_anchors.json"))
PR = {r["key"]: r for r in json.load(open(f"{OUT}/r4_x1_prompts.json"))}
RUN = collections.defaultdict(dict)
for r in L("r4_x1_runs.jsonl"):
    if not r.get("error"): RUN[r["mode"]][r["key"]] = r
LATER = sorted(I["claimnov"]["later"]); EXACT = [k for k in LATER if PR[k]["src"] == "exact"]; RECON = [k for k in LATER if PR[k]["src"] == "recon"]
anc = lambda k: k.split("::")[0]
def fmt(k, n): lo, hi = wilson(k, n); return f"{k}/{n} = {100*k/n:.1f}% [{100*lo:.1f}, {100*hi:.1f}]"
def cmp(base, new, keys):
    b = {k: base[k] for k in keys}; n = {k: new[k] for k in keys}; m = mcnemar_exact(n, b)
    return {"n": len(keys), "without": fmt(sum(b.values()), len(keys)), "with": fmt(sum(n.values()), len(keys)),
            "mcnemar_with_only/without_only": f"{m['a_only']}/{m['b_only']}", "p": round(m["p_exact"], 4)}
res = {"anchor_claims": {s: {"src": a["claims_src"], "n": len(a["claims"]), "dims": dict(collections.Counter(c["dimension"] for c in a["claims"]))} for s, a in A.items()}}
cn0 = {k: bool(I["claimnov"]["later"][k]["rej"]) for k in LATER}
cnA = {k: RUN["cn_anchor"][k]["rej"] for k in LATER}
ctrl = {k: RUN["cn_recon_noanchor"][k]["rej"] for k in RECON}
base400 = {**{k: cn0[k] for k in EXACT}, **ctrl}
cn = {"exact341_paper_vs_anchor": cmp(cn0, cnA, EXACT), "all400_paper_vs_anchor": cmp(cn0, cnA, LATER),
      "all400_promptmatched_vs_anchor": cmp(base400, cnA, LATER), "recon59_ctrl_vs_anchor": cmp(ctrl, cnA, RECON),
      "recon59_paper_vs_ctrl": cmp(cn0, ctrl, RECON)}
cn["verdict_cites_anchor"] = {"on_rejection": sum(RUN["cn_anchor"][k]["rej"] and RUN["cn_anchor"][k]["cites_anchor"] for k in LATER),
                              "any_decision": sum(RUN["cn_anchor"][k]["cites_anchor"] for k in LATER)}
cn["new_FA_items"] = [{"key": k, "src": PR[k]["src"], "anticipating": RUN["cn_anchor"][k]["anticipating_title"], "anchor_max_level": PR[k]["anchor_max_level"]}
                      for k in LATER if cnA[k] and not cn0[k]]
cn["lost_FA_items"] = [k for k in LATER if cn0[k] and not cnA[k]]
lv = collections.Counter(PR[k]["anchor_max_level"] for k in LATER)
cn["insertion"] = {"ideas_with_same_dim_anchor_claim": sum(bool(PR[k]["insertions"]) for k in LATER),
                   "anchor_shown_in_some_claim_top5": sum(any(x["pos"] is not None for x in PR[k]["insertions"]) for k in LATER),
                   "anchor_in_papers_block(top8)": sum(PR[k]["anchor_in_papers_block"] for k in LATER),
                   "anchor_first_paper": sum(PR[k]["anchor_rank_in_papers"] == 1 for k in LATER),
                   "anchor_max_level_hist": {str(a): b for a, b in sorted(lv.items(), key=lambda x: (x[0] is None, x[0] or 0))},
                   "rej_by_anchor_level": {str(l): f"{sum(cnA[k] for k in LATER if PR[k]['anchor_max_level'] == l)}/{sum(1 for k in LATER if PR[k]['anchor_max_level'] == l)}" for l in sorted(lv, key=lambda x: (x is None, x or 0))}}
rep = {}
for f in ("r3_rep1.jsonl", "r3_rep2.jsonl"):
    d = {r["key"]: r["rej"] for r in L(f) if r.get("set") == "later" and not r.get("error")}
    ks = [k for k in EXACT if k in d]
    m = mcnemar_exact(d, cn0)
    rep[f] = {"n": len(ks), "FA": sum(d[k] for k in ks), "flips_vs_paper": f"{m['a_only']}/{m['b_only']}"}
cn["variance_reference_D3"] = rep
res["claimnov"] = cn
if RUN.get("cn_anchor_bump1"):
    PRB = {r["key"]: r for r in json.load(open(f"{OUT}/r4_x1_prompts_bump1.json"))}
    have = [k for k in LATER if k in RUN["cn_anchor_bump1"] or PRB[k]["prompt_anchor"] == PR[k]["prompt_anchor"]]
    cnB = {k: (RUN["cn_anchor_bump1"][k]["rej"] if k in RUN["cn_anchor_bump1"] else cnA[k]) for k in have}
    lvb = collections.Counter(PRB[k]["anchor_max_level"] for k in LATER)
    res["claimnov_bump1"] = {"n_complete": len(have), "new_calls": len(RUN["cn_anchor_bump1"]), "paper_vs_bump": cmp(cn0, cnB, have),
                             "promptmatched_vs_bump": cmp(base400, cnB, have), "anchor_vs_bump": cmp(cnA, cnB, have),
                             "exact341_paper_vs_bump": cmp(cn0, cnB, [k for k in EXACT if k in cnB]),
                             "cites_anchor_on_rejection": sum(RUN["cn_anchor_bump1"][k]["rej"] and RUN["cn_anchor_bump1"][k]["cites_anchor"] for k in RUN["cn_anchor_bump1"]),
                             "cites_anchor_any": sum(RUN["cn_anchor_bump1"][k]["cites_anchor"] for k in RUN["cn_anchor_bump1"]),
                             "anchor_max_level_hist": {str(a): b for a, b in sorted(lvb.items(), key=lambda x: (x[0] is None, x[0] or 0))},
                             "rej_by_anchor_level": {str(l): f"{sum(cnB[k] for k in have if PRB[k]['anchor_max_level'] == l)}/{sum(1 for k in have if PRB[k]['anchor_max_level'] == l)}" for l in sorted(lvb, key=lambda x: (x is None, x or 0))},
                             "anchor_in_papers_block": sum(PRB[k]["anchor_in_papers_block"] for k in LATER),
                             "rejections": [{"key": k, "cites_anchor": RUN["cn_anchor_bump1"].get(k, {}).get("cites_anchor"), "anticipating": RUN["cn_anchor_bump1"].get(k, RUN["cn_anchor"][k])["anticipating_title"]} for k in have if cnB[k]]}
E1 = {r["key"]: r for r in L("e1_judge_only_k8.jsonl") if r["set"] == "later"}
jo0 = {k: not E1[k]["novel"] for k in LATER}; joA = {k: RUN["jo_anchor"][k]["rej"] for k in LATER}
jo = {"all400": cmp(jo0, joA, LATER), "exact341": cmp(jo0, joA, EXACT),
      "verdict_cites_anchor": {"on_rejection": sum(RUN["jo_anchor"][k]["rej"] and RUN["jo_anchor"][k]["cites_anchor"] for k in LATER),
                               "any_decision": sum(RUN["jo_anchor"][k]["cites_anchor"] for k in LATER)},
      "rule": "anchor first in the top-8 list, E1's 8th paper dropped"}
def emb(t):
    fp = f"{PCACHE}/e_{hashlib.sha256(('text-embedding-3-small::' + t).encode()).hexdigest()[:24]}.json"
    return np.array(json.load(open(fp)), float) if os.path.exists(fp) else None
ranks = []
_ip = {}
for k in LATER:
    a = A[anc(k)]; ae = emb(f"{a['title']}. {a['abstract']}")
    if ae is None: continue
    s = anc(k)
    if s not in _ip:
        d = next(p for p in (f"{PILOT}/data/anchors/{s}", f"{PILOT}/data/anchors_full/{s}") if os.path.exists(f"{p}/ideas.json"))
        pool = json.load(open(f"{d}/pool_preT.json"))[:400]
        E = np.array([emb(f"{p['title']}. {p['abstract']}") for p in pool]); E /= np.linalg.norm(E, axis=1, keepdims=True)
        _ip[s] = (E, {i["idx"]: i for i in json.load(open(f"{d}/ideas.json"))})
    E, ideas = _ip[s]; i = ideas[int(k.split("::")[1])]
    q = emb(f"{i['title']}: {i['description']}"); q /= np.linalg.norm(q)
    ca = float(ae @ q / np.linalg.norm(ae)); ranks.append(int((E @ q > ca).sum()) + 1)
jo["true_cosine_rank_of_anchor"] = {"ideas_with_cached_anchor_embedding": len(ranks), "anchors": sum(1 for s in A if emb(f"{A[s]['title']}. {A[s]['abstract']}") is not None),
                                    "rank1": sum(r == 1 for r in ranks), "in_top8": sum(r <= 8 for r in ranks), "median_rank": float(np.median(ranks)) if ranks else None}
JC = {r["key"]: r["rej"] for r in L("r4_x1_jo_control.jsonl") if not r.get("error")}
if JC:
    ks = sorted(JC)
    jo["control_rerun_unchanged_prompt"] = {"subset": "seeded random (random.Random(7)) of the 400", "E1_vs_rerun": cmp(jo0, JC, ks),
                                            "rerun_vs_anchor": cmp(JC, joA, ks), "E1_vs_anchor_same_subset": cmp(jo0, joA, ks)}
res["judge_only"] = jo
ON0 = {r["key"]: r for r in L("r1_opennovelty.jsonl") if r["set"] == "later" and not r.get("error")}
ONA = {r["key"]: r for r in L("r4_x1_on.jsonl") if not r.get("error")}
on0 = {k: bool(ON0[k]["reject_all"]) for k in LATER}; onA = {k: (ONA[k]["reject_all"] if k in ONA else on0[k]) for k in LATER}
on0y = {k: bool(ON0[k]["reject_any"]) for k in LATER}; onAy = {k: (ONA[k]["reject_any"] if k in ONA else on0y[k]) for k in LATER}
res["opennovelty"] = {"items_rerun_anchor_in_some_top10": len(ONA), "reject_all": cmp(on0, onA, LATER), "reject_any": cmp(on0y, onAy, LATER),
                      "reject_all_rerun_items_only": cmp(on0, onA, sorted(ONA)),
                      "contributions_with_anchor_in_top10": sum(v["anchor_in_top10"] for r in ONA.values() for v in r["verdicts"]),
                      "contributions_total_rerun": sum(len(r["verdicts"]) for r in ONA.values()),
                      "cites_anchor_can_refute_items": sum(r["cites_anchor_can_refute"] for r in ONA.values()),
                      "cites_anchor_any_verdict_items": sum(r["cites_anchor_any"] for r in ONA.values()),
                      "reject_all_and_cites_anchor": sum(r["reject_all"] and r["cites_anchor_can_refute"] for r in ONA.values()),
                      "contribution_can_refute_by_anchor": sum(1 for r in ONA.values() for v in r["verdicts"] if v["verdict"] == "can_refute" and v["refuting_id"] == A[r["anchor"]]["id"]),
                      "contribution_any_verdict_naming_anchor": sum(1 for r in ONA.values() for v in r["verdicts"] if v["refuting_id"] == A[r["anchor"]]["id"])}
CO = {r["key"]: r for r in L("r4_x1_coi.jsonl") if not r.get("error")}
ck = sorted(CO); co0 = {k: bool(I["coi"]["later"][k]["rej"]) for k in ck}; coA = {k: CO[k]["rej"] for k in ck}
res["coi"] = {"subset": "seeded random 120 of 400 (random.Random(4))", **cmp(co0, coA, ck),
              "cites_anchor_on_rejection": sum(CO[k]["cites_anchor"] for k in ck), "names_anchor_as_most_similar_any_decision": sum(CO[k]["names_anchor_any"] for k in ck),
              "claimnov_on_same_120": cmp(cn0, cnA, ck), "judge_only_on_same_120": cmp(jo0, joA, ck),
              "paper_check_prompt_reproduced": sum(CO[k]["paper_check_cached"] for k in ck)}
pa = {}
for s in sorted({anc(k) for k in LATER}, key=lambda s: -sum(anc(k) == s for k in LATER)):
    ks = [k for k in LATER if anc(k) == s]
    pa[s] = {"n": len(ks), "claimnov": f"{sum(cn0[k] for k in ks)} -> {sum(cnA[k] for k in ks)}", "judge_only": f"{sum(jo0[k] for k in ks)} -> {sum(joA[k] for k in ks)}",
             "opennovelty_all": f"{sum(on0[k] for k in ks)} -> {sum(onA[k] for k in ks)}",
             "coi(120 subset)": f"{sum(co0[k] for k in ks if k in co0)} -> {sum(coA[k] for k in ks if k in coA)} of {sum(1 for k in ks if k in co0)}",
             "anchor_claims_src": A[s]["claims_src"]}
res["per_anchor_FA_without_to_with"] = pa
for src in ("cached_shipped_gpt-4o-mini", "haiku-4.5_substitute"):
    ks = [k for k in LATER if A[anc(k)]["claims_src"] == src]
    res.setdefault("by_anchor_claim_source", {})[src] = {"claimnov": cmp(cn0, cnA, ks), "judge_only": cmp(jo0, joA, ks)}
json.dump(res, open(f"{OUT}/r4_x1_summary.json", "w"), indent=1)
print(json.dumps(res.get("claimnov_bump1"), indent=1))
