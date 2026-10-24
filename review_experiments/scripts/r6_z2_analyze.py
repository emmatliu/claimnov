import json, os, collections
from r0_common import OUT, mcnemar_exact, wilson
B = {r["key"]: r for r in json.load(open(f"{OUT}/r6_z2_build.json"))}
RUN = collections.defaultdict(dict)
for l in open(f"{OUT}/r6_z2_runs.jsonl"):
    r = json.loads(l)
    if not r.get("error"): RUN[r["mode"]][r["key"]] = r
I = json.load(open(f"{OUT}/r0_items.json"))["items"]["claimnov"]["later"]
LATER = sorted(B)
def fmt(k, n):
    lo, hi = wilson(k, n); return f"{k}/{n} = {100*k/n:.1f}% [{100*lo:.1f}, {100*hi:.1f}]"
base = {k: bool(I[k]["rej"]) for k in LATER}
missing = [k for k in LATER if B[k]["changed"] and k not in RUN["cn_anchor"]]
withA = {k: (RUN["cn_anchor"][k]["rej"] if B[k]["changed"] else base[k]) for k in LATER if k not in missing}
def cmp(keys):
    keys = [k for k in keys if k in withA]
    m = mcnemar_exact({k: withA[k] for k in keys}, {k: base[k] for k in keys})
    return {"n": len(keys), "changed_profiles": sum(B[k]["changed"] for k in keys), "FA_without": fmt(sum(base[k] for k in keys), len(keys)),
            "FA_with": fmt(sum(withA[k] for k in keys), len(keys)), "mcnemar_with_only/without_only": f"{m['a_only']}/{m['b_only']}",
            "p": round(m["p_exact"], 4)}
EXACT = [k for k in LATER if B[k]["paper_src"] == "exact"]
CERT = [k for k in LATER if B[k]["prompt0_paper_cache_hit"]]
res = {"missing_verdicts": missing, "all400": cmp(LATER), "exact341": cmp(EXACT), "certified_baseline": cmp(CERT),
       "exact341_and_certified": cmp([k for k in EXACT if k in set(CERT)])}
nb = RUN["cn_base_new"]
pm_keys = [k for k in LATER if B[k]["prompt0_paper_cache_hit"] or k in nb]
pm_base = {k: (base[k] if B[k]["prompt0_paper_cache_hit"] else nb[k]["rej"]) for k in pm_keys}
pm_with = {k: (RUN["cn_anchor"][k]["rej"] if B[k]["changed"] else pm_base[k]) for k in pm_keys}
m = mcnemar_exact(pm_with, pm_base)
res["prompt_matched"] = {"n": len(pm_keys), "uncertified_with_fresh_baseline": f"{len(nb)}/{sum(not B[k]['prompt0_paper_cache_hit'] for k in LATER)}",
                         "FA_without": fmt(sum(pm_base.values()), len(pm_keys)), "FA_with": fmt(sum(pm_with.values()), len(pm_keys)),
                         "mcnemar_with_only/without_only": f"{m['a_only']}/{m['b_only']}", "p": round(m["p_exact"], 4),
                         "fresh_baseline_vs_paper_on_uncertified": f"fresh rej {sum(r['rej'] for r in nb.values())}, paper rej {sum(base[k] for k in nb)} (n={len(nb)})"}
res["baseline_reproduction"] = {"prompt0_equals_paper_rebuilt_prompt_exact341": f"{sum(B[k]['prompt0_equals_r3'] for k in EXACT)}/{len(EXACT)}",
                                "prompt0_hits_paper_verdict_cache_all400": f"{len(CERT)}/{len(LATER)}",
                                "by_src": dict(collections.Counter(f"{B[k]['paper_src']}_cert={B[k]['prompt0_paper_cache_hit']}" for k in LATER))}
A = lambda f, keys=LATER: sum(1 for k in keys if f(B[k]))
res["anchor_in_profile"] = {
    "anchor_retrieved_in_some_claim_topk": A(lambda r: (r["anchor_retrieved_claims"] or 0) > 0),
    "anchor_shown_in_some_claim_top5_profile": A(lambda r: (r["anchor_shown_claims"] or 0) > 0),
    "anchor_in_papers_block_top8": A(lambda r: r["anchor_in_papers_block"]),
    "anchor_first_paper_in_block": A(lambda r: r["anchor_rank_in_papers"] == 1),
    "profile_prompt_changed": A(lambda r: r["changed"]),
    "changed_without_anchor_shown": A(lambda r: r["changed"] and not (r["anchor_shown_claims"] or 0)),
    "anchor_max_level_judged_hist": dict(sorted(collections.Counter(str(B[k]["anchor_max_level"]) for k in LATER).items())),
    "anchor_max_level_shown_hist": dict(sorted(collections.Counter(str(B[k]["anchor_max_level_shown"]) for k in LATER).items())),
    "exact341": {"retrieved": A(lambda r: (r["anchor_retrieved_claims"] or 0) > 0, EXACT), "shown": A(lambda r: (r["anchor_shown_claims"] or 0) > 0, EXACT),
                 "changed": A(lambda r: r["changed"], EXACT)},
    "anchor_claims": {s: B[next(k for k in LATER if B[k]["anchor"] == s)]["anchor_claims_src"] + f" ({B[next(k for k in LATER if B[k]['anchor'] == s)]['n_anchor_claims']})"
                      for s in sorted({B[k]["anchor"] for k in LATER})}}
cn = RUN["cn_anchor"]
res["verdict_cites_anchor"] = {"rejections_citing_anchor": sum(r["rej"] and r["cites_anchor"] for r in cn.values()),
                               "any_decision_citing_anchor": sum(bool(r["cites_anchor"]) for r in cn.values()), "n_new_verdicts": len(cn)}
res["new_FA"] = [{"key": k, "anticipating": cn[k]["anticipating_title"], "cites_anchor": cn[k]["cites_anchor"], "anchor_max_level_shown": B[k]["anchor_max_level_shown"],
                  "src": B[k]["paper_src"]} for k in LATER if k in withA and withA[k] and not base[k]]
res["lost_FA"] = [{"key": k, "anchor_shown": bool(B[k]["anchor_shown_claims"])} for k in LATER if k in withA and base[k] and not withA[k]]
res["rej_by_anchor_level_shown"] = {str(l): f"{sum(withA[k] for k in LATER if k in withA and B[k]['anchor_max_level_shown'] == l)}/{sum(1 for k in LATER if B[k]['anchor_max_level_shown'] == l)}"
                                    for l in sorted({B[k]["anchor_max_level_shown"] for k in LATER}, key=lambda x: (x is None, x or 0))}
ct = RUN["ctrl_unchanged"]
if ct:
    res["control_unchanged_reissued"] = {"n": len(ct), "paper_rej": sum(base[k] for k in ct), "reissue_rej": sum(r["rej"] for r in ct.values()),
                                         "flips_new_rej/new_pass": f"{sum(r['rej'] and not base[k] for k, r in ct.items())}/{sum((not r['rej']) and base[k] for k, r in ct.items())}",
                                         "identical_decision": sum(r["rej"] == base[k] for k, r in ct.items())}
pa = {}
for s in sorted({B[k]["anchor"] for k in LATER}):
    ks = [k for k in LATER if B[k]["anchor"] == s and k in withA]
    pa[s] = {"n": len(ks), "changed": sum(B[k]["changed"] for k in ks), "FA": f"{sum(base[k] for k in ks)} -> {sum(withA[k] for k in ks)}"}
res["per_anchor"] = pa
res["endpoints"] = dict(collections.Counter((r.get("meta") or {}).get("endpoint") for m in RUN.values() for r in m.values()))
json.dump(res, open(f"{OUT}/r6_z2_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("per_anchor",)}, indent=1))
