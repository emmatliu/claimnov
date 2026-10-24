import json, glob, collections
from r0_common import *
L = lambda fp: [json.loads(l) for l in open(fp)] if os.path.exists(fp) else []
I = json.load(open(f"{OUT}/r0_items.json"))["items"]
res = {}
R = {}
for r in L(f"{OUT}/r2_e11_run.jsonl"):
    if not r.get("error"): R[(r["key"], r["pool"], r["check"])] = r
errs = [r for r in L(f"{OUT}/r2_e11_run.jsonl") if r.get("error")]
srcs = sorted({k.rsplit("_B_", 1)[0] for k, p, c in R})
full = [s for s in srcs if all((f"{s}_B_{c}", p, ch) in R for c in ("in", "out") for p in ("20", "L") for ch in ("claimnov_sub", "coi"))]
e11 = {"n_sources_complete": len(full), "n_error_rows": len(errs), "error_examples": [e["error"] for e in errs[:3]]}
for ch in ("claimnov_sub", "coi"):
    o = {}
    for p in ("20", "L"):
        ins = {s: R[(f"{s}_B_in", p, ch)] for s in full}; outs = {s: R[(f"{s}_B_out", p, ch)] for s in full}
        o[p] = {"caught": sum(r["rej"] for r in ins.values()), "named": sum(r["named"] for r in ins.values()),
                "src_out": sum(r["rej"] for r in outs.values()), "n": len(full)}
        if ch == "claimnov_sub":
            o[p]["B_in_profile"] = sum(r["B_in_profile"] for r in ins.values())
            o[p]["B_level>=4_in_profile"] = sum(r["B_level_in_profile"] >= 4 for r in ins.values())
            o[p]["src_out_items_with_nonB_level>=4"] = sum(r["max_level_nonB"] >= 4 for r in outs.values())
            o[p]["median_pool"] = float(np.median([r["n_pool"] for r in ins.values()])) if ins else None
    for ev in ("rej_in", "named", "rej_out"):
        f = (lambda s, p: R[(f"{s}_B_in", p, ch)]["rej"]) if ev == "rej_in" else (lambda s, p: R[(f"{s}_B_in", p, ch)]["named"]) if ev == "named" else (lambda s, p: R[(f"{s}_B_out", p, ch)]["rej"])
        o[f"mcnemar_{ev}(20 vs L)"] = mcnemar_exact({s: f(s, "20") for s in full}, {s: f(s, "L") for s in full})
    e11[ch] = o
for g, nm in (("claimnov", "claimnov_paper"), ("coi", "coi_paper")):
    e11[nm + "_20"] = {"caught": sum(I[g]["disg_in"][f"{s}_B_in"]["rej"] for s in full), "named": sum(bool(I[g]["disg_in"][f"{s}_B_in"].get("named")) for s in full),
                       "src_out": sum(I[g]["disg_out"][f"{s}_B_out"]["rej"] for s in full)}
e11["mcnemar_substitute20_vs_paper_claimnov_catch"] = mcnemar_exact({s: R[(f"{s}_B_in", "20", "claimnov_sub")]["rej"] for s in full}, {s: I["claimnov"]["disg_in"][f"{s}_B_in"]["rej"] for s in full})
e11["claimnov_sub_vs_coi_L"] = {"catch": mcnemar_exact({s: R[(f"{s}_B_in", "L", "claimnov_sub")]["rej"] for s in full}, {s: R[(f"{s}_B_in", "L", "coi")]["rej"] for s in full}),
                                "src_out": mcnemar_exact({s: R[(f"{s}_B_out", "L", "claimnov_sub")]["rej"] for s in full}, {s: R[(f"{s}_B_out", "L", "coi")]["rej"] for s in full})}
res["E11"] = e11
LATER = set(later_set())
J = {r["key"]: r for r in L(f"{OUT}/r2_e12_judgeonly.jsonl") if not r.get("error")}
CN = {}
for r in jl("coupled_veto_ac_profile_k"):
    if r["item"] in J and r.get("novel") is not None and r["item"] not in CN: CN[r["item"]] = r
CK = {}
for r in jl("coupled_veto_ac_profile"):
    if r["item"] in J and r.get("novel") is not None and r["item"] not in CK: CK[r["item"]] = r
nt = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())[:50]
e12 = {"n_judge_only": len(J), "errors": sum(1 for r in L(f"{OUT}/r2_e12_judgeonly.jsonl") if r.get("error")),
       "judge_only_caught": fmt_ci(sum(not r["novel"] for r in J.values()), len(J)),
       "judge_only_named_match": sum((not r["novel"]) and r["names_match"] for r in J.values()),
       "judge_only_names_match_any_decision": sum(r["names_match"] for r in J.values()),
       "match_shown_in_top8": sum(r["match_shown"] for r in J.values()),
       "judge_only_later_FA_E1": "47/400 (11.8%)"}
ks = sorted(set(J) & set(CN))
e12["paired_with_shipped_ClaimNov"] = {"n": len(ks), "claimnov_caught": sum(not CN[k]["novel"] for k in ks), "judge_only_caught": sum(not J[k]["novel"] for k in ks),
                                       "mcnemar(claimnov,judge_only)": mcnemar_exact({k: not CN[k]["novel"] for k in ks}, {k: not J[k]["novel"] for k in ks}),
                                       "claimnov_named": sum((not CN[k]["novel"]) and nt(CN[k].get("evidence")) == nt(J[k]["match"]) for k in ks),
                                       "judge_only_named": sum((not J[k]["novel"]) and J[k]["names_match"] for k in ks)}
ks2 = sorted(set(J) & set(CK))
e12["paired_with_fixed_k10_ClaimNov"] = {"n": len(ks2), "claimnov_caught": sum(not CK[k]["novel"] for k in ks2), "judge_only_caught": sum(not J[k]["novel"] for k in ks2),
                                         "mcnemar(claimnov,judge_only)": mcnemar_exact({k: not CK[k]["novel"] for k in ks2}, {k: not J[k]["novel"] for k in ks2})}
e12["J_judge_only"] = round(sum(not r["novel"] for r in J.values()) / max(1, len(J)) - 47 / 400, 4)
res["E12_judge_only"] = e12
dump("r2_e11_e12_summary.json", res); print(json.dumps(res, indent=1))
