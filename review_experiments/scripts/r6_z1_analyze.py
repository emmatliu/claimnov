import json, os, collections
from r0_common import OUT, mcnemar_exact, wilson
L = lambda f: [json.loads(l) for l in open(f"{OUT}/{f}") if not json.loads(l).get("error")] if os.path.exists(f"{OUT}/{f}") else []
I = json.load(open(f"{OUT}/r0_items.json"))["items"]
Z = {r["key"]: r for r in L("r6_z1_run.jsonl")}
REP = os.environ.get("REPAIRED") == "1"
if REP: Z.update({r["key"]: r for r in L("r6_z1_seed55_repair.jsonl")})
X4 = L("r4_x4_run.jsonl"); E11 = L("r2_e11_run.jsonl")
for r in E11: r["rung"] = 3
def rows(src, chk, pool, rung): return {r["key"]: r for r in src if r["check"] == chk and r["pool"] == pool and r.get("rung", 3) == rung}
def fmt(k, n):
    lo, hi = wilson(k, n); return f"{k}/{n} = {100*k/n:.1f}% [{100*lo:.1f}, {100*hi:.1f}]"
def mc(a, b):
    m = mcnemar_exact(a, b); return f"{m['a_only']}/{m['b_only']}, p = {m['p_exact']:.3g}"
res = {}
allin = {}; allout = {}
for rung in (1, 2, 3, 4):
    zi = {k: r for k, r in Z.items() if r["rung"] == rung and k.endswith("B_in")}
    zo = {k: r for k, r in Z.items() if r["rung"] == rung and k.endswith("B_out")}
    if not zi: continue
    ki, ko = sorted(zi), sorted(zo)
    d = {"n_src_in": len(ki), "n_src_out": len(ko), "pool_median": sorted(r["n_pool"] for r in zi.values())[len(zi) // 2],
         "k_median": sorted(x for r in zi.values() for x in r["k_per_claim"])[len([x for r in zi.values() for x in r["k_per_claim"]]) // 2]}
    sh_c = {k: bool(zi[k]["rej"]) for k in ki}; sh_n = {k: bool(zi[k]["named"]) for k in ki}; sh_o = {k: bool(zo[k]["rej"]) for k in ko}
    allin.update(sh_c); allout.update(sh_o)
    p_c = {k: bool(I["claimnov"]["disg_in"][k]["rej"]) for k in ki}; p_n = {k: bool(I["claimnov"]["disg_in"][k].get("named")) for k in ki}
    p_o = {k: bool(I["claimnov"]["disg_out"][k]["rej"]) for k in ko}
    d["shipped_L"] = {"caught": fmt(sum(sh_c.values()), len(ki)), "named": fmt(sum(sh_n.values()), len(ki)), "src_out": fmt(sum(sh_o.values()), len(ko))}
    d["shipped_20_paper"] = {"caught": f"{sum(p_c.values())}/{len(ki)}", "named": f"{sum(p_n.values())}/{len(ki)}", "src_out": f"{sum(p_o.values())}/{len(ko)}",
                             "source_in_profile": f"{sum(bool(I['claimnov']['disg_in'][k].get('B_in_profile')) for k in ki)}/{len(ki)}"}
    d["mcnemar_20paper_vs_L"] = {"caught": mc(p_c, sh_c), "named": mc(p_n, sh_n), "src_out": mc(p_o, sh_o)}
    d["source_in_profile_L"] = f"{sum(zi[k]['B_in_profile'] for k in ki)}/{len(ki)}"
    d["source_retrieved_any_claim_topk_L"] = f"{sum(zi[k]['B_retrieved_any_claim'] for k in ki)}/{len(ki)}"
    d["source_at_level>=4_in_profile_L"] = f"{sum(zi[k]['B_level_in_profile'] >= 4 for k in ki)}/{len(ki)}"
    miss = [k for k in ki if not zi[k]["rej"]]
    d["misses_L"] = {"n": len(miss),
                     "not_retrieved_in_any_topk": sum(not zi[k]["B_retrieved_any_claim"] for k in miss),
                     "retrieved_but_not_in_top5_profile": sum(zi[k]["B_retrieved_any_claim"] and not zi[k]["B_in_profile"] for k in miss),
                     "in_profile_at<=3": sum(zi[k]["B_in_profile"] and zi[k]["B_level_in_profile"] <= 3 for k in miss),
                     "in_profile_at>=4": sum(zi[k]["B_in_profile"] and zi[k]["B_level_in_profile"] >= 4 for k in miss),
                     "keys": miss}
    d["caught_not_named_L"] = [{"key": k, "cited": zi[k]["anticipating"]} for k in ki if zi[k]["rej"] and not zi[k]["named"]]
    d["src_out_rejections_L"] = [{"key": k, "cited": zo[k]["anticipating"]} for k in ko if zo[k]["rej"]]
    sub = rows(X4 if rung != 3 else E11, "claimnov_sub", "L", rung)
    if sub:
        si = {k: bool(sub[k]["rej"]) for k in ki if k in sub}; so = {k: bool(sub[k]["rej"]) for k in ko if k in sub}
        sn = {k: bool(sub[k]["named"]) for k in ki if k in sub}
        d["substitute_L_same_keys"] = {"caught": f"{sum(si.values())}/{len(si)}", "named": f"{sum(sn.values())}/{len(sn)}", "src_out": f"{sum(so.values())}/{len(so)}",
                                       "source_in_profile": f"{sum(sub[k]['B_in_profile'] for k in si)}/{len(si)}"}
        d["mcnemar_substitute_vs_shipped_L"] = {"caught": mc(si, {k: sh_c[k] for k in si}), "src_out": mc(so, {k: sh_o[k] for k in so})}
        if len(si) < len(ki):
            d["shipped_L_on_substitute_keys"] = {"caught": f"{sum(sh_c[k] for k in si)}/{len(si)}", "src_out": f"{sum(sh_o[k] for k in so)}/{len(so)}",
                                                 "named": f"{sum(sh_n[k] for k in si)}/{len(si)}"}
    for chk in ("coi", "jo_bm25"):
        X = rows(X4 if rung != 3 else E11, chk, "L", rung)
        if not X: continue
        xi = {k: bool(X[k]["rej"]) for k in ki if k in X}; xo = {k: bool(X[k]["rej"]) for k in ko if k in X}
        xn = {k: bool(X[k]["named"]) for k in ki if k in X}
        d[f"{chk}_L"] = {"caught": f"{sum(xi.values())}/{len(xi)}", "named": f"{sum(xn.values())}/{len(xn)}", "src_out": f"{sum(xo.values())}/{len(xo)}"}
        d[f"mcnemar_shipped_vs_{chk}_L"] = {"caught": mc({k: sh_c[k] for k in xi}, xi), "named": mc({k: sh_n[k] for k in xn}, xn),
                                            "src_out": mc({k: sh_o[k] for k in xo}, xo)}
    res[f"rung{rung}"] = d
if allin:
    pc = {k: bool(I["claimnov"]["disg_in"][k]["rej"]) for k in allin}; po = {k: bool(I["claimnov"]["disg_out"][k]["rej"]) for k in allout}
    res["all_rungs"] = {"caught_L": fmt(sum(allin.values()), len(allin)), "caught_20_paper": f"{sum(pc.values())}/{len(pc)}", "mcnemar_caught": mc(pc, allin),
                        "src_out_L": fmt(sum(allout.values()), len(allout)), "src_out_20_paper": f"{sum(po.values())}/{len(po)}", "mcnemar_src_out": mc(po, allout)}
json.dump(res, open(f"{OUT}/r6_z1_summary{'_repaired' if REP else ''}.json", "w"), indent=1)
print(json.dumps(res, indent=1))
