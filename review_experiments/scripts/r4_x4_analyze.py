import json, collections
from r0_common import OUT, mcnemar_exact
I = json.load(open(f"{OUT}/r0_items.json"))["items"]
R = [json.loads(l) for l in open(f"{OUT}/r4_x4_run.jsonl") if not json.loads(l).get("error")]
E11 = [json.loads(l) for l in open(f"{OUT}/r2_e11_run.jsonl") if not json.loads(l).get("error")]
for r in E11: r["rung"] = 3
res = {}
def cell(rows, cond):
    rs = [r for r in rows if r["key"].endswith(cond)]
    return rs
def rate(rs, f="rej"): return f"{sum(bool(r[f]) for r in rs)}/{len(rs)}"
for rung in (1, 2, 3, 4):
    src = R if rung != 3 else E11
    out = {}
    for chk, paper in (("claimnov_sub", "claimnov"), ("coi", "coi"), ("jo_bm25", None)):
        L = {r["key"]: r for r in src if r["check"] == chk and r["pool"] == "L" and r.get("rung", 3) == rung}
        if not L: continue
        T = {r["key"]: r for r in src if r["check"] == chk and r["pool"] == "20" and r.get("rung", 3) == rung}
        d = {"n_sources": len({k.split("_")[0] for k in L}), "pool_median": sorted(r["n_pool"] for r in L.values())[len(L) // 2]}
        for cond, lab in (("B_in", "caught"), ("B_out", "src_out")):
            ks = sorted(k for k in L if k.endswith(cond))
            d[f"{lab}_L"] = rate([L[k] for k in ks])
            if cond == "B_in": d["named_L"] = rate([L[k] for k in ks], "named")
            base = None
            if T: base = {k: bool(T[k]["rej"]) for k in ks if k in T}; d[f"{lab}_20_same_check"] = f"{sum(base.values())}/{len(base)}"
            if paper:
                pb = {k: bool(I[paper]["disg_in" if cond == "B_in" else "disg_out"][k]["rej"]) for k in ks}
                d[f"{lab}_20_paper_{paper}"] = f"{sum(pb.values())}/{len(pb)}"
                if base is None: base = pb
                if cond == "B_in": d["named_20_paper"] = f"{sum(bool(I[paper]['disg_in'][k].get('named')) for k in ks)}/{len(ks)}"
            if base:
                m = mcnemar_exact(base, {k: bool(L[k]["rej"]) for k in base})
                d[f"{lab}_mcnemar_20only/Lonly"] = f"{m['a_only']}/{m['b_only']}, p={m['p_exact']:.3g}"
        if chk == "claimnov_sub":
            ks = [k for k in L if k.endswith("B_in")]
            d["source_in_profile_L"] = f"{sum(L[k]['B_in_profile'] for k in ks)}/{len(ks)}"
            d["source_at_level>=4_L"] = f"{sum(L[k]['B_level_in_profile'] >= 4 for k in ks)}/{len(ks)}"
            d["misses_L"] = {"source_not_in_profile": sum(1 for k in ks if not L[k]["rej"] and not L[k]["B_in_profile"]),
                             "source_in_profile_at<=3": sum(1 for k in ks if not L[k]["rej"] and L[k]["B_in_profile"] and L[k]["B_level_in_profile"] <= 3),
                             "source_in_profile_at>=4": sum(1 for k in ks if not L[k]["rej"] and L[k]["B_in_profile"] and L[k]["B_level_in_profile"] >= 4)}
        if chk == "jo_bm25":
            for p, D in (("20", T), ("L", L)):
                ks = [k for k in D if k.endswith("B_in")]
                d[f"source_in_bm25_top8_{p}"] = f"{sum(D[k]['B_in_top8'] for k in ks)}/{len(ks)}"
        out[chk] = d
    res[f"rung{rung}"] = out
for rung in (4,):
    A = {r["key"]: r["rej"] for r in R if r["check"] == "claimnov_sub" and r["pool"] == "L" and r["rung"] == rung}
    B = {r["key"]: r["rej"] for r in R if r["check"] == "coi" and r["pool"] == "L" and r.get("rung") == rung}
    J = {r["key"]: r["rej"] for r in R if r["check"] == "jo_bm25" and r["pool"] == "L" and r["rung"] == rung}
    for nm, X in (("coi", B), ("jo_bm25", J)):
        if not X: continue
        for cond in ("B_in", "B_out"):
            a = {k: v for k, v in A.items() if k.endswith(cond) and k in X}; b = {k: X[k] for k in a}
            m = mcnemar_exact(a, b); res.setdefault(f"rung{rung}_L_claimnov_vs", {})[f"{nm}_{cond}"] = f"{m['a_only']}/{m['b_only']}, p={m['p_exact']:.3g}"
json.dump(res, open(f"{OUT}/r4_x4_summary.json", "w"), indent=1)
print(json.dumps(res, indent=1))
