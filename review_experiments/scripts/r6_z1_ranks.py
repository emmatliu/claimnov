import json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["R6_EXP"] = "z1ranks"
sys.argv = ["x", "none"]
import r6_z1_largepool as Z
S = Z.S; retrieve, config = Z.retrieve, Z.config
Zr = {json.loads(l)["key"]: json.loads(l) for l in open(f"{Z.R.OUT}/r6_z1_run.jsonl")}
out = []
for u in Z.UU:
    big = Z.enlarge(u); fac = retrieve.build_pool(Z.corpus_of(big)); bid = u["B"]["id"]; best = None
    for c in Z.claims_of(u["title"], u["desc"]):
        c = {**c, "paper_abstract": u["desc"]}
        n = len({f["paper_id"] for f in fac if f["dimension"] == c["dimension"]})
        if not n: continue
        ranked = retrieve.top_k_works(c, fac, k=10**6)
        ids = [f["paper_id"] for f in ranked]
        k = config.top_k_for(n)
        if bid in ids:
            r = ids.index(bid) + 1
            if best is None or r - k < best[0] - best[1]: best = (r, k, c["dimension"])
    row = Zr[u["key"]]
    out.append({"key": u["key"], "rung": u["rung"], "caught": row["rej"], "B_in_profile": row["B_in_profile"],
                "best_rank": best[0] if best else None, "k_at_best": best[1] if best else None, "dim": best[2] if best else None})
agg = {}
for rung in (1, 2, 3, 4):
    rs = [r for r in out if r["rung"] == rung]
    miss = [r for r in rs if not r["B_in_profile"]]
    agg[f"rung{rung}"] = {"n": len(rs), "source_not_in_profile": len(miss),
                          "their_best_rank_vs_k": [(r["best_rank"], r["k_at_best"], r["dim"]) for r in miss],
                          "within_rank_40": sum(1 for r in miss if r["best_rank"] and r["best_rank"] <= 40),
                          "within_rank_60": sum(1 for r in miss if r["best_rank"] and r["best_rank"] <= 60),
                          "source_shares_no_dimension": sum(1 for r in miss if r["best_rank"] is None)}
json.dump({"agg": agg, "rows": out}, open(f"{Z.R.OUT}/r6_z1_ranks.json", "w"), indent=1)
print(json.dumps(agg, indent=1))
