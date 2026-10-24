import json
from r0_common import *
L = lambda fp: [json.loads(l) for l in open(fp)] if os.path.exists(fp) else []
DROP = {60, 83, 86, 87, 93}
res = {}
for V in ("A", "B"):
    R = {(r["set"], r["key"]): r for r in L(f"{OUT}/r2_e10_reword_{V}.jsonl") if not r.get("error")}
    o = {"n_rows": len(R), "parse_errors": sum(r["parse_error"] for r in R.values())}
    for lab, sel in (("later", lambda s, k: s == "later"), ("src_in", lambda s, k: s == "disg" and k.endswith("B_in") and seed_of(k) not in DROP),
                     ("src_out", lambda s, k: s == "disg" and k.endswith("B_out") and seed_of(k) not in DROP)):
        ks = [x for x in R if sel(*x)]
        a = {x: R[x]["orig_rej"] for x in ks}; b = {x: R[x]["rej"] for x in ks}
        o[lab] = {"n": len(ks), "orig_rej": sum(a.values()), "reworded_rej": sum(b.values()), "mcnemar(orig,reworded)": mcnemar_exact(a, b)}
        if lab == "src_in":
            o[lab]["orig_named"] = sum(R[x]["orig_rej"] and R[x]["orig_anticipating_id"] == R[x]["B_id"] for x in ks)
            o[lab]["reworded_named"] = sum(R[x]["rej"] and R[x]["anticipating_id"] == R[x]["B_id"] for x in ks)
        if lab == "later":
            o[lab]["reworded_FA_items"] = sorted(x[1] for x in ks if b[x])
    res[V] = o
dump("r2_e10_reword_summary.json", res); print(json.dumps(res, indent=1))
