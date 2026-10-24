import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r4_x3_ir as X
from r0_common import OUT
I = json.load(open(f"{OUT}/r0_items.json"))["items"]
res = {}
for st in ("claim", "hyb"):
    R = [r for r in X.rows if r["set"] == st]
    for chk in ("claimnov", "coi"):
        D = I[chk][st]
        miss = [r["key"] for r in R if r["key"] not in D]
        for rk in ("bm25", "dense", "claimnov"):
            hit = [r for r in R if r["key"] in D and r["rank"][rk][:1] == [r["B"]]]
            onp = [r for r in hit if not D[r["key"]]["rej"]]
            res[f"{st}|{chk} verdict|{rk} top1"] = {"n": len(R), "missing": len(miss), "B_top1": len(hit), "B_top1_on_pass": len(onp)}
    a = b = 0
    for r in R:
        if r["key"] not in I["claimnov"][st] or r["key"] not in I["coi"][st]: continue
        h = r["rank"]["bm25"][:1] == [r["B"]]
        x = h and not I["claimnov"][st][r["key"]]["rej"]; y = h and not I["coi"][st][r["key"]]["rej"]
        a += x and not y; b += y and not x
    res[f"{st}|paired ClaimNov+BM25 vs CoI+BM25 on pass"] = [a, b]
json.dump(res, open(f"{OUT}/r4_x3b_composite.json", "w"), indent=1)
for k, v in res.items(): print(k, v)
