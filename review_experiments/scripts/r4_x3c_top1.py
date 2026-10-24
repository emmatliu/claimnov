import json, sys, os
from math import comb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r4_x3_ir as X
from r0_common import OUT
I = json.load(open(f"{OUT}/r0_items.json"))["items"]
def mcn(a, b):
    n = a + b; k = min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
out = {}
for st in ("claim", "hyb"):
    R = [r for r in X.rows if r["set"] == st]
    for chk in ("claimnov", "graded", "central", "coi"):
        D = I[chk].get(st)
        if not D: out[f"{st}|{chk}"] = "no decisions"; continue
        R2 = [r for r in R if r["key"] in D]
        hit = [r for r in R2 if r["rank"]["claimnov"][:1] == [r["B"]]]
        out[f"{st}|{chk} verdict + ClaimNov top1"] = {"n": len(R2), "B": len(hit), "on_pass": sum(not D[r["key"]]["rej"] for r in hit), "passes": sum(not D[r["key"]]["rej"] for r in R2)}
    def onp(r, chk, rk): return r["rank"][rk][:1] == [r["B"]] and not I[chk][st][r["key"]]["rej"]
    pairs = {"ClaimNov+BM25 vs CoI+BM25": (("claimnov","bm25"),("coi","bm25")),
             "ClaimNov+BM25 vs ClaimNov top1": (("claimnov","bm25"),("claimnov","claimnov")),
             "ClaimNov top1 vs CoI+dense": (("claimnov","claimnov"),("coi","dense")),
             "ClaimNov+BM25 vs CoI+dense": (("claimnov","bm25"),("coi","dense")),
             "ClaimNov+dense vs CoI+dense": (("claimnov","dense"),("coi","dense"))}
    for nm, ((c1,r1),(c2,r2)) in pairs.items():
        a = sum(onp(r,c1,r1) and not onp(r,c2,r2) for r in R); b = sum(onp(r,c2,r2) and not onp(r,c1,r1) for r in R)
        out[f"{st}|{nm} on pass"] = {"disc": [a, b], "p": round(mcn(a, b), 5)}
json.dump(out, open(f"{OUT}/r4_x3c_top1.json", "w"), indent=1)
for k, v in out.items(): print(k, v)
