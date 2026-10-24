import json, os, re, hashlib, collections, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r4_shipped as S
from rank_bm25 import BM25Okapi
from r0_common import OUT, PILOT, PCACHE
STOP = set("""a an the and or of to in on for with by from as at is are was were be been being this that these those it its
we our us their they them which who whom whose what when where how than then there here such into onto over under
can could may might will would should shall do does did not no nor but if so also both each either neither via per
more most less least very much many few new novel propose proposed proposes approach method methods paper work using use
used based show shows shown""".split())
toks = lambda s: [t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(t) > 1 and t not in STOP]
def emb(t):
    fp = f"{PCACHE}/e_{hashlib.sha256(('text-embedding-3-small::' + t).encode()).hexdigest()[:24]}.json"
    return np.array(json.load(open(fp)), float) if os.path.exists(fp) else None
P3 = {(p["set"], p["key"]): p for p in map(json.loads, open(f"{OUT}/r3_prompts.jsonl"))}
ON = {(r["set"], r["key"]): r for r in map(json.loads, open(f"{OUT}/r1_opennovelty.jsonl")) if not r.get("error")}
HY = {h["item"]: h for h in json.load(open(f"{PILOT}/data/hybrids.json"))}
DROP = {60, 83, 86, 87, 93}
_ap = {}
def apool(s):
    if s not in _ap: _ap[s] = json.load(open(f"{S.anchor_dir(s)}/pool_preT.json"))[:400]
    return _ap[s]
def units():
    for (st, k), p in P3.items():
        if st == "claim":
            pool = json.load(open(f"{PILOT}/data/pools_full/{k}.json"))["pool"]; B = next((x["id"] for x in pool if x.get("is_B")), None)
            yield "claim", k, k, pool, B, p
        elif st == "hyb":
            pool = apool(HY[k]["anchor"]); yield "hyb", k, HY[k]["anchor"], pool, HY[k]["B_id"], p
        elif st == "disg" and k.endswith("B_in"):
            s = int(re.match(r"seed(\d+)_", k).group(1))
            if s in DROP: continue
            pool = json.load(open(f"{PILOT}/data/pools/seed{s}.json"))["pool"]; B = next(x["id"] for x in pool if x.get("is_B"))
            yield "disg", k, s, pool, B, p
rows = []
for st, k, cl, pool, B, p in units():
    if not p.get("prompt") or B is None: continue
    by_title = {x["title"]: x for x in pool}
    idea, parsed = S.parse_profile_prompt(p["prompt"], list(by_title))
    best = {}; order = []
    for _, _, ents in parsed:
        for t, _, lv in ents:
            pid = by_title[t]["id"]
            if pid not in best: order.append(pid); best[pid] = lv
            else: best[pid] = max(best[pid], lv)
    cn = sorted(order, key=lambda x: (-best[x], order.index(x)))
    q = emb(idea)
    if q is None: raise RuntimeError(f"no idea embedding {st} {k}")
    E = np.array([emb(f"{x['title']}. {x['abstract']}") for x in pool]); E /= np.linalg.norm(E, axis=1, keepdims=True)
    dense = [pool[i]["id"] for i in np.argsort(-(E @ (q / np.linalg.norm(q))), kind="stable")]
    sc = BM25Okapi([toks(f"{x['title']}. {x.get('abstract') or ''}") for x in pool]).get_scores(toks(idea))
    bm = [pool[i]["id"] for i in sorted(range(len(pool)), key=lambda i: -sc[i])]
    o = ON.get((st, k)); onr = []
    if o:
        for want in ("can_refute", "unclear"):
            for v in o["verdicts"]:
                if v["verdict"] == want and v.get("refuting_id") and v["refuting_id"] not in onr: onr.append(v["refuting_id"])
    rows.append({"set": st, "key": k, "cluster": cl, "src": p["src"], "B": B, "pool_n": len(pool),
                 "rank": {"claimnov": cn, "dense": dense, "bm25": bm, "opennov": onr}, "levels": best})
def metr(R, B):
    r = R.index(B) + 1 if B in R else None
    return {"P@1": (1.0 if r == 1 else 0.0) if R else None, "R@1": float(r == 1), "R@3": float(bool(r and r <= 3)), "R@5": float(bool(r and r <= 5)),
            "MRR": 1.0 / r if r else 0.0, "returned": float(bool(R))}
def summarize(sub, reps=10000, seed=0):
    rng = np.random.default_rng(seed); out = {}
    cls = sorted({r["cluster"] for r in sub}, key=str); idx = collections.defaultdict(list)
    for i, r in enumerate(sub): idx[r["cluster"]].append(i)
    PICKS = [np.concatenate([idx[cls[j]] for j in rng.integers(0, len(cls), len(cls))]) for _ in range(reps)]
    for m in ("claimnov", "dense", "bm25", "opennov"):
        M = [metr(r["rank"][m], r["B"]) for r in sub]
        res = {}
        for key in ("P@1", "R@1", "R@3", "R@5", "MRR"):
            vals = np.array([x[key] if x[key] is not None else np.nan for x in M], float)
            pt = np.nanmean(vals); bs = []
            for pick in PICKS:
                bs.append(np.nanmean(vals[pick]))
            den = int(np.sum(~np.isnan(vals)))
            res[key] = {"v": round(float(pt), 3), "CI": [round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)],
                        "num": int(np.nansum(vals)) if key != "MRR" else None, "den": den}
        res["items_returning"] = int(sum(x["returned"] for x in M))
        out[m] = res
    return out
def bands(sub):
    out = {}
    for name, f in ((">=3", lambda l: l >= 3), (">=4", lambda l: l >= 4), ("=5", lambda l: l == 5)):
        fire = [r for r in sub if any(f(l) for l in r["levels"].values())]
        papers = sum(sum(1 for l in r["levels"].values() if f(l)) for r in sub)
        relp = sum(1 for r in sub if f(r["levels"].get(r["B"], 0)))
        out[name] = {"items": len(sub), "items_firing": len(fire), "papers_at_band": papers, "relevant_papers_at_band": relp,
                     "precision": round(relp / papers, 3) if papers else None, "recall_items": f"{relp}/{len(sub)}",
                     "precision_frac": f"{relp}/{papers}"}
    return out
res = {}
SUBS = {"claim_all160": [r for r in rows if r["set"] == "claim"], "claim_exact": [r for r in rows if r["set"] == "claim" and r["src"] == "exact"],
        "hyb_all100": [r for r in rows if r["set"] == "hyb"],
        "disg_srcin_exact": [r for r in rows if r["set"] == "disg" and r["src"] == "exact"],
        "disg_srcin_all": [r for r in rows if r["set"] == "disg"]}
for n, sub in SUBS.items():
    res[n] = {"n": len(sub), "clusters": len({r["cluster"] for r in sub}), "metrics": summarize(sub), "claimnov_bands": bands(sub)}
    print(n, len(sub), {m: (v["R@1"]["v"], v["R@5"]["v"], v["MRR"]["v"], v["P@1"]["v"]) for m, v in res[n]["metrics"].items()}, flush=True)
    print("   bands", {b: (v["items_firing"], v["precision_frac"], v["recall_items"]) for b, v in res[n]["claimnov_bands"].items()}, flush=True)
lat = collections.Counter()
for (st, k), p in P3.items():
    if st != "later" or not p.get("prompt"): continue
    pool = apool(k.split("::")[0]); _, parsed = S.parse_profile_prompt(p["prompt"], [x["title"] for x in pool])
    mx = max([lv for _, _, e in parsed for _, _, lv in e] or [0])
    lat[">=3"] += mx >= 3; lat[">=4"] += mx >= 4; lat["=5"] += mx == 5; lat["n"] += 1; lat["exact"] += p["src"] == "exact"
res["later_band_firing"] = dict(lat)
json.dump(res, open(f"{OUT}/r4_x3_ir.json", "w"), indent=1)
