import json, re, ast, collections
import numpy as np
from rank_bm25 import BM25Okapi
from r0_common import *

I = json.load(open(f"{OUT}/r0_items.json"))["items"]
DROP = {60, 83, 86, 87, 93}
STOP = set("a an and are as at be by for from has have in is it of on or that the this to was were will with we our can into using use via".split())
tok = lambda s: [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]
src = open(f"{PILOT}/make_disguises.py").read()
R1 = next(ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "R1")
pairs = [json.loads(l) for l in open("eval/data/claim_disputed/testset_clean.jsonl")]

U = []
dz = {f"seed{d['seed_idx']}_r{d['rung']}": d for d in json.load(open(f"{PILOT}/data/disguises_v3.json"))}
for k in I["claimnov"]["disg_in"]:
    s = seed_of(k)
    if s in DROP: continue
    d = dz[k.rsplit("_B_", 1)[0]]
    pin = json.load(open(f"{PILOT}/data/pools/seed{s}.json"))["pool"]; B = next(p["id"] for p in pin if p.get("is_B"))
    U.append(("in", k, f"s{s}", f"{d['title']}: {d['description']}", pin, B))
    U.append(("out", k.replace("B_in", "B_out"), f"s{s}", f"{d['title']}: {d['description']}", [p for p in pin if not p.get("is_B")], None))
_ap = {}
def apool(slug):
    if slug not in _ap: _ap[slug] = json.load(open(f"{anchor_dir(slug)}/pool_preT.json"))[:400]
    return _ap[slug]
for k in I["claimnov"]["later"]:
    slug, idx = k.split("::")
    idea = next(i for i in json.load(open(f"{anchor_dir(slug)}/ideas.json")) if i["idx"] == int(idx))
    U.append(("later", k, f"a{slug}", f"{idea['title']}: {idea['description']}", apool(slug), None))
for k in I["claimnov"]["claim"]:
    i = int(k.split("_")[1])
    d = json.loads(re.search(r"\{.*\}", cached_bedrock(SONNET_46_BEDROCK, [{"role": "user", "content": R1.format(abstract=pairs[i]["A_abstract"])}], 3000, f"n2f_ideaify{i}"), re.S).group(0))
    pool = json.load(open(f"{PILOT}/data/pools_full/{k}.json"))["pool"]
    U.append(("claim", k, k, f"{d['title']}: {d['description']}", pool, next((p["id"] for p in pool if p.get("is_B")), None)))
for h in json.load(open(f"{PILOT}/data/hybrids.json")):
    U.append(("hyb", h["item"], h["anchor"], f"{h['title']}: {h['description']}", apool(h["anchor"]), h["B_id"]))

alldocs = {}
for u in U:
    for p in u[4]: alldocs[p["id"]] = f"{p['title']}. {p['abstract']}"
N = len(alldocs); df = collections.Counter()
for t in alldocs.values(): df.update(set(tok(t)))
avgdl = np.mean([len(tok(t)) for t in alldocs.values()])
gidf = {w: math.log((N - n + 0.5) / (n + 0.5) + 1) for w, n in df.items()}
def bm25_global(q, docs, k1=1.5, b=0.75):
    qt = tok(q); out = []
    for d in docs:
        dt = tok(d); tf = collections.Counter(dt); dl = len(dt)
        out.append(sum(gidf.get(w, 0) * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * dl / avgdl)) for w in qt if w in tf))
    return out

S = {}; miss = 0
for st, k, g, q, pool, B in U:
    docs = [f"{p['title']}. {p['abstract']}" for p in pool]
    if not pool:
        S[(st, k)] = {"cos": (-1, None), "bm25": (0, None), "bm25g": (0, None)}; continue
    qv = cached_embed(q)
    if qv is None: miss += 1; cosv = None
    else:
        qv = np.array(qv, float); qv /= np.linalg.norm(qv)
        cs = []
        for p, dtext in zip(pool, docs):
            v = cached_embed(dtext)
            if v is None: miss += 1; cs.append(-1); continue
            v = np.array(v, float); cs.append(float(qv @ v / np.linalg.norm(v)))
        cosv = cs
    bm = BM25Okapi([tok(d) for d in docs], k1=1.5, b=0.75).get_scores(tok(q))
    bg = bm25_global(q, docs)
    S[(st, k)] = {}
    for name, sc in (("cos", cosv), ("bm25", list(bm)), ("bm25g", bg)):
        if sc is None: S[(st, k)][name] = (None, None); continue
        j = int(np.argmax(sc)); S[(st, k)][name] = (float(sc[j]), pool[j]["id"])
print("embedding cache misses:", miss)

G = {(st, k): g for st, k, g, q, pool, B in U}; BB = {(st, k): B for st, k, g, q, pool, B in U}
def keys(st): return [k for (s, k) in S if s == st]
def table(m, th, sub=None):
    ok = (lambda st, k: True) if sub is None else (lambda st, k: G[(st, k)] in sub)
    rej = lambda st, k: S[(st, k)][m][0] is not None and S[(st, k)][m][0] >= th
    din = {k: rej("in", k) for k in keys("in") if ok("in", k)}
    dout = {k: rej("out", k) for k in keys("out") if ok("out", k)}
    lat = {k: rej("later", k) for k in keys("later") if ok("later", k)}
    named = {k: din[k] and S[("in", k)][m][1] == BB[("in", k)] for k in din}
    return din, dout, lat, named
def metr(din, dout, lat, named):
    c, so, fa = sum(din.values()), sum(dout.values()), sum(lat.values())
    n, nl = len(din) or 1, len(lat) or 1
    return {"caught": c, "named": sum(named.values()), "src_out": so, "FA": fa, "n_in": len(din), "n_later": len(lat),
            "J": round(c / n - fa / nl, 4), "Delta": round((c - so) / n, 4),
            "rungs": [sum(v for kk, v in din.items() if f"_r{q}_" in kk) for q in (1, 2, 3, 4)]}
def cands(m, sub=None):
    vals = sorted({S[(st, k)][m][0] for (st, k) in S if st in ("in", "out", "later") and S[(st, k)][m][0] is not None and (sub is None or G[(st, k)] in sub)})
    mids = [(a + b) / 2 for a, b in zip(vals, vals[1:])]
    return [vals[0] - 1e-6] + mids + [vals[-1] + 1e-6]
def best(m, crit="J", sub=None):
    bt, bv = None, -9
    for th in cands(m, sub):
        r = metr(*table(m, th, sub)); v = r[crit]
        if v > bv: bt, bv = th, v
    return bt
def passes(m, th, st):
    ks = keys(st); return sum(not (S[(st, k)][m][0] is not None and S[(st, k)][m][0] >= th) for k in ks), len(ks)
def names_B(m, th, st):
    ks = keys(st)
    return sum(S[(st, k)][m][1] == BB[(st, k)] for k in ks), sum(S[(st, k)][m][1] == BB[(st, k)] and not (S[(st, k)][m][0] >= th) for k in ks)

res = {"n_items": collections.Counter(s for s, k in S), "embedding_cache_misses": miss}
for m in ("cos", "bm25", "bm25g"):
    o = {}
    for crit in ("J", "Delta"):
        th = best(m, crit); r = metr(*table(m, th))
        r["theta"] = round(th, 4)
        r["claim_pass"] = passes(m, th, "claim"); r["hyb_pass"] = passes(m, th, "hyb")
        r["claim_top1_is_B / on pass"] = names_B(m, th, "claim"); r["hyb_top1_is_B / on pass"] = names_B(m, th, "hyb")
        o[f"insample_best_{crit}"] = r
    rng = np.random.default_rng(0)
    srcs = sorted({G[("in", k)] for k in keys("in")}); ancs = sorted({G[("later", k)] for k in keys("later")})
    rng.shuffle(srcs); rng.shuffle(ancs)
    fold = {g: i % 5 for i, g in enumerate(srcs)}; fold.update({g: i % 5 for i, g in enumerate(ancs)})
    agg = {"din": {}, "dout": {}, "lat": {}, "named": {}}; ths = []
    for f in range(5):
        train = {g for g, ff in fold.items() if ff != f}; test = {g for g, ff in fold.items() if ff == f}
        th = best(m, "J", train); ths.append(round(th, 4))
        din, dout, lat, named = table(m, th, test)
        agg["din"].update(din); agg["dout"].update(dout); agg["lat"].update(lat); agg["named"].update(named)
    r = metr(agg["din"], agg["dout"], agg["lat"], agg["named"]); r["fold_thetas"] = ths
    o["cv5_grouped_best_J"] = r
    def auc(pos, neg):
        pos, neg = np.array(pos), np.array(neg)
        return round(float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))), 4)
    sc = lambda st: [S[(st, k)][m][0] for k in keys(st)]
    o["AUC_in_vs_later"] = auc(sc("in"), sc("later")); o["AUC_in_vs_out"] = auc(sc("in"), sc("out"))
    o["argmax_is_source_srcin"] = sum(S[("in", k)][m][1] == BB[("in", k)] for k in keys("in"))
    o["median_score"] = {st: round(float(np.median(sc(st))), 4) for st in ("in", "out", "later", "claim", "hyb")}
    res[m] = o
dump("r2_e9_similarity.json", res)
print(json.dumps(res, indent=1))
