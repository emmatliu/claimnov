import json, os, sys, re, hashlib, random, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SCR = "review_experiments/out/scratch"
os.environ.setdefault("HF_HOME", f"{SCR}/hf")
from rank_bm25 import BM25Okapi
import r5_common as R
PILOT = "./eval"
OUT = R.OUT
STOP = set("""a an the and or of to in on for with by from as at is are was were be been being this that these those it its
we our us their they them which who whom whose what when where how than then there here such into onto over under
can could may might will would should shall do does did not no nor but if so also both each either neither via per
more most less least very much many few new novel propose proposed proposes approach method methods paper work using use
used based show shows shown""".split())
toks = lambda s: [t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(t) > 1 and t not in STOP]
PCACHE = f"{PILOT}/cache"
def emb3(t):
    fp = f"{PCACHE}/e_{hashlib.sha256(('text-embedding-3-small::' + t).encode()).hexdigest()[:24]}.json"
    return np.array(json.load(open(fp)), float) if os.path.exists(fp) else None
DROP = {60, 83, 86, 87, 93}
def adir(s): return next(p for p in (f"{PILOT}/data/anchors/{s}", f"{PILOT}/data/anchors_full/{s}") if os.path.exists(f"{p}/ideas.json"))
_ap = {}
def apool(s):
    if s not in _ap: _ap[s] = json.load(open(f"{adir(s)}/pool_preT.json"))[:400]
    return _ap[s]
P3 = {(p["set"], p["key"]): p for p in map(json.loads, open(f"{OUT}/r3_prompts.jsonl"))}
KEEP = json.load(open(f"{OUT}/r0_items.json"))["items"]["claimnov"]

items = []
for d in json.load(open(f"{PILOT}/data/disguises_v3.json")):
    k = f"seed{d['seed_idx']}_r{d['rung']}_B_in"
    if k not in KEEP["disg_in"] or d["seed_idx"] in DROP: continue
    pool = json.load(open(f"{PILOT}/data/pools/seed{d['seed_idx']}.json"))["pool"]
    B = next(x["id"] for x in pool if x.get("is_B"))
    p3 = P3.get(("disg", k), {})
    items.append(dict(set="disg_in", key=k, cluster=d["seed_idx"], title=d["title"], text=d["description"], pool=pool, rel=B, src=p3.get("src", "none")))
    items.append(dict(set="disg_out", key=k.replace("B_in", "B_out"), cluster=d["seed_idx"], title=d["title"], text=d["description"],
                      pool=[x for x in pool if not x.get("is_B")], rel=None, src=P3.get(("disg", k.replace("B_in", "B_out")), {}).get("src", "none")))
ideas = {}
for it in KEEP["later"]:
    s, i = it.split("::")
    if s not in ideas: ideas[s] = {x["idx"]: x for x in json.load(open(f"{adir(s)}/ideas.json"))}
    x = ideas[s][int(i)]
    items.append(dict(set="later", key=it, cluster=s, title=x["title"], text=x["description"], pool=apool(s), rel=None, src=P3.get(("later", it), {}).get("src")))
for h in json.load(open(f"{PILOT}/data/hybrids.json")):
    items.append(dict(set="hyb", key=h["item"], cluster=h["anchor"], title=h["title"], text=h["description"], pool=apool(h["anchor"]), rel=h["B_id"], src="all"))
import common as C
for u in C.claim_units():
    if u["B_id"] is None: continue
    items.append(dict(set="claim", key=u["key"], cluster=u["key"], title=u["idea"]["title"], text=u["idea"]["description"], pool=u["recs"], rel=u["B_id"],
                      src=P3.get(("claim", u["key"]), {}).get("src")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(collections.Counter(i["set"] for i in items), flush=True)

def all_texts():
    T = {}
    for it in items:
        T[("q", it["title"], it["text"])] = 1
        for p in it["pool"]: T[("d", p["title"], p.get("abstract") or "")] = 1
    return list(T)
TX = all_texts(); print(len(TX), "texts", flush=True)
hk = lambda t: hashlib.sha1(json.dumps(t).encode()).hexdigest()[:20]

def encode(name):
    fp = f"{OUT}/r5_cache/y3_{name}.npz"
    have = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}
    need = [t for t in TX if hk(t) not in have]
    if need:
        import torch
        from transformers import AutoTokenizer
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        if name == "specter2":
            from adapters import AutoAdapterModel
            tok = AutoTokenizer.from_pretrained("allenai/specter2_base")
            model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
            model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
        else:
            from transformers import AutoModel
            tok = AutoTokenizer.from_pretrained("malteos/scincl"); model = AutoModel.from_pretrained("malteos/scincl")
        model.to(dev).eval()
        bs = 32
        for i in range(0, len(need), bs):
            batch = need[i:i + bs]
            enc = tok([t[1] + tok.sep_token + t[2] for t in batch], padding=True, truncation=True, max_length=512, return_tensors="pt",
                      return_token_type_ids=False).to(dev)
            with torch.no_grad():
                out = model(**enc)
            v = out.last_hidden_state[:, 0, :].float().cpu().numpy()
            for t, e in zip(batch, v): have[hk(t)] = e
            if (i // bs) % 50 == 0: print(name, i, "/", len(need), flush=True)
        np.savez(fp, **have)
    return have

def nrm(v): return v / np.linalg.norm(v)
def ranks_for(E, it, kind):
    if kind == "bm25":
        sc = BM25Okapi([toks(f"{x['title']}. {x.get('abstract') or ''}") for x in it["pool"]]).get_scores(toks(f"{it['title']}: {it['text']}"))
        sc = np.array(sc)
    else:
        if kind == "3small":
            q = emb3(f"{it['title']}: {it['text']}")
            D = [emb3(f"{x['title']}. {x['abstract']}") for x in it["pool"]]
            if q is None or any(d is None for d in D): return None, None
            D = np.array(D)
        else:
            q = E[hk(("q", it["title"], it["text"]))]; D = np.array([E[hk(("d", x["title"], x.get("abstract") or ""))] for x in it["pool"]])
        D = D / np.linalg.norm(D, axis=1, keepdims=True); sc = D @ nrm(q)
    order = [it["pool"][i]["id"] for i in np.argsort(-sc, kind="stable")]
    return order, float(np.max(sc))

def boot(vals, clus, reps=10000, seed=0):
    rng = np.random.default_rng(seed); idx = collections.defaultdict(list)
    for i, c in enumerate(clus): idx[c].append(i)
    cl = list(idx); vals = np.asarray(vals, float); bs = []
    for _ in range(reps):
        pick = np.concatenate([idx[cl[j]] for j in rng.integers(0, len(cl), len(cl))]); bs.append(vals[pick].mean())
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]

res = {"items": dict(collections.Counter(i["set"] for i in items))}
EMB = {}
for name in ("specter2", "scincl"):
    try: EMB[name] = encode(name)
    except Exception as e:
        import traceback; traceback.print_exc(); res[f"{name}_error"] = str(e)[:300]
KINDS = list(EMB) + ["3small", "bm25"]
rows = {k: {} for k in KINDS}
for it in items:
    for k in KINDS:
        o, mx = ranks_for(EMB.get(k), it, k)
        rows[k][(it["set"], it["key"])] = (o, mx)

SUBS = {"claim_160": lambda i: i["set"] == "claim", "hyb_100": lambda i: i["set"] == "hyb",
        "disg_srcin_exact68": lambda i: i["set"] == "disg_in" and i["src"] == "exact",
        "disg_srcin_profiled347": lambda i: i["set"] == "disg_in" and i["src"] in ("exact", "recon"),
        "disg_srcin_all380": lambda i: i["set"] == "disg_in"}
res["ir"] = {}
for sn, f in SUBS.items():
    sub = [i for i in items if f(i)]; res["ir"][sn] = {"n": len(sub), "clusters": len({i["cluster"] for i in sub})}
    for k in KINDS:
        M = collections.defaultdict(list); clus = []; miss = 0
        for it in sub:
            o, _ = rows[k][(it["set"], it["key"])]
            if o is None: miss += 1; continue
            r = o.index(it["rel"]) + 1
            M["R@1"].append(r == 1); M["R@3"].append(r <= 3); M["R@5"].append(r <= 5); M["MRR"].append(1 / r); clus.append(it["cluster"])
        res["ir"][sn][k] = {m: {"v": round(float(np.mean(v)), 3), "CI": boot(v, clus), "num": int(np.sum(v)) if m != "MRR" else None, "den": len(v)} for m, v in M.items()}
        if miss: res["ir"][sn][k]["missing_embeddings"] = miss
        print(sn, k, {m: res["ir"][sn][k][m]["v"] for m in M}, flush=True)

pos = [i for i in items if i["set"] == "disg_in"]; neg = [i for i in items if i["set"] == "later"]; so = {i["key"].replace("B_out", "B_in"): i for i in items if i["set"] == "disg_out"}
res["maxcos"] = {}
for k in KINDS:
    S = lambda it: rows[k][(it["set"], it["key"])][1]
    if any(S(i) is None for i in pos + neg + list(so.values())): res["maxcos"][k] = "missing embeddings"; continue
    fold_of = {}
    for grp in (sorted({i["cluster"] for i in pos}), sorted({i["cluster"] for i in neg})):
        g = list(grp); random.Random(0).shuffle(g)
        for j, c in enumerate(g): fold_of[c] = j % 5
    out = {"catch": [], "srcout": [], "fa": [], "thr": []}; cl = {"catch": [], "srcout": [], "fa": []}
    for fo in range(5):
        tp = [S(i) for i in pos if fold_of[i["cluster"]] != fo]; tn = [S(i) for i in neg if fold_of[i["cluster"]] != fo]
        cands = sorted(set(tp + tn)); best = (-9, None)
        for t in cands:
            j = np.mean([s >= t for s in tp]) - np.mean([s >= t for s in tn])
            if j > best[0]: best = (j, t)
        t = best[1]; out["thr"].append(round(t, 4))
        for i in pos:
            if fold_of[i["cluster"]] == fo:
                out["catch"].append(S(i) >= t); cl["catch"].append(i["cluster"])
                out["srcout"].append(S(so[i["key"]]) >= t); cl["srcout"].append(i["cluster"])
        for i in neg:
            if fold_of[i["cluster"]] == fo: out["fa"].append(S(i) >= t); cl["fa"].append(i["cluster"])
    c, s_, f_ = (float(np.mean(out[x])) for x in ("catch", "srcout", "fa"))
    rng = np.random.default_rng(0)
    def cidx(cs):
        d = collections.defaultdict(list)
        for n, c0 in enumerate(cs): d[c0].append(n)
        return d
    dp, dn = cidx(cl["catch"]), cidx(cl["fa"]); kp, kn = list(dp), list(dn)
    A = {x: np.array(out[x], float) for x in out if x != "thr"}; Jb, Db = [], []
    for _ in range(10000):
        pp = np.concatenate([dp[kp[j]] for j in rng.integers(0, len(kp), len(kp))]); nn = np.concatenate([dn[kn[j]] for j in rng.integers(0, len(kn), len(kn))])
        Jb.append(A["catch"][pp].mean() - A["fa"][nn].mean()); Db.append(A["catch"][pp].mean() - A["srcout"][pp].mean())
    res["maxcos"][k] = {"catch": f"{sum(out['catch'])}/{len(out['catch'])}", "srcout": f"{sum(out['srcout'])}/{len(out['srcout'])}",
                        "fa": f"{sum(out['fa'])}/{len(out['fa'])}", "catch_rate": round(c, 3), "srcout_rate": round(s_, 3), "fa_rate": round(f_, 3),
                        "J": round(c - f_, 3), "J_CI": [round(float(np.percentile(Jb, q)), 3) for q in (2.5, 97.5)],
                        "Delta": round(c - s_, 3), "Delta_CI": [round(float(np.percentile(Db, q)), 3) for q in (2.5, 97.5)], "fold_thresholds": out["thr"],
                        "auc_srcin_vs_later": round(float(np.mean([[a > b for b in [S(i) for i in neg]] for a in [S(i) for i in pos]])), 3)}
    print("maxcos", k, res["maxcos"][k], flush=True)
json.dump(res, open(f"{OUT}/r5_y3_dense.json", "w"), indent=1)
