import json, os, re, sys, random, threading, hashlib, collections, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import common as C
from rank_bm25 import BM25Okapi
from novelty import config, entail, gate
llm = C.llm
STAGE = sys.argv[1]
SUBJ = os.environ.get("SUBJ", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
PRICE = {"us.anthropic.claude-haiku-4-5-20251001-v1:0": (1.0, 5.0), "openai.gpt-oss-120b-1:0": (0.15, 0.60)}
NADD = int(os.environ.get("NADD", "180"))
CAP = float(os.environ.get("CAP", "12"))
ncache = json.load(open(f"{C.ROOT}/src/.cache/novelty_cache.json"))
def _key(*p): return hashlib.md5("|".join(str(x) for x in p).encode()).hexdigest()[:16]
def claims_of(t, a): return ncache.get("claims4::" + _key(t, a))
def emb(t):
    fp = os.path.join(llm.CACHE, f"e_{hashlib.sha256(('text-embedding-3-small::' + t).encode()).hexdigest()[:24]}.json")
    return np.array(json.load(open(fp)), float) if os.path.exists(fp) else None
STOP = set("a an and are as at be by for from has have in is it of on or that the this to was were will with we our can into using use via".split())
tok = lambda s: [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]
nt = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())[:60]
DROP = {60, 83, 86, 87, 93}

def universe():
    fp = f"{C.OUTD}/r2_e11_universe.json"
    if os.path.exists(fp): return json.load(open(fp))
    import glob
    postT = set(); papers = {}
    for f in glob.glob(f"{C.P}/data/pools/seed*.json"):
        for p in json.load(open(f))["pool"]:
            if p.get("is_B"): postT.add(nt(p["title"]))
            else: papers.setdefault(p["id"], p)
    for root in ("anchors", "anchors_full"):
        for adir in glob.glob(f"{C.P}/data/{root}/*"):
            if os.path.exists(f"{adir}/pool_preT.json"):
                for p in json.load(open(f"{adir}/pool_preT.json"))[:400]: papers.setdefault(p["id"], p)
    for f in glob.glob(f"{C.P}/data/pools_full/n2f_*.json"):
        for p in json.load(open(f))["pool"]:
            if not p.get("is_B"): papers.setdefault(p["id"], p)
    keep = [{"id": p["id"], "title": p["title"], "abstract": p["abstract"]} for p in papers.values()
            if nt(p["title"]) not in postT and claims_of(p["title"], p["abstract"]) and emb(f"{p['title']}. {p['abstract']}") is not None]
    json.dump({"papers": keep, "postT_titles": sorted(postT), "n_seen": len(papers)}, open(fp, "w"))
    return json.load(open(fp))
U = universe(); UP = U["papers"]; POSTT = set(U["postT_titles"])
UV = np.array([emb(f"{p['title']}. {p['abstract']}") for p in UP]); UV /= np.linalg.norm(UV, axis=1, keepdims=True)

def disguises():
    dz = json.load(open(f"{C.P}/data/disguises_v3.json"))
    keys = json.load(open(f"{C.OUTD}/r0_items.json"))["items"]["claimnov"]["disg_in"]
    out = []
    for d in dz:
        k = f"seed{d['seed_idx']}_r{d['rung']}_B_in"
        if k not in keys or d["seed_idx"] in DROP: continue
        pin = json.load(open(f"{C.P}/data/pools/seed{d['seed_idx']}.json"))["pool"]
        out.append(dict(key=k, seed=d["seed_idx"], rung=d["rung"], title=d["title"], desc=d["description"], pool=pin,
                        B=next(p for p in pin if p.get("is_B"))))
    return out

def enlarge(u):
    q = emb(f"{u['title']}: {u['desc']}")
    q = q / np.linalg.norm(q)
    have = {p["id"] for p in u["pool"]} | {nt(p["title"]) for p in u["pool"]}
    sims = UV @ q; order = np.argsort(-sims); add = []
    for i in order:
        p = UP[i]
        if p["id"] in have or nt(p["title"]) in have or nt(p["title"]) == nt(u["B"]["title"]): continue
        have.add(p["id"]); have.add(nt(p["title"])); add.append(p)
        if len(add) == NADD: break
    return u["pool"] + add

def facets_of(pool):
    fs = []
    for p in pool:
        for c in (claims_of(p["title"], p["abstract"]) or []):
            fs.append({**c, "paper_id": p["id"], "paper_title": p["title"], "paper_abstract": p["abstract"]})
    return fs
ftext = lambda f: f"{f['attribute']}: {f['value']}" if f.get("value") else f["attribute"]

def bm25_topk(tc, fs):
    cands = [f for f in fs if f["dimension"] == tc["dimension"]]
    if not cands: return []
    k = config.top_k_for(len({f["paper_id"] for f in cands}))
    sc = BM25Okapi([tok(ftext(f)) for f in cands], k1=1.5, b=0.75).get_scores(tok(ftext(tc)))
    best = {}
    for f, s in zip(cands, sc):
        if f["paper_id"] not in best or s > best[f["paper_id"]][1]: best[f["paper_id"]] = (f, s)
    return [f for f, s in sorted(best.values(), key=lambda x: -x[1])[:k]]

ECF = f"{C.OUTD}/r2_e11_entail_cache.json"
ecache = json.load(open(ECF)) if os.path.exists(ECF) else {}
elock = threading.Lock(); ecost = {"in": 0, "out": 0, "calls": 0, "usd": 0.0}
import boto3
_br = {}
def brc():
    reg = os.environ.get("SUBJ_REGION", "us-east-2")
    if reg not in _br: _br[reg] = boto3.client("bedrock-runtime", region_name=reg)
    return _br[reg]
def ekey(t, c):
    return hashlib.md5((t["claim"] + "||" + c["claim"] + "||" + entail._context(t) + "||" + entail._context(c)).encode()).hexdigest()[:16]
def sub_level(t, c):
    k = SUBJ + "::" + ekey(t, c)
    with elock:
        if k in ecache: return ecache[k]
    if ecost["usd"] > CAP: raise RuntimeError("cap")
    pr = entail._prompt(t, c)
    for i in range(8):
        try:
            r = brc().converse(modelId=SUBJ, messages=[{"role": "user", "content": [{"text": pr}]}],
                               inferenceConfig={"maxTokens": 16 if "anthropic" in SUBJ else 2000, "temperature": 0.0}); break
        except Exception as e:
            if i == 7: raise
            time.sleep(min(60, 2 ** i))
    u = r.get("usage", {}); txt = "".join(b.get("text", "") for b in r["output"]["message"]["content"] if "text" in b)
    d = [ch for ch in txt if ch.isdigit()]; lv = int(d[0]) if d else config.ENTAIL_MIN
    lv = max(1, min(5, lv))
    with elock:
        ecost["in"] += u.get("inputTokens", 0); ecost["out"] += u.get("outputTokens", 0); ecost["calls"] += 1
        p = PRICE[SUBJ]; ecost["usd"] = ecost["in"] / 1e6 * p[0] + ecost["out"] / 1e6 * p[1]
        ecache[k] = lv
        if ecost["calls"] % 50 == 0: json.dump(ecache, open(ECF, "w"))
    return lv
def orig_level(t, c): return ncache.get("entail::" + ekey(t, c))

def report(u, pool, judge=sub_level):
    tcl = claims_of(u["title"], u["desc"]) or []
    fs = facets_of(pool); claims = []
    for c in tcl:
        t = {**c, "paper_abstract": u["desc"]}
        cands = bm25_topk(t, fs)
        with ThreadPoolExecutor(4) as ex: lv = list(ex.map(lambda f: judge(t, f), cands))
        js = [{"paper_id": f["paper_id"], "paper_title": f["paper_title"], "claim": f["claim"], "entailment_level": l} for f, l in zip(cands, lv)]
        js = sorted(js, key=lambda x: -x["entailment_level"])
        claims.append({"claim": c["claim"], "dimension": c["dimension"], "judged": js, "n_retrieved": len(cands)})
    return {"claims": claims}

if STAGE == "retr":
    res = collections.Counter(); per = []
    for u in disguises():
        if not claims_of(u["title"], u["desc"]): res["no_target_claims"] += 1; continue
        tcl = claims_of(u["title"], u["desc"])
        big = enlarge(u)
        row = {"key": u["key"], "rung": u["rung"], "pool20": len(u["pool"]), "poolL": len(big)}
        for nm, pool in (("20", u["pool"]), ("L", big)):
            fs = facets_of(pool)
            hit = any(u["B"]["id"] in [f["paper_id"] for f in bm25_topk({**c, "paper_abstract": u["desc"]}, fs)] for c in tcl)
            row[f"B_in_topk_{nm}"] = hit
            row[f"k_{nm}"] = [config.top_k_for(len({f['paper_id'] for f in fs if f['dimension'] == c['dimension']})) for c in tcl]
        q = emb(f"{u['title']}: {u['desc']}"); q /= np.linalg.norm(q)
        sims = sorted(((float(emb(f"{p['title']}. {p['abstract']}") @ q / np.linalg.norm(emb(f"{p['title']}. {p['abstract']}"))), p["id"]) for p in big), reverse=True)
        row["B_cos_rank_L"] = [pid for _, pid in sims].index(u["B"]["id"]) + 1
        per.append(row)
    agg = {"n": len(per), **dict(res)}
    for nm in ("20", "L"):
        agg[f"B_in_some_claim_topk_{nm}"] = sum(r[f"B_in_topk_{nm}"] for r in per)
        agg[f"by_rung_{nm}"] = {q: f"{sum(r[f'B_in_topk_{nm}'] for r in per if r['rung'] == q)}/{sum(1 for r in per if r['rung'] == q)}" for q in (1, 2, 3, 4)}
    agg["poolL_size_median"] = float(np.median([r["poolL"] for r in per])); agg["k_L_median"] = float(np.median([x for r in per for x in r["k_L"]]))
    agg["B_cos_rank_L_median"] = float(np.median([r["B_cos_rank_L"] for r in per]))
    agg["universe_size"] = len(UP)
    json.dump({"agg": agg, "per": per}, open(f"{C.OUTD}/r2_e11_retrieval.json", "w"), indent=1)
    print(json.dumps(agg, indent=1))

elif STAGE == "val":
    rng = random.Random(0); pairs = []
    for u in disguises():
        tcl = claims_of(u["title"], u["desc"]) or []
        fs = facets_of(u["pool"])
        for c in tcl:
            t = {**c, "paper_abstract": u["desc"]}
            for f in fs:
                if f["dimension"] == c["dimension"]:
                    o = orig_level(t, f)
                    if o is not None: pairs.append((t, f, o))
    byl = collections.defaultdict(list)
    for p in pairs: byl[p[2]].append(p)
    samp = []
    for l in sorted(byl): samp += rng.sample(byl[l], min(len(byl[l]), 45))
    with ThreadPoolExecutor(6) as ex: sub = list(ex.map(lambda p: sub_level(p[0], p[1]), samp))
    json.dump(ecache, open(ECF, "w"))
    o = [p[2] for p in samp]
    cm = collections.Counter(zip(o, sub))
    w = lambda a, b: (a - b) ** 2
    import itertools
    n = len(o); po = collections.Counter(o); ps = collections.Counter(sub)
    obs = sum(w(a, b) for a, b in zip(o, sub)) / n; exp = sum(po[a] * ps[b] * w(a, b) for a in range(1, 6) for b in range(1, 6)) / n / n
    out = {"SUBJ": SUBJ, "n": n, "exact": round(sum(a == b for a, b in zip(o, sub)) / n, 3), "within1": round(sum(abs(a - b) <= 1 for a, b in zip(o, sub)) / n, 3),
           "quadratic_kappa": round(1 - obs / exp, 3), "mean_orig": round(float(np.mean(o)), 3), "mean_sub": round(float(np.mean(sub)), 3),
           ">=4_agreement": f"{sum((a >= 4) == (b >= 4) for a, b in zip(o, sub))}/{n}",
           "confusion(orig,sub)": {f"{a}{b}": v for (a, b), v in sorted(cm.items())}, "pool_pairs_available": len(pairs),
           "sample_per_level": {l: min(len(byl[l]), 45) for l in sorted(byl)}, "cost": ecost}
    json.dump(out, open(f"{C.OUTD}/r2_e11_val_{SUBJ.split('.')[1][:20]}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))

elif STAGE == "run":
    from harness import gates
    from harness.sandbox import Pool
    def cache_only_embed(texts, model="text-embedding-3-small"):
        out = []
        for t in texts:
            fp = os.path.join(llm.CACHE, f"e_{hashlib.sha256((model + '::' + t).encode()).hexdigest()[:24]}.json")
            if not os.path.exists(fp): raise RuntimeError("embedding cache miss")
            out.append(json.load(open(fp)))
        return out
    llm.embed = cache_only_embed
    SON = llm.SONNET_46
    def chat(model, messages, max_tokens=2000, temperature=0.0, cache_tag=""):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system") or None
        msgs = [m for m in messages if m["role"] != "system"]
        for fp in (C.bedrock_cache_path(SON, msgs[0]["content"], max_tokens, cache_tag, system), C.ant_cache_path("claude-sonnet-4-6", msgs[0]["content"], max_tokens, cache_tag, system)):
            if len(msgs) == 1 and os.path.exists(fp): return json.load(open(fp))["text"]
        if cache_tag.startswith("coiq"): raise RuntimeError("CoI query call not cached")
        return C.sonnet(msgs, max_tokens, cache_tag, system=system)
    gates.llm.chat = chat
    NSRC = int(os.environ.get("NSRC", "40")); RUNG = int(os.environ.get("RUNG", "3"))
    allu = [u for u in disguises() if u["rung"] == RUNG and claims_of(u["title"], u["desc"])]
    seeds = sorted({u["seed"] for u in allu if u["seed"] >= 10}); random.Random(0).shuffle(seeds); seeds = set(seeds[:NSRC])
    UU = [u for u in allu if u["seed"] in seeds]
    OUTF = f"{C.OUTD}/r2_e11_run.jsonl"
    done = {(json.loads(l)["key"], json.loads(l)["pool"], json.loads(l)["check"]) for l in open(OUTF) if not json.loads(l).get("error")} if os.path.exists(OUTF) else set()
    fo = open(OUTF, "a"); olock = threading.Lock()
    def one(u):
        big = enlarge(u)
        for pname, pin in (("20", u["pool"]), ("L", big)):
            pout = [p for p in pin if p["id"] != u["B"]["id"]]
            for cond, recs in (("B_in", pin), ("B_out", pout)):
                key = u["key"].replace("B_in", cond)
                if (key, pname, "claimnov_sub") not in done:
                    try:
                        rep = report(u, recs)
                        tgt = {"id": key, "title": u["title"], "abstract": u["desc"]}
                        box = {}
                        def cap_chat(pr):
                            box["p"] = pr; return C.sonnet([{"role": "user", "content": pr}], 1500, f"r2e11_{pname}_{key}")
                        r = gate.profile_veto(tgt, recs, rep, chat=cap_chat)
                        ev = r.get("evidence") or {}
                        lvB = max([j["entailment_level"] for c in rep["claims"] for j in c["judged"][:5] if j["paper_id"] == u["B"]["id"]] or [0])
                        rec = {"key": key, "pool": pname, "check": "claimnov_sub", "n_pool": len(recs), "rej": bool(r["veto"]),
                               "named": bool(r["veto"] and ev.get("id") == u["B"]["id"]), "B_in_profile": u["B"]["id"] in r["candidates"],
                               "B_level_in_profile": lvB, "n_judged": sum(len(c["judged"]) for c in rep["claims"]),
                               "max_level_nonB": max([j["entailment_level"] for c in rep["claims"] for j in c["judged"][:5] if j["paper_id"] != u["B"]["id"]] or [0])}
                    except Exception as e:
                        rec = {"key": key, "pool": pname, "check": "claimnov_sub", "error": str(e)[:200]}
                    with olock: fo.write(json.dumps(rec) + "\n"); fo.flush()
                if (key, pname, "coi") not in done:
                    try:
                        idea = {"title": u["title"], "description": u["desc"]}
                        r = gates.coi_gate(idea, Pool(recs), tag=f"sn_{key}" if pname == "20" else f"sn_{key}")
                        rec = {"key": key, "pool": pname, "check": "coi", "n_pool": len(recs), "rej": not r["novel"],
                               "named": bool((not r["novel"]) and nt(r.get("evidence")) == nt(u["B"]["title"]))}
                    except Exception as e:
                        rec = {"key": key, "pool": pname, "check": "coi", "error": str(e)[:200]}
                    with olock: fo.write(json.dumps(rec) + "\n"); fo.flush()
    print(len(UU), "disguises", flush=True)
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "3"))) as ex: list(ex.map(one, UU))
    json.dump(ecache, open(ECF, "w"))
    print("DONE entail cost", ecost, "sonnet cost", llm.cost(), flush=True)
