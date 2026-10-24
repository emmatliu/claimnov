import json, os, sys, re, random, threading, hashlib, collections
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("R4_EXP", "x4")
import r4_common as R
import r4_shipped as S
from rank_bm25 import BM25Okapi
config, gate, entail = S.config, S.gate, S.entail
MODE = sys.argv[1]; RUNGS = [int(x) for x in (sys.argv[2:] or ["1", "2", "4"])]
PILOT = S.PILOT; PC = S.PCACHE
def emb(t):
    fp = f"{PC}/e_{hashlib.sha256(('text-embedding-3-small::' + t).encode()).hexdigest()[:24]}.json"
    return np.array(json.load(open(fp)), float) if os.path.exists(fp) else None
STOPF = set("a an and are as at be by for from has have in is it of on or that the this to was were will with we our can into using use via".split())
tokf = lambda s: [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOPF]
nt = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())[:60]
U = json.load(open(f"{R.OUT}/r2_e11_universe.json")); UP = U["papers"]
UV = np.array([emb(f"{p['title']}. {p['abstract']}") for p in UP]); UV /= np.linalg.norm(UV, axis=1, keepdims=True)
DZ = json.load(open(f"{PILOT}/data/disguises_v3.json"))
KEYS = json.load(open(f"{R.OUT}/r0_items.json"))["items"]["claimnov"]["disg_in"]
DROP = {60, 83, 86, 87, 93}
def units_for(rung):
    out = []
    for d in DZ:
        k = f"seed{d['seed_idx']}_r{d['rung']}_B_in"
        if d["rung"] != rung or k not in KEYS or d["seed_idx"] in DROP: continue
        pin = json.load(open(f"{PILOT}/data/pools/seed{d['seed_idx']}.json"))["pool"]
        out.append(dict(key=k, seed=d["seed_idx"], rung=rung, title=d["title"], desc=d["description"], pool=pin, B=next(p for p in pin if p.get("is_B"))))
    return out
r3u = [u for u in units_for(3) if S.claims_of(u["title"], u["desc"])]
seeds = sorted({u["seed"] for u in r3u if u["seed"] >= 10}); random.Random(0).shuffle(seeds); SEEDS = set(seeds[:40])
def enlarge(u, nadd=180):
    q = emb(f"{u['title']}: {u['desc']}"); q = q / np.linalg.norm(q)
    have = {p["id"] for p in u["pool"]} | {nt(p["title"]) for p in u["pool"]}
    add = []
    for i in np.argsort(-(UV @ q)):
        p = UP[i]
        if p["id"] in have or nt(p["title"]) in have or nt(p["title"]) == nt(u["B"]["title"]): continue
        have.add(p["id"]); have.add(nt(p["title"])); add.append(p)
        if len(add) == nadd: break
    return u["pool"] + add
def facets_of(pool):
    return [{**c, "paper_id": p["id"], "paper_title": p["title"], "paper_abstract": p["abstract"]} for p in pool for c in (S.claims_of(p["title"], p["abstract"]) or [])]
ftext = lambda f: f"{f['attribute']}: {f['value']}" if f.get("value") else f["attribute"]
def bm25_topk(tc, fs):
    cands = [f for f in fs if f["dimension"] == tc["dimension"]]
    if not cands: return []
    k = config.top_k_for(len({f["paper_id"] for f in cands}))
    sc = BM25Okapi([tokf(ftext(f)) for f in cands], k1=1.5, b=0.75).get_scores(tokf(ftext(tc)))
    best = {}
    for f, s in zip(cands, sc):
        if f["paper_id"] not in best or s > best[f["paper_id"]][1]: best[f["paper_id"]] = (f, s)
    return [f for f, s in sorted(best.values(), key=lambda x: -x[1])[:k]]
def report(u, pool, judge=True):
    fs = facets_of(pool); claims = []; pairs = 0
    for c in (S.claims_of(u["title"], u["desc"]) or []):
        t = {**c, "paper_abstract": u["desc"]}
        cands = bm25_topk(t, fs); pairs += len(cands)
        if not judge: continue
        with ThreadPoolExecutor(4) as ex: lv = list(ex.map(lambda f: S.haiku_level(t, f, tag="r4x4ent")[0], cands))
        js = sorted([{"paper_id": f["paper_id"], "paper_title": f["paper_title"], "claim": f["claim"], "entailment_level": l} for f, l in zip(cands, lv)],
                    key=lambda x: -x["entailment_level"])
        claims.append({"claim": c["claim"], "dimension": c["dimension"], "judged": js})
    return {"claims": claims}, pairs
OUTF = f"{R.OUT}/r4_x4_run.jsonl"
done = {(json.loads(l)["key"], json.loads(l)["pool"], json.loads(l)["check"]) for l in open(OUTF) if not json.loads(l).get("error")} if os.path.exists(OUTF) else set()
fo = open(OUTF, "a"); lock = threading.Lock()
def emit(rec):
    with lock: fo.write(json.dumps(rec) + "\n"); fo.flush()
NSUB = int(os.environ.get("NSUB_R12", "20"))
SEEDS_SUB = set(seeds[:NSUB])
UU = [u for r in RUNGS for u in units_for(r) if (u["seed"] in (SEEDS if r in (3, 4) else SEEDS_SUB)) and S.claims_of(u["title"], u["desc"])]
print(len(UU), "disguises", collections.Counter(u["rung"] for u in UU), flush=True)
TOKS_ON = None
if MODE == "dry":
    tot = 0; newp = 0
    for u in UU:
        big = enlarge(u)
        for recs in (big, [p for p in big if p["id"] != u["B"]["id"]]):
            fs = facets_of(recs)
            for c in S.claims_of(u["title"], u["desc"]):
                t = {**c, "paper_abstract": u["desc"]}
                for f in bm25_topk(t, fs):
                    tot += 1; newp += (S.HAIKU_E11 + "::" + S.ekey(t, f)) not in S.E11
    print("pairs", tot, "not in E11 cache (upper bound on new Haiku calls; in/out overlap dedups further)", newp)
if MODE == "claimnov":
    def one(u):
        big = enlarge(u)
        for cond in ("B_in", "B_out"):
            key = u["key"].replace("B_in", cond)
            if (key, "L", "claimnov_sub") in done: continue
            recs = big if cond == "B_in" else [p for p in big if p["id"] != u["B"]["id"]]
            try:
                rep, _ = report(u, recs)
                tgt = {"id": key, "title": u["title"], "abstract": u["desc"]}
                r = gate.profile_veto(tgt, recs, rep, chat=lambda pr: R.call("sonnet", pr, 1500, f"r4x4_L_{key}")[0])
                ev = r.get("evidence") or {}
                lvB = max([j["entailment_level"] for c in rep["claims"] for j in c["judged"][:5] if j["paper_id"] == u["B"]["id"]] or [0])
                emit({"key": key, "rung": u["rung"], "seed": u["seed"], "pool": "L", "check": "claimnov_sub", "n_pool": len(recs), "rej": bool(r["veto"]),
                      "named": bool(r["veto"] and ev.get("id") == u["B"]["id"]), "B_in_profile": u["B"]["id"] in r["candidates"], "B_level_in_profile": lvB})
            except R.Budget: return
            except Exception as e: emit({"key": key, "pool": "L", "check": "claimnov_sub", "error": str(e)[:300]})
    try:
        with ThreadPoolExecutor(int(os.environ.get("WORKERS", "4"))) as ex: list(ex.map(one, UU))
    finally: print(R.summary(), flush=True)
if MODE == "jo":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    STOP = set("""a an the and or of to in on for with by from as at is are was were be been being this that these those it its
we our us their they them which who whom whose what when where how than then there here such into onto over under
can could may might will would should shall do does did not no nor but if so also both each either neither via per
more most less least very much many few new novel propose proposed proposes approach method methods paper work using use
used based show shows shown""".split())
    toks = lambda s: [t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(t) > 1 and t not in STOP]
    JO = open(f"{R.EXP}/prompts/e1_judge_only.txt").read()
    def one(u):
        big = enlarge(u)
        for pname, pin in (("20", u["pool"]), ("L", big)):
            for cond in ("B_in", "B_out"):
                key = u["key"].replace("B_in", cond)
                if (key, pname, "jo_bm25") in done: continue
                recs = pin if cond == "B_in" else [p for p in pin if p["id"] != u["B"]["id"]]
                idea = f"{u['title']}: {u['desc']}"
                sc = BM25Okapi([toks(f"{r['title']}. {r.get('abstract') or ''}") for r in recs]).get_scores(toks(idea))
                top = [recs[i] for i in sorted(range(len(recs)), key=lambda i: -sc[i])[:8]]
                pr = JO.format(idea=idea, papers="\n".join(f"- {p['title']}: {(p.get('abstract') or '')[:700]}" for p in top))
                try:
                    txt, meta = R.call("sonnet", pr, 1500, f"r4x4jo_{pname}_{key}")
                    v = R.parse_verdict(txt, recs)
                    emit({"key": key, "rung": u["rung"], "seed": u["seed"], "pool": pname, "check": "jo_bm25", "n_pool": len(recs), "rej": v["rej"],
                          "named": bool(v["rej"] and v["anticipating_id"] == u["B"]["id"]), "B_in_top8": any(p["id"] == u["B"]["id"] for p in top), "meta": meta})
                except R.Budget: return
                except Exception as e: emit({"key": key, "pool": pname, "check": "jo_bm25", "error": str(e)[:300]})
    try:
        with ThreadPoolExecutor(int(os.environ.get("WORKERS", "6"))) as ex: list(ex.map(one, UU))
    finally: print(R.summary(), flush=True)
if MODE == "coi":
    import common as C
    from harness import gates
    from harness.sandbox import Pool
    import harness.sandbox as SB
    llm = C.llm
    def cache_only_embed(texts, model="text-embedding-3-small"):
        out = []
        for t in texts:
            fp = os.path.join(llm.CACHE, f"e_{hashlib.sha256((model + '::' + t).encode()).hexdigest()[:24]}.json")
            if not os.path.exists(fp): raise RuntimeError("embedding cache miss")
            out.append(json.load(open(fp)))
        return out
    llm.embed = cache_only_embed; SB.llm.embed = cache_only_embed
    SON = llm.SONNET_46
    def chat(model, messages, max_tokens=2000, temperature=0.0, cache_tag=""):
        msgs = [m for m in messages if m["role"] != "system"]
        for fp in (C.bedrock_cache_path(SON, msgs[0]["content"], max_tokens, cache_tag), C.ant_cache_path("claude-sonnet-4-6", msgs[0]["content"], max_tokens, cache_tag)):
            if os.path.exists(fp): return json.load(open(fp))["text"]
        if cache_tag.startswith("coiq"): raise RuntimeError("CoI query call not cached")
        return R.call("sonnet", msgs[0]["content"], max_tokens, f"r4x4_{cache_tag}")[0]
    gates.llm.chat = chat; gates.JUDGE = SON
    def one(u):
        big = enlarge(u)
        for cond in ("B_in", "B_out"):
            key = u["key"].replace("B_in", cond)
            if (key, "L", "coi") in done: continue
            recs = big if cond == "B_in" else [p for p in big if p["id"] != u["B"]["id"]]
            try:
                r = gates.coi_gate({"title": u["title"], "description": u["desc"]}, Pool(recs), tag=f"sn_{key}")
                emit({"key": key, "rung": u["rung"], "seed": u["seed"], "pool": "L", "check": "coi", "n_pool": len(recs), "rej": not r["novel"],
                      "named": bool((not r["novel"]) and nt(r.get("evidence")) == nt(u["B"]["title"]))})
            except R.Budget: return
            except Exception as e: emit({"key": key, "pool": "L", "check": "coi", "error": str(e)[:300]})
    try:
        with ThreadPoolExecutor(int(os.environ.get("WORKERS", "6"))) as ex: list(ex.map(one, UU))
    finally: print(R.summary(), flush=True)
