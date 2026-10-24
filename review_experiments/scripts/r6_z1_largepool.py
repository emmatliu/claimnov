import json, os, sys, re, random, threading, hashlib, collections
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("R6_EXP", "z1")
import r6_shipped as S
import r6_common as R
config, gate, retrieve, pipeline, entail, llm = S.config, S.gate, S.retrieve, S.pipeline, S.entail, S.llm
MODE = sys.argv[1]; RUNGS = [int(x) for x in (sys.argv[2:] or ["1", "2", "3", "4"])]
PILOT = f"{S.ROOT}/eval"; PC = f"{PILOT}/cache"
def emb3s(t):
    fp = f"{PC}/e_{hashlib.sha256(('text-embedding-3-small::' + t).encode()).hexdigest()[:24]}.json"
    return np.array(json.load(open(fp)), float) if os.path.exists(fp) else None
nt = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())[:60]
def claims_of(t, a): return llm.cache_get("claims4::" + S.extract._key(t, a))
U = json.load(open(f"{R.OUT}/r2_e11_universe.json")); UP = U["papers"]
UV = np.array([emb3s(f"{p['title']}. {p['abstract']}") for p in UP]); UV /= np.linalg.norm(UV, axis=1, keepdims=True)
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
r3u = [u for u in units_for(3) if claims_of(u["title"], u["desc"])]
seeds = sorted({u["seed"] for u in r3u if u["seed"] >= 10}); random.Random(0).shuffle(seeds); SEEDS = set(seeds[:40])
def enlarge(u, nadd=180):
    q = emb3s(f"{u['title']}: {u['desc']}"); q = q / np.linalg.norm(q)
    have = {p["id"] for p in u["pool"]} | {nt(p["title"]) for p in u["pool"]}
    add = []
    for i in np.argsort(-(UV @ q)):
        p = UP[i]
        if p["id"] in have or nt(p["title"]) in have or nt(p["title"]) == nt(u["B"]["title"]): continue
        have.add(p["id"]); have.add(nt(p["title"])); add.append(p)
        if len(add) == nadd: break
    return u["pool"] + add
UU = [u for r in RUNGS for u in units_for(r) if u["seed"] in SEEDS and claims_of(u["title"], u["desc"])]
if os.environ.get("ONLY_SEEDS"): UU = [u for u in UU if u["seed"] in {int(x) for x in os.environ["ONLY_SEEDS"].split(",")}]
CONDS = os.environ.get("CONDS", "B_in,B_out").split(",")
if os.environ.get("REPAIR_EMPTY") == "1":
    for u in UU:
        for p in u["pool"]:
            ck = "claims4::" + S.extract._key(p["title"], p["abstract"])
            if llm.cache_get(ck) == []:
                with llm._cache_lock: llm._cache.pop(ck, None)
                print("re-extract", p["title"][:60], "->", len(S.extract.extract_claims({"id": p["id"], "title": p["title"], "abstract": p["abstract"]})), flush=True)
    S.flush()
print(len(UU), "disguises", dict(collections.Counter(u["rung"] for u in UU)), flush=True)
def corpus_of(recs): return [{"id": r["id"], "title": r["title"], "abstract": r["abstract"]} for r in recs]
def ekey(t, c):
    return "entail::" + hashlib.md5((t["claim"] + "||" + c["claim"] + "||" + entail._context(t) + "||" + entail._context(c)).encode()).hexdigest()[:16]
def target_of(u, key): return {"id": key, "title": u["title"], "abstract": u["desc"]}
def pairs_needed(u, recs, key):
    fac = retrieve.build_pool(corpus_of(recs)); tgt = target_of(u, key); tot = new = 0
    for c in claims_of(u["title"], u["desc"]):
        c = {**c, "paper_abstract": u["desc"]}
        for f in retrieve.top_k_works(c, fac):
            tot += 1; new += llm.cache_get(ekey(c, f)) is None
    return tot, new
def lvB_of(rep, B):
    return max([j["entailment_level"] for c in rep["claims"] for j in c["judged"][:5] if j["paper_id"] == B] or [0])

if MODE == "dry":
    tot = new = 0; seen = set()
    for i, u in enumerate(UU):
        big = enlarge(u)
        for cond, recs in (("B_in", big), ("B_out", [p for p in big if p["id"] != u["B"]["id"]])):
            fac = retrieve.build_pool(corpus_of(recs))
            for c in claims_of(u["title"], u["desc"]):
                c = {**c, "paper_abstract": u["desc"]}
                for f in retrieve.top_k_works(c, fac):
                    k = ekey(c, f); tot += 1
                    if llm.cache_get(k) is None and k not in seen: new += 1; seen.add(k)
        if i % 10 == 0: print(i, tot, new, S.summary(), flush=True)
    print("pairs", tot, "new unique uncached", new, S.summary(), flush=True)

if MODE == "val20":
    P3 = {p["key"]: p for p in map(json.loads, open(f"{R.OUT}/r3_prompts.jsonl")) if p["set"] == "disg"}
    I = json.load(open(f"{R.OUT}/r0_items.json"))["items"]["claimnov"]
    res = []; cnt = collections.Counter()
    def one(u):
        rows = []
        for cond in ("B_in", "B_out"):
            key = u["key"].replace("B_in", cond)
            recs = u["pool"] if cond == "B_in" else [p for p in u["pool"] if not p.get("is_B")]
            fac = retrieve.build_pool(corpus_of(recs)); tgt = target_of(u, key)
            rep = pipeline.score_paper(tgt, fac)
            box = {}
            def cap(pr): box["p"] = pr; return '{"decision": "novel"}'
            r = gate.profile_veto(tgt, recs, rep, chat=cap)
            p3 = P3.get(key) or {}
            rows.append({"key": key, "rung": u["rung"], "src": p3.get("src"), "prompt_equal_exact": (box["p"] == p3.get("prompt")) if p3.get("src") == "exact" else None,
                         "prompt_equal_recon": (box["p"] == p3.get("prompt")) if p3.get("src") == "recon" else None,
                         "B_in_profile": u["B"]["id"] in r["candidates"], "paper_B_in_profile": (I["disg_in"].get(key) or {}).get("B_in_profile"),
                         "B_level": lvB_of(rep, u["B"]["id"]) if cond == "B_in" else None, "prompt": box["p"]})
        return rows
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "6"))) as ex:
        for rows in ex.map(one, UU): res += rows
    S.flush()
    for r in res:
        cnt[f"{r['src']}"] += 1
        if r["prompt_equal_exact"] is not None: cnt[f"exact_equal_{r['prompt_equal_exact']}"] += 1
        if r["prompt_equal_recon"] is not None: cnt[f"recon_equal_{r['prompt_equal_recon']}"] += 1
        if r["key"].endswith("B_in"): cnt[f"B_in_profile_{r['B_in_profile']}_paper_{r['paper_B_in_profile']}"] += 1
    json.dump({"counts": dict(cnt), "rows": res}, open(f"{R.OUT}/r6_z1_val20.json", "w"), indent=1)
    print(dict(cnt), S.summary(), flush=True)

if MODE == "run":
    OUTF = os.environ.get("R6_Z1_OUT", f"{R.OUT}/r6_z1_run.jsonl")
    done = {(json.loads(l)["key"], json.loads(l)["pool"]) for l in open(OUTF) if not json.loads(l).get("error")} if os.path.exists(OUTF) else set()
    fo = open(OUTF, "a"); lock = threading.Lock()
    def emit(rec):
        with lock: fo.write(json.dumps(rec) + "\n"); fo.flush()
    def one(u):
        big = enlarge(u)
        for cond in CONDS:
            key = u["key"].replace("B_in", cond)
            if (key, "L") in done: continue
            recs = big if cond == "B_in" else [p for p in big if p["id"] != u["B"]["id"]]
            try:
                fac = retrieve.build_pool(corpus_of(recs)); tgt = target_of(u, key)
                rep = pipeline.score_paper(tgt, fac)
                box = {}
                def chat(pr):
                    box["p"] = pr; txt, meta = R.call("sonnet", pr, 1500, os.environ.get("TAGP", "r6z1_L_") + key); box["meta"] = meta; box["txt"] = txt; return txt
                r = gate.profile_veto(tgt, recs, rep, chat=chat)
                ev = r.get("evidence") or {}
                bid = u["B"]["id"]
                emit({"key": key, "rung": u["rung"], "seed": u["seed"], "pool": "L", "check": "claimnov_shipped", "n_pool": len(recs),
                      "rej": bool(r["veto"]), "named": bool(r["veto"] and ev.get("id") == bid), "anticipating": ev.get("title"),
                      "B_in_profile": bid in r["candidates"], "B_level_in_profile": lvB_of(rep, bid),
                      "B_retrieved_any_claim": any(j["paper_id"] == bid for c in rep["claims"] for j in c["judged"]),
                      "B_max_level_judged": max([j["entailment_level"] for c in rep["claims"] for j in c["judged"] if j["paper_id"] == bid] or [0]),
                      "k_per_claim": [c["candidates_retrieved"] for c in rep["claims"]], "n_claims": rep["n_claims"],
                      "profile": [[(j["paper_id"], j["entailment_level"]) for j in c["judged"][:5]] for c in rep["claims"]],
                      "verdict_text": box.get("txt"), "meta": box.get("meta")})
            except (R.Budget, S.OAIBudget) as e:
                print("BUDGET", e, flush=True); return
            except Exception as e:
                emit({"key": key, "pool": "L", "check": "claimnov_shipped", "error": str(e)[:300]})
        S.flush()
    try:
        with ThreadPoolExecutor(int(os.environ.get("WORKERS", "4"))) as ex: list(ex.map(one, UU))
    finally:
        S.flush(); print(R.summary(), S.summary(), flush=True)
