import json, os, sys, re, threading, hashlib, collections, random
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("R6_EXP", "z2")
import r6_shipped as S
import r6_common as R
config, gate, retrieve, pipeline, entail, llm, extract = S.config, S.gate, S.retrieve, S.pipeline, S.entail, S.llm, S.extract
STAGE = sys.argv[1]
PILOT = f"{S.ROOT}/eval"; PC = f"{PILOT}/cache"
LATER = sorted(json.load(open(f"{PILOT}/data/later_realized_set.json")))
I = json.load(open(f"{R.OUT}/r0_items.json"))["items"]["claimnov"]["later"]
P3 = {p["key"]: p for p in map(json.loads, open(f"{R.OUT}/r3_prompts.jsonl")) if p["set"] == "later"}
def anchor_dir(slug):
    return next(p for p in (f"{PILOT}/data/anchors/{slug}", f"{PILOT}/data/anchors_full/{slug}") if os.path.exists(f"{p}/ideas.json"))
def paper_cache_hit(prompt, tag):
    msgs = [{"role": "user", "content": prompt}]
    for h in (hashlib.sha256(json.dumps(["bedrock", "us.anthropic.claude-sonnet-4-6", None, msgs, 1500, tag]).encode()).hexdigest()[:24],
              hashlib.sha256(json.dumps(["antapi", "claude-sonnet-4-6", None, msgs, 1500, tag]).encode()).hexdigest()[:24]):
        if os.path.exists(f"{PC}/c_{h}.json"): return True
    return False
def ekey(t, c):
    return "entail::" + hashlib.md5((t["claim"] + "||" + c["claim"] + "||" + entail._context(t) + "||" + entail._context(c)).encode()).hexdigest()[:16]
BF = f"{R.OUT}/r6_z2_build.json"

if STAGE in ("dry", "build"):
    slugs = sorted({k.split("::")[0] for k in LATER})
    rows = []; lock = threading.Lock()
    def per_anchor(slug):
        ad = anchor_dir(slug)
        a = json.load(open(f"{ad}/anchor.json"))
        an = {"id": "ANCHOR_" + a["paperId"], "title": a["title"], "abstract": a["abstract"]}
        cached_claims = llm.cache_get("claims4::" + extract._key(an["title"], an["abstract"])) is not None
        pool = json.load(open(f"{ad}/pool_preT.json"))[:400]
        corpus0 = [{"id": r["id"], "title": r["title"], "abstract": r["abstract"]} for r in pool]
        corpus1 = corpus0 + [an]
        assert an["id"] not in {r["id"] for r in pool} and an["title"].lower() not in {r["title"].lower() for r in pool}
        an_claims = extract.extract_claims(an)
        fac0 = retrieve.build_pool(corpus0); fac1 = retrieve.build_pool(corpus1)
        ideas = {i["idx"]: i for i in json.load(open(f"{ad}/ideas.json"))}
        out = []
        for k in [k for k in LATER if k.split("::")[0] == slug]:
            idea = ideas[int(k.split("::")[1])]
            tgt = {"id": "idea", "title": idea["title"], "abstract": idea["description"]}
            rec = {"key": k, "anchor": slug, "anchor_claims_src": "cached_shipped" if cached_claims else "new_shipped_gpt-4o-mini",
                   "n_anchor_claims": len(an_claims)}
            if STAGE == "dry":
                for nm, fac in (("0", fac0), ("1", fac1)):
                    tot = new = 0; a_ret = 0
                    for c in llm.cache_get("claims4::" + extract._key(idea["title"], idea["description"])) or []:
                        c = {**c, "paper_abstract": idea["description"]}
                        for f in retrieve.top_k_works(c, fac):
                            tot += 1; new += llm.cache_get(ekey(c, f)) is None; a_ret += f["paper_id"] == an["id"]
                    rec[f"pairs_{nm}"] = tot; rec[f"uncached_{nm}"] = new; rec[f"anchor_retrieved_{nm}"] = a_ret
                out.append(rec); continue
            prompts = {}
            for nm, fac, corp in (("0", fac0, corpus0), ("1", fac1, corpus1)):
                rep = pipeline.score_paper(tgt, fac)
                box = {}
                gate.profile_veto(tgt, corp, rep, chat=lambda pr: box.setdefault("p", pr) and '{"decision": "novel"}')
                prompts[nm] = box["p"]
                if nm == "1":
                    rec["anchor_retrieved_claims"] = sum(any(j["paper_id"] == an["id"] for j in c["judged"]) for c in rep["claims"])
                    rec["anchor_shown_claims"] = sum(any(j["paper_id"] == an["id"] for j in c["judged"][:5]) for c in rep["claims"])
                    lv = [j["entailment_level"] for c in rep["claims"] for j in c["judged"] if j["paper_id"] == an["id"]]
                    rec["anchor_max_level"] = max(lv) if lv else None
                    shown = [j["entailment_level"] for c in rep["claims"] for j in c["judged"][:5] if j["paper_id"] == an["id"]]
                    rec["anchor_max_level_shown"] = max(shown) if shown else None
                    seen = []
                    for c in rep["claims"]:
                        for j in c["judged"][:5]:
                            if j["paper_id"] not in seen: seen.append(j["paper_id"])
                    rec["anchor_in_papers_block"] = an["id"] in seen[:8]
                    rec["anchor_rank_in_papers"] = seen.index(an["id"]) + 1 if an["id"] in seen else None
                    rec["k_per_claim"] = [c["candidates_retrieved"] for c in rep["claims"]]
            p3 = P3.get(k, {})
            rec.update(prompt0=prompts["0"], prompt1=prompts["1"], paper_src=p3.get("src"),
                       prompt0_equals_r3=prompts["0"] == p3.get("prompt"),
                       prompt0_paper_cache_hit=paper_cache_hit(prompts["0"], p3.get("tag") or f"vsq_vs_pf_{slug}_{k.split('::')[1]}"),
                       changed=prompts["1"] != prompts["0"], paper_rej=bool(I[k]["rej"]), anchor_id=an["id"], anchor_title=an["title"])
            out.append(rec)
        S.flush()
        with lock: rows.extend(out)
        print(slug, len(out), S.summary(), flush=True)
    try:
        with ThreadPoolExecutor(int(os.environ.get("WORKERS", "3"))) as ex: list(ex.map(per_anchor, slugs))
    finally:
        S.flush()
    if STAGE == "dry":
        agg = {f: sum(r[f] for r in rows) for f in ("pairs_0", "uncached_0", "pairs_1", "uncached_1")}
        agg["anchor_retrieved_any"] = sum(r["anchor_retrieved_1"] > 0 for r in rows)
        print(len(rows), agg, S.summary(), flush=True)
        json.dump(rows, open(f"{R.OUT}/r6_z2_dry.json", "w"), indent=1)
    else:
        json.dump(sorted(rows, key=lambda r: r["key"]), open(BF, "w"))
        c = collections.Counter()
        for r in rows:
            c["changed"] += r["changed"]; c["p0_eq_r3"] += r["prompt0_equals_r3"]; c["p0_paper_hit"] += r["prompt0_paper_cache_hit"]
            c[f"src_{r['paper_src']}_hit_{r['prompt0_paper_cache_hit']}"] += 1
            c["anchor_shown"] += (r["anchor_shown_claims"] or 0) > 0; c["anchor_retrieved"] += (r["anchor_retrieved_claims"] or 0) > 0
        print(dict(c), S.summary(), flush=True)

if STAGE == "run":
    B = json.load(open(BF)); which = sys.argv[2:] or ["cn", "ctrl"]
    OUTF = f"{R.OUT}/r6_z2_runs.jsonl"
    done = {(json.loads(l)["mode"], json.loads(l)["key"]) for l in open(OUTF) if not json.loads(l).get("error")} if os.path.exists(OUTF) else set()
    jobs = []
    if "cn" in which: jobs += [("cn_anchor", r, r["prompt1"], f"r6z2cn_{r['key']}") for r in B if r["changed"]]
    if "ctrl" in which:
        unch = sorted(r["key"] for r in B if not r["changed"] and r["prompt0_paper_cache_hit"])
        pick = set(random.Random(6).sample(unch, min(30, len(unch))))
        jobs += [("ctrl_unchanged", r, r["prompt0"], f"r6z2ctrl_{r['key']}") for r in B if r["key"] in pick]
    if "base" in which:
        jobs += [("cn_base_new", r, r["prompt0"], f"r6z2base_{r['key']}") for r in B if not r["prompt0_paper_cache_hit"]]
    jobs = [j for j in jobs if (j[0], j[1]["key"]) not in done]
    print(len(jobs), "calls", collections.Counter(j[0] for j in jobs), flush=True)
    fo = open(OUTF, "a"); lock = threading.Lock()
    _pools = {}
    def corpus(r):
        if r["anchor"] not in _pools: _pools[r["anchor"]] = json.load(open(f"{anchor_dir(r['anchor'])}/pool_preT.json"))[:400]
        return _pools[r["anchor"]] + [{"id": r["anchor_id"], "title": r["anchor_title"]}]
    def one(j):
        mode, r, pr, tag = j
        try:
            txt, meta = R.call("sonnet", pr, 1500, tag)
            v = R.parse_verdict(txt, corpus(r))
            rec = {"mode": mode, "key": r["key"], "anchor": r["anchor"], **v, "cites_anchor": v["anticipating_id"] == r["anchor_id"],
                   "text": txt, "meta": meta}
        except R.Budget:
            return
        except Exception as e:
            rec = {"mode": mode, "key": r["key"], "error": str(e)[:300]}
        with lock: fo.write(json.dumps(rec) + "\n"); fo.flush()
    try:
        with ThreadPoolExecutor(int(os.environ.get("WORKERS", "6"))) as ex: list(ex.map(one, jobs))
    finally:
        print(R.summary(), flush=True)
