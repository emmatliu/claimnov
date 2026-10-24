import sys, os, json, re, glob, hashlib, collections
sys.argv = [sys.argv[0], "none"] + sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_profile_rebuild as RB
from r0_common import PILOT, OUT, jl, anchor_dir, PCACHE
ARGS = sys.argv[2:]
MAXLEAF_E12 = int(os.environ.get("MAXLEAF_E12", "20000"))
I = json.load(open(f"{OUT}/r0_items.json"))["items"]["claimnov"]
E1 = {(r["set"], r["key"]): r for r in map(json.loads, open(f"{OUT}/e1_judge_only_k8.jsonl"))}
JO = open(f"{OUT}/../prompts/e1_judge_only.txt").read()
OUTF = f"{OUT}/r3_prompts.jsonl"
done = {(json.loads(l)["set"], json.loads(l)["key"]) for l in open(OUTF)} if os.path.exists(OUTF) else set()
fo = open(OUTF, "a")
_ap = {}
def apool(slug):
    if slug not in _ap: _ap[slug] = json.load(open(f"{anchor_dir(slug)}/pool_preT.json"))[:400]
    return _ap[slug]

def block(ids, recs):
    by = {r["id"]: r for r in recs}
    return "\n".join(f"- {by[i]['title']}: {(by[i].get('abstract') or '')[:700]}" for i in ids)

def jo_cached(prompt, tag):
    msgs = [{"role": "user", "content": prompt}]
    for key in (["bedrock", "us.anthropic.claude-sonnet-4-6", None, msgs, 1500, tag], ["antapi", "claude-sonnet-4-6", None, msgs, 1500, tag]):
        if os.path.exists(f"{PCACHE}/c_{hashlib.sha256(json.dumps(key).encode()).hexdigest()[:24]}.json"): return True
    return False

def first_leaf(u, cands):
    orig = RB.cache_hit
    RB.cache_hit = lambda pr, tag: ("recon", None)
    try:
        r = RB.rebuild(u, cands) if cands is not None else {"status": "not_found"}
        if r["status"] == "byte_identical": return r["prompt"], "constrained"
        r = RB.rebuild(u, None)
        if r["status"] == "byte_identical": return r["prompt"], "unconstrained"
        if r["status"] == "claim_without_lists":
            tcl = RB.claims_of(u["title"], u["description"])
            facets = [{**c, "paper_id": p["id"], "paper_title": p["title"], "paper_abstract": p["abstract"]}
                      for p in u["recs"] for c in (RB.claims_of(p["title"], p["abstract"]) or [])]
            lists = []
            for c in tcl:
                t = {**c, "paper_abstract": u["description"]}
                L = next(iter(RB.claim_lists(t, [f for f in facets if f["dimension"] == c["dimension"]])), [])
                lists.append(L)
            pr, _ = RB.build_prompt(f"{u['title']}: {u['description']}", tcl, lists, {x["id"]: x for x in u["recs"]})
            return pr, "partial_lists"
        return None, r["status"]
    finally:
        RB.cache_hit = orig

def emit(rec):
    fo.write(json.dumps(rec) + "\n"); fo.flush()

EXACT = {(r["set"], r["key"]): r for fp in ("r1_profile_prompts.jsonl", "r2_claim_profile_prompts.jsonl") for r in map(json.loads, open(f"{OUT}/{fp}"))}
cnt = collections.Counter()

if not ARGS or "main" in ARGS:
    units = []
    RB.SETS = ["disg", "later", "hyb"]
    for u in RB.units():
        units.append(u)
    import ast
    from r0_common import cached_bedrock, SONNET_46_BEDROCK
    src = open(f"{PILOT}/make_disguises.py").read()
    R1 = next(ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "R1")
    pairs = [json.loads(l) for l in open("eval/data/claim_disputed/testset_clean.jsonl")]
    for k in sorted(I["claim"]):
        i = int(k.split("_")[1])
        d = json.loads(re.search(r"\{.*\}", cached_bedrock(SONNET_46_BEDROCK, [{"role": "user", "content": R1.format(abstract=pairs[i]["A_abstract"])}], 3000, f"n2f_ideaify{i}"), re.S).group(0))
        units.append(dict(set="claim", key=k, title=d["title"], description=d["description"], recs=json.load(open(f"{PILOT}/data/pools_full/{k}.json"))["pool"],
                          tag=f"vsq_sonnet_nf{i}", cands=I["claim"][k]["candidates"]))
    for n, u in enumerate(units):
        if (u["set"], u["key"]) in done: continue
        ex = EXACT.get((u["set"], u["key"]))
        e1 = E1[(u["set"], u["key"])]
        tb = block(e1["shown_ids"], u["recs"])
        idea = f"{u['title']}: {u['description']}"
        jo_ok = jo_cached(JO.format(idea=idea, papers=tb), f"e1jo_k8_{u['set']}_{u['key']}")
        rec = {"set": u["set"], "key": u["key"], "tag": u["tag"], "shown_ids": e1["shown_ids"], "top8_block": tb, "jo_prompt_verified": jo_ok,
               "profile_k": "shipped"}
        rl, how = first_leaf(u, u["cands"])
        rec["recon_prompt"] = rl; rec["recon_how"] = how
        if ex and ex["status"] == "byte_identical":
            rec.update(src="exact", prompt=ex["prompt"], paper_verdict_text=ex["paper_verdict_text"], recon_equals_exact=(rl == ex["prompt"]))
        elif rl:
            rec.update(src="recon", prompt=rl)
        else:
            rec.update(src="none", prompt=None, why=how)
        cnt[(u["set"], rec["src"], jo_ok)] += 1
        emit(rec)
        if n % 50 == 0: print(n, dict(cnt), flush=True)

if not ARGS or "e12" in ARGS:
    import common as C
    def cache_only_embed(texts, model="text-embedding-3-small"):
        out = []
        for t in texts:
            fp = os.path.join(C.llm.CACHE, f"e_{hashlib.sha256((model + '::' + t).encode()).hexdigest()[:24]}.json")
            if not os.path.exists(fp): raise RuntimeError("embedding cache miss")
            out.append(json.load(open(fp)))
        return out
    C.llm.embed = cache_only_embed
    import harness.sandbox as SB
    SB.llm.embed = cache_only_embed
    from harness.sandbox import Pool
    LATER = set(json.load(open(f"{PILOT}/data/later_realized_set.json")))
    RK = {}; R10 = {}
    for r in jl("coupled_veto_ac_profile_k"):
        if r.get("novel") is not None and r["item"] not in RK: RK[r["item"]] = r
    for r in jl("coupled_veto_ac_profile"):
        if r.get("novel") is not None and r["item"] not in R10: R10[r["item"]] = r
    units = []
    for root in ("anchors", "anchors_full"):
        for adir in sorted(glob.glob(f"{PILOT}/data/{root}/*")):
            slug = os.path.basename(adir)
            if slug == "cut" or not os.path.exists(f"{adir}/ideas.json"): continue
            sf = f"{adir}/sorted_full.json" if os.path.exists(f"{adir}/sorted_full.json") else f"{adir}/sorted.json"
            if not os.path.exists(sf): continue
            ideas = {i["idx"]: i for i in json.load(open(f"{adir}/ideas.json"))}
            for s in json.load(open(sf)):
                k = f"{slug}::{s['idx']}"
                if s["bucket"] == "a" and k not in LATER and ("e12", k) not in done:
                    units.append((k, slug, ideas[s["idx"]], s["match"]))
    pools = {}
    print(len(units), "E12 units", flush=True)
    for n, (k, slug, idea, match) in enumerate(units):
        recs = apool(slug)
        if slug not in pools: pools[slug] = Pool(recs)
        itxt = f"{idea['title']}: {idea['description']}"
        top = pools[slug].search(itxt, k=8)
        ids = [p["id"] for p in top]; tb = block(ids, recs)
        jo_ok = jo_cached(JO.format(idea=itxt, papers=tb), f"r2e12jo_{k}")
        row = RK.get(k); pk = "shipped"
        if row is None: row = R10.get(k); pk = "k10"
        u = dict(set="e12", key=k, title=idea["title"], description=idea["description"], recs=recs, tag=f"vsq_vs_pf_{slug}_{k.split('::')[1]}",
                 cands=row["candidates"] if row else None)
        rec = {"set": "e12", "key": k, "tag": u["tag"], "match": match, "shown_ids": ids, "top8_block": tb, "jo_prompt_verified": jo_ok,
               "profile_k": pk, "paper_row_rej": (not row["novel"]) if row else None, "paper_row_evidence": row.get("evidence") if row else None}
        RB.MAXLEAF = MAXLEAF_E12
        r = RB.rebuild(u, u["cands"]) if row else {"status": "no_row"}
        if r["status"] == "byte_identical":
            rec.update(src="exact", prompt=r["prompt"], paper_verdict_text=r["paper_verdict_text"])
        else:
            rl, how = first_leaf(u, u["cands"])
            rec.update(src="recon" if rl else "none", prompt=rl, recon_how=how, exact_status=r["status"])
        cnt[("e12", pk, rec["src"], jo_ok)] += 1
        emit(rec)
        if n % 25 == 0: print(n, dict(cnt), flush=True)
print("DONE", dict(cnt))
