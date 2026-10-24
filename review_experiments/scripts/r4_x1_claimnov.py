import json, os, sys, re, threading, collections
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("R4_EXP", "x1")
import r4_common as R
import r4_shipped as S
STAGE = sys.argv[1]
OUT = R.OUT
LATER = sorted(json.load(open(f"{S.PILOT}/data/later_realized_set.json")))
SLUGS = sorted({k.split("::")[0] for k in LATER})
AF = f"{OUT}/r4_x1_anchors.json"
P3 = {p["key"]: p for p in map(json.loads, open(f"{OUT}/r3_prompts.jsonl")) if p["set"] == "later"}
_ideas = {}
def idea_of(key):
    slug, idx = key.split("::")
    if slug not in _ideas: _ideas[slug] = {i["idx"]: i for i in json.load(open(f"{S.anchor_dir(slug)}/ideas.json"))}
    return _ideas[slug][int(idx)]
_pools = {}
def pool(slug):
    if slug not in _pools: _pools[slug] = json.load(open(f"{S.anchor_dir(slug)}/pool_preT.json"))[:400]
    return _pools[slug]

if STAGE == "anchors":
    out = {}
    for s in SLUGS:
        a = json.load(open(f"{S.anchor_dir(s)}/anchor.json"))
        rec = {"id": "ANCHOR_" + a["paperId"], "paperId": a["paperId"], "title": a["title"], "abstract": a["abstract"]}
        cl = S.claims_of(a["title"], a["abstract"])
        if cl is not None:
            rec.update(claims=cl, claims_src="cached_shipped_gpt-4o-mini")
        else:
            cl, meta = S.haiku_extract(a["title"], a["abstract"])
            rec.update(claims=cl, claims_src="haiku-4.5_substitute", meta=meta)
        out[s] = rec
        print(s, rec["claims_src"], len(cl), collections.Counter(c["dimension"] for c in cl), flush=True)
    json.dump(out, open(AF, "w"), indent=1)
    print(R.summary())

A = json.load(open(AF)) if os.path.exists(AF) else {}
LF = f"{OUT}/r4_x1_levels.json"

if STAGE == "entail":
    jobs = []
    for k in LATER:
        slug = k.split("::")[0]; idea = idea_of(k)
        tcl = S.claims_of(idea["title"], idea["description"]) or []
        for ci, c in enumerate(tcl):
            t = {**c, "paper_abstract": idea["description"]}
            for ai, ac in enumerate(A[slug]["claims"]):
                if ac["dimension"] != c["dimension"]: continue
                f = {**ac, "paper_id": A[slug]["id"], "paper_title": A[slug]["title"], "paper_abstract": A[slug]["abstract"]}
                jobs.append((k, ci, ai, t, f))
    print(len(jobs), "pairs", flush=True)
    res = {}; lock = threading.Lock()
    def one(j):
        k, ci, ai, t, f = j
        try:
            lv, meta = S.haiku_level(t, f)
        except R.Budget: return
        with lock: res.setdefault(k, []).append({"ci": ci, "ai": ai, "level": lv})
    try:
        with ThreadPoolExecutor(8) as ex: list(ex.map(one, jobs))
    finally:
        json.dump(res, open(LF, "w")); print(R.summary(), flush=True)

BUMP = int(os.environ.get("BUMP", "0"))
if STAGE == "build":
    LV = json.load(open(LF))
    E1 = {r["key"]: r for r in map(json.loads, open(f"{OUT}/e1_judge_only_k8.jsonl")) if r["set"] == "later"}
    JO = open(f"{R.EXP}/prompts/e1_judge_only.txt").read()
    recs = []; cnt = collections.Counter()
    for k in LATER:
        slug = k.split("::")[0]; idea = idea_of(k); p3 = P3[k]; an = A[slug]
        recsP = pool(slug); by_title = {r["title"]: r for r in recsP}
        by_title_a = {**by_title, an["title"]: an}
        ideatxt, parsed = S.parse_profile_prompt(p3["prompt"], list(by_title))
        tcl = S.claims_of(idea["title"], idea["description"]) or []
        assert len(tcl) == len(parsed), k
        base_lists = [(tc, ents) for tc, (_, _, ents) in zip(tcl, parsed)]
        rt, _ = S.build_profile_prompt(ideatxt, base_lists, by_title)
        assert rt == p3["prompt"], f"round trip failed {k}"
        lv = collections.defaultdict(dict)
        for e in LV.get(k, []): lv[e["ci"]][e["ai"]] = e["level"]
        new_lists, ins = [], []
        for ci, (tc, ents) in enumerate(base_lists):
            if lv.get(ci):
                ai, L = max(lv[ci].items(), key=lambda x: (x[1], -x[0]))
                L = min(5, L + BUMP)
                pos = next((i for i, e in enumerate(ents) if e[2] <= L), len(ents))
                ne = (ents[:pos] + [(an["title"], an["claims"][ai]["claim"], L)] + ents[pos:])[:min(5, len(ents) + 1)]
                shown = any(e[0] == an["title"] for e in ne)
                ins.append({"ci": ci, "level": L, "pos": pos if shown else None})
                new_lists.append((tc, ne))
            else:
                new_lists.append((tc, ents))
        pr, seen = S.build_profile_prompt(ideatxt, new_lists, by_title_a)
        a_lv = [x["level"] for x in ins]
        cnt["has_same_dim_anchor_claim"] += bool(ins); cnt["anchor_shown"] += any(x["pos"] is not None for x in ins)
        cnt["anchor_in_papers_block"] += an["title"] in seen[:8]
        e1 = E1[k]; ids7 = e1["shown_ids"][:7]; byid = {r["id"]: r for r in recsP}
        papers = "\n".join([f"- {an['title']}: {an['abstract'][:700]}"] + [f"- {byid[i]['title']}: {(byid[i].get('abstract') or '')[:700]}" for i in ids7])
        jo = JO.format(idea=ideatxt, papers=papers)
        recs.append({"key": k, "anchor": slug, "src": p3["src"], "prompt_anchor": pr, "prompt_base": p3["prompt"],
                     "insertions": ins, "anchor_max_level": max(a_lv) if a_lv else None,
                     "anchor_in_papers_block": an["title"] in seen[:8], "anchor_rank_in_papers": seen.index(an["title"]) + 1 if an["title"] in seen else None,
                     "jo_prompt_anchor": jo})
    json.dump(recs, open(f"{OUT}/r4_x1_prompts{'_bump%d' % BUMP if BUMP else ''}.json", "w"))
    ml = collections.Counter(r["anchor_max_level"] for r in recs)
    print(dict(cnt), "anchor max level", dict(ml), flush=True)

if STAGE == "run":
    PR = json.load(open(f"{OUT}/r4_x1_prompts.json"))
    PRB = {r["key"]: r for r in json.load(open(f"{OUT}/r4_x1_prompts_bump1.json"))} if os.path.exists(f"{OUT}/r4_x1_prompts_bump1.json") else {}
    which = sys.argv[2:] or ["cn", "ctrl", "jo"]
    OUTF = f"{OUT}/r4_x1_runs.jsonl"
    done = {(json.loads(l)["mode"], json.loads(l)["key"]) for l in open(OUTF) if not json.loads(l).get("error")} if os.path.exists(OUTF) else set()
    jobs = []
    for r in PR:
        if "cn" in which: jobs.append(("cn_anchor", r, r["prompt_anchor"], f"r4x1cn_{r['key']}"))
        if "bump" in which and r["key"] in PRB and PRB[r["key"]]["prompt_anchor"] != r["prompt_anchor"]:
            jobs.append(("cn_anchor_bump1", r, PRB[r["key"]]["prompt_anchor"], f"r4x1cnb1_{r['key']}"))
        if "ctrl" in which and r["src"] == "recon": jobs.append(("cn_recon_noanchor", r, r["prompt_base"], f"r4x1ctrl_{r['key']}"))
        if "jo" in which: jobs.append(("jo_anchor", r, r["jo_prompt_anchor"], f"r4x1jo_{r['key']}"))
    jobs = [j for j in jobs if (j[0], j[1]["key"]) not in done]
    print(len(jobs), "calls", flush=True)
    fo = open(OUTF, "a"); lock = threading.Lock()
    def one(j):
        mode, r, pr, tag = j
        corpus = pool(r["anchor"]) + [A[r["anchor"]]]
        try:
            txt, meta = R.call("sonnet", pr, 1500, tag)
            v = R.parse_verdict(txt, corpus)
            if v["anticipating_id"] is None and v["anticipating_title"]:
                nt = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())[:50]
                m = next((c for c in corpus if nt(c["title"]) == nt(v["anticipating_title"])), None)
                v["anticipating_id"] = m["id"] if m else None
            rec = {"mode": mode, "key": r["key"], "anchor": r["anchor"], "src": r["src"], **v,
                   "cites_anchor": v["anticipating_id"] == A[r["anchor"]]["id"], "text": txt, "meta": meta}
        except R.Budget:
            return
        except Exception as e:
            rec = {"mode": mode, "key": r["key"], "error": str(e)[:300]}
        with lock: fo.write(json.dumps(rec) + "\n"); fo.flush()
    try:
        with ThreadPoolExecutor(int(os.environ.get("WORKERS", "6"))) as ex: list(ex.map(one, jobs))
    finally:
        print(R.summary(), flush=True)
