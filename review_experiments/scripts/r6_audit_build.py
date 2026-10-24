import json, os, sys, random, hashlib, glob, collections, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from r0_common import OUT, PILOT, anchor_dir, jl
import harness.sandbox as SB
from harness.sandbox import Pool
def cache_only_embed(texts, model="text-embedding-3-small"):
    out = []
    for t in texts:
        fp = os.path.join(C.llm.CACHE, f"e_{hashlib.sha256((model + '::' + t).encode()).hexdigest()[:24]}.json")
        if not os.path.exists(fp): raise RuntimeError("embedding cache miss")
        out.append(json.load(open(fp)))
    return out
C.llm.embed = cache_only_embed; SB.llm.embed = cache_only_embed
AUD = f"{C.EXP}/../audit_v2"; os.makedirs(AUD, exist_ok=True)
L = lambda fp: [json.loads(l) for l in open(fp)] if os.path.exists(fp) else []
I = json.load(open(f"{OUT}/r0_items.json"))["items"]
FREE = json.load(open(f"{OUT}/r3_free.json")); CONFL = set(FREE["conflicts_101"])
LATER = sorted(json.load(open(f"{PILOT}/data/later_realized_set.json")))
def dec_map():
    D = {}
    for g in I:
        if "later" in I[g]:
            D[g] = {k: {"rej": bool(v["rej"]), "paper": v.get("evidence")} for k, v in I[g]["later"].items()}
    ON = {r["key"]: r for r in L(f"{OUT}/r1_opennovelty.jsonl") if r["set"] == "later" and not r.get("error")}
    D["opennovelty_all"] = {k: {"rej": bool(r["reject_all"]), "paper": [v.get("refuting_title") for v in r.get("verdicts", []) if v.get("verdict") == "can_refute"]} for k, r in ON.items()}
    D["opennovelty_any"] = {k: {"rej": bool(r["reject_any"]), "paper": None} for k, r in ON.items()}
    for K in (8, 20):
        D[f"judge_only_top{K}"] = {r["key"]: {"rej": not r["novel"], "paper": r.get("anticipating_paper")} for r in L(f"{OUT}/e1_judge_only_k{K}.jsonl") if r["set"] == "later"}
    for fp in glob.glob(f"{OUT}/r1_nonclaude_judgeonly_*.jsonl"):
        D["judge_only_top8_" + os.path.basename(fp)[len("r1_nonclaude_judgeonly_"):-6]] = {r["key"]: {"rej": bool(r["rej"]), "paper": r.get("anticipating_paper")} for r in L(fp) if r["set"] == "later"}
    D["claimnov_ctx"] = {r["key"]: {"rej": r["rej"], "paper": r.get("anticipating_title")} for r in L(f"{OUT}/r3_ctx.jsonl") if r["set"] == "later" and not r.get("error")}
    return D
D = dec_map()
LOWFA = ["claimnov", "graded", "claimnov_k10", "claimnov_ctx", "coi", "opennovelty_all", "judge_only_top8", "judge_only_top20", "judge_only_top8_gpt-oss-120b-1"]
LOWFA = [c for c in LOWFA if c in D]
rng = random.Random(20261012)
conf = sorted(CONFL); rng.shuffle(conf); s_conf = sorted(conf[:10])
fa_pool = sorted(k for k in LATER if k not in CONFL and any(D[c].get(k, {}).get("rej") for c in LOWFA)); rng.shuffle(fa_pool); s_fa = sorted(fa_pool[:10])
rest = sorted(k for k in LATER if k not in set(s_conf) | set(s_fa)); rng.shuffle(rest); s_uni = sorted(rest[:25])
P3 = {(r["set"], r["key"]): r for r in L(f"{OUT}/r3_prompts.jsonl")}
e12 = sorted(k for (s, k) in P3 if s == "e12"); rng.shuffle(e12); s_e12 = sorted(e12[:15])
strata = [("later_uniform", k) for k in s_uni] + [("later_sorter_conflict", k) for k in s_conf] + [("later_some_lowFA_check_fired", k) for k in s_fa] + [("e12_bucket_a", k) for k in s_e12]
assert len({k for _, k in strata}) == 60
CS = {}
for r in L(f"{PILOT}/results/claim_sort_full_indep.jsonl"):
    CS.setdefault(r["item"], r)
SORT = {}
for root in ("anchors", "anchors_full"):
    for adir in glob.glob(f"{PILOT}/data/{root}/*"):
        slug = os.path.basename(adir)
        sf = f"{adir}/sorted_full.json" if os.path.exists(f"{adir}/sorted_full.json") else f"{adir}/sorted.json"
        if slug == "cut" or not os.path.exists(sf) or not os.path.exists(f"{adir}/ideas.json"): continue
        for s in json.load(open(sf)): SORT[f"{slug}::{s['idx']}"] = s
E2 = {r["key"]: r for r in L(f"{OUT}/e2_relabel.jsonl")}
CTXE = {r["key"]: r for r in L(f"{OUT}/r3_ctx.jsonl") if r["set"] == "e12" and not r.get("error")}
JO12 = {r["key"]: r for r in L(f"{OUT}/r2_e12_judgeonly.jsonl") if not r.get("error")}
nt = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())[:60]
order = list(range(60)); rng.shuffle(order)
items, key = [], {}
_pools = {}
for n, i in enumerate(order):
    st, k = strata[i]; slug, idx = k.split("::")
    idea = next(x for x in json.load(open(f"{anchor_dir(slug)}/ideas.json")) if x["idx"] == int(idx))
    recs = json.load(open(f"{anchor_dir(slug)}/pool_preT.json"))[:400]
    if slug not in _pools: _pools[slug] = Pool(recs)
    top = _pools[slug].search(f"{idea['title']}: {idea['description']}", k=20)
    iid = f"Q{n + 1:02d}"
    F = []
    if st.startswith("later"):
        Fd = {nt(f["title"]): f for f in json.load(open(f"{anchor_dir(slug)}/F.json"))}
        for c in CS[k].get("post", []):
            if c.get("established") == "yes" and c.get("by"):
                f = Fd.get(nt(c["by"]))
                if f and f["title"] not in [x["title"] for x in F]: F.append(f)
    items.append((iid, idea, top, F))
    pidx = {p["title"]: j + 1 for j, p in enumerate(top)}
    def pnum(t):
        if not t: return None
        if isinstance(t, list): return [pnum(x) for x in t]
        return next((j for tt, j in pidx.items() if nt(tt) == nt(t)), "not_in_top20")
    checks = {}
    if st.startswith("later"):
        for c, d in D.items():
            if k in d: checks[c] = {"rej": d[k]["rej"], "paper": d[k]["paper"], "paper_num": pnum(d[k]["paper"]) if d[k]["rej"] else None}
    else:
        for fn, nm in (("coupled_veto_ac_profile_k", "claimnov"), ("coupled_veto_ac_profile", "claimnov_k10")):
            r = next((r for r in jl(fn) if r["item"] == k and r.get("novel") is not None), None)
            if r: checks[nm] = {"rej": not r["novel"], "paper": r.get("evidence"), "paper_num": pnum(r.get("evidence")) if not r["novel"] else None}
        if k in CTXE: checks["claimnov_ctx"] = {"rej": CTXE[k]["rej"], "paper": CTXE[k]["anticipating_title"], "paper_num": pnum(CTXE[k]["anticipating_title"]) if CTXE[k]["rej"] else None}
        if k in JO12: checks["judge_only_top8"] = {"rej": not JO12[k]["novel"], "paper": JO12[k]["anticipating_paper"], "paper_num": pnum(JO12[k]["anticipating_paper"]) if not JO12[k]["novel"] else None}
    srt = SORT.get(k, {})
    grp = None if not st.startswith("later") else ("conflict" if k in CONFL else ("lowFA_fired" if k in set(fa_pool) else "other"))
    key[iid] = {"item": k, "stratum": st, "population_group": grp, "set": "later" if st.startswith("later") else "e12",
                "opus_claim_labeler": ({"pre_established_claims": [c for c in CS[k].get("pre", []) if c.get("established") == "yes"],
                                        "post_established": [c.get("by") for c in CS[k].get("post", []) if c.get("established") == "yes"],
                                        "label": "novel_vs_pool"} if k in CS else None),
                "opus_sorter": {"bucket": srt.get("bucket"), "match": srt.get("match"), "match_num": pnum(srt.get("match"))},
                "gpt5": ({"pool_contained": E2[k].get("pool_contained"), "F_realized": E2[k].get("F_realized")} if k in E2 else None),
                "checks": checks, "top20_ids": [p["id"] for p in top], "top20_titles": [p["title"] for p in top],
                "F_titles": [f["title"] for f in F]}
by_st = collections.defaultdict(list)
for iid, *_ in sorted(items): by_st[key[iid]["stratum"]].append(iid)
assign, j = {}, 0
for st_ in sorted(by_st):
    for iid in by_st[st_]: assign[iid] = "ABC"[j % 3]; j += 1
for iid in key: key[iid]["packet"] = assign[iid]
for P in "ABC":
    md = [f"# Novelty audit packet {P}", "", "Read INSTRUCTIONS.md first. Items are in random order. For each item, answer in labels_" + P + ".csv.", ""]
    for iid, idea, top, F in sorted(items):
        if assign[iid] != P: continue
        md += [f"---", "", f"## {iid}", "", f"**Idea: {idea['title']}**", "", idea["description"].strip(), "",
               "### Prior papers (published before the cutoff), most similar first", ""]
        for jj, p in enumerate(top, 1):
            md += [f"**P{jj}. {p['title']}**  ", (p.get("abstract") or "").strip().replace("\n", " "), ""]
        if F:
            md += ["### Later paper(s) (published after the cutoff), for reference only", ""]
            for f in F:
                md += [f"**F. {f['title']}**  ", (f.get("abstract") or "").strip().replace("\n", " "), ""]
    open(f"{AUD}/packet_{P}.md", "w").write("\n".join(md))
    with open(f"{AUD}/labels_{P}.csv", "w") as f:
        f.write("item_id,annotator,done_by_listed_paper,paper_number,contribution_established,contribution_text,contribution_paper_number,confidence,notes\n")
        for iid in sorted(i for i in assign if assign[i] == P): f.write(f"{iid},,,,,,,,\n")
json.dump({"seed": 20261012, "low_FA_checks_for_stratum": LOWFA, "n_fa_pool": len(fa_pool), "population_groups": {"conflict": len(CONFL), "lowFA_fired": len(fa_pool), "other": len(LATER) - len(CONFL) - len(fa_pool)}, "n_conflicts": len(CONFL), "n_later": len(LATER),
           "n_e12_pool": len(e12), "items": key}, open(f"{AUD}/key.json", "w"), indent=1)
print("strata", collections.Counter(s for s, _ in strata), "fa_pool", len(fa_pool), "packets", collections.Counter(assign.values()), collections.Counter((assign[i], key[i]["stratum"]) for i in assign))
