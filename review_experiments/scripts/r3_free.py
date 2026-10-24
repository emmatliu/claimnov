import json, re, glob, collections
from r0_common import *

I = json.load(open(f"{OUT}/r0_items.json")); M = I["meta"]; I = I["items"]
ST = json.load(open(f"{OUT}/r0_stats.json")); CLEAN = ST["clean_subset"]["items"]
DEC = M["claim_decision"]
L = lambda fp: [json.loads(l) for l in open(fp)] if os.path.exists(fp) else []
ncache = json.load(open(NCACHE))
def _key(*p): return hashlib.md5("|".join(str(x) for x in p).encode()).hexdigest()[:16]
def claims_of(t, a): return ncache.get("claims4::" + _key(t, a))
def ctx(f):
    parts = [f"Facet — attribute: {f.get('attribute', '')}; value: {f.get('value', '')}"]
    if f.get("paper_abstract"): parts.append(f"Abstract: {f['paper_abstract'][:1200]}")
    return "\n".join(parts)
def level(t, c):
    return ncache.get("entail::" + hashlib.md5((t["claim"] + "||" + c["claim"] + "||" + ctx(t) + "||" + ctx(c)).encode()).hexdigest()[:16])

def parse_profile_ids(prompt, pool):
    import difflib
    titles = {p["title"]: p["id"] for p in pool}
    body = prompt.split("NOVELTY PROFILE:\n", 1)[1].split("\n\nPAPERS IN THE PROFILE", 1)[0]
    out = []
    for line in body.split("\n"):
        if line.startswith("    - ["):
            lv = int(re.search(r"-> coverage (\d)/5$", line).group(1)); rest = line[len("    - ["):]
            pid = None
            for j in [m.start() for m in re.finditer(r"\] ", rest)]:
                if rest[:j] in titles: pid = titles[rest[:j]]; break
            if pid is None:
                cand = difflib.get_close_matches(rest.split("] ")[0], list(titles), n=1, cutoff=0.9)
                pid = titles[cand[0]] if cand else None
            out.append((pid, lv))
    return out

def reconstruct_entries(title, desc, pool, cands):
    tcl = claims_of(title, desc)
    if tcl is None: return []
    facets = [{**c, "paper_id": p["id"], "paper_abstract": p["abstract"]} for p in pool for c in (claims_of(p["title"], p["abstract"]) or [])]
    out = []
    for c in tcl:
        t = {**c, "paper_abstract": desc}; best = {}
        for f in facets:
            if f["dimension"] != c["dimension"] or f["paper_id"] not in cands: continue
            lv = level(t, f)
            if lv is not None: best[f["paper_id"]] = max(best.get(f["paper_id"], 0), lv)
        out += sorted(best.items(), key=lambda x: (-x[1], cands.index(x[0]) if x[0] in cands else 999))[:5]
    return out

res = {}
P3 = {(r["set"], r["key"]): r for r in L(f"{OUT}/r3_prompts.jsonl")}
EXC = {r["key"]: r for r in L(f"{OUT}/r2_claim_profile_prompts.jsonl")}
import ast
R1 = next(ast.literal_eval(n.value) for n in ast.parse(open(f"{PILOT}/make_disguises.py").read()).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "R1")
pairs = [json.loads(l) for l in open("eval/data/claim_disputed/testset_clean.jsonl")]
names3 = {}; src = collections.Counter()
for k in sorted(I["claimnov"]["claim"]):
    pool = json.load(open(f"{PILOT}/data/pools_full/{k}.json"))["pool"]; B = next((p["id"] for p in pool if p.get("is_B")), None)
    if EXC[k]["status"] == "byte_identical":
        ent = parse_profile_ids(EXC[k]["prompt"], pool); src["exact"] += 1
    else:
        i = int(k.split("_")[1])
        dd = json.loads(re.search(r"\{.*\}", cached_bedrock(SONNET_46_BEDROCK, [{"role": "user", "content": R1.format(abstract=pairs[i]["A_abstract"])}], 3000, f"n2f_ideaify{i}"), re.S).group(0))
        ent = reconstruct_entries(dd["title"], dd["description"], pool, I["claimnov"]["claim"][k]["candidates"]); src["reconstructed"] += 1
    names3[k] = bool(B) and any(pid == B and lv >= 3 for pid, lv in ent)
def t2(rej, items):
    acc = [k for k in items if DEC.get(k) == "acc"]; rj = [k for k in items if DEC.get(k) == "rej"]
    return {"n": len(items), "rej_acc": f"{sum(rej[k] for k in acc)}/{len(acc)}", "rej_rej": f"{sum(rej[k] for k in rj)}/{len(rj)}",
            "names_B_lvl3": fmt_ci(sum(names3[k] for k in items), len(items)),
            "on_pass": fmt_ci(sum(names3[k] and not rej[k] for k in items), len(items))}
b3 = {"profile_source": dict(src)}
for g, lab in (("claimnov", "ClaimNov"), ("graded", "w/o verdict judge (graded check)"), ("central", "per-claim rule")):
    rej = {k: bool(I[g]["claim"][k]["rej"]) for k in I["claimnov"]["claim"]}
    b3[lab] = {"all160": t2(rej, sorted(rej)), "clean36": t2(rej, CLEAN)}
b3["mcnemar_on_pass_vs_claimnov"] = {
    lab: mcnemar_exact({k: names3[k] and not I["claimnov"]["claim"][k]["rej"] for k in names3}, {k: names3[k] and not I[g]["claim"][k]["rej"] for k in names3})
    for g, lab in (("graded", "graded"), ("central", "per-claim"))}
res["B3"] = b3
res["names3_claim_per_item"] = names3

LATER = set(later_set()); conflicts = set()
for root in ("anchors", "anchors_full"):
    for adir in glob.glob(f"{PILOT}/data/{root}/*"):
        slug = os.path.basename(adir)
        if slug == "cut" or not os.path.exists(f"{adir}/ideas.json"): continue
        sf = f"{adir}/sorted_full.json" if os.path.exists(f"{adir}/sorted_full.json") else f"{adir}/sorted.json"
        if not os.path.exists(sf): continue
        for s in json.load(open(sf)):
            k = f"{slug}::{s['idx']}"
            if s["bucket"] == "a" and k in LATER: conflicts.add(k)
E2 = {r["key"]: r for r in L(f"{OUT}/e2_relabel.jsonl")}
agree = {k for k, r in E2.items() if r.get("pool_parse_ok") and r.get("F_parse_ok") and r.get("pool_contained") is False and r.get("F_realized") is True}
checks = {g: {k: bool(v["rej"]) for k, v in I[g]["later"].items()} for g in I if "later" in I[g]}
ON = {r["key"]: r for r in L(f"{OUT}/r1_opennovelty.jsonl") if r["set"] == "later" and not r.get("error")}
checks["opennovelty_all"] = {k: bool(r["reject_all"]) for k, r in ON.items()}
checks["opennovelty_any"] = {k: bool(r["reject_any"]) for k, r in ON.items()}
for K in (8, 20):
    checks[f"judge_only_top{K}"] = {r["key"]: not r["novel"] for r in L(f"{OUT}/e1_judge_only_k{K}.jsonl") if r["set"] == "later"}
for fp in glob.glob(f"{OUT}/r1_nonclaude_judgeonly_*.jsonl"):
    checks["judge_only_top8_" + os.path.basename(fp)[len("r1_nonclaude_judgeonly_"):-6]] = {r["key"]: bool(r["rej"]) for r in L(fp) if r["set"] == "later"}
CTX = {r["key"]: r for r in L(f"{OUT}/r3_ctx.jsonl") if r["set"] == "later" and not r.get("error")}
if CTX: checks["claimnov_ctx (r3)"] = {k: bool(r["rej"]) for k, r in CTX.items()}
subs = {"all400": LATER, "excl_101_conflicts": LATER - conflicts, "conflicts_101": conflicts, "opus_gpt5_agree": agree & LATER,
        "agree_and_excl_conflicts": (agree & LATER) - conflicts}
a3 = {"n": {s: len(v) for s, v in subs.items()}, "FA": {}}
for g, d in checks.items():
    a3["FA"][g] = {s: (fmt_ci(sum(d[k] for k in v if k in d), sum(k in d for k in v)) if any(k in d for k in v) else None) for s, v in subs.items()}
res["A3"] = a3
res["conflicts_101"] = sorted(conflicts); res["agree_292"] = sorted(agree & LATER)
dump("r3_free.json", res)
print(json.dumps({k: v for k, v in res.items() if k in ("B3", "A3")}, indent=1))
