import json, re, ast, difflib, collections
import numpy as np
from r0_common import *

I = json.load(open(f"{OUT}/r0_items.json")); M = I["meta"]; I = I["items"]
ncache = json.load(open(NCACHE))
L = lambda fp: [json.loads(l) for l in open(fp)] if os.path.exists(fp) else []
def _key(*p): return hashlib.md5("|".join(str(x) for x in p).encode()).hexdigest()[:16]
def claims_of(t, a): return ncache.get("claims4::" + _key(t, a))
def ctx(f):
    parts = [f"Facet — attribute: {f.get('attribute', '')}; value: {f.get('value', '')}"]
    if f.get("paper_abstract"): parts.append(f"Abstract: {f['paper_abstract'][:1200]}")
    return "\n".join(parts)
def level(t, c):
    return ncache.get("entail::" + hashlib.md5((t["claim"] + "||" + c["claim"] + "||" + ctx(t) + "||" + ctx(c)).encode()).hexdigest()[:16])
def cos_rank(qtext, pool):
    q = np.array(cached_embed(qtext), float); q /= np.linalg.norm(q)
    sc = []
    for p in pool:
        v = np.array(cached_embed(f"{p['title']}. {p['abstract']}"), float)
        sc.append((float(q @ v / np.linalg.norm(v)), p["id"]))
    return [pid for _, pid in sorted(sc, key=lambda x: -x[0])]

src = open(f"{PILOT}/make_disguises.py").read()
R1 = next(ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "R1")
pairs = [json.loads(l) for l in open("eval/data/claim_disputed/testset_clean.jsonl")]
HYB = {h["item"]: h for h in json.load(open(f"{PILOT}/data/hybrids.json"))}
_ap = {}
def apool(slug):
    if slug not in _ap: _ap[slug] = json.load(open(f"{anchor_dir(slug)}/pool_preT.json"))[:400]
    return _ap[slug]
ITEMS = {}
for k in I["claimnov"]["later"]:
    slug, idx = k.split("::")
    idea = next(i for i in json.load(open(f"{anchor_dir(slug)}/ideas.json")) if i["idx"] == int(idx))
    ITEMS[("later", k)] = dict(title=idea["title"], desc=idea["description"], pool=apool(slug), B=None)
for k in I["claimnov"]["claim"]:
    i = int(k.split("_")[1])
    d = json.loads(re.search(r"\{.*\}", cached_bedrock(SONNET_46_BEDROCK, [{"role": "user", "content": R1.format(abstract=pairs[i]["A_abstract"])}], 3000, f"n2f_ideaify{i}"), re.S).group(0))
    pool = json.load(open(f"{PILOT}/data/pools_full/{k}.json"))["pool"]
    ITEMS[("claim", k)] = dict(title=d["title"], desc=d["description"], pool=pool, B=next((p["id"] for p in pool if p.get("is_B")), None),
                               dim=(M["claim_dimension"].get(k) or "").lower())
for k, h in HYB.items():
    ITEMS[("hyb", k)] = dict(title=h["title"], desc=h["description"], pool=apool(h["anchor"]), B=h["B_id"], spliced=h.get("spliced_claim"))

def parse_profile(prompt, pool):
    titles = {p["title"]: p["id"] for p in pool}
    body = prompt.split("NOVELTY PROFILE:\n", 1)[1].split("\n\nPAPERS IN THE PROFILE", 1)[0]
    claims = []
    for line in body.split("\n"):
        m = re.match(r"^Claim (\d+) \[([^\]]*)\]: (.*)$", line)
        if m:
            claims.append({"dim": m.group(2), "claim": m.group(3), "entries": []}); continue
        if line.startswith("    - ["):
            lv = int(re.search(r"-> coverage (\d)/5$", line).group(1))
            rest = line[len("    - ["):]
            pid = None
            for j in [m.start() for m in re.finditer(r"\] ", rest)]:
                if rest[:j] in titles: pid = titles[rest[:j]]; break
            if pid is None:
                cand = difflib.get_close_matches(rest.split("] ")[0], list(titles), n=1, cutoff=0.9)
                pid = titles[cand[0]] if cand else "?" + rest[:40]
            claims[-1]["entries"].append((pid, lv))
    return claims

def verdict_json(txt):
    try: return json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
    except Exception: return {}

def title_id(t, pool):
    if not t: return None
    c = next((x for x in pool if x["title"].lower()[:60] == str(t).lower()[:60]), None)
    return c["id"] if c else None

def reconstruct(it, cands=None):
    tcl = claims_of(it["title"], it["desc"])
    if tcl is None: return None
    facets = [{**c, "paper_id": p["id"], "paper_abstract": p["abstract"]} for p in it["pool"] for c in (claims_of(p["title"], p["abstract"]) or [])]
    out = []
    for c in tcl:
        t = {**c, "paper_abstract": it["desc"]}; best = {}
        for f in facets:
            if f["dimension"] != c["dimension"] or (cands is not None and f["paper_id"] not in cands): continue
            lv = level(t, f)
            if lv is not None: best[f["paper_id"]] = max(best.get(f["paper_id"], 0), lv)
        srt = sorted(best.items(), key=lambda x: (-x[1], (cands.index(x[0]) if cands and x[0] in cands else 999)))
        top = srt[:5]; th = top[-1][1] if top else 99
        out.append({"dim": c["dimension"], "claim": c["claim"], "entries": top, "tie_extra": [p for p, l in srt[5:] if l >= th]})
    return out

PROF = {}; SRC = collections.Counter(); VERD = {}
for r in L(f"{OUT}/r1_profile_prompts.jsonl") + L(f"{OUT}/r2_claim_profile_prompts.jsonl"):
    if r["status"] != "byte_identical": continue
    st = r["set"]
    if st == "disg":
        seed = seed_of(r["key"]); pin = json.load(open(f"{PILOT}/data/pools/seed{seed}.json"))["pool"]
        pool = pin if r["key"].endswith("B_in") else [p for p in pin if not p.get("is_B")]
        B = next(p["id"] for p in pin if p.get("is_B"))
        ITEMS[("disg", r["key"])] = dict(pool=pool, B=B if r["key"].endswith("B_in") else None, srcB=B)
    PROF[(st, r["key"])] = parse_profile(r["prompt"], ITEMS[(st, r["key"])]["pool"])
    VERD[(st, r["key"])] = verdict_json(r["paper_verdict_text"])
    SRC[(st, "exact")] += 1
val = collections.Counter()
for st in ("later", "claim"):
    for k in I["claimnov"][st]:
        if (st, k) in PROF:
            rc = reconstruct(ITEMS[(st, k)], I["claimnov"][st][k]["candidates"])
            ex = PROF[(st, k)]
            for t in (3, 4, 5):
                a = {p for c in ex for p, l in c["entries"] if l >= t}; b = {p for c in (rc or []) for p, l in c["entries"] if l >= t}
                val[(st, t, "set_equal")] += a == b; val[(st, t, "fire_equal")] += bool(a) == bool(b); val[(st, t, "n")] += 1
for st in ("later", "claim"):
    for k in I["claimnov"][st]:
        if (st, k) not in PROF:
            PROF[(st, k)] = reconstruct(ITEMS[(st, k)], I["claimnov"][st][k]["candidates"]) or []
            SRC[(st, "reconstructed")] += 1
hyb_exact = {k for (s, k) in PROF if s == "hyb"}
for k in HYB:
    rc = reconstruct(ITEMS[("hyb", k)])
    if k in hyb_exact:
        ex = PROF[("hyb", k)]
        for t in (3, 4):
            val[("hyb", t, "set_equal")] += {p for c in ex for p, l in c["entries"] if l >= t} == {p for c in rc for p, l in c["entries"] if l >= t}
            val[("hyb", t, "n")] += 1
        PROF[("hyb", k, "tie")] = rc
    else:
        PROF[("hyb", k)] = rc; PROF[("hyb", k, "tie")] = rc; SRC[("hyb", "reconstructed")] += 1

def cn_set(st, k, t, eq=False, tie=False):
    pr = PROF[(st, k)]; s = set()
    for c in pr:
        ents = list(c["entries"])
        if tie and st == "hyb" and ("hyb", k, "tie") in PROF:
            pass
        for p, l in ents:
            if (l == t if eq else l >= t): s.add(p)
        if tie:
            th_ok = [p for p in c.get("tie_extra", [])]
            if th_ok and ents and (ents[-1][1] == t if eq else ents[-1][1] >= t): s |= set(th_ok)
    return s

def cn_top1(st, k):
    best, bl = None, 0
    for c in PROF[(st, k)]:
        for p, l in c["entries"]:
            if l > bl: best, bl = p, l
    return best, bl

def rejected(st, k): return bool(I["claimnov"][st][k]["rej"])

def score(name, att, passes, sets=("later", "claim", "hyb")):
    row = {"method": name}
    for st in sets:
        ks = sorted(k for (s, k, *x) in [(a[0], a[1]) for a in ITEMS] if s == st)
        A = {k: att(st, k) for k in ks}
        n = len(ks); fire = sum(bool(A[k]) for k in ks); tot = sum(len(A[k]) for k in ks)
        o = {"n": n, "fire": fire, "fire_rate": round(fire / n, 4), "mean_attributed": round(tot / n, 3),
             "fire_on_pass": sum(bool(A[k]) and passes(st, k) for k in ks), "n_pass": sum(passes(st, k) for k in ks)}
        if st != "later":
            hit = {k: ITEMS[(st, k)]["B"] in A[k] for k in ks}
            o.update({"recall_B": sum(hit.values()), "recall_B_on_pass": sum(hit[k] and passes(st, k) for k in ks),
                      "precision_micro": round(sum(hit.values()) / tot, 4) if tot else None,
                      "precision_macro_over_firing": round(float(np.mean([hit[k] / len(A[k]) for k in ks if A[k]])), 4) if fire else None,
                      "_hit": hit})
            o["_onpass"] = {k: hit[k] and passes(st, k) for k in ks}
        o["_fire"] = {k: bool(A[k]) for k in ks}
        row[st] = o
    return row

cn_pass = lambda st, k: not rejected(st, k)
ROWS = []
for t in (3, 4):
    ROWS.append(score(f"ClaimNov profile level>={t}", lambda st, k, t=t: cn_set(st, k, t), cn_pass))
ROWS.append(score("ClaimNov profile level=5", lambda st, k: cn_set(st, k, 5, eq=True), cn_pass))
ROWS.append(score("ClaimNov profile level>=3 (hyb tie-inclusive upper bound)", lambda st, k: cn_set(st, k, 3, tie=True), cn_pass))
ROWS.append(score("ClaimNov top-1 profile paper (any level)", lambda st, k: {cn_top1(st, k)[0]} - {None}, cn_pass))
ROWS.append(score("ClaimNov top-1 profile paper if level>=3", lambda st, k: {cn_top1(st, k)[0]} if cn_top1(st, k)[1] >= 3 else set(), cn_pass))
ROWS.append(score("ClaimNov top-1 profile paper if level>=4", lambda st, k: {cn_top1(st, k)[0]} if cn_top1(st, k)[1] >= 4 else set(), cn_pass))

def cn_verdict(st, k):
    if (st, k) in VERD:
        return {title_id(VERD[(st, k)].get("anticipating_paper"), ITEMS[(st, k)]["pool"])} - {None}
    r = I["claimnov"][st][k]
    if st in ("claim", "hyb") and r.get("verdict_names_B"): return {ITEMS[(st, k)]["B"]}
    if r.get("rej") and r.get("evidence"): return {title_id(r["evidence"], ITEMS[(st, k)]["pool"]) or "?ev"}
    return set()
ROWS.append(score("ClaimNov verdict anticipating_paper (any decision; non-exact items: row fields, reject-only)", cn_verdict, cn_pass))

def spliced_idx(k):
    sc = HYB[k].get("spliced_claim") or ""
    pr = PROF[("hyb", k)]
    if not pr or not sc: return None
    return max(range(len(pr)), key=lambda i: difflib.SequenceMatcher(None, pr[i]["claim"], sc).ratio())
def cn_dimclaim(st, k, t):
    pr = PROF[(st, k)]
    if st == "claim":
        d = ITEMS[(st, k)]["dim"]
        return {p for c in pr if c["dim"].lower() == d for p, l in c["entries"] if l >= t}
    if st == "hyb":
        i = spliced_idx(k)
        return set() if i is None else {p for p, l in pr[i]["entries"] if l >= t}
    return cn_set(st, k, t)
for t in (3, 4):
    ROWS.append(score(f"ClaimNov per-claim: level>={t} on B's claim (claim set: claims of the disputed dimension; hybrids: spliced claim; later: any claim)",
                      lambda st, k, t=t: cn_dimclaim(st, k, t), cn_pass))

ON = {(r["set"], r["key"]): r for r in L(f"{OUT}/r1_opennovelty.jsonl") if not r.get("error")}
on_pass = lambda st, k: not ON[(st, k)]["reject_all"]
def on_set(st, k, strict):
    return {v["refuting_id"] for v in ON[(st, k)].get("verdicts", []) if v["refuting_id"] and (not strict or v["verdict"] == "can_refute")}
def on_first(st, k, strict):
    vs = ON[(st, k)].get("verdicts", [])
    c = [v for v in vs if v["verdict"] == "can_refute" and v["refuting_id"]] + ([] if strict else [v for v in vs if v["verdict"] == "unclear" and v["refuting_id"]])
    return {c[0]["refuting_id"]} if c else set()
ROWS.append(score("OpenNovelty-style: can_refute refuters (strict)", lambda st, k: on_set(st, k, True), on_pass))
ROWS.append(score("OpenNovelty-style: any named refuter (paper's Names B rule)", lambda st, k: on_set(st, k, False), on_pass))
ROWS.append(score("OpenNovelty-style: first can_refute refuter", lambda st, k: on_first(st, k, True), on_pass))
ROWS.append(score("OpenNovelty-style: first can_refute, else first unclear refuter", lambda st, k: on_first(st, k, False), on_pass))

COS = {key: cos_rank(f"{it['title']}: {it['desc']}", it["pool"]) for key, it in ITEMS.items() if key[0] in ("later", "claim", "hyb")}
for t in (3, 4):
    ROWS.append(score(f"Embedding top-m, m = |ClaimNov level>={t} set| (verdict: ClaimNov)",
                      lambda st, k, t=t: set(COS[(st, k)][:len(cn_set(st, k, t))]), cn_pass))
ROWS.append(score("Embedding top-1 (verdict: ClaimNov)", lambda st, k: {COS[(st, k)][0]}, cn_pass))

def wp_set(g):
    def f(st, k):
        r = I[g][st][k]
        if not r["rej"] or not r.get("evidence"): return set()
        ev = r["evidence"] if isinstance(r["evidence"], list) else [r["evidence"]]
        return {title_id(e, ITEMS[(st, k)]["pool"]) or ("?" + str(e)[:30]) for e in ev}
    return f
for g, lab in (("coi", "CoI-Agent"), ("si", "Si et al."), ("v1", "AI Scientist v1 (all <=3 returned titles)")):
    ROWS.append(score(f"{lab}: named paper on rejection", wp_set(g), lambda st, k, g=g: not I[g][st][k]["rej"]))

coi_pass = lambda st, k: not I["coi"][st][k]["rej"]
ROWS.append(score("E13 CoI verdict + embedding top-m (m = |ClaimNov level>=3 set|)", lambda st, k: set(COS[(st, k)][:len(cn_set(st, k, 3))]), coi_pass))
ROWS.append(score("E13 CoI verdict + embedding top-m (m = |ClaimNov level>=4 set|)", lambda st, k: set(COS[(st, k)][:len(cn_set(st, k, 4))]), coi_pass))
ROWS.append(score("E13 CoI verdict + embedding top-1", lambda st, k: {COS[(st, k)][0]}, coi_pass))
ROWS.append(score("E13 CoI verdict + its named paper on rejection, else embedding top-1",
                  lambda st, k: wp_set("coi")(st, k) or {COS[(st, k)][0]}, coi_pass))
for m in (1, 2, 3):
    ROWS.append(score(f"E13 CoI verdict + embedding top-{m} (fixed m)", lambda st, k, m=m: set(COS[(st, k)][:m]), coi_pass))

R = {r["method"]: r for r in ROWS}
def mc(a, b, st, fld):
    return mcnemar_exact(R[a][st][fld], R[b][st][fld])
MC = {}
for a, b in [("ClaimNov profile level>=4", "OpenNovelty-style: can_refute refuters (strict)"),
             ("ClaimNov profile level>=3", "OpenNovelty-style: can_refute refuters (strict)"),
             ("ClaimNov profile level>=3", "OpenNovelty-style: any named refuter (paper's Names B rule)"),
             ("ClaimNov top-1 profile paper (any level)", "OpenNovelty-style: first can_refute refuter"),
             ("ClaimNov profile level>=3", "E13 CoI verdict + embedding top-m (m = |ClaimNov level>=3 set|)"),
             ("ClaimNov top-1 profile paper (any level)", "E13 CoI verdict + embedding top-1")]:
    MC[f"{a} || {b}"] = {"later_fire": mc(a, b, "later", "_fire"), "claim_recall": mc(a, b, "claim", "_hit"), "claim_on_pass": mc(a, b, "claim", "_onpass"),
                         "hyb_recall": mc(a, b, "hyb", "_hit"), "hyb_on_pass": mc(a, b, "hyb", "_onpass")}

dx = {"src_in": collections.Counter(), "src_out": collections.Counter()}
for (s, k, *x), pr in list(PROF.items()):
    if s != "disg" or x: continue
    side = "src_in" if k.endswith("B_in") else "src_out"; B = ITEMS[("disg", k)]["srcB"]
    dx[side]["n"] += 1
    for t in (3, 4, 5):
        S = cn_set("disg", k, t, eq=(t == 5))
        dx[side][f"fire_lvl{'=' if t == 5 else '>='}{t}"] += bool(S)
        if side == "src_in": dx[side][f"source_in_set_lvl{'=' if t == 5 else '>='}{t}"] += B in S; dx[side][f"papers_lvl{'=' if t == 5 else '>='}{t}"] += len(S)
    dx[side]["rejected"] += bool(I["claimnov"]["disg_in" if side == "src_in" else "disg_out"][k]["rej"])
    if side == "src_out":
        ok = ("disg", k) in ON or True

pc = collections.Counter()
for k in I["claimnov"]["later"]:
    for c in PROF[("later", k)]:
        pc["claims"] += 1
        mx = max([l for _, l in c["entries"]] or [0])
        for t in (3, 4, 5): pc[f"claims_lvl>={t}"] += mx >= t
lvl_dist = collections.Counter(max([l for c in PROF[("later", k)] for _, l in c["entries"]] or [0]) for k in I["claimnov"]["later"])
lvl_dist_exact = collections.Counter(max([l for c in PROF[("later", k)] for _, l in c["entries"]] or [0]) for k in I["claimnov"]["later"] if ("later", k) in VERD)
exact_later = [k for k in I["claimnov"]["later"] if ("later", k) in VERD]
fire_exact = {t: sum(bool(cn_set("later", k, t, eq=(t == 5))) for k in exact_later) for t in (3, 4, 5)}

def strip(o):
    if isinstance(o, dict): return {k: strip(v) for k, v in o.items() if not str(k).startswith("_")}
    return o
out = {"sources": {f"{a}:{b}": v for (a, b), v in SRC.items()}, "reconstruction_validation": {f"{a}:{t}:{c}": v for (a, t, c), v in val.items()},
       "rows": [strip(r) for r in ROWS], "mcnemar(a=first,b=second)": MC, "disguise_exact_133": {k: dict(v) for k, v in dx.items()},
       "later_per_claim": dict(pc), "later_max_level_dist_all400": dict(sorted(lvl_dist.items())), "later_max_level_dist_exact341": dict(sorted(lvl_dist_exact.items())),
       "later_fire_exact341": fire_exact, "n_exact_later": len(exact_later)}
dump("r2_e8_attribution.json", out)
for r in out["rows"]:
    s = r["method"][:95]
    lt = r["later"]; cl = r["claim"]; hy = r["hyb"]
    print(f"{s:95s} | later {lt['fire']:3d} ({lt['mean_attributed']:.2f}) | claim {cl['recall_B']:3d}/{cl['recall_B_on_pass']:3d} P={cl['precision_micro']} m={cl['mean_attributed']:.2f} | hyb {hy['recall_B']:3d}/{hy['recall_B_on_pass']:3d} P={hy['precision_micro']} m={hy['mean_attributed']:.2f}")
print(json.dumps({k: out[k] for k in ("sources", "reconstruction_validation", "disguise_exact_133", "later_per_claim", "later_max_level_dist_all400", "later_max_level_dist_exact341", "later_fire_exact341")}, indent=0))
for k, v in MC.items(): print(k, {a: (b["a_only"], b["b_only"], round(b["p_exact"], 5)) for a, b in v.items()})
