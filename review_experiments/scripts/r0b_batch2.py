import ast, json, hashlib, difflib, glob, os
import numpy as np
from r0_common import *

D = json.load(open(f"{OUT}/r0_items.json")); I = D["items"]; M = D["meta"]
PROF = json.load(open(f"{OUT}/r0_claim_profiles.json"))["profiles"]
ncache = json.load(open(NCACHE))
T1 = ["v1", "coi", "si", "arc", "ra", "scholareval", "v2_reflection", "claimnov", "graded", "central", "claimnov_k10"]
NAMERS = {"v1", "coi", "si", "arc", "claimnov", "graded", "central", "claimnov_k10"}
PILOT_ANCH = sorted(os.listdir(f"{PILOT}/data/anchors"))
REPS = 10000
res = {}

def boot(checks, keyset, reps=REPS, seed=0):
    rng = np.random.default_rng(seed)
    srcs = sorted(keyset); ancs = sorted({anchor_of(k) for k in I["claimnov"]["later"]})
    A = {}
    for c in checks:
        ci = collections.defaultdict(list); co = collections.defaultdict(list); ca = collections.defaultdict(list)
        for k, v in I[c]["disg_in"].items(): ci[seed_of(k)].append(v["rej"])
        for k, v in I[c]["disg_out"].items(): co[seed_of(k)].append(v["rej"])
        for k, v in I[c]["later"].items(): ca[anchor_of(k)].append(v["rej"])
        A[c] = (np.array([sum(ci[s]) for s in srcs], float), np.array([len(ci[s]) for s in srcs], float),
                np.array([sum(co[s]) for s in srcs], float), np.array([sum(ca[a]) for a in ancs], float),
                np.array([len(ca[a]) for a in ancs], float))
    Js = {c: [] for c in checks}; Ds = {c: [] for c in checks}
    for _ in range(reps):
        si = rng.integers(0, len(srcs), len(srcs)); ai = rng.integers(0, len(ancs), len(ancs))
        for c in checks:
            cin, n, cout, fa, nf = A[c]
            Js[c].append(cin[si].sum() / n[si].sum() - fa[ai].sum() / nf[ai].sum()); Ds[c].append((cin[si].sum() - cout[si].sum()) / n[si].sum())
    out = {}
    for c in checks:
        cin, n, cout, fa, nf = A[c]
        pc = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
        out[c] = {"J": round(cin.sum() / n.sum() - fa.sum() / nf.sum(), 4), "J_CI": pc(Js[c]),
                  "Delta": round((cin.sum() - cout.sum()) / n.sum(), 4), "Delta_CI": pc(Ds[c])}
        if c != "claimnov":
            out[c]["CN_minus_J_CI"] = pc(np.array(Js["claimnov"]) - np.array(Js[c]))
            out[c]["CN_minus_Delta_CI"] = pc(np.array(Ds["claimnov"]) - np.array(Ds[c]))
    return out

allsrc = {seed_of(k) for k in I["claimnov"]["disg_in"]}
psize = {s: len(pool_B(s)[1]) for s in allsrc}
ONE = {s for s, n in psize.items() if n == 1}
res["B1_pool_sizes"] = {"single_paper_pools": sorted(ONE), "short_pools(2-19)": {s: n for s, n in sorted(psize.items()) if 1 < n < 20}}
for lab, ks in (("i_95src_excl_single_paper_pools", allsrc - ONE), ("ii_85src_also_excl_pilot_seeds0-9", {s for s in allsrc - ONE if s >= 10})):
    b = boot(T1, ks); tab = {}
    for c in T1:
        din = {k: v for k, v in I[c]["disg_in"].items() if seed_of(k) in ks}
        dout = {k: v for k, v in I[c]["disg_out"].items() if seed_of(k) in ks}
        lat = I[c]["later"]
        tab[c] = {"caught": fmt_ci(sum(v["rej"] for v in din.values()), len(din)),
                  "named": fmt_ci(sum(bool(v.get("named")) for v in din.values()), len(din)) if c in NAMERS else None,
                  "src_out": fmt_ci(sum(v["rej"] for v in dout.values()), len(dout)),
                  "FA": fmt_ci(sum(v["rej"] for v in lat.values()), len(lat)),
                  "rungs": [sum(v["rej"] for k, v in din.items() if f"_r{q}_" in k) for q in (1, 2, 3, 4)],
                  "per_rung_n": sum(1 for k in din if "_r1_" in k), **b[c]}
    mc = {}
    for tn, s, f in (("catch", "disg_in", lambda v: v["rej"]), ("src_out", "disg_out", lambda v: v["rej"]), ("named", "disg_in", lambda v: bool(v.get("named")))):
        a = {k: f(v) for k, v in I["claimnov"][s].items() if seed_of(k) in ks}
        o = {k: f(v) for k, v in I["coi"][s].items() if seed_of(k) in ks}
        mc[tn] = mcnemar_exact(a, o)
    res[f"B1_{lab}"] = {"n_sources": len(ks), "n_disguises": 4 * len(ks), "rows": tab, "mcnemar_claimnov_vs_coi": mc}

def _key(*p): return hashlib.md5("|".join(str(x) for x in p).encode()).hexdigest()[:16]
def claims_of(t, a): return ncache.get("claims4::" + _key(t, a))
def ctx(f):
    parts = [f"Facet — attribute: {f.get('attribute', '')}; value: {f.get('value', '')}"]
    if f.get("paper_abstract"): parts.append(f"Abstract: {f['paper_abstract'][:1200]}")
    return "\n".join(parts)
def level(t, c):
    return ncache.get("entail::" + hashlib.md5((t["claim"] + "||" + c["claim"] + "||" + ctx(t) + "||" + ctx(c)).encode()).hexdigest()[:16])
def cos_rank(qtext, pool):
    q = cached_embed(qtext); q = [float(x) for x in q]; qn = sum(x * x for x in q) ** .5
    sc = []
    for p in pool:
        v = cached_embed(f"{p['title']}. {p['abstract']}")
        sc.append((sum(a * b for a, b in zip(q, v)) / (qn * sum(b * b for b in v) ** .5), p["id"]))
    return [pid for _, pid in sorted(sc, key=lambda x: -x[0])]
src = open(f"{PILOT}/make_disguises.py").read()
R1 = next(ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "R1")
pairs = [json.loads(l) for l in open("eval/data/claim_disputed/testset_clean.jsonl")]

def size_matched(rows, label):
    o = {}
    for thr in (3, 4):
        cn_hit, cos_hit, ms, cn_sz = {}, {}, [], 0
        for r in rows:
            S = [p for p, l in r["prof"].items() if l >= thr]; m = len(S); ms.append(m)
            cn_hit[r["id"]] = r["B"] in S; cos_hit[r["id"]] = r["B"] in r["cos"][:m]; cn_sz += m
        o[f"level>={thr}"] = {"mean_m": round(float(np.mean(ms)), 3), "n_items_m>0": sum(1 for x in ms if x),
                              "total_papers_in_sets": cn_sz,
                              "claimnov_set_contains_B": sum(cn_hit.values()), "cosine_top_m_contains_B": sum(cos_hit.values()),
                              "precision_claimnov": round(sum(cn_hit.values()) / cn_sz, 4) if cn_sz else None,
                              "precision_cosine": round(sum(cos_hit.values()) / cn_sz, 4) if cn_sz else None,
                              "mcnemar": mcnemar_exact(cn_hit, cos_hit)}
    t1, t1exp, c1 = {}, 0.0, {}
    for r in rows:
        if not r["prof"]: t1[r["id"]] = False; c1[r["id"]] = r["B"] in r["cos"][:1]; continue
        top = max(r["prof"].values()); ties = [p for p in r["order"] if r["prof"].get(p) == top]
        t1[r["id"]] = ties[0] == r["B"]; t1exp += (1 / len(ties)) if r["B"] in ties else 0
        c1[r["id"]] = r["B"] in r["cos"][:1]
    o["m=1"] = {"claimnov_top1_is_B(profile-order tiebreak)": sum(t1.values()), "claimnov_top1_expected(uniform tiebreak)": round(t1exp, 2),
                "cosine_top1_is_B": sum(c1.values()), "mcnemar": mcnemar_exact(t1, c1)}
    res[f"B2_{label}"] = o

crow = []
for k, p in PROF.items():
    cand = I["claimnov"]["claim"][k]["candidates"]
    lv = {}
    for r in p["claims"]:
        for pid, l in r["levels"].items():
            if pid in cand: lv[pid] = max(lv.get(pid, 0), l)
    i = int(k.split("_")[1])
    d = json.loads(re.search(r"\{.*\}", cached_bedrock(SONNET_46_BEDROCK, [{"role": "user", "content": R1.format(abstract=pairs[i]["A_abstract"])}], 3000, f"n2f_ideaify{i}"), re.S).group(0))
    pool = json.load(open(f"{PILOT}/data/pools_full/{k}.json"))["pool"]
    crow.append({"id": k, "B": p["B_id"], "prof": lv, "order": cand, "cos": cos_rank(f"{d['title']}: {d['description']}", pool)})
size_matched(crow, "claim160")

H = json.load(open(f"{PILOT}/data/hybrids.json")); hy = I["claimnov"]["hyb"]
pools = {}; hrow = []; b4 = collections.Counter(); b4_items = []
for h in H:
    a = h["anchor"]
    if a not in pools: pools[a] = json.load(open(f"{anchor_dir(a)}/pool_preT.json"))[:400]
    pool = pools[a]; B = h["B_id"]
    tcl = claims_of(h["title"], h["description"]) or []
    facets = [{**c, "paper_id": p["id"], "paper_abstract": p["abstract"]} for p in pool for c in (claims_of(p["title"], p["abstract"]) or [])]
    shown, order, per_claim = {}, [], []
    for c in tcl:
        t = {**c, "paper_abstract": h["description"]}; best = {}
        for f in facets:
            if f["dimension"] != c["dimension"]: continue
            l = level(t, f)
            if l is not None: best[f["paper_id"]] = max(best.get(f["paper_id"], 0), l)
        srt = sorted(best.items(), key=lambda x: -x[1]); per_claim.append(srt)
        for pid, l in srt[:5]:
            shown[pid] = max(shown.get(pid, 0), l)
            if pid not in order: order.append(pid)
    hrow.append({"id": h["item"], "B": B, "prof": shown, "order": order, "cos": cos_rank(f"{h['title']}: {h['description']}", pool)})
    if tcl:
        sim = [difflib.SequenceMatcher(None, c.get("claim", "").lower(), h["spliced_claim"].lower()).ratio() for c in tcl]
        si = max(range(len(tcl)), key=lambda i: sim[i]); srt = per_claim[si]
        if srt:
            top = srt[0][1]; ties = [pid for pid, l in srt if l == top]
            Bl = dict(srt).get(B)
            passed = not hy[h["item"]]["rej"]
            if B in ties:
                b4["B_tied_top_on_spliced_claim"] += 1; b4["...among_passes"] += passed
                b4["expected_B_top_uniform_tiebreak"] += 1 / len(ties)
                if len(ties) == 1: b4["B_unique_top_on_spliced_claim"] += 1; b4["....unique_among_passes"] += passed
                b4_items.append({"item": h["item"], "B_level": Bl, "n_tied_at_top": len(ties), "passed": passed})
            if B in [pid for pid, _ in srt[:5]]: b4["B_in_top5_of_spliced_claim(reconstructed)"] += 1
b4["row_B_on_spliced_claim(paper 'right'=30)"] = sum(bool(v["B_on_spliced_claim"]) for v in hy.values())
b4["expected_B_top_uniform_tiebreak"] = round(b4["expected_B_top_uniform_tiebreak"], 2)
res["B4_hyb_claim_attached"] = {"counts": b4, "items": b4_items,
                                "note": "spliced claim = argmax difflib ratio to hybrids.json spliced_claim (run_hybrids.py); levels from src/.cache entail:: (= exactly the judged pairs for hybrid claims)"}
size_matched(hrow, "hyb100")
res["B2_hyb100_check"] = {"reconstructed_B_in_lvl>=3_set": sum(1 for r in hrow if r["prof"].get(r["B"], 0) >= 3),
                          "row_based_named_and_B_level>=3": sum(1 for v in hy.values() if v["named"] and v["B_level"] >= 3),
                          "reconstructed_B_in_lvl>=4_set": sum(1 for r in hrow if r["prof"].get(r["B"], 0) >= 4),
                          "row_based_named_and_B_level>=4": sum(1 for v in hy.values() if v["named"] and v["B_level"] >= 4)}

tdim = {"Problem": 33, "Methodology": 35, "Resource": 26, "Result": 33}; trel = {"anticipation": 45, "comparison": 62, "overlap": 20}
b3 = []
for fp in sorted(glob.glob(f"{RES}/coupled_veto_neg*.jsonl")):
    f = os.path.basename(fp)[:-6]; d = {}
    for r in jl(f): d.setdefault(r["item"], r)
    for lab, fn in (("B in candidates", lambda r: r["B_id"] in (r["candidates"] or [])),
                    ("B in candidates[:8] (= whole-idea top-8 for cand* files)", lambda r: r["B_id"] in (r["candidates"] or [])[:8]),
                    ("B_rank <= checked", lambda r: r["B_rank"] is not None and r["B_rank"] <= r["checked"])):
        hits = [k for k, r in d.items() if fn(r)]
        dims = dict(collections.Counter(M["claim_dimension"][k] for k in hits)); rel = dict(collections.Counter(M["claim_relation"][k] for k in hits))
        b3.append({"file": f, "n": len(d), "field": lab, "count": len(hits), "by_dim": dims, "by_rel": rel,
                   "exact_match_paper": dims == tdim and rel == trel})
cos8 = [r for r in crow if r["B"] in r["cos"][:8]]
b3.append({"file": "(computed) whole-proposal cosine top-8, cached 3-small", "count": len(cos8),
           "by_dim": dict(collections.Counter(M["claim_dimension"][r["id"]] for r in cos8)),
           "by_rel": dict(collections.Counter(M["claim_relation"][r["id"]] for r in cos8))})
res["B3_127_search"] = {"paper": {"total": 127, "by_dim": tdim, "by_rel": trel}, "candidates": b3,
                        "exact_matches": [x for x in b3 if x.get("exact_match_paper")]}

b5 = {"later_excl_pilot_anchors": {}, "claim_excl_pilot20": {}, "hyb_excl_pilot_anchors": {}}
for c in T1:
    lat = {k: v for k, v in I[c]["later"].items() if anchor_of(k) not in PILOT_ANCH}
    b5["later_excl_pilot_anchors"][c] = fmt_ci(sum(v["rej"] for v in lat.values()), len(lat))
pilot20 = set(json.load(open(f"{OUT}/r0_pilot_overlap.json"))["pilot_20_n2_pairs_in_claim160"])
dec = M["claim_decision"]
for c in ["v1", "coi", "si", "arc", "ra", "scholareval", "v2_reflection", "claimnov", "graded", "central"]:
    cl = {k: v for k, v in I[c]["claim"].items() if k not in pilot20}
    acc = [k for k in cl if dec[k] == "acc"]; rj = [k for k in cl if dec[k] == "rej"]
    nb = sum(bool(cl[k].get("named")) for k in cl); nrej = sum(cl[k]["rej"] for k in cl)
    elig = len(cl) if c in ("claimnov", "graded", "central") else nrej
    b5["claim_excl_pilot20"][c] = {"rej_acc": fmt_ci(sum(cl[k]["rej"] for k in acc), len(acc)), "rej_rej": fmt_ci(sum(cl[k]["rej"] for k in rj), len(rj)),
                                   "names_B": fmt_ci(nb, len(cl)) if c in NAMERS else None,
                                   "elig": (fmt_ci(nb, elig) if elig else "0/0") if c in NAMERS else None}
hanch = {h["item"]: h["anchor"] for h in H}
b5["hyb_n_from_pilot_anchors"] = sum(1 for a in hanch.values() if a in PILOT_ANCH)
for c in ["v1", "coi", "si", "arc", "ra", "scholareval", "v2_reflection", "claimnov"]:
    xs = {k: v for k, v in I[c]["hyb"].items() if hanch[k] not in PILOT_ANCH}; n = len(xs)
    p = sum(not v["rej"] for v in xs.values()); nb = sum(bool(v["named"]) for v in xs.values())
    elig = n if c == "claimnov" else n - p
    b5["hyb_excl_pilot_anchors"][c] = {"n": n, "passes": fmt_ci(p, n), "names_B": fmt_ci(nb, n) if c in NAMERS else None,
                                       "of_elig": (fmt_ci(nb, elig) if elig else "0/0") if c in NAMERS else None}
    if c == "claimnov":
        b5["hyb_excl_pilot_anchors"][c]["names_and_passes"] = sum(1 for v in xs.values() if v["named"] and not v["rej"])
        b5["hyb_excl_pilot_anchors"][c]["on_spliced"] = sum(1 for v in xs.values() if v["B_on_spliced_claim"])
res["B5_heldout"] = b5

anc = collections.Counter(anchor_of(k) for k in I["claimnov"]["later"]); ancs = sorted(anc)
rng = np.random.default_rng(1); b6 = {}
idx = [rng.integers(0, len(ancs), len(ancs)) for _ in range(REPS)]
for c in T1:
    s = np.array([sum(v["rej"] for k, v in I[c]["later"].items() if anchor_of(k) == a) for a in ancs], float)
    n = np.array([anc[a] for a in ancs], float)
    bs = np.array([s[i].sum() / n[i].sum() for i in idx])
    k = int(s.sum())
    b6[c] = {"FA": f"{k}/400", "wilson_iid": fmt_ci(k, 400),
             "cluster_boot_CI_pct": [round(100 * float(np.percentile(bs, 2.5)), 2), round(100 * float(np.percentile(bs, 97.5)), 2)]}
res["B6_per_anchor_counts"] = dict(sorted(anc.items(), key=lambda x: -x[1]))
res["B6_FA_cluster_CI"] = b6
dump("r0b_batch2.json", res)
print(json.dumps({k: v for k, v in res.items() if not k.startswith("B1")}, indent=1, default=str)[:9000])
