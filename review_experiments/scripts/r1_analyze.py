import ast, difflib, json, os, glob
import numpy as np
from r0_common import *

I = json.load(open(f"{OUT}/r0_items.json")); M = I["meta"]; I = I["items"]
ATT = json.load(open(f"{OUT}/r0_attribution.json"))
ST = json.load(open(f"{OUT}/r0_stats.json"))
CLEAN = ST["clean_subset"]["items"]
DROP = {60, 83, 86, 87, 93}
DEC = M["claim_decision"]
fmt_ci0 = fmt_ci
fmt_ci = lambda k, n: fmt_ci0(k, n) if n else f"{k}/0"
res = {}
L = lambda fp: [json.loads(l) for l in open(fp)] if os.path.exists(fp) else []


def boot_JD(din, dout, lat, reps=10000, seed=0):
    rng = np.random.default_rng(seed)
    srcs = sorted({seed_of(k) for k in din}); ancs = sorted({anchor_of(k) for k in lat})
    ci = {s: [0, 0, 0] for s in srcs}
    for k, v in din.items(): ci[seed_of(k)][0] += v; ci[seed_of(k)][1] += 1
    for k, v in dout.items(): ci[seed_of(k)][2] += v
    ca = {a: [0, 0] for a in ancs}
    for k, v in lat.items(): ca[anchor_of(k)][0] += v; ca[anchor_of(k)][1] += 1
    A = np.array([ci[s] for s in srcs], float); B = np.array([ca[a] for a in ancs], float)
    Js, Ds = [], []
    for _ in range(reps):
        si = rng.integers(0, len(srcs), len(srcs)); ai = rng.integers(0, len(ancs), len(ancs))
        a = A[si].sum(0); b = B[ai].sum(0)
        Js.append(a[0] / a[1] - b[0] / b[1]); Ds.append((a[0] - a[2]) / a[1])
    pc = lambda x: [round(float(np.percentile(x, 2.5)), 3), round(float(np.percentile(x, 97.5)), 3)]
    return pc(Js), pc(Ds)


def t1row(din, dout, lat, named):
    n = len(din); c = sum(din.values()); so = sum(dout.values()); fa = sum(lat.values())
    jci, dci = boot_JD(din, dout, lat)
    return {"caught": f"{c}/{n}", "named": f"{sum(named.values())}/{n}" if named is not None else None, "src_out": f"{so}/{len(dout)}",
            "FA": fmt_ci(fa, len(lat)), "J": round(c / n - fa / len(lat), 3), "J_CI": jci, "Delta": round((c - so) / n, 3), "Delta_CI": dci,
            "rungs": [sum(v for k, v in din.items() if f"_r{q}_" in k) for q in (1, 2, 3, 4)]}


def keep(k): return seed_of(k) not in DROP


CN = {"in": {k: v["rej"] for k, v in I["claimnov"]["disg_in"].items() if keep(k)},
      "out": {k: v["rej"] for k, v in I["claimnov"]["disg_out"].items() if keep(k)},
      "named": {k: bool(v["named"]) for k, v in I["claimnov"]["disg_in"].items() if keep(k)},
      "later": {k: v["rej"] for k, v in I["claimnov"]["later"].items()},
      "claim_rej": {k: v["rej"] for k, v in I["claimnov"]["claim"].items()},
      "claim_names3": {k: ATT["per_item_claim"][k]["b_lvl3"] for k in I["claimnov"]["claim"]},
      "hyb_pass": {k: not v["rej"] for k, v in I["claimnov"]["hyb"].items()},
      "hyb_names3": {k: bool(v["named"] and (v["B_level"] or 0) >= 3) for k, v in I["claimnov"]["hyb"].items()}}
CN["claim_onpass"] = {k: CN["claim_names3"][k] and not CN["claim_rej"][k] for k in CN["claim_rej"]}
CN["hyb_onpass"] = {k: CN["hyb_names3"][k] and CN["hyb_pass"][k] for k in CN["hyb_pass"]}
res["claimnov_reference_T1"] = t1row(CN["in"], CN["out"], CN["later"], CN["named"])
COI = {"in": {k: v["rej"] for k, v in I["coi"]["disg_in"].items() if keep(k)}, "out": {k: v["rej"] for k, v in I["coi"]["disg_out"].items() if keep(k)},
       "later": {k: v["rej"] for k, v in I["coi"]["later"].items()}}
res["coi_reference_T1"] = t1row(COI["in"], COI["out"], COI["later"], {k: bool(v["named"]) for k, v in I["coi"]["disg_in"].items() if keep(k)})


def t2(rej, names, onpass, items):
    acc = [k for k in items if DEC.get(k) == "acc"]; rj = [k for k in items if DEC.get(k) == "rej"]
    return {"n": len(items), "rej_acc": fmt_ci(sum(rej[k] for k in acc), len(acc)), "rej_rej": fmt_ci(sum(rej[k] for k in rj), len(rj)),
            "rej_all": f"{sum(rej[k] for k in items)}/{len(items)}",
            "names_B": fmt_ci(sum(names[k] for k in items), len(items)) if names else None,
            "on_pass": fmt_ci(sum(onpass[k] for k in items), len(items)) if onpass else None}


ON = {(r["set"], r["key"]): r for r in L(f"{OUT}/r1_opennovelty.jsonl") if not r.get("error")}
ONerr = [r for r in L(f"{OUT}/r1_opennovelty.jsonl") if r.get("error")]
if ON:
    e5 = {"n_rows": len(ON), "n_error_rows": len(ONerr), "errors_still_missing": sorted({(r["set"], r["key"]) for r in ONerr} - set(ON))}
    allv = [v["verdict"] for r in ON.values() for v in r.get("verdicts", [])]
    e5["verdict_counts"] = dict(collections.Counter(allv))
    e5["n_contrib_dist"] = dict(collections.Counter(r.get("n_contrib", 0) for r in ON.values()))
    e5["refuter_outside_candidates"] = sum(1 for r in ON.values() for v in r.get("verdicts", []) if v["refuting_id"] and not v["refuter_in_candidates"])
    e5["refuting_title_unmatched"] = sum(1 for r in ON.values() for v in r.get("verdicts", []) if v.get("refuting_title") and not v["refuting_id"])

    def names(r, strict=False):
        return bool(r["B_id"]) and any(v["refuting_id"] == r["B_id"] and (not strict or v["verdict"] == "can_refute") for v in r.get("verdicts", []))
    for rule in ("reject_all", "reject_any"):
        din = {k: ON[("disg", k)][rule] for k in CN["in"] if ("disg", k) in ON}
        dout = {k: ON[("disg", k)][rule] for k in CN["out"] if ("disg", k) in ON}
        lat = {k: ON[("later", k)][rule] for k in CN["later"] if ("later", k) in ON}
        nm = {k: din[k] and names(ON[("disg", k)], strict=True) for k in din}
        row = {"T1": t1row(din, dout, lat, nm) if din and dout and lat else None,
               "n": {"in": len(din), "out": len(dout), "later": len(lat)}}
        row["T1_names_source_any_verdict"] = sum(names(ON[("disg", k)]) for k in din)
        mc = {"catch": mcnemar_exact(CN["in"], din), "src_out": mcnemar_exact(CN["out"], dout), "later_FA": mcnemar_exact(CN["later"], lat),
              "named": mcnemar_exact(CN["named"], nm)}
        cl = {k: ON[("claim", k)] for k in CN["claim_rej"] if ("claim", k) in ON}
        if cl:
            rej = {k: r[rule] for k, r in cl.items()}
            nb = {k: names(r) for k, r in cl.items()}; nbs = {k: names(r, True) for k, r in cl.items()}
            op = {k: nb[k] and not rej[k] for k in cl}; ops = {k: nbs[k] and not rej[k] for k in cl}
            row["T2"] = t2(rej, nb, op, sorted(cl)); row["T2_strict_can_refute_only"] = t2(rej, nbs, ops, sorted(cl))
            cle = [k for k in CLEAN if k in cl]
            row["T2_clean36"] = t2(rej, nb, op, cle); row["T2_clean36_strict"] = t2(rej, nbs, ops, cle)
            mc.update({"claim_rejected": mcnemar_exact(CN["claim_rej"], rej), "claim_names_B(any verdict)": mcnemar_exact(CN["claim_names3"], nb),
                       "claim_names_B_on_pass": mcnemar_exact(CN["claim_onpass"], op),
                       "clean36_names_B": mcnemar_exact({k: CN["claim_names3"][k] for k in cle}, {k: nb[k] for k in cle})})
            row["claim_mean_distinct_refuting_papers"] = round(float(np.mean([len({v["refuting_id"] for v in r.get("verdicts", []) if v["refuting_id"]}) for r in cl.values()])), 3)
        hy = {k: ON[("hyb", k)] for k in CN["hyb_pass"] if ("hyb", k) in ON}
        if hy:
            ps = {k: not r[rule] for k, r in hy.items()}; nb = {k: names(r) for k, r in hy.items()}; nbs = {k: names(r, True) for k, r in hy.items()}
            row["T3"] = {"n": len(hy), "passes": sum(ps.values()), "names_B": sum(nb.values()), "on_pass": sum(nb[k] and ps[k] for k in hy),
                         "names_B_strict": sum(nbs.values()), "on_pass_strict": sum(nbs[k] and ps[k] for k in hy)}
            mc.update({"hyb_passed": mcnemar_exact(CN["hyb_pass"], ps), "hyb_names_B": mcnemar_exact(CN["hyb_names3"], nb),
                       "hyb_names_B_on_pass": mcnemar_exact(CN["hyb_onpass"], {k: nb[k] and ps[k] for k in hy})})
            row["hyb_mean_distinct_refuting_papers"] = round(float(np.mean([len({v["refuting_id"] for v in r.get("verdicts", []) if v["refuting_id"]}) for r in hy.values()])), 3)
        row["mcnemar_vs_claimnov(a=ClaimNov,b=OpenNovelty)"] = mc
        e5[rule] = row
    res["E5_opennovelty"] = e5

_HYB = {h["item"]: h for h in json.load(open(f"{PILOT}/data/hybrids.json"))}
norm2 = lambda x: re.sub(r"[^a-z0-9]", "", (x or "").lower())[:50]
def Btitle_of(st, k):
    if st == "hyb": return _HYB[k]["B_title"]
    if st == "disg" and k.endswith("B_in"): return pool_B(seed_of(k))[0]["title"]
    return None

for mode in ("profile", "judgeonly"):
    for fp in sorted(glob.glob(f"{OUT}/r1_nonclaude_{mode}_*.jsonl")):
        R = {(r["set"], r["key"]): r for r in L(fp)}
        for (st, k), r in R.items():
            bt = Btitle_of(st, k)
            ap = r.get("anticipating_paper")
            r["names_B_anydec"] = bool(bt and ap and (str(ap).lower()[:60] == bt.lower()[:60] or norm2(ap) == norm2(bt)))
        out = {"file": os.path.basename(fp), "n_rows": len(R), "parse_errors": sum(r["parse_error"] for r in R.values()),
               "empty": sum(r["empty"] for r in R.values())}
        if mode == "judgeonly":
            out["byte_identical_to_e1"] = sum(r.get("byte_identical_to_e1", False) for r in R.values())
        for who in ("new", "sonnet"):
            f = (lambda r: r["rej"]) if who == "new" else (lambda r: r["sonnet_rej"])
            fn = (lambda r: r["names_B"]) if who == "new" else (lambda r: r["sonnet_names_B"])
            din = {k: f(r) for (s, k), r in R.items() if s == "disg" and k.endswith("B_in") and keep(k)}
            dout = {k: f(r) for (s, k), r in R.items() if s == "disg" and k.endswith("B_out") and keep(k)}
            lat = {k: f(r) for (s, k), r in R.items() if s == "later"}
            hyb = {k: r for (s, k), r in R.items() if s == "hyb"}
            o = {"n": {"in": len(din), "out": len(dout), "later": len(lat), "hyb": len(hyb)},
                 "caught": sum(din.values()), "named": sum(fn(R[("disg", k)]) for k in din), "src_out": sum(dout.values()), "FA": sum(lat.values())}
            if din: o["catch_rate"] = round(o["caught"] / len(din), 4)
            if dout: o["src_out_rate"] = round(o["src_out"] / len(dout), 4)
            if lat: o["FA_rate"] = round(o["FA"] / len(lat), 4)
            if din and lat: o["J"] = round(o["caught"] / len(din) - o["FA"] / len(lat), 3)
            if din and dout: o["Delta"] = round((o["caught"] - o["src_out"]) / len(din), 3)
            if din and dout and lat and mode == "judgeonly": o["J_CI"], o["Delta_CI"] = boot_JD(din, dout, lat)
            if who == "new":
                o["named_anydec_srcin"] = sum(R[("disg", k)]["names_B_anydec"] for k in din)
            if hyb and who == "new":
                o["hyb_names_B_anydec"] = sum(r["names_B_anydec"] for r in hyb.values())
                o["hyb_names_B_anydec_on_pass"] = sum(r["names_B_anydec"] and not r["rej"] for r in hyb.values())
            if hyb:
                o["hyb_passes"] = sum(not f(r) for r in hyb.values()); o["hyb_names_B"] = sum(fn(r) for r in hyb.values())
                o["hyb_names_B_on_pass"] = sum(fn(r) and not f(r) for r in hyb.values())
            out[who] = o
        out["agree"] = {s: f"{sum(r['rej'] == r['sonnet_rej'] for (ss, k), r in R.items() if ss == s)}/{sum(1 for (ss, k) in R if ss == s)}" for s in ("disg", "later", "hyb")}
        res[f"E7_{mode}_{os.path.basename(fp)}"] = out

for fp in sorted(glob.glob(f"{OUT}/r1_nonclaude_profile_*.jsonl")):
    m = fp.split("r1_nonclaude_profile_")[1]
    P = {(r["set"], r["key"]): r for r in L(fp)}
    J = {(r["set"], r["key"]): r for r in L(f"{OUT}/r1_nonclaude_judgeonly_{m}")}
    chk = collections.Counter()
    for (s, k), r in P.items():
        row = I["claimnov"]["disg_in" if k.endswith("B_in") else "disg_out"][k] if s == "disg" else I["claimnov"][s][k]
        chk[(s, r["sonnet_rej"] == row["rej"])] += 1
    out = {"paper_verdict_vs_result_row_agreement": {f"{a}:{b}": v for (a, b), v in chk.items()}}
    for lab, sel in (("disg_in", lambda s, k: s == "disg" and k.endswith("B_in")), ("disg_out", lambda s, k: s == "disg" and k.endswith("B_out")),
                     ("later", lambda s, k: s == "later"), ("hyb", lambda s, k: s == "hyb")):
        ks = [x for x in P if sel(*x) and x in J and (x[0] != "disg" or keep(x[1]))]
        a = {x: P[x]["rej"] for x in ks}; b = {x: J[x]["rej"] for x in ks}
        sa = {x: P[x]["sonnet_rej"] for x in ks}; sb = {x: J[x]["sonnet_rej"] for x in ks}
        out[lab] = {"n": len(ks), "nonclaude_profile_rej": sum(a.values()), "nonclaude_judgeonly_rej": sum(b.values()),
                    "sonnet_profile_rej": sum(sa.values()), "sonnet_judgeonly_rej": sum(sb.values()),
                    "mcnemar_nonclaude_profile_vs_judgeonly": mcnemar_exact(a, b),
                    "mcnemar_sonnet_profile_vs_judgeonly": mcnemar_exact(sa, sb)}
        if lab == "hyb" or lab == "disg_in":
            out[lab]["nonclaude_profile_names_B"] = sum(P[x]["names_B"] for x in ks); out[lab]["nonclaude_judgeonly_names_B"] = sum(J[x]["names_B"] for x in ks)
    res[f"E7_paired_{m}"] = out

ncache = json.load(open(NCACHE))
def _key(*p): return hashlib.md5("|".join(str(x) for x in p).encode()).hexdigest()[:16]
def claims_of(t, a): return ncache.get("claims4::" + _key(t, a))
def ctx(f):
    parts = [f"Facet — attribute: {f.get('attribute', '')}; value: {f.get('value', '')}"]
    if f.get("paper_abstract"): parts.append(f"Abstract: {f['paper_abstract'][:1200]}")
    return "\n".join(parts)
def level(t, c):
    return ncache.get("entail::" + hashlib.md5((t["claim"] + "||" + c["claim"] + "||" + ctx(t) + "||" + ctx(c)).encode()).hexdigest()[:16])
def cos_rank(qtext, pool):
    q = [float(x) for x in cached_embed(qtext)]; qn = sum(x * x for x in q) ** .5; sc = []
    for p in pool:
        v = cached_embed(f"{p['title']}. {p['abstract']}")
        sc.append((sum(a * b for a, b in zip(q, v)) / (qn * sum(b * b for b in v) ** .5), p["id"]))
    return [pid for _, pid in sorted(sc, key=lambda x: -x[0])]

src = open(f"{PILOT}/make_disguises.py").read()
R1 = next(ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "R1")
pairs = [json.loads(l) for l in open("eval/data/claim_disputed/testset_clean.jsonl")]
PROF = json.load(open(f"{OUT}/r0_claim_profiles.json"))["profiles"]
rows = {"claim": {}, "hyb": {}}
for k, p in PROF.items():
    cand = I["claimnov"]["claim"][k]["candidates"]; lv = {}
    for r in p["claims"]:
        for pid, l in r["levels"].items():
            if pid in cand: lv[pid] = max(lv.get(pid, 0), l)
    i = int(k.split("_")[1])
    d = json.loads(re.search(r"\{.*\}", cached_bedrock(SONNET_46_BEDROCK, [{"role": "user", "content": R1.format(abstract=pairs[i]["A_abstract"])}], 3000, f"n2f_ideaify{i}"), re.S).group(0))
    pool = json.load(open(f"{PILOT}/data/pools_full/{k}.json"))["pool"]
    rows["claim"][k] = {"B": p["B_id"], "prof": lv, "order": cand, "cos": cos_rank(f"{d['title']}: {d['description']}", pool), "titles": {x["id"]: x["title"] for x in pool}}
H = json.load(open(f"{PILOT}/data/hybrids.json")); pools = {}
for h in H:
    a = h["anchor"]
    if a not in pools: pools[a] = json.load(open(f"{anchor_dir(a)}/pool_preT.json"))[:400]
    pool = pools[a]
    tcl = claims_of(h["title"], h["description"]) or []
    facets = [{**c, "paper_id": p["id"], "paper_abstract": p["abstract"]} for p in pool for c in (claims_of(p["title"], p["abstract"]) or [])]
    shown, order = {}, []
    for c in tcl:
        t = {**c, "paper_abstract": h["description"]}; best = {}
        for f in facets:
            if f["dimension"] != c["dimension"]: continue
            l = level(t, f)
            if l is not None: best[f["paper_id"]] = max(best.get(f["paper_id"], 0), l)
        for pid, l in sorted(best.items(), key=lambda x: -x[1])[:5]:
            shown[pid] = max(shown.get(pid, 0), l)
            if pid not in order: order.append(pid)
    rows["hyb"][h["item"]] = {"B": h["B_id"], "prof": shown, "order": order, "cos": cos_rank(f"{h['title']}: {h['description']}", pool),
                              "titles": {x["id"]: x["title"] for x in pool}}

E1 = {(r["set"], r["key"]): r for r in L(f"{OUT}/e1_judge_only_k8.jsonl")}
a2 = {}
for S in ("claim", "hyb"):
    R = rows[S]; ks = sorted(R)
    o = {"n": len(ks)}
    def top1_cn(r):
        if not r["prof"]: return None
        top = max(r["prof"].values()); return next(p for p in r["order"] if r["prof"].get(p) == top)
    o["ClaimNov top-1 (highest level, profile-order tiebreak) = B"] = sum(top1_cn(R[k]) == R[k]["B"] for k in ks)
    m3 = {k: [p for p, l in R[k]["prof"].items() if l >= 3] for k in ks}
    o["ClaimNov level>=3 set contains B"] = sum(R[k]["B"] in m3[k] for k in ks)
    o["ClaimNov level>=3 set: mean size"] = round(float(np.mean([len(m3[k]) for k in ks])), 3)
    o["whole-proposal embedding top-1 = B"] = sum(R[k]["cos"][0] == R[k]["B"] for k in ks)
    o["whole-proposal embedding top-m3 contains B (m3 = ClaimNov level>=3 set size)"] = sum(R[k]["B"] in R[k]["cos"][:len(m3[k])] for k in ks)
    for g in ("v1", "coi", "si"):
        it = I[g][S]
        def first(ev): return (ev[0] if isinstance(ev, list) and ev else ev) if ev else None
        o[f"{g}: named paper top-1 = B (on rejection)"] = sum(bool(it[k]["rej"]) and norm(first(it[k]["evidence"])) == norm(R[k]["titles"][R[k]["B"]]) for k in ks)
        o[f"{g}: names B (any returned title, on rejection; = paper's Names B)"] = sum(bool(it[k]["named"]) for k in ks)
        o[f"{g}: mean papers named per item"] = round(float(np.mean([(len(it[k]["evidence"]) if isinstance(it[k]["evidence"], list) else 1) if (it[k]["rej"] and it[k]["evidence"]) else 0 for k in ks])), 3)
    o["judge-only top-8 (E1): anticipating paper = B"] = sum(E1[(S, k)]["names_B"] for k in ks if (S, k) in E1)
    o["judge-only top-8 (E1): anticipating paper = B on pass"] = sum(E1[(S, k)]["names_B"] and E1[(S, k)]["novel"] for k in ks if (S, k) in E1)
    if ON:
        for strict in (False, True):
            lab = "can_refute only" if strict else "any verdict"
            hit = {k: any(v["refuting_id"] == R[k]["B"] and (not strict or v["verdict"] == "can_refute") for v in ON[(S, k)].get("verdicts", [])) for k in ks if (S, k) in ON}
            o[f"OpenNovelty: some contribution's refuting paper = B ({lab})"] = sum(hit.values())
            first = {}
            for k in ks:
                if (S, k) not in ON: continue
                vs = ON[(S, k)].get("verdicts", [])
                cand = [v for v in vs if v["verdict"] == "can_refute" and v["refuting_id"]] + ([] if strict else [v for v in vs if v["verdict"] == "unclear" and v["refuting_id"]])
                first[k] = bool(cand) and cand[0]["refuting_id"] == R[k]["B"]
            o[f"OpenNovelty top-1 (first can_refute{'' if strict else ', else first unclear'} contribution's paper) = B"] = sum(first.values())
        o["OpenNovelty: mean distinct refuting papers per item"] = round(float(np.mean([len({v["refuting_id"] for v in ON[(S, k)].get("verdicts", []) if v["refuting_id"]}) for k in ks if (S, k) in ON])), 3)
        o["OpenNovelty: n items"] = sum((S, k) in ON for k in ks)
    a2[S] = o
t2c = {}
for g in ("v1", "coi", "si", "arc", "ra", "scholareval", "v2_reflection"):
    it = I[g]["claim"]
    nm = {k: bool(it[k].get("named")) for k in CLEAN} if g in ("v1", "coi", "si", "arc") else None
    t2c[g] = t2({k: it[k]["rej"] for k in CLEAN}, nm, {k: False for k in CLEAN} if nm else None, CLEAN)
for g in ("claimnov", "graded", "central"):
    rej = {k: I[g]["claim"][k]["rej"] for k in CLEAN}
    t2c[g + " (Names B = profile level>=3)"] = t2(rej, CN["claim_names3"], {k: CN["claim_names3"][k] and not rej[k] for k in CLEAN}, CLEAN)
rej = {k: not E1[("claim", k)]["novel"] for k in CLEAN}; nb = {k: E1[("claim", k)]["names_B"] for k in CLEAN}
t2c["judge-only top-8 (E1)"] = t2(rej, nb, {k: nb[k] and not rej[k] for k in CLEAN}, CLEAN)
a2["T2_clean36"] = t2c
a2["T2_clean36_mcnemar_names_B_vs_claimnov"] = {g: mcnemar_exact({k: CN["claim_names3"][k] for k in CLEAN}, {k: bool(I[g]["claim"][k].get("named")) for k in CLEAN}) for g in ("v1", "coi", "si")}
res["A2"] = a2
dump("r1_summary.json", res)
print(json.dumps(res, indent=1)[:20000])
