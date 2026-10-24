import json, collections, glob
import numpy as np
from r0_common import *
_fc = fmt_ci
fmt_ci = lambda k, n: _fc(k, n) if n else f"{k}/0"

I = json.load(open(f"{OUT}/r0_items.json")); M = I["meta"]; I = I["items"]
DEC = M["claim_decision"]; DROP = {60, 83, 86, 87, 93}
L = lambda fp: [json.loads(l) for l in open(fp)] if os.path.exists(fp) else []
FREE = json.load(open(f"{OUT}/r3_free.json"))
NAMES3 = FREE["names3_claim_per_item"]; CONFL = set(FREE["conflicts_101"]); AGREE = set(FREE["agree_292"])
P = {(r["set"], r["key"]): r for r in L(f"{OUT}/r3_prompts.jsonl")}
CTX = {(r["set"], r["key"]): r for r in L(f"{OUT}/r3_ctx.jsonl") if not r.get("error")}
E1 = {(r["set"], r["key"]): r for r in L(f"{OUT}/e1_judge_only_k8.jsonl")}
nt = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())[:60]
res = {"coverage": {"|".join(map(str, a)): b for a, b in collections.Counter((s, P[(s, k)]["src"], P[(s, k)].get("profile_k")) for (s, k) in CTX).items()},
       "parse_errors": sum(r["parse_error"] for r in CTX.values())}

def prof_ids(s, k):
    pr = P[(s, k)]["prompt"]; papers = pr.split("PAPERS IN THE PROFILE:\n", 1)[1].split("\n\nReturn JSON only", 1)[0]
    return {nt(l[2:].split(": ")[0]) for l in papers.split("\n") if l.startswith("- ")}
def g_of(s, k): return ("disg_in" if k.endswith("B_in") else "disg_out") if s == "disg" else s
def keep(k): return seed_of(k) not in DROP
def shipped(s, k): return bool(I["claimnov"][g_of(s, k)][k]["rej"])
def shipped_named(k): return bool(I["claimnov"]["disg_in"][k]["named"])
def jo(s, k): return not E1[(s, k)]["novel"]
def jo_named(s, k): return bool(E1[(s, k)]["names_B"]) and jo(s, k)
def cx(s, k): return bool(CTX[(s, k)]["rej"])
def cx_named(s, k): return cx(s, k) and bool(CTX[(s, k)]["names_B"])

def boot(din, dout, lat, reps=10000, seed=0):
    rng = np.random.default_rng(seed)
    srcs = sorted({seed_of(k) for k in list(din) + list(dout)}); ancs = sorted({anchor_of(k) for k in lat})
    ci = {s: [0, 0, 0, 0] for s in srcs}
    for k, v in din.items(): ci[seed_of(k)][0] += v; ci[seed_of(k)][1] += 1
    for k, v in dout.items(): ci[seed_of(k)][2] += v; ci[seed_of(k)][3] += 1
    ca = {a: [0, 0] for a in ancs}
    for k, v in lat.items(): ca[anchor_of(k)][0] += v; ca[anchor_of(k)][1] += 1
    A = np.array([ci[s] for s in srcs], float); B = np.array([ca[a] for a in ancs], float)
    si = rng.integers(0, len(srcs), (reps, len(srcs))); ai = rng.integers(0, len(ancs), (reps, len(ancs)))
    a = A[si].sum(1); b = B[ai].sum(1)
    J = a[:, 0] / a[:, 1] - b[:, 0] / b[:, 1]; D = a[:, 0] / a[:, 1] - a[:, 2] / np.maximum(a[:, 3], 1)
    pc = lambda x: [round(float(np.percentile(x, 2.5)), 3), round(float(np.percentile(x, 97.5)), 3)]
    return pc(J), pc(D)

def t1(din, dout, lat, named):
    c, so, fa = sum(din.values()), sum(dout.values()), sum(lat.values())
    jci, dci = boot(din, dout, lat)
    return {"caught": f"{c}/{len(din)}", "named": f"{sum(named.values())}/{len(din)}", "src_out": f"{so}/{len(dout)}",
            "FA": fmt_ci(fa, len(lat)), "J": round(c / len(din) - fa / len(lat), 3), "J_CI": jci,
            "Delta": round(c / len(din) - so / len(dout), 3), "Delta_CI": dci,
            "rungs": [sum(v for k, v in din.items() if f"_r{q}_" in k) for q in (1, 2, 3, 4)]}

def mc(a, b): return mcnemar_exact(a, b)

C1 = {}
for scope in ("exact", "all_with_profile"):
    ok = lambda s, k: (s, k) in CTX and (scope != "exact" or P[(s, k)]["src"] == "exact")
    kin = [k for (s, k) in CTX if s == "disg" and k.endswith("B_in") and keep(k) and ok(s, k)]
    kout = [k for (s, k) in CTX if s == "disg" and k.endswith("B_out") and keep(k) and ok(s, k)]
    klat = [k for (s, k) in CTX if s == "later" and ok(s, k)]
    rows = {}
    for nm, f, fn in (("ClaimNov+ctx", cx, lambda k: cx_named("disg", k)), ("ClaimNov shipped", shipped, shipped_named),
                      ("judge-only top-8", jo, lambda k: jo_named("disg", k))):
        rows[nm] = t1({k: f("disg", k) for k in kin}, {k: f("disg", k) for k in kout}, {k: f("later", k) for k in klat}, {k: fn(k) for k in kin})
    tests = {}
    for other, f, fn in (("shipped", shipped, shipped_named), ("judge_only", jo, lambda k: jo_named("disg", k))):
        tests[f"ctx_vs_{other}"] = {
            "catch": mc({k: cx("disg", k) for k in kin}, {k: f("disg", k) for k in kin}),
            "named": mc({k: cx_named("disg", k) for k in kin}, {k: fn(k) for k in kin}),
            "src_out_rej": mc({k: cx("disg", k) for k in kout}, {k: f("disg", k) for k in kout}),
            "later_FA": mc({k: cx("later", k) for k in klat}, {k: f("later", k) for k in klat})}
    C1[scope] = {"n": {"src_in": len(kin), "src_out": len(kout), "later": len(klat)}, "rows": rows, "mcnemar": tests}
C1["reference_full_paper_sets"] = {
    "ClaimNov shipped": t1({k: v["rej"] for k, v in I["claimnov"]["disg_in"].items() if keep(k)}, {k: v["rej"] for k, v in I["claimnov"]["disg_out"].items() if keep(k)},
                           {k: v["rej"] for k, v in I["claimnov"]["later"].items()}, {k: bool(v["named"]) for k, v in I["claimnov"]["disg_in"].items() if keep(k)})}
lat_all = [k for (s, k) in CTX if s == "later"]
fa_sub = {}
for nm, f in (("ClaimNov+ctx", cx), ("ClaimNov shipped", shipped), ("judge-only top-8", jo)):
    o = {}
    for sn, ks in (("all", lat_all), ("excl_101_conflicts", [k for k in lat_all if k not in CONFL]), ("conflicts_101", [k for k in lat_all if k in CONFL]),
                   ("opus_gpt5_agree", [k for k in lat_all if k in AGREE]), ("exact341", [k for k in lat_all if P[("later", k)]["src"] == "exact"])):
        o[sn] = fmt_ci(sum(f("later", k) for k in ks), len(ks))
    fa_sub[nm] = o
fa_sub["mcnemar_ctx_vs_shipped_agree"] = mc({k: cx("later", k) for k in lat_all if k in AGREE}, {k: shipped("later", k) for k in lat_all if k in AGREE})
fa_sub["mcnemar_ctx_vs_shipped_excl_conflicts"] = mc({k: cx("later", k) for k in lat_all if k not in CONFL}, {k: shipped("later", k) for k in lat_all if k not in CONFL})
C1["later_FA_subsets"] = fa_sub
C1["ctx_FA_items"] = [{"key": k, "anticipating": CTX[("later", k)]["anticipating_title"], "in_top8": CTX[("later", k)]["anticipating_in_top8"],
                       "in_profile_papers": nt(CTX[("later", k)]["anticipating_title"]) in prof_ids("later", k), "conflict": k in CONFL,
                       "agree": k in AGREE, "shipped_rej": shipped("later", k), "jo_rej": jo("later", k), "src": P[("later", k)]["src"]}
                      for k in lat_all if cx("later", k)]

def claim_block(keys):
    acc = [k for k in keys if DEC.get(k) == "acc"]; rj = [k for k in keys if DEC.get(k) == "rej"]
    o = {}
    for nm, rej, vB in (("ClaimNov+ctx", lambda k: cx("claim", k), lambda k: bool(CTX[("claim", k)]["names_B"])),
                        ("ClaimNov shipped", lambda k: shipped("claim", k), lambda k: bool(I["claimnov"]["claim"][k]["verdict_names_B"])),
                        ("judge-only top-8", lambda k: jo("claim", k), lambda k: bool(E1[("claim", k)]["names_B"]))):
        o[nm] = {"rej_acc": fmt_ci(sum(rej(k) for k in acc), len(acc)), "rej_rej": fmt_ci(sum(rej(k) for k in rj), len(rj)),
                 "rej_all": f"{sum(rej(k) for k in keys)}/{len(keys)}",
                 "verdict_names_B": f"{sum(vB(k) for k in keys)}/{len(keys)}",
                 "verdict_names_B_on_pass": f"{sum(vB(k) and not rej(k) for k in keys)}/{len(keys)}",
                 "verdict_names_B_on_rejection": f"{sum(vB(k) and rej(k) for k in keys)}/{len(keys)}",
                 "profile_names_B_lvl3": f"{sum(NAMES3[k] for k in keys)}/{len(keys)}" if nm != "judge-only top-8" else None,
                 "profile_names_B_lvl3_on_pass": f"{sum(NAMES3[k] and not rej(k) for k in keys)}/{len(keys)}" if nm != "judge-only top-8" else None}
    o["mcnemar"] = {
        "rej_ctx_vs_shipped": mc({k: cx("claim", k) for k in keys}, {k: shipped("claim", k) for k in keys}),
        "rej_ctx_vs_jo": mc({k: cx("claim", k) for k in keys}, {k: jo("claim", k) for k in keys}),
        "onpass_ctx_vs_shipped": mc({k: NAMES3[k] and not cx("claim", k) for k in keys}, {k: NAMES3[k] and not shipped("claim", k) for k in keys}),
        "verdict_names_B_ctx_vs_shipped": mc({k: bool(CTX[("claim", k)]["names_B"]) for k in keys}, {k: bool(I["claimnov"]["claim"][k]["verdict_names_B"]) for k in keys}),
        "verdict_names_B_ctx_vs_jo": mc({k: bool(CTX[("claim", k)]["names_B"]) for k in keys}, {k: bool(E1[("claim", k)]["names_B"]) for k in keys})}
    return o
ck = sorted(k for (s, k) in CTX if s == "claim")
CLEAN = json.load(open(f"{OUT}/r0_stats.json"))["clean_subset"]["items"]
C1["claim"] = {"all": claim_block(ck), "exact147": claim_block([k for k in ck if P[("claim", k)]["src"] == "exact"]), "clean36": claim_block([k for k in ck if k in CLEAN])}

hk = sorted(k for (s, k) in CTX if s == "hyb")
H = I["claimnov"]["hyb"]
hn3 = {k: bool(H[k]["named"] and (H[k]["B_level"] or 0) >= 3) for k in hk}
hyb = {}
for nm, rej, vB in (("ClaimNov+ctx", lambda k: cx("hyb", k), lambda k: bool(CTX[("hyb", k)]["names_B"])),
                    ("ClaimNov shipped", lambda k: bool(H[k]["rej"]), lambda k: bool(H[k]["verdict_names_B"])),
                    ("judge-only top-8", lambda k: jo("hyb", k), lambda k: bool(E1[("hyb", k)]["names_B"]))):
    hyb[nm] = {"passes": f"{sum(not rej(k) for k in hk)}/{len(hk)}", "verdict_names_B": sum(vB(k) for k in hk),
               "verdict_names_B_on_pass": sum(vB(k) and not rej(k) for k in hk),
               "profile_names_B_lvl3": sum(hn3.values()) if nm != "judge-only top-8" else None,
               "profile_names_B_lvl3_on_pass": sum(hn3[k] and not rej(k) for k in hk) if nm != "judge-only top-8" else None}
hyb["mcnemar"] = {"pass_ctx_vs_shipped": mc({k: not cx("hyb", k) for k in hk}, {k: not H[k]["rej"] for k in hk}),
                  "pass_ctx_vs_jo": mc({k: not cx("hyb", k) for k in hk}, {k: not jo("hyb", k) for k in hk}),
                  "verdict_names_B_ctx_vs_jo": mc({k: bool(CTX[("hyb", k)]["names_B"]) for k in hk}, {k: bool(E1[("hyb", k)]["names_B"]) for k in hk}),
                  "onpass_ctx_vs_shipped": mc({k: hn3[k] and not cx("hyb", k) for k in hk}, {k: hn3[k] and not H[k]["rej"] for k in hk})}
C1["hybrids"] = hyb

JO12 = {r["key"]: r for r in L(f"{OUT}/r2_e12_judgeonly.jsonl") if not r.get("error")}
R10 = {}
for r in jl("coupled_veto_ac_profile"):
    if r.get("novel") is not None and r["item"] not in R10: R10[r["item"]] = r
fa = {"ClaimNov+ctx": sum(cx("later", k) for k in lat_all) / len(lat_all),
      "ClaimNov shipped": sum(v["rej"] for v in I["claimnov"]["later"].values()) / 400,
      "judge-only top-8": sum(jo("later", k) for k in lat_all) / len(lat_all),
      "ClaimNov fixed k=10": sum(v["rej"] for v in I["claimnov_k10"]["later"].values()) / 400}
e12 = {"later_FA_used_for_J": {k: round(v, 4) for k, v in fa.items()}}
for sub, cond in (("shipped_profile_407", lambda p: p["profile_k"] == "shipped"), ("shipped_exact_283", lambda p: p["profile_k"] == "shipped" and p["src"] == "exact"),
                  ("k10_profile_329", lambda p: p["profile_k"] == "k10"), ("all_736", lambda p: True)):
    ks = sorted(k for (s, k) in CTX if s == "e12" and cond(P[(s, k)]))
    if not ks: continue
    m = {k: P[("e12", k)]["match"] for k in ks}
    dec = {"ClaimNov+ctx": {k: cx("e12", k) for k in ks},
           "judge-only top-8": {k: not JO12[k]["novel"] for k in ks if k in JO12},
           "ClaimNov paper verdict (profile of this subset)": {k: bool(P[("e12", k)]["paper_row_rej"]) for k in ks}}
    named = {"ClaimNov+ctx": {k: cx("e12", k) and nt(CTX[("e12", k)]["anticipating_title"]) == nt(m[k]) for k in ks},
             "judge-only top-8": {k: (not JO12[k]["novel"]) and JO12[k]["names_match"] for k in ks if k in JO12},
             "ClaimNov paper verdict (profile of this subset)": {k: bool(P[("e12", k)]["paper_row_rej"]) and nt(P[("e12", k)]["paper_row_evidence"]) == nt(m[k]) for k in ks}}
    o = {"n": len(ks)}
    for nm in dec:
        n = len(dec[nm]); c = sum(dec[nm].values())
        fan = "ClaimNov shipped" if nm.startswith("ClaimNov paper") and sub != "k10_profile_329" else ("ClaimNov fixed k=10" if nm.startswith("ClaimNov paper") else nm)
        o[nm] = {"caught": fmt_ci(c, n), "named": sum(named[nm].values()), "J_vs_later": round(c / n - fa[fan], 3), "FA_used": fan}
    o["mcnemar_ctx_vs_paper_verdict"] = mc(dec["ClaimNov+ctx"], dec["ClaimNov paper verdict (profile of this subset)"])
    o["mcnemar_ctx_vs_jo"] = mc(dec["ClaimNov+ctx"], dec["judge-only top-8"])
    o["ctx_rejections_anticipating_in_top8"] = sum(cx("e12", k) and CTX[("e12", k)]["anticipating_in_top8"] for k in ks)
    e12[sub] = o
C1["E12"] = e12
use = collections.Counter()
for (s, k), r in CTX.items():
    if r["rej"]:
        inprof = nt(r["anticipating_title"]) in {x for x in prof_ids(s, k)} if r["anticipating_title"] else False
        use[(s, "rej", "anticipating_in_profile_papers" if inprof else ("only_in_top8" if r["anticipating_in_top8"] else "other/none"))] += 1
C1["ctx_rejection_source"] = {str(k): v for k, v in sorted(use.items())}
res["C1"] = C1

REP = {r: {(x["set"], x["key"]): x for x in L(f"{OUT}/r3_rep{r}.jsonl") if not x.get("error")} for r in (1, 2)}
COI = collections.defaultdict(dict)
for x in L(f"{OUT}/r3_coi_rerun.jsonl"):
    if not x.get("error"): COI[x["run"]][x["key"]] = x
if REP[1]:
    ex = sorted((s, k) for (s, k), p in P.items() if p["src"] == "exact" and s in ("later", "disg"))
    d3 = {"n_exact": len(ex), "runs_done": {r: len(REP[r]) for r in REP}}
    for r in (1, 2):
        d3[f"rep{r}_identical_text_to_paper"] = sum(bool(x.get("identical_to_paper_text")) for x in REP[r].values())
        d3[f"rep{r}_decision_flips_vs_paper"] = sum(x["rej"] != shipped(s, k) for (s, k), x in REP[r].items())
    both = [sk for sk in ex if sk in REP[1] and sk in REP[2]]
    d3["rep1_vs_rep2_decision_flips"] = sum(REP[1][sk]["rej"] != REP[2][sk]["rej"] for sk in both)
    d3["rep1_vs_rep2_identical_text"] = sum(REP[1][sk]["text"] == REP[2][sk]["text"] for sk in both)
    def runs_cn(r):
        if r == 0: return {sk: shipped(*sk) for sk in ex}
        return {sk: REP[r][sk]["rej"] for sk in ex if sk in REP[r]}
    def runs_coi(r):
        if r == 0: return {(s, k): bool(I["coi"][g_of(s, k)][k]["rej"]) for (s, k) in ex}
        return {(x["set"], x["key"]): x["rej"] for x in COI[r].values()}
    def core(coi_rr):
        o = {}
        common = set(ex)
        for r in (0, 1, 2): common &= set(runs_cn(r))
        for r in coi_rr: common &= set(runs_coi(r))
        common = sorted(common)
        if not (any(k.endswith("B_in") for s, k in common) and any(k.endswith("B_out") for s, k in common) and any(s == "later" for s, k in common)):
            return {"skipped": "incomplete", "n": len(common)}
        def JD(dec):
            din = {k: dec[("disg", k)] for (s, k) in common if s == "disg" and k.endswith("B_in")}
            dout = {k: dec[("disg", k)] for (s, k) in common if s == "disg" and k.endswith("B_out")}
            lat = {k: dec[("later", k)] for (s, k) in common if s == "later"}
            return din, dout, lat
        per = {}
        for nm, fn, rr in (("ClaimNov", runs_cn, (0, 1, 2)), ("CoI", runs_coi, coi_rr)):
            for r in rr:
                din, dout, lat = JD(fn(r))
                per[f"{nm} run{r}"] = t1(din, dout, lat, {k: False for k in din})
        o["common_items"] = {"n": len(common), "src_in": sum(1 for s, k in common if k.endswith("B_in")), "src_out": sum(1 for s, k in common if k.endswith("B_out")),
                             "later": sum(1 for s, k in common if s == "later")}
        o["per_run"] = per
        rng = np.random.default_rng(1)
        srcs = sorted({seed_of(k) for s, k in common if s == "disg"}); ancs = sorted({anchor_of(k) for s, k in common if s == "later"})
        din = [k for s, k in common if k.endswith("B_in")]; dout = [k for s, k in common if k.endswith("B_out")]; lat = [k for s, k in common if s == "later"]
        def arr(dec):
            A = np.array([[sum(dec[("disg", k)] for k in din if seed_of(k) == s), sum(seed_of(k) == s for k in din),
                           sum(dec[("disg", k)] for k in dout if seed_of(k) == s), sum(seed_of(k) == s for k in dout)] for s in srcs], float)
            B = np.array([[sum(dec[("later", k)] for k in lat if anchor_of(k) == a), sum(anchor_of(k) == a for k in lat)] for a in ancs], float)
            return A, B
        CN_ = [arr(runs_cn(r)) for r in (0, 1, 2)]; CO_ = [arr(runs_coi(r)) for r in coi_rr]
        def JDv(AB, si, ai):
            a = AB[0][si].sum(0); b = AB[1][ai].sum(0)
            return a[0] / a[1] - b[0] / b[1], a[0] / a[1] - a[2] / max(a[3], 1)
        allsi, allai = np.arange(len(srcs)), np.arange(len(ancs))
        jc = [JDv(x, allsi, allai)[0] for x in CN_]; jo_ = [JDv(x, allsi, allai)[0] for x in CO_]
        dc = [JDv(x, allsi, allai)[1] for x in CN_]; do_ = [JDv(x, allsi, allai)[1] for x in CO_]
        dJ, dD = [], []
        for _ in range(10000):
            si = rng.integers(0, len(srcs), len(srcs)); ai = rng.integers(0, len(ancs), len(ancs))
            vc = [JDv(x, si, ai) for x in CN_]; vo = [JDv(x, si, ai) for x in CO_]
            dJ.append(np.mean([v[0] for v in vc]) - np.mean([v[0] for v in vo])); dD.append(np.mean([v[1] for v in vc]) - np.mean([v[1] for v in vo]))
        o["J_claimnov_runs"] = [round(x, 3) for x in jc]; o["J_coi_runs"] = [round(x, 3) for x in jo_]
        o["Delta_claimnov_runs"] = [round(x, 3) for x in dc]; o["Delta_coi_runs"] = [round(x, 3) for x in do_]
        o["DeltaJ_mean_over_runs"] = round(float(np.mean(jc) - np.mean(jo_)), 3)
        o["DeltaJ_paired_cluster_bootstrap_CI"] = [round(float(np.percentile(dJ, 2.5)), 3), round(float(np.percentile(dJ, 97.5)), 3)]
        o["DeltaJ_frac_boot_le_0"] = round(float(np.mean(np.array(dJ) <= 0)), 4)
        o["DeltaDelta_mean_over_runs"] = round(float(np.mean(dc) - np.mean(do_)), 3)
        o["DeltaDelta_paired_cluster_bootstrap_CI"] = [round(float(np.percentile(dD, 2.5)), 3), round(float(np.percentile(dD, 97.5)), 3)]
        return o
    d3["all474_claimnov_runs_0_1_2_vs_coi_paper_run"] = core((0,))
    if COI[1]:
        d3["coi_runs_0_1_on_283"] = core((0, 1))
        if COI[2]: d3["coi_runs_0_1_2_on_run2_subset"] = core((0, 1, 2))
        d3["coi_check_prompt_is_papers"] = sum(bool(x.get("check_prompt_is_papers")) for r in COI for x in COI[r].values())
        d3["coi_identical_text_to_paper"] = {r: sum(bool(x.get("identical_to_paper_text")) for x in COI[r].values()) for r in COI}
        d3["coi_flips_vs_paper"] = {r: sum(x["rej"] != runs_coi(0)[(x["set"], x["key"])] for x in COI[r].values()) for r in COI}
        if COI[1] and COI[2]:
            kk = set(COI[1]) & set(COI[2]); d3["coi_flips_run1_vs_run2"] = sum(COI[1][k]["rej"] != COI[2][k]["rej"] for k in kk)
    res["D3"] = d3
dump("r3_summary.json", res)
print(json.dumps(res, indent=1, default=str)[:30000])
