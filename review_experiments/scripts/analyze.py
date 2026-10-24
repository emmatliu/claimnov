import json, os, statistics as st, collections
import common as C, oldrows as O
OUTD = C.OUTD
L = lambda f: C.load_jsonl(f"{OUTD}/{f}")
S = {}
CN = O.claimnov(); BL = O.baselines()
units = {(u["set"], u["key"]): u for u in C.disguise_units() + C.later_units() + C.claim_units() + C.hybrid_units()}
NAMES = ["v1", "coi", "si", "arc", "ra", "scholareval", "v2_reflection"]


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.1f}%)" if b else f"{a}/0"


def claimset(d):
    ks = [k for k in d if k[0] == "claim"]
    out = {"n": len(ks)}
    for dec in ("accept", "reject", None):
        kk = [k for k in ks if units[k]["meta"]["decision"] == dec]
        out[f"rej_{dec}"] = (sum(not d[k]["novel"] for k in kk), len(kk))
    out["pass"] = sum(d[k]["novel"] for k in ks)
    out["names_B"] = sum(bool(d[k]["named"]) for k in ks)
    return out


def hybset(d):
    ks = [k for k in d if k[0] == "hyb"]
    return {"n": len(ks), "pass": sum(d[k]["novel"] for k in ks), "names_B": sum(bool(d[k]["named"]) for k in ks),
            "names_B_and_pass": sum(bool(d[k]["named"]) and d[k]["novel"] for k in ks)}


def rungs(d):
    r = collections.Counter()
    for k, v in d.items():
        if k[0] == "disg" and k[1].endswith("B_in") and not v["novel"]:
            r[units[k]["meta"]["rung"]] += 1
    return [r[i] for i in (1, 2, 3, 4)]


for K in (8, 20):
    rows = L(f"e1_judge_only_k{K}.jsonl")
    if not rows:
        continue
    d = {(r["set"], r["key"]): {"novel": r["novel"], "named": (r["names_B"] if (r["set"] != "disg" or not r["novel"]) else False),
                                "names_B_any": r["names_B"], "B_in_shown": r["B_in_shown"], "parse_error": r["parse_error"]}
         for r in rows}
    e = {"n_rows": len(rows), "parse_errors": sum(r["parse_error"] for r in rows),
         "whole": O.whole(d), "rungs_caught": rungs(d), "claim": claimset(d), "hyb": hybset(d)}
    e["claim"]["B_in_shown"] = sum(v["B_in_shown"] for k, v in d.items() if k[0] == "claim")
    e["hyb"]["B_in_shown"] = sum(v["B_in_shown"] for k, v in d.items() if k[0] == "hyb")
    e["disg_B_in_shown_srcin"] = sum(v["B_in_shown"] for k, v in d.items() if k[0] == "disg" and k[1].endswith("B_in"))
    agree = collections.Counter()
    for k, v in d.items():
        if k in CN:
            agree[(k[0], CN[k]["novel"], v["novel"])] += 1
    e["vs_claimnov"] = {f"{s}|claimnov_{'pass' if a else 'rej'}|jo_{'pass' if b else 'rej'}": n for (s, a, b), n in sorted(agree.items(), key=str)}
    S[f"E1_k{K}"] = e
S["paper_claimnov"] = {"whole": O.whole(CN), "rungs_caught": rungs(CN), "claim": claimset(CN), "hyb": hybset(CN)}

rows = L("e3_claimnov_runs.jsonl")
if rows:
    e = {"n_items": len(rows), "prompt_source": dict(collections.Counter(str(r["prompt_matches_paper"]) for r in rows)),
         "backend": dict(collections.Counter(str(r.get("backend", "api")) for r in rows))}
    mism = [r["key"] for r in rows if "run0" in r and (r["set"], r["key"]) in CN and r["run0"]["novel"] != CN[(r["set"], r["key"])]["novel"]]
    e["run0_cache_vs_paper_rows_mismatch"] = len(mism)
    for r in rows:
        c = CN[(r["set"], r["key"])]
        r["run0"] = {"novel": c["novel"], "names_B": c["named"]}
    runs = ["run0", "run1", "run2"]
    per = {}
    for rn in runs:
        d = {(r["set"], r["key"]): {"novel": r[rn]["novel"], "named": (not r[rn]["novel"]) and r[rn]["names_B"]} for r in rows}
        per[rn] = O.whole(d)
    e["per_run_on_completed_items"] = per
    stat = {}
    for m in ("caught", "src_out", "fa", "named"):
        vals = [per[rn][m] for rn in runs]
        stat[m] = {"mean": round(st.mean(vals), 3), "sd": round(st.stdev(vals), 3), "values": vals}
    for m in ("J", "Delta"):
        vals = [per[rn].get(m) for rn in runs]
        if None not in vals:
            stat[m] = {"mean": round(st.mean(vals), 4), "sd": round(st.stdev(vals), 4), "values": vals}
    e["mean_sd"] = stat
    flips = {}
    for s_, cond in (("disg", "B_in"), ("disg", "B_out"), ("later", "set")):
        rr = [r for r in rows if r["set"] == s_ and r["cond"] == cond]
        f = {"n": len(rr)}
        for i, a in enumerate(runs):
            for b in runs[i + 1:]:
                f[f"{a}_vs_{b}"] = sum(r[a]["novel"] != r[b]["novel"] for r in rr)
        f["any_flip"] = sum(len({r[rn]["novel"] for rn in runs}) > 1 for r in rr)
        f["rejected_in_any_run"] = sum(any(not r[rn]["novel"] for rn in runs) for r in rr)
        flips[f"{s_}_{cond}"] = f
    e["flips"] = flips
    S["E3_claimnov"] = e
    m = [r for r in rows if r["prompt_matches_paper"]]
    pm = {}
    for rn in runs:
        pm[rn] = O.whole({(r["set"], r["key"]): {"novel": r[rn]["novel"], "named": (not r[rn]["novel"]) and r[rn]["names_B"]} for r in m})
    fm = {}
    for s_, cond in (("disg", "B_in"), ("disg", "B_out"), ("later", "set")):
        rr = [r for r in m if r["set"] == s_ and r["cond"] == cond]
        fm[f"{s_}_{cond}"] = {"n": len(rr), **{f"{a}_vs_{b}": sum(r[a]["novel"] != r[b]["novel"] for r in rr)
                                                for i, a in enumerate(runs) for b in runs[i + 1:]}}
    e["byte_identical_subset"] = {"n": len(m), "per_run": pm, "flips": fm,
                                  "flipped_keys": [r["key"] for r in m if len({r[rn]["novel"] for rn in runs}) > 1]}
    um = [r for r in rows if not r["prompt_matches_paper"]]
    e["rebuilt_profile_differs_subset"] = {"n": len(um), "flipped_run0_vs_run1": sum(r["run0"]["novel"] != r["run1"]["novel"] for r in um),
                                           "flipped_run1_vs_run2": sum(r["run1"]["novel"] != r["run2"]["novel"] for r in um)}
rows = L("e3_coi_rerun.jsonl")
if rows:
    d1 = {(r["set"], r["key"]): {"novel": r["novel"], "named": r["names_B"]} for r in rows if r["novel"] is not None}
    d0 = {k: BL["coi"][k] for k in d1 if k in BL["coi"]}
    e = {"n": len(rows), "errors": sum(r["novel"] is None for r in rows), "run0_paper": O.whole(d0), "run1": O.whole(d1)}
    fl = {}
    for s, cond in (("disg", "B_in"), ("disg", "B_out"), ("later", "set")):
        ks = [k for k in d1 if k[0] == s and units[k]["cond"] == cond and k in d0]
        fl[f"{s}_{cond}"] = {"n": len(ks), "flips": sum(d1[k]["novel"] != d0[k]["novel"] for k in ks)}
    e["flips"] = fl
    S["E3_coi"] = e

rows = L("e2_relabel.jsonl")
if rows:
    ok = [r for r in rows if r["pool_contained"] is not None and r["F_realized"] is not None]
    n = len(ok)
    notant = {r["key"] for r in ok if r["pool_contained"] is False}
    real = {r["key"] for r in ok if r["F_realized"] is True}
    both = notant & real
    e = {"n": len(rows), "n_parsed_both": n, "not_anticipated_by_pool": pct(len(notant), n),
         "realized_in_F": pct(len(real), n), "confirmed_both": pct(len(both), n)}
    checks = {"claimnov": CN, **{g: BL[g] for g in NAMES if g in BL}}
    e1 = {("later", r["key"]): {"novel": r["novel"]} for r in L("e1_judge_only_k8.jsonl") if r["set"] == "later"}
    if e1:
        checks["judge_only_k8"] = e1
    e1b = {("later", r["key"]): {"novel": r["novel"]} for r in L("e1_judge_only_k20.jsonl") if r["set"] == "later"}
    if e1b:
        checks["judge_only_k20"] = e1b
    fa = {}
    allk = {r["key"] for r in ok}
    for g, d in checks.items():
        def f(sub):
            ks = [k for k in sub if ("later", k) in d]
            return pct(sum(not d[("later", k)]["novel"] for k in ks), len(ks))
        if not any(("later", k) in d for k in allk):
            continue
        fa[g] = {"all_parsed": f(allk), "confirmed_not_anticipated": f(notant), "confirmed_both": f(both)}
        rej = [k for k in allk if ("later", k) in d and not d[("later", k)]["novel"]]
        fa[g]["gpt5_says_anticipated_on_its_rejections"] = pct(sum(k not in notant for k in rej), len(rej))
    e["FA"] = fa
    anyrej = {k for k in allk if any(("later", k) in d and not d[("later", k)]["novel"] for g, d in checks.items() if g in ("claimnov", "v1", "coi", "si", "ra", "scholareval", "v2_reflection"))}
    e["items_rejected_by_any_paper_check"] = len(anyrej)
    e["gpt5_anticipated_among_any_rejected"] = pct(sum(k not in notant for k in anyrej), len(anyrej))
    e["gpt5_anticipated_among_never_rejected"] = pct(sum(k not in notant for k in allk - anyrej), len(allk - anyrej))
    cnrej = [k for k in allk if ("later", k) in CN and not CN[("later", k)]["novel"]]
    e["claimnov_FA_items"] = [{"key": k, "gpt5_pool_contained": next(r["pool_contained"] for r in ok if r["key"] == k),
                               "gpt5_F_realized": next(r["F_realized"] for r in ok if r["key"] == k)} for k in sorted(cnrej)]
    S["E2"] = e

H = {u["key"]: u for u in C.hybrid_units()}
e4 = {}
checks = {"claimnov": CN, **{g: BL[g] for g in NAMES if g in BL}}
e1 = {(r["set"], r["key"]): {"novel": r["novel"]} for r in L("e1_judge_only_k8.jsonl")}
if e1:
    checks["judge_only_k8"] = e1
for g, d in checks.items():
    base_in = [k for k in H if ("later", k) in d and ("hyb", k) in d]
    if not base_in:
        continue
    bp = [k for k in base_in if d[("later", k)]["novel"]]
    e4[g] = {"n_with_base": len(base_in), "base_pass": len(bp), "hyb_pass_given_base_pass": pct(sum(d[("hyb", k)]["novel"] for k in bp), len(bp)),
             "hyb_pass_overall": pct(sum(d[("hyb", k)]["novel"] for k in base_in), len(base_in))}
S["E4"] = e4
json.dump(S, open(f"{OUTD}/round0_summary.json", "w"), indent=1, default=str)
print(json.dumps(S, indent=1, default=str))
