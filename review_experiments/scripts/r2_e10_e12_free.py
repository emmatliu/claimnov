import json, glob, collections
from r0_common import *

I = json.load(open(f"{OUT}/r0_items.json"))["items"]
L = lambda fp: [json.loads(l) for l in open(fp)] if os.path.exists(fp) else []
res = {}
E2 = {r["key"]: r for r in L(f"{OUT}/e2_relabel.jsonl")}
agree = sorted(k for k, r in E2.items() if r.get("pool_parse_ok") and r.get("F_parse_ok") and r.get("pool_contained") is False and r.get("F_realized") is True)
notpool = sorted(k for k, r in E2.items() if r.get("pool_parse_ok") and r.get("pool_contained") is False)
lab = sorted(k for k, r in E2.items() if r.get("pool_parse_ok") and r.get("F_parse_ok"))
checks = {g: {k: bool(v["rej"]) for k, v in I[g]["later"].items()} for g in I if "later" in I[g]}
ON = {r["key"]: r for r in L(f"{OUT}/r1_opennovelty.jsonl") if r["set"] == "later" and not r.get("error")}
checks["opennovelty_all"] = {k: bool(r["reject_all"]) for k, r in ON.items()}
checks["opennovelty_any"] = {k: bool(r["reject_any"]) for k, r in ON.items()}
for K in (8, 20):
    E1 = {r["key"]: r for r in L(f"{OUT}/e1_judge_only_k{K}.jsonl") if r["set"] == "later"}
    checks[f"judge_only_top{K}"] = {k: not r["novel"] for k, r in E1.items()}
for fp in glob.glob(f"{OUT}/r1_nonclaude_judgeonly_*.jsonl"):
    nm = os.path.basename(fp)[len("r1_nonclaude_judgeonly_"):-6]
    checks[f"judge_only_top8_{nm}"] = {r["key"]: bool(r["rej"]) for r in L(fp) if r["set"] == "later"}
t = {}
for g, d in checks.items():
    row = {}
    for nm, ks in (("all400", sorted(d)), ("gpt5_labeled", lab), ("gpt5_not_in_pool", notpool), ("opus_and_gpt5_agree", agree)):
        ks = [k for k in ks if k in d]
        row[nm] = fmt_ci(sum(d[k] for k in ks), len(ks)) if ks else None
    t[g] = row
res["E10i"] = {"n_labeled": len(lab), "n_not_in_pool": len(notpool), "n_agree": len(agree), "FA": t,
               "mcnemar_claimnov_vs_coi_on_agree": mcnemar_exact({k: checks["claimnov"][k] for k in agree}, {k: checks["coi"][k] for k in agree})}

LATER = set(later_set())
A = {}
for root in ("anchors", "anchors_full"):
    for adir in glob.glob(f"{PILOT}/data/{root}/*"):
        slug = os.path.basename(adir)
        if slug == "cut" or not os.path.exists(f"{adir}/ideas.json"): continue
        sf = f"{adir}/sorted_full.json" if os.path.exists(f"{adir}/sorted_full.json") else f"{adir}/sorted.json"
        if not os.path.exists(sf): continue
        for s in json.load(open(sf)):
            if s["bucket"] == "a": A[f"{slug}::{s['idx']}"] = s["match"]
pool_titles = {}
def pool_of(slug):
    if slug not in pool_titles: pool_titles[slug] = json.load(open(f"{anchor_dir(slug)}/pool_preT.json"))[:400]
    return pool_titles[slug]
nt = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())[:60]
match_in_pool = {k: any(nt(p["title"]) == nt(m) for p in pool_of(k.split("::")[0])) for k, m in A.items()}
mid = {k: next((p["id"] for p in pool_of(k.split("::")[0]) if nt(p["title"]) == nt(m)), None) for k, m in A.items()}
e12 = {"n_bucket_a": len(A), "n_in_later_realized_set": len(set(A) & LATER), "matched_paper_in_pool400": sum(match_in_pool.values()),
       "anchors": len({k.split('::')[0] for k in A}), "pool_size_median": float(np.median([len(pool_of(k.split('::')[0])) for k in A]))}
subsets = {"all837": sorted(A), "excl_later_realized": sorted(set(A) - LATER), "overlap_with_later": sorted(set(A) & LATER)}
fa_cn = {k: v["rej"] for k, v in I["claimnov"]["later"].items()}; fa_k10 = {k: v["rej"] for k, v in I["claimnov_k10"]["later"].items()}
rows = {}
for name, fn, lat in (("ClaimNov shipped (k-scaled), coupled_veto_ac_profile_k", "coupled_veto_ac_profile_k", fa_cn),
                      ("ClaimNov fixed k=10 (Table-1 'fixed k=10' row), coupled_veto_ac_profile", "coupled_veto_ac_profile", fa_k10)):
    R = {}
    for r in jl(fn):
        if r["item"] in A and r.get("novel") is not None and r["item"] not in R: R[r["item"]] = r
    o = {"later_FA": f"{sum(lat.values())}/{len(lat)}"}
    for sn, ks in subsets.items():
        ks2 = [k for k in ks if k in R]
        rej = sum(not R[k]["novel"] for k in ks2)
        named = sum((not R[k]["novel"]) and nt(R[k].get("evidence")) == nt(A[k]) for k in ks2)
        inprof = sum(mid[k] in (R[k].get("candidates") or []) for k in ks2)
        o[sn] = {"n": len(ks2), "caught": fmt_ci(rej, len(ks2)) if ks2 else None, "named_matched_paper": named,
                 "matched_paper_in_profile": inprof,
                 "J_vs_later400": round(rej / len(ks2) - sum(lat.values()) / len(lat), 4) if ks2 else None}
    rows[name] = o
R1 = {r["item"]: r for r in jl("coupled_veto_ac_profile_k") if r["item"] in A}
R2 = {r["item"]: r for r in jl("coupled_veto_ac_profile") if r["item"] in A}
com = sorted((set(R1) & set(R2)) - LATER)
e12["paired_shipped_vs_k10_on_shared_excl_later"] = {"n": len(com), "shipped_caught": sum(not R1[k]["novel"] for k in com),
                                                       "k10_caught": sum(not R2[k]["novel"] for k in com),
                                                       "mcnemar": mcnemar_exact({k: not R1[k]["novel"] for k in com}, {k: not R2[k]["novel"] for k in com})}
for g in ("v1", "coi", "si", "ra", "scholareval", "v2_reflection", "arc"):
    d = I[g]["later"]
    ks = [k for k in subsets["overlap_with_later"] if k in d]
    rows[f"{g} (Sonnet rows exist only on the later-set overlap)"] = {"overlap_with_later": {"n": len(ks), "caught": sum(d[k]["rej"] for k in ks)}}
ks = subsets["overlap_with_later"]
rows["ClaimNov paper verdict on the overlap (Table-1 FA rows)"] = {"overlap_with_later": {"n": len(ks), "caught": sum(fa_cn[k] for k in ks)}}
rows["OpenNovelty-all on the overlap"] = {"overlap_with_later": {"n": len(ks), "caught": sum(ON[k]["reject_all"] for k in ks if k in ON)}}
VS = {}
for r in jl("positive_full"):
    if r["gate"] == "veto_sonnet" and r["item"] in A and r.get("novel") is not None and r["item"] not in VS: VS[r["item"]] = r
vlat = {}
for r in jl("positive_full"):
    if r["gate"] == "veto_sonnet" and r["item"] in LATER and r.get("novel") is not None and r["item"] not in vlat: vlat[r["item"]] = not r["novel"]
o = {"later_FA": f"{sum(vlat.values())}/{len(vlat)}", "note": "same MATCH_PROMPT as the Opus sorter that defines bucket (a): circular, reference only"}
for sn, ks in subsets.items():
    ks2 = [k for k in ks if k in VS]; rej = sum(not VS[k]["novel"] for k in ks2)
    o[sn] = {"n": len(ks2), "caught": fmt_ci(rej, len(ks2)) if ks2 else None,
             "named_matched_paper": sum((not VS[k]["novel"]) and nt(VS[k].get("evidence"))[:50] == nt(A[k])[:50] for k in ks2)}
rows["Sonnet MATCH veto over top-8 (veto_sonnet, positive_full)"] = o
e12["rows"] = rows
res["E12"] = e12
dump("r2_e10_e12_free.json", res)
print(json.dumps(res, indent=1))
