import ast, json, hashlib
import numpy as np
from r0_common import *

D = json.load(open(f"{OUT}/r0_items.json")); I = D["items"]; M = D["meta"]
PROF = json.load(open(f"{OUT}/r0_claim_profiles.json"))["profiles"]
ncache = json.load(open(NCACHE))
res = {}

def _key(*parts): return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:16]
def claims_of(t, a): return ncache.get("claims4::" + _key(t, a))
def ctx(f):
    parts = [f"Facet — attribute: {f.get('attribute', '')}; value: {f.get('value', '')}"]
    if f.get("paper_abstract"): parts.append(f"Abstract: {f['paper_abstract'][:1200]}")
    return "\n".join(parts)
def level(t, c):
    return ncache.get("entail::" + hashlib.md5((t["claim"] + "||" + c["claim"] + "||" + ctx(t) + "||" + ctx(c)).encode()).hexdigest()[:16])

def cos_rank(qtext, pool):
    q = cached_embed(qtext)
    if q is None: return None
    q = np.array(q); out = []
    for p in pool:
        v = cached_embed(f"{p['title']}. {p['abstract']}")
        if v is None: return None
        v = np.array(v); out.append((float(q @ v / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-9)), p["id"]))
    return [pid for _, pid in sorted(out, key=lambda x: -x[0])]

src = open(f"{PILOT}/make_disguises.py").read()
R1 = next(ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "R1")
pairs = [json.loads(l) for l in open("eval/data/claim_disputed/testset_clean.jsonl")]

cn = I["claimnov"]["claim"]; C = sorted(cn)
cl = {}
for k in C:
    p = PROF[k]; B = p["B_id"]; cand = cn[k]["candidates"]
    pool = json.load(open(f"{PILOT}/data/pools_full/{k}.json"))["pool"]
    cl_levels = {}
    for r in p["claims"]:
        for pid, lv in r["levels"].items():
            if pid in cand: cl_levels[pid] = max(cl_levels.get(pid, 0), lv)
    bl = cl_levels.get(B)
    top = max(cl_levels.values()) if cl_levels else None
    i = int(k.split("_")[1])
    d = json.loads(re.search(r"\{.*\}", cached_bedrock(SONNET_46_BEDROCK, [{"role": "user", "content": R1.format(abstract=pairs[i]["A_abstract"])}], 3000, f"n2f_ideaify{i}"), re.S).group(0))
    rank = cos_rank(f"{d['title']}: {d['description']}", pool)
    m = len(cand)
    cl[k] = {"a_B_in_profile": B in cand, "b_lvl3": B in cand and (bl or 0) >= 3, "c_lvl4": B in cand and (bl or 0) >= 4,
             "d_verdict_names_B": bool(cn[k]["verdict_names_B"]),
             "e_B_top_tied": B in cand and bl is not None and bl == top,
             "e_B_top_unique": B in cand and bl is not None and bl == top and sum(1 for v in cl_levels.values() if v == top) == 1,
             "e_B_first_in_profile_order": bool(cand) and cand[0] == B,
             "m_profile_papers": m, "pool_size": len(pool), "chance": m / len(pool),
             "cos_rank_B": (rank.index(B) + 1) if rank and B in rank else None,
             "cos_topm_hit": bool(rank) and B in rank[:m], "cos_top5": bool(rank) and B in rank[:5], "cos_top8": bool(rank) and B in rank[:8],
             "B_cached_any_samedim": any(r["B_level"] is not None for r in p["claims"]),
             "B_max_cached_level": max([r["B_level"] for r in p["claims"] if r["B_level"] is not None] or [0]),
             "B_level_shown": bl, "rej": cn[k]["rej"]}
def summ(rows, keys):
    n = len(rows)
    return {key: fmt_ci(sum(bool(r[key]) for r in rows), n) for key in keys}
KEYS = ["a_B_in_profile", "b_lvl3", "c_lvl4", "d_verdict_names_B", "e_B_top_tied", "e_B_top_unique", "e_B_first_in_profile_order"]
rows = list(cl.values())
res["claim160"] = {"claimnov_variants": summ(rows, KEYS),
                   "claimnov_variants_among_passes": summ([r for r in rows if not r["rej"]], KEYS),
                   "paper_reported": {"B_in_profile": "127/160 (79%)", "lvl>=3": "54/160 (34%)", "verdict_names_B": "2/160"},
                   "upper_bound_B_judged_any_run_same_dim": fmt_ci(sum(r["B_cached_any_samedim"] for r in rows), len(rows)),
                   "upper_bound_B_judged_any_run_lvl>=3": fmt_ci(sum(r["B_max_cached_level"] >= 3 for r in rows), len(rows)),
                   "chance_B_in_random_m_subset": {"expected_count": round(sum(r["chance"] for r in rows), 2),
                                                   "expected_rate": round(float(np.mean([r["chance"] for r in rows])), 4),
                                                   "mean_m": round(float(np.mean([r["m_profile_papers"] for r in rows])), 2),
                                                   "m_dist": sorted(collections.Counter(r["m_profile_papers"] for r in rows).items()),
                                                   "pool_size_dist": sorted(collections.Counter(r["pool_size"] for r in rows).items())},
                   "cosine_baseline": {"top_m_matched": fmt_ci(sum(r["cos_topm_hit"] for r in rows), len(rows)),
                                       "top5": fmt_ci(sum(r["cos_top5"] for r in rows), len(rows)),
                                       "top8": fmt_ci(sum(r["cos_top8"] for r in rows), len(rows)),
                                       "median_cos_rank_of_B": float(np.median([r["cos_rank_B"] for r in rows if r["cos_rank_B"]])),
                                       "n_ranked": sum(1 for r in rows if r["cos_rank_B"])},
                   "mcnemar_profile_vs_cosine_topm": mcnemar_exact({k: v["a_B_in_profile"] for k, v in cl.items()}, {k: v["cos_topm_hit"] for k, v in cl.items()})}
len_b = {}
for c in ["v1", "coi", "si"]:
    xs = I[c]["claim"]
    len_b[c] = {"names_B_on_reject (paper metric)": fmt_ci(sum(v["named"] for v in xs.values()), 160),
                "B_in_returned_evidence_any_verdict": fmt_ci(sum(v["seen_B"] for v in xs.values()), 160),
                "B_in_evidence_on_pass": sum(1 for v in xs.values() if v["seen_B"] and not v["rej"])}
len_b["note"] = "v1 evidence = titles of the top-3 results of its LAST search (recorded on pass and fail); CoI/Si evidence exists only on rejection, so lenient == strict for them. What v1/CoI/Si saw in all rounds is not stored (trace stripped)."
res["claim160"]["baselines_lenient"] = len_b

H = json.load(open(f"{PILOT}/data/hybrids.json")); hy = I["claimnov"]["hyb"]
pools = {}
hrows = {}; recon_check = collections.Counter()
for h in H:
    a = h["anchor"]
    if a not in pools: pools[a] = json.load(open(f"{anchor_dir(a)}/pool_preT.json"))[:400]
    pool = pools[a]; B = h["B_id"]
    tcl = claims_of(h["title"], h["description"])
    facets = []
    for p in pool:
        for c in (claims_of(p["title"], p["abstract"]) or []):
            facets.append({**c, "paper_id": p["id"], "paper_abstract": p["abstract"]})
    shown = {}; shown_tie = set(); judged_any = set()
    for c in (tcl or []):
        t = {**c, "paper_abstract": h["description"]}
        best = {}
        for f in facets:
            if f["dimension"] != c["dimension"]: continue
            lv = level(t, f)
            if lv is not None: best[f["paper_id"]] = max(best.get(f["paper_id"], 0), lv)
        judged_any |= set(best)
        srt = sorted(best.items(), key=lambda x: -x[1])
        if srt:
            th = srt[min(5, len(srt)) - 1][1]
            for pid, lv in srt[:5]: shown[pid] = max(shown.get(pid, 0), lv)
            for pid, lv in srt:
                if lv >= th: shown_tie.add(pid)
    r = hy[h["item"]]
    bl_row = r["B_level"]
    recon_check["named_row_vs_tieinclusive_agree"] += (B in shown_tie) == bool(r["named"]) or (not r["named"] and B not in shown_tie)
    recon_check["row_named_but_B_not_judged"] += bool(r["named"]) and B not in judged_any
    recon_check["n_claims_match"] += (len(tcl or []) == r["n_claims"])
    top = max(shown.values()) if shown else None
    rank = cos_rank(f"{h['title']}: {h['description']}", pool)
    m = len(shown)
    hrows[h["item"]] = {"a_B_in_profile": bool(r["named"]), "b_lvl3": bool(r["named"]) and bl_row >= 3, "c_lvl4": bool(r["named"]) and bl_row >= 4,
                        "d_verdict_names_B": bool(r["verdict_names_B"]),
                        "e_B_top_tied": bool(r["named"]) and top is not None and bl_row >= top,
                        "e_B_top_unique": bool(r["named"]) and top is not None and bl_row >= top and sum(1 for v in shown.values() if v == top) == 1,
                        "B_on_spliced_claim": bool(r["B_on_spliced_claim"]), "rej": r["rej"],
                        "m_profile_papers_est": m, "m_tie_inclusive": len(shown_tie), "pool_size": len(pool), "chance": m / len(pool),
                        "cos_rank_B": (rank.index(B) + 1) if rank and B in rank else None,
                        "cos_topm_hit": bool(rank) and B in rank[:m], "cos_top5": bool(rank) and B in rank[:5], "cos_top10": bool(rank) and B in rank[:10]}
hr = list(hrows.values())
KH = ["a_B_in_profile", "b_lvl3", "c_lvl4", "d_verdict_names_B", "e_B_top_tied", "e_B_top_unique", "B_on_spliced_claim"]
res["hyb100"] = {"claimnov_variants": summ(hr, KH), "claimnov_variants_among_passes": summ([r for r in hr if not r["rej"]], KH),
                 "B_level_dist_when_named": sorted(collections.Counter(hy[k]["B_level"] for k in hy if hy[k]["named"]).items()),
                 "reconstruction_check": recon_check,
                 "chance_B_in_random_m_subset": {"expected_count": round(sum(r["chance"] for r in hr), 2),
                                                 "mean_m_est": round(float(np.mean([r["m_profile_papers_est"] for r in hr])), 2),
                                                 "expected_count_tie_inclusive_m": round(sum(r["m_tie_inclusive"] / r["pool_size"] for r in hr), 2),
                                                 "pool_size_dist": sorted(collections.Counter(r["pool_size"] for r in hr).items())},
                 "cosine_baseline": {"top_m_matched": fmt_ci(sum(r["cos_topm_hit"] for r in hr), 100), "top5": fmt_ci(sum(r["cos_top5"] for r in hr), 100),
                                     "top10": fmt_ci(sum(r["cos_top10"] for r in hr), 100),
                                     "median_cos_rank_of_B": float(np.median([r["cos_rank_B"] for r in hr if r["cos_rank_B"]])),
                                     "note": "B was chosen as a pool paper ranked 5-15 by cosine to the ORIGINAL idea (build_hybrids); the hybrid text then contains B's contribution"},
                 "mcnemar_profile_vs_cosine_topm": mcnemar_exact({k: v["a_B_in_profile"] for k, v in hrows.items()}, {k: v["cos_topm_hit"] for k, v in hrows.items()})}
lb = {}
for c in ["v1", "coi", "si"]:
    xs = I[c]["hyb"]
    lb[c] = {"names_B_on_reject (paper metric)": sum(v["named"] for v in xs.values()), "B_in_returned_evidence_any_verdict": sum(v["seen_B"] for v in xs.values()),
             "B_in_evidence_on_pass": sum(1 for v in xs.values() if v["seen_B"] and not v["rej"])}
res["hyb100"]["baselines_lenient"] = lb
res["hyb100"]["note"] = ("(a),(b),(c),(d) and on_spliced come straight from hybrids_eval rows (B in top-5 judged of some claim; B_level = max level of B there). "
                         "(e) and m use a cache reconstruction of the judged lists (src/.cache entail:: keys; only run_hybrids judged these claims, so cached pairs = judged pairs); "
                         "ties at the 5th slot are broken arbitrarily since retrieval cosines (3-large) are not cached.")
dump("r0_attribution.json", {**res, "per_item_claim": cl, "per_item_hyb": hrows})
print(json.dumps(res, indent=1, default=str)[:6000])
