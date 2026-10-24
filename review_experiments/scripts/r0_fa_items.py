import json, os, random
from r0_common import *

D = json.load(open(f"{OUT}/r0_items.json")); I = D["items"]
LAB = {r["item"]: r for r in jl("claim_sort_full_indep")}
ideas_cache = {}
def idea(item):
    a, idx = item.split("::")
    if a not in ideas_cache: ideas_cache[a] = {x["idx"]: x for x in json.load(open(f"{anchor_dir(a)}/ideas.json"))}
    return ideas_cache[a][int(idx)]
pool_cache = {}
def pool_title(a, title):
    if a not in pool_cache: pool_cache[a] = json.load(open(f"{anchor_dir(a)}/pool_preT.json"))[:400]
    return any(norm(p["title"]) == norm(title) for p in pool_cache[a]) if title else None

out = {"note": "later-realized false alarms (rejections) on the 400-idea set; 'opus_labels' = claim_sort_full_indep.jsonl (Opus 4.6, independent claims; pre = 8 nearest pre-T pool papers, post = 8 nearest post-T follow-ups)", "claimnov": [], "coi": []}
for c in ("claimnov", "coi"):
    for k, v in sorted(I[c]["later"].items()):
        if not v["rej"]: continue
        it = idea(k); lab = LAB.get(k, {})
        rec = {"item": k, "anchor": anchor_of(k), "title": it["title"], "description": it["description"],
               "named_paper": v.get("evidence"), "named_paper_in_preT_pool400": pool_title(anchor_of(k), v.get("evidence")),
               "sorter_bucket": lab.get("bucket"),
               "opus_labels": {"claims": lab.get("claims"), "pre": lab.get("pre"), "post": lab.get("post")}}
        if c == "claimnov":
            rec["claimnov_justification"] = v.get("justification"); rec["claimnov_established_claims"] = v.get("evidence_levels")
        rec["also_rejected_by"] = sorted(g for g in ("claimnov", "coi", "v1", "si", "ra", "v2_reflection", "graded", "claimnov_k10", "central")
                                         if g != c and I[g]["later"][k]["rej"])
        out[c].append(rec)
out["overlap_claimnov_coi"] = sorted({r["item"] for r in out["claimnov"]} & {r["item"] for r in out["coi"]})
dump("fa_items.json", out)
print(len(out["claimnov"]), len(out["coi"]), out["overlap_claimnov_coi"])

S = set(later_set()); ov = {}
pa = sorted(os.listdir(f"{PILOT}/data/anchors"))
ov["pilot_anchors"] = pa
ov["later_set_items_from_pilot_anchors"] = {a: sum(1 for s in S if anchor_of(s) == a) for a in pa}
abuck = {}
for a in pa:
    for s in json.load(open(f"{PILOT}/data/anchors/{a}/sorted.json")):
        abuck[f"{a}::{s['idx']}"] = s["bucket"]
ov["pilot_bucket_counts"] = collections.Counter(abuck.values())
ov["pilot_a106_in_later_set"] = sorted(k for k, b in abuck.items() if b == "a" and k in S)
ov["pilot_b139_in_later_set"] = sum(1 for k, b in abuck.items() if b == "b" and k in S)
dec = []
for a in pa:
    bs = [s for s in json.load(open(f"{PILOT}/data/anchors/{a}/sorted.json")) if s["bucket"] == "b"]
    rng = random.Random(0); rng.shuffle(bs); dec += [f"{a}::{s['idx']}" for s in bs[:15]]
ov["pilot_32_decoys(run_panel selection)"] = {"n": len(dec), "in_later_set": sum(d in S for d in dec)}
n2 = json.load(open(f"{PILOT}/data/n2_pairs.json"))
pairs = [json.loads(l) for l in open("eval/data/claim_disputed/testset_clean.jsonl")]
cl = set(I["claimnov"]["claim"])
aid = {pairs[int(k.split("_")[1])]["A_id"]: k for k in cl}
ov["pilot_20_n2_pairs_in_claim160"] = sorted(aid[x["A_id"]] for x in n2 if x["A_id"] in aid)
ov["pilot_disguise_sources_in_table1"] = "seeds 0-9 (data/disguises_v2.json, identical texts in disguises_v3.json) = 40 of 400 disguises; smuggle_ladder / pilot veto calibration used them"
po = {}
for c in ("claimnov", "coi", "v1", "si", "graded", "central"):
    lat = I[c]["later"]; pil = [k for k in lat if anchor_of(k) in pa]; rest = [k for k in lat if anchor_of(k) not in pa]
    cla = I[c]["claim"]; p20 = ov["pilot_20_n2_pairs_in_claim160"]; r140 = [k for k in cla if k not in p20]
    po[c] = {"later_FA_pilot_anchors": f"{sum(lat[k]['rej'] for k in pil)}/{len(pil)}", "later_FA_other_anchors": f"{sum(lat[k]['rej'] for k in rest)}/{len(rest)}",
             "claim_rej_pilot20": f"{sum(cla[k]['rej'] for k in p20)}/{len(p20)}", "claim_rej_other140": f"{sum(cla[k]['rej'] for k in r140)}/{len(r140)}",
             "claim_namesB_pilot20": f"{sum(bool(cla[k].get('named')) for k in p20)}/{len(p20)}", "claim_namesB_other140": f"{sum(bool(cla[k].get('named')) for k in r140)}/{len(r140)}"}
ov["split_results"] = po
dump("r0_pilot_overlap.json", ov)
print(json.dumps(ov, indent=1, default=str)[:3000])
