import csv, json, collections
from r0_common import *

csv.field_size_limit(10**9)
chosen, dz = disguise_keys()
DIN = sorted(f"seed{d['seed_idx']}_r{d['rung']}_B_in" for d in dz)
DOUT = sorted(f"seed{d['seed_idx']}_r{d['rung']}_B_out" for d in dz)
LATER = later_set()
Bcache = {}
def Btitle(item):
    s = seed_of(item)
    if s not in Bcache:
        Bcache[s] = pool_B(s)[0]
    return Bcache[s]

def ev_names(r, btitle):
    ev = r.get("evidence"); evs = ev if isinstance(ev, list) else [ev]
    return any(e and norm(e) == norm(btitle) for e in evs)

items = collections.defaultdict(lambda: collections.defaultdict(dict))
meta = {"selection": {}}

DEP, DEPby = final_rows(DEPLOYED_FILES, pick="first")
for g in ["v1", "coi", "si", "ra", "scholareval", "v2_reflection"]:
    for k in DIN:
        r = DEP[(k, g)]
        items[g]["disg_in"][k] = {"rej": r["novel"] is False if r["novel"] is not None else None,
                                  "named": (r["novel"] is False) and ev_names(r, Btitle(k)["title"]),
                                  "evidence": r.get("evidence"), "timeout": None}
    for k in DOUT:
        r = DEP[(k, g)]
        items[g]["disg_out"][k] = {"rej": r["novel"] is False if r["novel"] is not None else None,
                                   "evidence": r.get("evidence")}
    for k in LATER:
        r = DEP[(k, g)]
        items[g]["later"][k] = {"rej": r["novel"] is False if r["novel"] is not None else None,
                                "evidence": r.get("evidence")}
meta["selection"]["deployed"] = "deployed_sonnet*.jsonl; first valid (novel!=None) row per (item,gate) in order " + ",".join(DEPLOYED_FILES)

SM, SMby = final_rows(SMUGGLE_FILES, pick="first")
CV_SM = {r["item"]: r for r in jl("coupled_veto_smuggle_profile")}
CC_SM = {}
for r in jl("coupled_veto_smuggle_candcentral"):
    CC_SM.setdefault(r["item"], r)
def sm_row(k, g):
    if (k, g) in SM: return SM[(k, g)]
    if g == "claimnov": return CV_SM[k]
    if g == "central_rule": return CC_SM[k]
    raise KeyError((k, g))
for g, name in [("arc", "arc"), ("claimnov", "claimnov"), ("module", "graded"), ("central_rule", "central"),
                ("claimnov", "claimnov_k10")]:
    for k in DIN:
        r = sm_row(k, g)
        named = r["hit_B"] if ("hit_B" in r and r["novel"] is False) else ((r["novel"] is False) and ev_names(r, Btitle(k)["title"]))
        items[name]["disg_in"][k] = {"rej": r["novel"] is False, "named": bool(named), "evidence": r.get("evidence"),
                                     "B_in_profile": r.get("B_in_profile", (Btitle(k)["id"] in r["candidates"]) if "candidates" in r else None)}
    for k in DOUT:
        r = sm_row(k, g)
        items[name]["disg_out"][k] = {"rej": r["novel"] is False, "evidence": r.get("evidence")}
meta["selection"]["smuggle"] = "smuggle_full+smuggle_v2+smuggle_v2_api first valid row; seeds 0-9 ClaimNov from coupled_veto_smuggle_profile, per-claim rule from coupled_veto_smuggle_candcentral"

def union(files):
    d = {}
    for f in files:
        for r in jl(f):
            d.setdefault(r["item"], r)
    return d
CN_K = union(["coupled_veto_fa_profile_k", "coupled_veto_ac_profile_k", "coupled_veto_set_profile_k"])
CN_10 = union(["coupled_veto_fa_profile", "coupled_veto_ac_profile"])
CC = union(["coupled_veto_fa_candcentral", "coupled_veto_set_candcentral"])
POS = {}
for r in jl("positive_full"):
    POS.setdefault((r["item"], r["gate"]), r)
for k in LATER:
    r = CN_K[k]; items["claimnov"]["later"][k] = {"rej": r["novel"] is False, "evidence": r["evidence"], "candidates": r["candidates"], "justification": r.get("justification"), "evidence_levels": r.get("evidence_levels")}
    r = CN_10[k]; items["claimnov_k10"]["later"][k] = {"rej": r["novel"] is False, "evidence": r["evidence"]}
    r = CC[k]; items["central"]["later"][k] = {"rej": r["novel"] is False, "evidence": r["evidence"]}
    r = POS[(k, "module")]; items["graded"]["later"][k] = {"rej": r["novel"] is False, "evidence": r.get("evidence")}
    r = POS[(k, "arc")]; items["arc"]["later"][k] = {"rej": r["novel"] is False, "evidence": r.get("evidence")}
meta["selection"]["later"] = ("ClaimNov: coupled_veto_{fa,ac,set}_profile_k; fixed k10: coupled_veto_{fa,ac}_profile; "
                              "per-claim rule: coupled_veto_{fa,set}_candcentral; graded gate + ARC: positive_full (module/arc rows)")

P = {r["id"]: r for r in csv.DictReader(open("claim_disputed_mining/iclr_novelty_validation_data/papers.csv"))}
pairs = [json.loads(l) for l in open("eval/data/claim_disputed/testset_clean.jsonl")]
idx = json.load(open(f"{PILOT}/data/n2_full_index.json"))
CLAIM = [f"n2f_{i}" for i in sorted(int(k) for k, v in idx.items() if v["status"] == "ok")]
def decision(item):
    d = P.get(pairs[int(item.split("_")[1])]["A_id"], {}).get("decision_raw")
    return None if d is None else ("rej" if d == "Reject" else "acc")
CS, CSby = final_rows(["claim_sonnet"], pick="first")
for g in ["v1", "coi", "si", "ra", "scholareval", "v2_reflection"]:
    for k in CLAIM:
        r = CS[(k, g)]
        items[g]["claim"][k] = {"rej": r["novel"] is False, "named": (r["novel"] is False) and ev_names(r, r["B_title"]),
                                "seen_B": ev_names(r, r["B_title"]), "evidence": r.get("evidence")}
NF = {}
for r in jl("negative_full"):
    NF.setdefault((r["item"], r["gate"]), r)
NP = union(["coupled_veto_neg_profile"]); NC = union(["coupled_veto_neg_candcentral"])
for k in CLAIM:
    pd = json.load(open(f"{PILOT}/data/pools_full/{k}.json"))["pool"]
    B = next(p for p in pd if p.get("is_B"))
    r = NF[(k, "arc")]; items["arc"]["claim"][k] = {"rej": r["novel"] is False, "named": (r["novel"] is False) and ev_names(r, B["title"])}
    r = NP[k]
    items["claimnov"]["claim"][k] = {"rej": r["novel"] is False, "named": B["id"] in r["candidates"],
                                     "verdict_names_B": (r["novel"] is False) and norm(r["evidence"]) == norm(B["title"]),
                                     "candidates": r["candidates"], "B_id": B["id"], "B_rank": r["B_rank"],
                                     "evidence": r["evidence"], "n_claims": r["n_claims"], "justification": r.get("justification")}
    r = NF[(k, "module")]; items["graded"]["claim"][k] = {"rej": r["novel"] is False, "named": items["claimnov"]["claim"][k]["named"]}
    r = NC[k]; items["central"]["claim"][k] = {"rej": r["novel"] is False, "named": items["claimnov"]["claim"][k]["named"]}
meta["claim_decision"] = {k: decision(k) for k in CLAIM}
meta["claim_dimension"] = {k: pairs[int(k.split("_")[1])]["novelty_dimension"] for k in CLAIM}
meta["claim_relation"] = {k: pairs[int(k.split("_")[1])]["relation"] for k in CLAIM}
meta["claim_A_id"] = {k: pairs[int(k.split("_")[1])]["A_id"] for k in CLAIM}
meta["claim_B_title"] = {k: pairs[int(k.split("_")[1])]["B_referenced_work"] for k in CLAIM}

HY, HYby = final_rows(["hybrids_eval"], pick="first")
H = json.load(open(f"{PILOT}/data/hybrids.json"))
for h in H:
    for g in ["v1", "coi", "si", "arc", "ra", "scholareval", "v2_reflection", "claimnov"]:
        r = HY[(h["item"], g)]
        d = {"rej": r["novel"] is False, "named": r["named_B"], "evidence": r.get("evidence")}
        if g == "claimnov":
            d.update({k: r[k] for k in ("B_on_spliced_claim", "B_level", "verdict_names_B", "n_claims")})
        else:
            d["seen_B"] = ev_names(r, h["B_title"])
        items[g]["hyb"][h["item"]] = d
meta["hyb_B_id"] = {h["item"]: h["B_id"] for h in H}

dump("r0_items.json", {"items": items, "meta": meta})
print({c: {s: len(v) for s, v in d.items()} for c, d in items.items()})
