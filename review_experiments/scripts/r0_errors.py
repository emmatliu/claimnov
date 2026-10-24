import json
from r0_common import *

D = json.load(open(f"{OUT}/r0_items.json")); I = D["items"]
chosen, dz = disguise_keys()
DIN = {f"seed{d['seed_idx']}_r{d['rung']}_B_in" for d in dz}; DOUT = {k.replace("B_in", "B_out") for k in DIN}
LATER = set(later_set())
res = {}

def set_of(item):
    if item in DIN: return "disg_in"
    if item in DOUT: return "disg_out"
    if item in LATER: return "later"
    if item.startswith("n2f_"): return "claim"
    return "other"

def account(files, used_sets, label):
    rows = []
    for f in files:
        for r in jl(f):
            r["_f"] = f; rows.append(r)
    by = collections.defaultdict(list)
    for r in rows: by[(r["item"], r.get("gate"))].append(r)
    acc = {"n_rows": len(rows), "n_error_rows": sum(r.get("novel") is None for r in rows),
           "error_messages": collections.Counter((r.get("error") or "<novel=None, no error key>")[:70] for r in rows if r.get("novel") is None).most_common(),
           "error_rows_by_file": collections.Counter(r["_f"] for r in rows if r.get("novel") is None),
           "per_gate_set": {}}
    for (item, g), v in by.items():
        s = set_of(item)
        if s not in used_sets: continue
        key = f"{g}|{s}"
        d = acc["per_gate_set"].setdefault(key, {"cells": 0, "cells_with_any_error": 0, "error_rows": 0, "cells_never_valid": 0,
                                                 "cells_dup_valid": 0, "cells_conflicting_valid": 0})
        ok = [r for r in v if r.get("novel") is not None]; er = [r for r in v if r.get("novel") is None]
        d["cells"] += 1; d["error_rows"] += len(er); d["cells_with_any_error"] += bool(er)
        d["cells_never_valid"] += not ok; d["cells_dup_valid"] += len(ok) > 1
        d["cells_conflicting_valid"] += len({r["novel"] for r in ok}) > 1
    alt = {}
    for (item, g), v in by.items():
        s = set_of(item)
        if s not in used_sets: continue
        ok = [r for r in v if r.get("novel") is not None]; er = [r for r in v if r.get("novel") is None]
        a = alt.setdefault(f"{g}|{s}", {"n": 0, "rej_as_scored": 0, "rej_if_errored_cells=pass": 0, "rej_if_errored_cells=fail": 0,
                                       "rej_if_errored_cells_dropped": 0, "n_if_dropped": 0, "rej_last_valid": 0})
        a["n"] += 1
        rej = ok[0]["novel"] is False if ok else False
        a["rej_as_scored"] += rej; a["rej_last_valid"] += (ok[-1]["novel"] is False) if ok else False
        a["rej_if_errored_cells=pass"] += (rej if not er else False)
        a["rej_if_errored_cells=fail"] += (rej if not er else True)
        if not er: a["rej_if_errored_cells_dropped"] += rej; a["n_if_dropped"] += 1
    acc["alternative_treatments"] = alt
    acc["conflicts"] = [{"item": k[0], "gate": k[1], "values": [(r["_f"], r["novel"]) for r in v if r.get("novel") is not None]}
                        for k, v in by.items() if len({r["novel"] for r in v if r.get("novel") is not None}) > 1 and set_of(k[0]) in used_sets]
    res[label] = acc

account(DEPLOYED_FILES, {"disg_in", "disg_out", "later"}, "deployed_sonnet (Table 1 rows v1/coi/si/ra/scholareval/v2)")
account(["claim_sonnet"], {"claim"}, "claim_sonnet (Table 2 rows)")
account(SMUGGLE_FILES, {"disg_in", "disg_out"}, "smuggle (Table 1 ARC/ClaimNov/ablations, seeds 0-99)")
hy = jl("hybrids_eval")
res["hybrids_eval"] = {"n_rows": len(hy), "n_error_rows": sum(r.get("novel") is None for r in hy),
                       "dups": len(hy) - len({(r["item"], r["gate"]) for r in hy})}
for f in ["coupled_veto_neg_profile", "coupled_veto_fa_profile_k", "coupled_veto_ac_profile_k", "coupled_veto_set_profile_k", "coupled_veto_smuggle_profile"]:
    R = jl(f)
    res[f] = {"n": len(R), "n_claims_0": sum(r["n_claims"] == 0 for r in R), "novel_None": sum(r["novel"] is None for r in R)}
for f, gates in [("positive_full", ["module"]), ("negative_full", ["module"])]:
    R = [r for r in jl(f) if r["gate"] in gates]
    res[f"{f}_module_unresolved"] = {"n": len(R), "novel_None": sum(r["novel"] is None for r in R),
                                     "items_None": [r["item"] for r in R if r["novel"] is None],
                                     "None_in_later_set": [r["item"] for r in R if r["novel"] is None and r["item"] in LATER]}
smf = [r for f in SMUGGLE_FILES for r in jl(f) if r["gate"] == "module"]
res["smuggle_module_None"] = sum(r["novel"] is None for r in smf)

v3 = json.load(open(f"{PILOT}/data/disguises_v3.json"))
used = {(d["seed_idx"], d["rung"]) for d in dz}
ov = {}
for q in (1, 2, 3, 4):
    xs = [d for d in v3 if d["rung"] == q and (d["seed_idx"], d["rung"]) in used]
    vals = sorted(d["overlap6"] for d in xs if "overlap6" in d)
    ov[f"r{q}"] = {"n_used": len(xs), "n_with_overlap6": len(vals),
                   "min": vals[0] if vals else None, "median": vals[len(vals)//2] if vals else None, "max": vals[-1] if vals else None,
                   "mean": round(sum(vals)/len(vals), 4) if vals else None,
                   "n_overlap6_gt_0.02": sum(v > 0.02 for v in vals),
                   "n_verified_same_false": sum(1 for d in xs if d.get("verified_same") is False),
                   "n_verified_same_missing": sum(1 for d in xs if "verified_same" not in d),
                   "n_r34_gt0.02_or_unverified": sum(1 for d in xs if q >= 3 and (d.get("overlap6", 0) > 0.02 or d.get("verified_same") is not True))}
    allq = [d for d in v3 if d["rung"] == q]
    ov[f"r{q}"]["all_generated_n"] = len(allq)
    ov[f"r{q}"]["all_generated_verified_false"] = sum(1 for d in allq if d.get("verified_same") is False)
res["overlap6"] = ov
res["overlap6_note"] = "seeds 0-9 (pilot disguises_v2 texts) carry no overlap6/verified_same fields in disguises_v3.json"
res["overlap6_seeds0_9_missing"] = sum(1 for d in v3 if d["seed_idx"] < 10 and "overlap6" not in d)

vt = {}
for s in ("disg_in", "disg_out", "later", "claim"):
    xs = I["v1"][s]
    vt[s] = {"rejections": sum(v["rej"] for v in xs.values()),
             "rej_with_evidence_None (timeout upper bound)": sum(1 for v in xs.values() if v["rej"] and v.get("evidence") is None)}
vt["hyb"] = {"rejections": sum(v["rej"] for v in I["v1"]["hyb"].values()),
             "rej_with_evidence_None": sum(1 for v in I["v1"]["hyb"].values() if v["rej"] and v.get("evidence") is None)}
nf = [r for r in jl("negative_full") if r["gate"] == "v1"]
vt["negative_full_4omini_exact"] = {"rejections": sum(r["novel"] is False for r in nf),
                                    "timeouts": sum(1 for r in nf if (r.get("trace") or {}).get("timeout")),
                                    "evidence_None_rej": sum(1 for r in nf if r["novel"] is False and r.get("evidence") is None)}
res["v1_timeouts"] = vt
res["v1_timeouts_note"] = "trace (rounds/timeout) is NOT stored in deployed_sonnet*/claim_sonnet/hybrids rows; only an upper bound (rejection with evidence=None) is computable"

DEP, _ = final_rows(DEPLOYED_FILES, pick="first")
se = {}
for s, keys in (("disg_in", sorted(DIN)), ("disg_out", sorted(DOUT)), ("later", sorted(LATER))):
    rows = [DEP[(k, "scholareval")] for k in keys]
    se[s] = {"score_dist": sorted(collections.Counter(str(r.get("score")) for r in rows).items()),
             "rating_dist": sorted(collections.Counter(str(r.get("rating")) for r in rows).items()),
             "verdict_slot_imposed": collections.Counter(str(r.get("verdict_slot_imposed")) for r in rows),
             "rejected": sum(r["novel"] is False for r in rows),
             "rejected_items": [k for k, r in zip(keys, rows) if r["novel"] is False]}
rin = set(k.replace("_B_in", "") for k in se["disg_in"]["rejected_items"]); rout = set(k.replace("_B_out", "") for k in se["disg_out"]["rejected_items"])
se["same_items_in_and_out"] = {"in": len(rin), "out": len(rout), "both": len(rin & rout)}
se["rejected_sources"] = sorted({seed_of(k) for k in se["disg_in"]["rejected_items"]})
se["rejected_by_rung_in"] = collections.Counter(int(re.search(r"_r(\d)_", k).group(1)) for k in se["disg_in"]["rejected_items"])
ex = DEP[(se["disg_in"]["rejected_items"][0], "scholareval")] if se["disg_in"]["rejected_items"] else None
se["example_rejected_row"] = {k: v for k, v in ex.items() if k != "_f"} if ex else None
res["scholareval"] = se
dump("r0_errors.json", res)
print(json.dumps({k: v for k, v in res.items() if k in ("overlap6", "v1_timeouts")}, indent=1))
print(json.dumps(res["scholareval"], indent=0, default=str)[:2500])
