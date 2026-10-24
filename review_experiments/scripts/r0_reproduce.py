import json
from r0_common import *

D = json.load(open(f"{OUT}/r0_items.json")); I = D["items"]; M = D["meta"]
CHECKS = ["v1", "coi", "si", "arc", "ra", "scholareval", "v2_reflection", "claimnov", "graded", "central", "claimnov_k10"]
PAPER_T1 = {
    "v1": (400, 312, 91, 51.8, .48, .77), "coi": (388, 382, 31, 3.8, .93, .89), "si": (400, 399, 359, 96.0, .04, .10),
    "arc": (90, 90, 0, 0.0, .23, .23), "ra": (354, None, 265, 98.5, -.10, .22), "scholareval": (20, None, 20, 0.0, .05, .00),
    "v2_reflection": (379, None, 117, 39.2, .55, .66), "claimnov": (386, 385, 18, 1.5, .95, .92),
    "graded": (368, 364, 20, 1.8, .90, .87), "central": (400, 380, 197, 56.8, .43, .51), "claimnov_k10": (386, 385, 18, 2.0, .94, .92)}
PAPER_RUNG = {"v1": [100]*4, "coi": [100, 99, 96, 93], "si": [100]*4, "arc": [90, 0, 0, 0], "ra": [77, 91, 94, 92],
              "scholareval": [5]*4, "v2_reflection": [94, 96, 97, 92], "claimnov": [99, 97, 98, 92]}
PAPER_T2 = {"v1": (53, 40, 20, 45), "coi": (10, 11, 6, 53), "si": (81, 78, 47, 59), "arc": (3, 0, 0, 0),
            "ra": (47, 66, None, None), "scholareval": (0, 0, None, None), "v2_reflection": (16, 14, None, None),
            "claimnov": (3, 3, 79, 79), "graded": (6, 9, 79, 79), "central": (21, 20, 79, 79)}
PAPER_T3 = {"v1": (18, 23, 28), "coi": (76, 11, 46), "si": (0, 63, 63), "arc": (100, 0, None), "ra": (2, None, None),
            "scholareval": (100, None, None), "v2_reflection": (29, None, None), "claimnov": (91, 45, 45)}
NAMES = {"v1", "coi", "si", "arc", "claimnov", "graded", "central", "claimnov_k10"}

res = {"table1": {}, "rungs": {}, "table2": {}, "table3": {}, "mismatches": []}
def mm(where, ours, paper):
    res["mismatches"].append({"where": where, "reproduced": ours, "paper": paper})

for c in CHECKS:
    din, dout, lat = I[c]["disg_in"], I[c]["disg_out"], I[c]["later"]
    caught = sum(v["rej"] for v in din.values()); named = sum(bool(v.get("named")) for v in din.values())
    out = sum(v["rej"] for v in dout.values()); fa = sum(v["rej"] for v in lat.values())
    J = caught / 400 - fa / 400; Dl = (caught - out) / 400
    rungs = [sum(v["rej"] for k, v in din.items() if f"_r{q}_" in k) for q in (1, 2, 3, 4)]
    res["table1"][c] = {"caught": caught, "named": named, "src_out": out, "fa": fa, "fa_pct": round(100*fa/400, 2),
                        "J": round(J, 4), "Delta": round(Dl, 4), "rungs": rungs}
    p = PAPER_T1[c]
    ours = (caught, named if c in NAMES else None, out, round(100*fa/400, 1), round(J + 1e-9, 2), round(Dl + 1e-9, 2))
    for lab, o, q in zip(("caught", "named", "src_out", "FA%", "J", "Delta"), ours, p):
        if q is not None and o != q:
            mm(f"T1 {c} {lab}", o, q)
    if c in PAPER_RUNG and rungs != PAPER_RUNG[c]:
        mm(f"rung table {c}", rungs, PAPER_RUNG[c])

dec = M["claim_decision"]
for c in PAPER_T2:
    cl = I[c]["claim"]
    acc = [k for k in cl if dec[k] == "acc"]; rej = [k for k in cl if dec[k] == "rej"]
    ra = sum(cl[k]["rej"] for k in acc); rr = sum(cl[k]["rej"] for k in rej)
    nb = sum(bool(cl[k].get("named")) for k in cl)
    nrej = sum(cl[k]["rej"] for k in cl)
    elig = len(cl) if c in ("claimnov", "graded", "central") else nrej
    res["table2"][c] = {"rej_acc": f"{ra}/{len(acc)}", "rej_rej": f"{rr}/{len(rej)}", "rej_total": f"{nrej}/{len(cl)}",
                        "names_B": f"{nb}/{len(cl)}", "elig": f"{nb}/{elig}"}
    p = PAPER_T2[c]
    ours = (round(100*ra/len(acc)), round(100*rr/len(rej)),
            round(100*nb/len(cl)) if c in NAMES else None, (round(100*nb/elig) if elig else 0) if c in NAMES else None)
    for lab, o, q in zip(("rej_acc%", "rej_rej%", "names_B%", "elig%"), ours, p):
        if q is not None and o != q:
            mm(f"T2 {c} {lab}", o, q)

for c in PAPER_T3:
    hy = I[c]["hyb"]
    passes = sum(not v["rej"] for v in hy.values()); nb = sum(bool(v["named"]) for v in hy.values())
    nrej = 100 - passes; elig = 100 if c == "claimnov" else nrej
    d = {"passes": passes, "names_B": nb, "elig": f"{nb}/{elig}"}
    if c == "claimnov":
        d["names_and_passes"] = sum(v["named"] and not v["rej"] for v in hy.values())
        d["on_spliced"] = sum(bool(v["B_on_spliced_claim"]) for v in hy.values())
        d["verdict_names_B"] = sum(bool(v["verdict_names_B"]) for v in hy.values())
    res["table3"][c] = d
    p = PAPER_T3[c]
    ours = (passes, nb if c in NAMES else None, (round(100*nb/elig) if elig else None) if c in NAMES else None)
    for lab, o, q in zip(("passes", "names_B", "elig%"), ours, p):
        if q is not None and o != q:
            mm(f"T3 {c} {lab}", o, q)
if res["table3"]["claimnov"]["names_and_passes"] != 37: mm("text: names B among passes", res["table3"]["claimnov"]["names_and_passes"], 37)
if res["table3"]["claimnov"]["on_spliced"] != 30: mm("text: B on spliced claim", res["table3"]["claimnov"]["on_spliced"], 30)

cn = I["claimnov"]["claim"]
bydim = {}
for k, v in cn.items():
    for key in (M["claim_dimension"][k], M["claim_relation"][k]):
        bydim.setdefault(key, [0, 0]); bydim[key][0] += 1; bydim[key][1] += v["named"]
res["attr_dim_B_in_profile"] = bydim
dump("r0_reproduce.json", res)
print(json.dumps(res, indent=1))
