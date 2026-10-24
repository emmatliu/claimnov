import json, collections
from r0_common import *
exec(open("r2_e8_attribution.py").read().split("# ---------------- generic scorer")[0])
ATT = json.load(open(f"{OUT}/r0_attribution.json"))["per_item_claim"]
ST = json.load(open(f"{OUT}/r0_stats.json")); CLEAN = ST["clean_subset"]["items"]
res = {}
d = collections.Counter()
for k in I["claimnov"]["claim"]:
    ex = ("claim", k) in VERD
    new = ITEMS[("claim", k)]["B"] in cn_set("claim", k, 3)
    old = ATT[k]["b_lvl3"]
    d[(ex, old, new)] += 1
res["claim_B_lvl3_(exact_item, r0_reconstruction, now)"] = {str(k): v for k, v in d.items()}
rej = lambda k: I["claimnov"]["claim"][k]["rej"]
res["claim_now"] = {"names_B": sum(ITEMS[("claim", k)]["B"] in cn_set("claim", k, 3) for k in I["claimnov"]["claim"]),
                    "on_pass": sum(ITEMS[("claim", k)]["B"] in cn_set("claim", k, 3) and not rej(k) for k in I["claimnov"]["claim"])}
res["clean36_now"] = {"names_B": sum(ITEMS[("claim", k)]["B"] in cn_set("claim", k, 3) for k in CLEAN),
                      "on_pass": sum(ITEMS[("claim", k)]["B"] in cn_set("claim", k, 3) and not rej(k) for k in CLEAN), "n": len(CLEAN)}
ON = {(r["set"], r["key"]): r for r in L(f"{OUT}/r1_opennovelty.jsonl") if not r.get("error")}
on_nb = {k: any(v["refuting_id"] == ITEMS[("claim", k)]["B"] for v in ON[("claim", k)]["verdicts"]) for k in CLEAN}
cn_nb = {k: ITEMS[("claim", k)]["B"] in cn_set("claim", k, 3) for k in CLEAN}
res["clean36_mcnemar_names_B_claimnov_vs_ON"] = mcnemar_exact(cn_nb, on_nb)
si_nb = {k: bool(I["si"]["claim"][k]["named"]) for k in CLEAN}
res["clean36_mcnemar_names_B_claimnov_vs_Si"] = mcnemar_exact(cn_nb, si_nb)
DROP = {60, 83, 86, 87, 93}
so = [k for (s, k) in ON if s == "disg" and k.endswith("B_out") and seed_of(k) not in DROP]
res["ON_srcout_any_can_refute_380"] = sum(any(v["verdict"] == "can_refute" for v in ON[("disg", k)]["verdicts"]) for k in so)
ex_out = [k for (s, k, *x) in PROF if s == "disg" and not x and k.endswith("B_out")]
res["ON_srcout_any_can_refute_on_65_exact"] = sum(any(v["verdict"] == "can_refute" for v in ON[("disg", k)]["verdicts"]) for k in ex_out if ("disg", k) in ON)
res["ClaimNov_srcout_65_lvl>=4"] = sum(bool(cn_set("disg", k, 4)) for k in ex_out)
res["ClaimNov_srcout_65_lvl>=3"] = sum(bool(cn_set("disg", k, 3)) for k in ex_out)
hy = I["claimnov"]["hyb"]
res["hyb_rowbased"] = {t: {"names_B": sum(bool(v["named"]) and (v["B_level"] or 0) >= t for v in hy.values()),
                           "on_pass": sum(bool(v["named"]) and (v["B_level"] or 0) >= t and not v["rej"] for v in hy.values())} for t in (3, 4, 5)}
dump("r2_e8_extra.json", res); print(json.dumps(res, indent=1))
