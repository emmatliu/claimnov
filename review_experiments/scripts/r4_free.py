import json, os, re, glob, collections
import numpy as np
from r0_common import OUT, PILOT, RES, mcnemar_exact, wilson, jl
I = json.load(open(f"{OUT}/r0_items.json"))["items"]
DROP = {60, 83, 86, 87, 93}
seed_of = lambda k: int(re.match(r"seed(\d+)_", k).group(1))
res = {}

def from_items(c):
    d = {}
    for which in ("disg_in", "disg_out", "later"):
        for k, v in I[c][which].items():
            if which != "later" and seed_of(k) in DROP: continue
            d[(which, k)] = {"rej": bool(v["rej"]), "named": bool(v.get("named"))}
    return d
CHK = {"claimnov": from_items("claimnov"), "coi": from_items("coi")}
jo = {}
for r in map(json.loads, open(f"{OUT}/e1_judge_only_k8.jsonl")):
    if r["set"] == "disg":
        if seed_of(r["key"]) in DROP: continue
        w = "disg_in" if r["cond"] == "B_in" else "disg_out"
    elif r["set"] == "later": w = "later"
    else: continue
    jo[(w, r["key"])] = {"rej": not r["novel"], "named": (not r["novel"]) and r["names_B"]}
CHK["judge_only"] = jo
on = {}
for r in map(json.loads, open(f"{OUT}/r1_opennovelty.jsonl")):
    if r.get("error"): continue
    if r["set"] == "disg":
        if seed_of(r["key"]) in DROP: continue
        w = "disg_in" if r["cond"] == "B_in" else "disg_out"
    elif r["set"] == "later": w = "later"
    else: continue
    named = r["reject_all"] and any(v["verdict"] == "can_refute" and v["refuting_id"] == r.get("B_id") for v in r["verdicts"])
    on[(w, r["key"])] = {"rej": bool(r["reject_all"]), "named": bool(named)}
CHK["opennovelty_all"] = on
for c, d in CHK.items():
    assert sum(1 for k in d if k[0] == "disg_in") == 380 and sum(1 for k in d if k[0] == "later") == 400, c

SRCS = sorted({seed_of(k) for (w, k) in CHK["claimnov"] if w == "disg_in"})
ANCS = sorted({k.split("::")[0] for (w, k) in CHK["claimnov"] if w == "later"})
def arrays(d):
    cin = np.array([sum(v["rej"] for (w, k), v in d.items() if w == "disg_in" and seed_of(k) == s) for s in SRCS], float)
    nam = np.array([sum(v["named"] for (w, k), v in d.items() if w == "disg_in" and seed_of(k) == s) for s in SRCS], float)
    cout = np.array([sum(v["rej"] for (w, k), v in d.items() if w == "disg_out" and seed_of(k) == s) for s in SRCS], float)
    nin = np.array([sum(1 for (w, k) in d if w == "disg_in" and seed_of(k) == s) for s in SRCS], float)
    fa = np.array([sum(v["rej"] for (w, k), v in d.items() if w == "later" and k.split("::")[0] == a) for a in ANCS], float)
    nfa = np.array([sum(1 for (w, k) in d if w == "later" and k.split("::")[0] == a) for a in ANCS], float)
    return cin, nam, cout, nin, fa, nfa
AR = {c: arrays(d) for c, d in CHK.items()}
def metrics(a, si=None, ai=None):
    cin, nam, cout, nin, fa, nfa = a
    if si is not None: cin, nam, cout, nin = cin[si], nam[si], cout[si], nin[si]
    if ai is not None: fa, nfa = fa[ai], nfa[ai]
    c = cin.sum() / nin.sum(); o = cout.sum() / nin.sum(); f = fa.sum() / nfa.sum()
    return {"catch": c, "named": nam.sum() / nin.sum(), "src_out": o, "FA": f, "J": c - f, "Delta": c - o}
REPS = 10000; rng = np.random.default_rng(0)
boots = {c: collections.defaultdict(list) for c in CHK}
for _ in range(REPS):
    si = rng.integers(0, len(SRCS), len(SRCS)); ai = rng.integers(0, len(ANCS), len(ANCS))
    for c in CHK:
        for m, v in metrics(AR[c], si, ai).items(): boots[c][m].append(v)
def perm_p(c, m, R=100000, seed=1):
    a, b = AR["claimnov"], AR[c]
    cinA, namA, coutA, ninA, faA, nfaA = a; cinB, namB, coutB, ninB, faB, nfaB = b
    g = np.random.default_rng(seed)
    obs = metrics(a)[m] - metrics(b)[m]
    fs = g.integers(0, 2, (R, len(SRCS))).astype(bool); fa_ = g.integers(0, 2, (R, len(ANCS))).astype(bool)
    def rate(x, y, n, f): return (np.where(f, y, x).sum(1) - np.where(f, x, y).sum(1)) / n.sum()
    dcin = rate(cinA, cinB, ninA, fs); dcout = rate(coutA, coutB, ninA, fs); dnam = rate(namA, namB, ninA, fs)
    dfa = rate(faA, faB, nfaA, fa_)
    stat = {"catch": dcin, "named": dnam, "src_out": dcout, "FA": dfa, "J": dcin - dfa, "Delta": dcin - dcout}[m]
    return float((np.abs(stat) >= abs(obs) - 1e-12).mean())
x2 = {"n_sources": len(SRCS), "n_anchors": len(ANCS), "reps_bootstrap": REPS, "reps_permutation": 100000, "point": {}, "diff": {}}
for c in CHK:
    pt = metrics(AR[c])
    x2["point"][c] = {m: round(v, 4) for m, v in pt.items()}
    x2["point"][c]["CI"] = {m: [round(float(np.percentile(boots[c][m], 2.5)), 4), round(float(np.percentile(boots[c][m], 97.5)), 4)] for m in pt}
for c in ("coi", "judge_only", "opennovelty_all"):
    out = {}
    for m in ("catch", "named", "src_out", "FA", "J", "Delta"):
        d = np.array(boots["claimnov"][m]) - np.array(boots[c][m])
        obs = metrics(AR["claimnov"])[m] - metrics(AR[c])[m]
        pb = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
        w = {"catch": "disg_in", "named": "disg_in", "src_out": "disg_out", "FA": "later"}.get(m)
        mc = None
        if w:
            key = "named" if m == "named" else "rej"
            A = {k: v[key] for (ww, k), v in CHK["claimnov"].items() if ww == w}; B = {k: v[key] for (ww, k), v in CHK[c].items() if ww == w}
            mm = mcnemar_exact(A, B); mc = f"{mm['a_only']}/{mm['b_only']}, p={mm['p_exact']:.3g}"
        out[m] = {"diff": round(obs, 4), "boot_CI": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                  "boot_p": round(pb, 4), "perm_p": round(perm_p(c, m), 5), "item_mcnemar": mc}
    x2["diff"][f"claimnov_minus_{c}"] = out
res["X2"] = x2

pa = {}
for a in ANCS:
    row = {"n": int(sum(1 for (w, k) in CHK["claimnov"] if w == "later" and k.split("::")[0] == a))}
    for c in CHK: row[c] = int(sum(v["rej"] for (w, k), v in CHK[c].items() if w == "later" and k.split("::")[0] == a))
    pa[a] = row
res["per_anchor_FA"] = dict(sorted(pa.items(), key=lambda x: -x[1]["n"]))
res["per_anchor_FA_concentration"] = {c: {"anchors_with_any_FA": sum(1 for r in pa.values() if r[c]), "max_anchor_share": round(max(r[c] for r in pa.values()) / max(1, sum(r[c] for r in pa.values())), 3),
                                          "top_anchor": max(pa, key=lambda a: pa[a][c])} for c in CHK}

def first_valid(name, key="item"):
    d = {}
    for r in jl(name):
        if r.get("novel") is not None and r[key] not in d: d[r[key]] = r
    return d
ALL = {}
for root in ("anchors", "anchors_full"):
    for adir in sorted(glob.glob(f"{PILOT}/data/{root}/*")):
        s = os.path.basename(adir)
        if s == "cut" or not os.path.exists(f"{adir}/ideas.json"): continue
        sf = f"{adir}/sorted_full.json" if os.path.exists(f"{adir}/sorted_full.json") else f"{adir}/sorted.json"
        bk = {x["idx"]: x["bucket"] for x in json.load(open(sf))}
        for i in json.load(open(f"{adir}/ideas.json")): ALL[f"{s}::{i['idx']}"] = bk.get(i["idx"])
k10 = {**first_valid("coupled_veto_fa_profile"), **first_valid("coupled_veto_ac_profile")}
ksc = {**first_valid("coupled_veto_fa_profile_k"), **first_valid("coupled_veto_ac_profile_k")}
pf = collections.defaultdict(dict)
for r in jl("positive_full"):
    if r.get("novel") is not None and r["item"] not in pf[r["gate"]]: pf[r["gate"]][r["item"]] = r
X6C = {"claimnov_k10": k10, "claimnov_shipped_kscaled": ksc, "sonnet_match_veto_top8": pf["veto_sonnet"], "arc_lexical": pf["arc"]}
x6 = {"n_ideas": len(ALL), "buckets": dict(collections.Counter(ALL.values())), "checks": {}}
for c, d in X6C.items():
    items = [k for k in ALL if k in d]
    rej = {k: not d[k]["novel"] for k in items}
    per = collections.defaultdict(lambda: [0, 0])
    for k in items: per[k.split("::")[0]][0] += rej[k]; per[k.split("::")[0]][1] += 1
    byb = collections.defaultdict(lambda: [0, 0])
    for k in items: byb[ALL[k]][0] += rej[k]; byb[ALL[k]][1] += 1
    tot = sum(rej.values()); lo, hi = wilson(tot, len(items))
    rates = [v[0] / v[1] for v in per.values()]
    x6["checks"][c] = {"n": len(items), "anchors": len(per), "rejected": tot, "rate": round(tot / len(items), 4), "wilson": [round(lo, 4), round(hi, 4)],
                       "by_bucket": {b: f"{v[0]}/{v[1]}" for b, v in sorted(byb.items())},
                       "per_anchor_rate_median": round(float(np.median(rates)), 4), "per_anchor_rate_range": [round(min(rates), 3), round(max(rates), 3)],
                       "per_anchor": {a: f"{v[0]}/{v[1]}" for a, v in sorted(per.items())}}
res["X6"] = x6
json.dump(res, open(f"{OUT}/r4_free.json", "w"), indent=1)
print(json.dumps({"X2": res["X2"]["diff"], "points": res["X2"]["point"]}, indent=0)[:6000])
print(json.dumps({c: {k: v for k, v in x.items() if k != "per_anchor"} for c, x in x6["checks"].items()}, indent=0))
