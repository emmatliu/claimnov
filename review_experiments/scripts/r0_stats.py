import json
import numpy as np
from r0_common import *

D = json.load(open(f"{OUT}/r0_items.json")); I = D["items"]; M = D["meta"]
T1 = ["v1", "coi", "si", "arc", "ra", "scholareval", "v2_reflection", "claimnov", "graded", "central", "claimnov_k10"]
NAMERS = {"v1", "coi", "si", "arc", "claimnov", "graded", "central", "claimnov_k10"}
res = {}
REPS = 10000

def src_clusters(c, which, keyset=None):
    d = collections.defaultdict(list)
    for k, v in I[c][which].items():
        if keyset is not None and seed_of(k) not in keyset: continue
        d[seed_of(k)].append((k, v["rej"]))
    return d

def anc_clusters(c):
    d = collections.defaultdict(list)
    for k, v in I[c]["later"].items():
        d[anchor_of(k)].append((k, v["rej"]))
    return d

def boot_J_D(checks, keyset=None, reps=REPS, seed=0):
    rng = np.random.default_rng(seed)
    srcs = sorted({seed_of(k) for k in I["claimnov"]["disg_in"] if keyset is None or seed_of(k) in keyset})
    ancs = sorted({anchor_of(k) for k in I["claimnov"]["later"]})
    A = {}
    for c in checks:
        ci = src_clusters(c, "disg_in", keyset); co = src_clusters(c, "disg_out", keyset); ca = anc_clusters(c)
        A[c] = (np.array([sum(x for _, x in ci[s]) for s in srcs], float), np.array([len(ci[s]) for s in srcs], float),
                np.array([sum(x for _, x in co[s]) for s in srcs], float),
                np.array([sum(x for _, x in ca[a]) for a in ancs], float), np.array([len(ca[a]) for a in ancs], float))
    Js = {c: [] for c in checks}; Ds = {c: [] for c in checks}
    for _ in range(reps):
        si = rng.integers(0, len(srcs), len(srcs)); ai = rng.integers(0, len(ancs), len(ancs))
        for c in checks:
            cin, n_in, cout, fa, n_fa = A[c]
            catch = cin[si].sum() / n_in[si].sum(); out = cout[si].sum() / n_in[si].sum()
            far = fa[ai].sum() / n_fa[ai].sum()
            Js[c].append(catch - far); Ds[c].append(catch - out)
    out = {}
    for c in checks:
        cin, n_in, cout, fa, n_fa = A[c]
        J = cin.sum() / n_in.sum() - fa.sum() / n_fa.sum(); Dl = (cin.sum() - cout.sum()) / n_in.sum()
        Ja, Da = np.array(Js[c]), np.array(Ds[c])
        out[c] = {"J": round(J, 4), "J_CI": [round(float(np.percentile(Ja, 2.5)), 4), round(float(np.percentile(Ja, 97.5)), 4)],
                  "Delta": round(Dl, 4), "Delta_CI": [round(float(np.percentile(Da, 2.5)), 4), round(float(np.percentile(Da, 97.5)), 4)]}
        if c != "claimnov" and "claimnov" in checks:
            dj = np.array(Js["claimnov"]) - Ja; dd = np.array(Ds["claimnov"]) - Da
            out[c]["claimnov_minus_J_CI"] = [round(float(np.percentile(dj, 2.5)), 4), round(float(np.percentile(dj, 97.5)), 4)]
            out[c]["claimnov_minus_Delta_CI"] = [round(float(np.percentile(dd, 2.5)), 4), round(float(np.percentile(dd, 97.5)), 4)]
            out[c]["boot_P_claimnov_J_le"] = round(float((dj <= 0).mean()), 4)
    return out, {"n_sources": len(srcs), "n_anchors": len(ancs)}

t1 = {}
for c in T1:
    din, dout, lat = I[c]["disg_in"], I[c]["disg_out"], I[c]["later"]
    caught = sum(v["rej"] for v in din.values()); out = sum(v["rej"] for v in dout.values())
    fa = sum(v["rej"] for v in lat.values())
    t1[c] = {"caught": fmt_ci(caught, len(din)), "src_out": fmt_ci(out, len(dout)), "FA": fmt_ci(fa, len(lat))}
    if c in NAMERS:
        named = sum(bool(v.get("named")) for v in din.values())
        t1[c]["named"] = fmt_ci(named, len(din))
        t1[c]["named_of_caught"] = fmt_ci(named, caught) if caught else None
bJ, binfo = boot_J_D(T1)
for c in T1: t1[c].update(bJ[c])
res["table1"] = t1; res["table1_boot_info"] = {**binfo, "reps": REPS, "cluster": "disguises by source paper (in/out resampled jointly), later-realized by anchor; one joint resample for all checks (paired differences)"}

held = {s for s in {seed_of(k) for k in I["claimnov"]["disg_in"]} if s >= 10}
t1h = {}
bJh, binfoh = boot_J_D(T1, keyset=held)
for c in T1:
    din = {k: v for k, v in I[c]["disg_in"].items() if seed_of(k) in held}
    dout = {k: v for k, v in I[c]["disg_out"].items() if seed_of(k) in held}
    caught = sum(v["rej"] for v in din.values()); out = sum(v["rej"] for v in dout.values())
    t1h[c] = {"caught": fmt_ci(caught, len(din)), "src_out": fmt_ci(out, len(dout)),
              "named": fmt_ci(sum(bool(v.get("named")) for v in din.values()), len(din)) if c in NAMERS else None, **bJh[c]}
    pil = {k: v for k, v in I[c]["disg_in"].items() if seed_of(k) < 10}
    pilo = {k: v for k, v in I[c]["disg_out"].items() if seed_of(k) < 10}
    t1h[c]["pilot_seeds0-9_caught"] = f"{sum(v['rej'] for v in pil.values())}/{len(pil)}"
    t1h[c]["pilot_seeds0-9_src_out"] = f"{sum(v['rej'] for v in pilo.values())}/{len(pilo)}"
res["table1_heldout_seeds_ge10"] = t1h; res["table1_heldout_info"] = {**binfoh, "n_disguises": 4 * len(held)}


psize = {s: len(pool_B(s)[1]) for s in {seed_of(k) for k in I["claimnov"]["disg_in"]}}
res["disguise_pool_sizes"] = {"size_counts": collections.Counter(psize.values()), "small": {s: n for s, n in psize.items() if n < 20}}
for lab, ks in (("excl_1paper_pools", {s for s, n in psize.items() if n > 1}), ("only_full_20paper_pools", {s for s, n in psize.items() if n == 20}),
                ("heldout_and_full20", {s for s, n in psize.items() if n == 20 and s >= 10})):
    bj, bi = boot_J_D(T1, keyset=ks)
    tab = {}
    for c in T1:
        din = {k: v for k, v in I[c]["disg_in"].items() if seed_of(k) in ks}
        dout = {k: v for k, v in I[c]["disg_out"].items() if seed_of(k) in ks}
        tab[c] = {"caught": fmt_ci(sum(v["rej"] for v in din.values()), len(din)),
                  "src_out": fmt_ci(sum(v["rej"] for v in dout.values()), len(dout)),
                  "named": fmt_ci(sum(bool(v.get("named")) for v in din.values()), len(din)) if c in NAMERS else None, **bj[c]}
    res[f"table1_{lab}"] = {"n_sources": len(ks), "rows": tab}

dec = M["claim_decision"]
def t2rows(keys):
    o = {}
    for c in ["v1", "coi", "si", "arc", "ra", "scholareval", "v2_reflection", "claimnov", "graded", "central"]:
        cl = {k: I[c]["claim"][k] for k in keys}
        acc = [k for k in cl if dec[k] == "acc"]; rej = [k for k in cl if dec[k] == "rej"]
        nb = sum(bool(cl[k].get("named")) for k in cl); nrej = sum(cl[k]["rej"] for k in cl)
        elig = len(cl) if c in ("claimnov", "graded", "central") else nrej
        o[c] = {"rej_acc": fmt_ci(sum(cl[k]["rej"] for k in acc), len(acc)) if acc else None,
                "rej_rej": fmt_ci(sum(cl[k]["rej"] for k in rej), len(rej)) if rej else None,
                "rej_all": fmt_ci(nrej, len(cl)),
                "names_B": fmt_ci(nb, len(cl)) if c in NAMERS else None,
                "elig": (fmt_ci(nb, elig) if elig else f"0/0") if c in NAMERS else None}
    return o
CLAIM = sorted(I["claimnov"]["claim"])
res["table2"] = t2rows(CLAIM)
res["table2_note"] = "ClaimNov/graded/central names_B uses the row-recorded profile (papers shown to the verdict judge): 104/160, not the paper's 127/160 (see attribution script)."

t3 = {}
for c in ["v1", "coi", "si", "arc", "ra", "scholareval", "v2_reflection", "claimnov"]:
    hy = I[c]["hyb"]; p = sum(not v["rej"] for v in hy.values()); nb = sum(bool(v["named"]) for v in hy.values())
    elig = 100 if c == "claimnov" else 100 - p
    t3[c] = {"passes": fmt_ci(p, 100), "names_B": fmt_ci(nb, 100) if c in NAMERS else None,
             "of_elig": (fmt_ci(nb, elig) if elig else "0/0") if c in NAMERS else None}
res["table3"] = t3

def outcome(c, s, f):
    return {k: f(v) for k, v in I[c][s].items()}
tests = {
    "disguise_catch(src_in)": ("disg_in", lambda v: v["rej"]),
    "src_out_rejection": ("disg_out", lambda v: v["rej"]),
    "later_FA": ("later", lambda v: v["rej"]),
    "disguise_named_source": ("disg_in", lambda v: bool(v.get("named"))),
    "claim_names_B(row profile, 104)": ("claim", lambda v: bool(v.get("named"))),
    "claim_rejection": ("claim", lambda v: v["rej"]),
    "hybrid_passes": ("hyb", lambda v: not v["rej"]),
    "hybrid_names_B": ("hyb", lambda v: bool(v["named"])),
}
mc = {}
for tname, (s, f) in tests.items():
    mc[tname] = {}
    a = outcome("claimnov", s, f)
    for c in T1:
        if c == "claimnov" or s not in I[c] or not I[c][s]: continue
        if "named" in tname and c not in NAMERS: continue
        mc[tname][c] = mcnemar_exact(a, outcome(c, s, f))
res["mcnemar_claimnov_vs"] = mc
res["mcnemar_note"] = "a_only = ClaimNov success & other failure ('success' = the event named: rejection for catch/src-out/FA, pass for hybrid_passes). Two-sided exact binomial on discordant pairs."

mch = {}
for tname in ["disguise_catch(src_in)", "src_out_rejection", "disguise_named_source"]:
    s, f = tests[tname]; mch[tname] = {}
    a = {k: f(v) for k, v in I["claimnov"][s].items() if seed_of(k) in held}
    for c in T1:
        if c == "claimnov" or ("named" in tname and c not in NAMERS): continue
        mch[tname][c] = mcnemar_exact(a, {k: f(v) for k, v in I[c][s].items() if seed_of(k) in held})
res["mcnemar_heldout"] = mch

lab = json.load(open(f"{REPO}/data/annotation/n2_majority_labels.json"))
clean = sorted({r["item"] for r in lab["rows"] if r["q1"] == "yes" and r["q2"] == "yes" and r.get("item")})
clean = [k for k in clean if k in I["claimnov"]["claim"]]
res["clean_subset"] = {"n": len(clean), "items": clean, "n_clean_pairs_total": sum(1 for r in lab["rows"] if r["q1"] == "yes" and r["q2"] == "yes"),
                       "decisions": collections.Counter(dec[k] for k in clean), "table2": t2rows(clean)}
mcc = {}
for tname in ["claim_names_B(row profile, 104)", "claim_rejection"]:
    s, f = tests[tname]; mcc[tname] = {}
    a = {k: f(I["claimnov"][s][k]) for k in clean}
    for c in ["v1", "coi", "si", "arc", "ra", "scholareval", "v2_reflection", "graded", "central"]:
        if "names" in tname and c not in NAMERS: continue
        mcc[tname][c] = mcnemar_exact(a, {k: f(I[c][s][k]) for k in clean})
res["clean_subset"]["mcnemar"] = mcc

anc = collections.Counter(anchor_of(k) for k in I["claimnov"]["later"])
pa = {}
for a in sorted(anc, key=lambda x: -anc[x]):
    pa[a] = {"n": anc[a]}
    for c in T1:
        pa[a][c] = sum(v["rej"] for k, v in I[c]["later"].items() if anchor_of(k) == a)
conc = {}
for c in T1:
    per = sorted(((pa[a][c], a) for a in pa), reverse=True)
    tot = sum(x for x, _ in per)
    conc[c] = {"total_FA": tot, "n_anchors_with_FA": sum(1 for x, _ in per if x),
               "top1_share": round(per[0][0] / tot, 3) if tot else None,
               "top3_share": round(sum(x for x, _ in per[:3]) / tot, 3) if tot else None,
               "top3": per[:3]}
res["per_anchor"] = {"counts": dict(anc), "n_anchors": len(anc), "min_n": min(anc.values()), "max_n": max(anc.values()),
                     "median_n": float(np.median(list(anc.values()))), "table": pa, "concentration": conc}
dump("r0_stats.json", res)
print(json.dumps({k: res[k] for k in ("table1",)}, indent=1)[:3000])
