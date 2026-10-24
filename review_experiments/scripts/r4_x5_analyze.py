import json, os, re, collections
import numpy as np
from r0_common import OUT, mcnemar_exact, wilson
L = lambda f: [json.loads(l) for l in open(f"{OUT}/{f}")]
I = json.load(open(f"{OUT}/r0_items.json"))["items"]["claimnov"]
seed_of = lambda k: int(re.match(r"seed(\d+)_", k).group(1))
def w_of(st, k): return "later" if st == "later" else ("disg_in" if k.endswith("B_in") else "disg_out")
PROF = {"gptoss": {(w_of(r["set"], r["key"]), r["key"]): r for r in L("r4_x5_profile_gptoss.jsonl") if not r.get("error")},
        "maverick": {(w_of(r["set"], r["key"]), r["key"]): r for r in L("r4_x5_profile_maverick.jsonl") if not r.get("error")}}
SRC = {(w_of(r["set"], r["key"]), r["key"]): r["src"] for r in PROF["gptoss"].values()}
JO = {"gptoss": {}, "maverick": {}, "sonnet": {}}
for m, f in (("gptoss", "r1_nonclaude_judgeonly_gpt-oss-120b-1.jsonl"), ("maverick", "r1_nonclaude_judgeonly_meta.jsonl")):
    for r in L(f):
        if r["set"] in ("disg", "later"): JO[m][(w_of(r["set"], r["key"]), r["key"])] = {"rej": r["rej"], "names_B": r["names_B"]}
for r in L("e1_judge_only_k8.jsonl"):
    if r["set"] in ("disg", "later"): JO["sonnet"][(w_of(r["set"], r["key"]), r["key"])] = {"rej": not r["novel"], "names_B": (not r["novel"]) and r["names_B"]}
SON = {}
for w in ("disg_in", "disg_out", "later"):
    for k, v in I[w].items(): SON[(w, k)] = {"rej": bool(v["rej"]), "names_B": bool(v.get("named"))}
PROF["sonnet"] = SON
def table(dec, keys, reps=10000, seed=0):
    srcs = sorted({seed_of(k) for w, k in keys if w == "disg_in"}); ancs = sorted({k.split("::")[0] for w, k in keys if w == "later"})
    cin = np.array([sum(dec[(w, k)]["rej"] for w, k in keys if w == "disg_in" and seed_of(k) == s) for s in srcs], float)
    nin = np.array([sum(1 for w, k in keys if w == "disg_in" and seed_of(k) == s) for s in srcs], float)
    cout = np.array([sum(dec[(w, k)]["rej"] for w, k in keys if w == "disg_out" and seed_of(k) == s) for s in srcs], float)
    nout = np.array([sum(1 for w, k in keys if w == "disg_out" and seed_of(k) == s) for s in srcs], float)
    fa = np.array([sum(dec[(w, k)]["rej"] for w, k in keys if w == "later" and k.split("::")[0] == a) for a in ancs], float)
    nfa = np.array([sum(1 for w, k in keys if w == "later" and k.split("::")[0] == a) for a in ancs], float)
    rng = np.random.default_rng(seed); Js, Ds = [], []
    for _ in range(reps):
        si = rng.integers(0, len(srcs), len(srcs)); ai = rng.integers(0, len(ancs), len(ancs))
        c = cin[si].sum() / nin[si].sum(); Js.append(c - fa[ai].sum() / nfa[ai].sum()); Ds.append(c - cout[si].sum() / nout[si].sum())
    c = cin.sum() / nin.sum()
    named = sum(bool(dec[(w, k)].get("names_B")) for w, k in keys if w == "disg_in")
    pct = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
    return {"caught": f"{int(cin.sum())}/{int(nin.sum())}", "named": named, "src_out": f"{int(cout.sum())}/{int(nout.sum())}",
            "FA": f"{int(fa.sum())}/{int(nfa.sum())} = {100*fa.sum()/nfa.sum():.1f}%", "J": round(c - fa.sum() / nfa.sum(), 3), "J_CI": pct(Js),
            "Delta": round(c - cout.sum() / nout.sum(), 3), "Delta_CI": pct(Ds)}
res = {}
for sub in ("all_with_profile", "exact", "recon"):
    keys = [k for k in PROF["gptoss"] if sub == "all_with_profile" or SRC[k] == sub]
    keys = [k for k in keys if all(k in PROF[m] and k in JO[m] for m in ("gptoss", "maverick", "sonnet"))]
    out = {"n": dict(collections.Counter(w for w, _ in keys))}
    for m in ("gptoss", "maverick", "sonnet"):
        out[m] = {"profile": table(PROF[m], keys), "judge_only": table(JO[m], keys)}
        mc = {}
        for w, lab in (("disg_in", "catch"), ("disg_out", "src_out"), ("later", "FA")):
            ks = [k for k in keys if k[0] == w]
            r = mcnemar_exact({k: PROF[m][k]["rej"] for k in ks}, {k: JO[m][k]["rej"] for k in ks})
            mc[lab] = f"{r['a_only']}/{r['b_only']}, p={r['p_exact']:.2g}"
        out[m]["mcnemar_profile_vs_judgeonly"] = mc
        out[m]["parse_errors_profile"] = sum(bool(PROF[m][k].get("parse_error")) for k in keys)
    res[sub] = out
res["note"] = "sonnet profile = the paper's own verdict rows (true profile); non-Claude profile = exact rebuilt prompt where available (round-1 E7 outputs), else the reconstructed prompt (new calls)."
json.dump(res, open(f"{OUT}/r4_x5_summary.json", "w"), indent=1)
for sub in ("all_with_profile", "exact", "recon"):
    print(sub, res[sub]["n"])
    for m in ("gptoss", "maverick", "sonnet"):
        print("  ", m, "PROFILE", res[sub][m]["profile"], "\n      JO", res[sub][m]["judge_only"], "\n      mc", res[sub][m]["mcnemar_profile_vs_judgeonly"], "perr", res[sub][m]["parse_errors_profile"])
