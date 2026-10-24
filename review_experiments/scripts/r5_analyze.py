import json, os, sys, collections, math
import numpy as np
from scipy.stats import binomtest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = "review_experiments/out"
def wilson(k, n, z=1.96):
    if n == 0: return None
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]
def mcn(a, b):
    ks = sorted(set(a) & set(b)); x = sum(1 for k in ks if a[k] and not b[k]); y = sum(1 for k in ks if b[k] and not a[k])
    return {"n": len(ks), "a_only": x, "b_only": y, "p": float("%.3g" % binomtest(x, x + y, .5).pvalue) if x + y else 1.0}
S = {}
L = [json.loads(l) for l in open(f"{OUT}/r5_y1_loop.jsonl")]
L = list({(r["anchor"], r["idx"], r["arm"]): r for r in L}.values())
y1 = {}
for arm in ("claimnov", "coi", "none"):
    rr = [r for r in L if r["arm"] == arm]
    com = [r for r in rr if r["committed"]]
    first = sum(1 for r in rr if r["rounds"] and r["rounds"][0]["check"]["rej"])
    tin = sum(r["cost"]["in"] for r in rr); tout = sum(r["cost"]["out"] for r in rr); usd = sum(r["cost"]["usd"] for r in rr)
    chk = [x for r in rr for x in r["rounds"]]
    d = {"n": len(rr), "first_rejected": first, "first_rej_rate": round(first / len(rr), 3) if rr else None, "first_rej_CI": wilson(first, len(rr)),
         "committed": len(com), "giveups": len(rr) - len(com),
         "regen_dist": dict(collections.Counter(r["n_regen"] for r in com)),
         "mean_regen_per_committed": round(np.mean([r["n_regen"] for r in com]), 3) if com else None,
         "mean_checks_per_committed": round(np.mean([len(r["rounds"]) for r in com]), 3) if com and arm != "none" else 0,
         "committed_unchanged_from_first": sum(1 for r in com if r["n_regen"] == 0),
         "tokens_in": tin, "tokens_out": tout, "usd": round(usd, 4),
         "per_committed": {"tokens_in": round(tin / len(com)), "tokens_out": round(tout / len(com)), "usd": round(usd / len(com), 4),
                           "wall_s": round(sum(r["wall_s"] for r in rr) / len(com), 1),
                           "sum_call_latency_s": round(sum(r["cost"]["sum_latency_s"] for r in rr) / len(com), 1)} if com else None,
         "per_check": {"usd_mean": round(np.mean([x["cost"]["usd"] for x in chk]), 4), "wall_check_s_median": round(float(np.median([x["wall_check_s"] for x in chk])), 1),
                       "calls_mean": round(np.mean([x["cost"]["calls"] for x in chk]), 1)} if chk else None,
         "regen_call": {"n": sum(1 for x in chk if x["round"] > 0), "wall_gen_s_median": round(float(np.median([x["wall_gen_s"] for x in chk if x["round"] > 0])), 1) if any(x["round"] > 0 for x in chk) else None},
         "by_anchor": {a: {"n": sum(1 for r in rr if r["anchor"] == a), "first_rejected": sum(1 for r in rr if r["anchor"] == a and r["rounds"] and r["rounds"][0]["check"]["rej"]),
                           "giveups": sum(1 for r in rr if r["anchor"] == a and not r["committed"])} for a in sorted({r["anchor"] for r in rr})}}
    if arm == "claimnov":
        d["claims_src_first"] = dict(collections.Counter(r["rounds"][0]["check"]["claims_src"] for r in rr))
        d["claims_src_regen"] = dict(collections.Counter(x["check"]["claims_src"] for r in rr for x in r["rounds"][1:]))
        d["pairs_per_check_mean"] = round(np.mean([x["check"]["n_pairs"] for x in chk]), 1)
    if arm != "none":
        d["rejections"] = [{"anchor": r["anchor"][:24], "idx": r["idx"], "rounds": [(x["round"], x["check"]["rej"], x["check"].get("paper_title")) for x in r["rounds"]]}
                           for r in rr if any(x["check"]["rej"] for x in r["rounds"])]
    y1[arm] = d
fr = {arm: {(r["anchor"], r["idx"]): r["rounds"][0]["check"]["rej"] for r in L if r["arm"] == arm} for arm in ("claimnov", "coi")}
y1["first_rej_mcnemar_claimnov_vs_coi"] = mcn(fr["claimnov"], fr["coi"])
y1["same_first_paper_cited"] = sum(1 for k in fr["claimnov"] if fr["claimnov"][k] and fr["coi"].get(k))
S["Y1"] = y1
if os.path.exists(f"{OUT}/r5_y1_review.jsonl"):
    RV = [json.loads(l) for l in open(f"{OUT}/r5_y1_review.jsonl")]
    rev = {}
    for a, b in (("claimnov", "none"), ("coi", "none"), ("claimnov", "coi")):
        rows = [r for r in RV if r["a"] == a and r["b"] == b]
        seeds = sorted({(r["anchor"], r["idx"]) for r in rows})
        sc = {s: np.mean([1.0 if r["winner"] == a else 0.0 if r["winner"] == b else 0.5 for r in rows if (r["anchor"], r["idx"]) == s]) for s in seeds}
        v = np.array(list(sc.values())); rng = np.random.default_rng(0)
        bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(10000)] if len(v) else []
        matched = sum(1 for s in {(r["anchor"], r["idx"]) for r in L} if all(any(x["anchor"] == s[0] and x["idx"] == s[1] and x["arm"] == arm and x["committed"] for x in L) for arm in (a, b)))
        rev[f"{a}_vs_{b}"] = {"matched_seeds": matched, "seeds_with_different_committed_idea": len(seeds), "judge_calls": len(rows),
                              "unparsed": sum(1 for r in rows if r["pick"] is None),
                              "a_win_rate_on_differing": round(float(v.mean()), 3) if len(v) else None,
                              "CI": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)] if len(v) else None,
                              "order_consistent_seeds": sum(1 for s in seeds if len({r["winner"] for r in rows if (r["anchor"], r["idx"]) == s}) == 1),
                              "position1_pick_rate": round(np.mean([r["pick"] == 1 for r in rows if r["pick"]]), 3) if rows else None,
                              "per_seed": {f"{s[0][:20]}::{s[1]}": sc[s] for s in seeds}}
    S["Y1_review_proxy"] = rev
Y = {(r["set"], r["key"]): r for r in map(json.loads, open(f"{OUT}/r5_y2_top1.jsonl"))}
CN = json.load(open(f"{OUT}/r0_items.json"))["items"]
cn = {"disg_in": CN["claimnov"]["disg_in"], "disg_out": CN["claimnov"]["disg_out"], "later": CN["claimnov"]["later"], "hyb": CN["claimnov"]["hyb"]}
coi = {"disg_in": CN["coi"]["disg_in"], "disg_out": CN["coi"]["disg_out"], "later": CN["coi"]["later"], "hyb": CN["coi"]["hyb"]}
def jboot(pos, neg, reps=10000):
    rng = np.random.default_rng(0)
    dp = collections.defaultdict(list); dn = collections.defaultdict(list)
    for i, (c, *_r) in enumerate(pos): dp[c].append(i)
    for i, (c, _f) in enumerate(neg): dn[c].append(i)
    kp, kn = list(dp), list(dn); P = np.array([[x[1], x[2]] for x in pos], float); N = np.array([x[1] for x in neg], float)
    J, D = [], []
    for _ in range(reps):
        pp = np.concatenate([dp[kp[j]] for j in rng.integers(0, len(kp), len(kp))]); nn = np.concatenate([dn[kn[j]] for j in rng.integers(0, len(kn), len(kn))])
        J.append(P[pp, 0].mean() - N[nn].mean()); D.append(P[pp, 0].mean() - P[pp, 1].mean())
    q = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
    return q(J), q(D)
y2 = {}
for sub in ("exact", "full"):
    ok = lambda r: r["exact"] or sub == "full"
    din = {k: r for (s, k), r in Y.items() if s == "disg_in" and ok(r)}
    dout = {k: r for (s, k), r in Y.items() if s == "disg_out" and ok(r)}
    lat = {k: r for (s, k), r in Y.items() if s == "later" and ok(r)}
    if sub == "exact" and (len(din), len(dout), len(lat)) != (68, 65, 341): print("WARN exact sizes", len(din), len(dout), len(lat))
    def block(dec_in, dec_out, dec_lat, named):
        c = sum(dec_in.values()); so = sum(dec_out.values()); fa = sum(dec_lat.values())
        pos = [(int(k.split("_")[0][4:]), dec_in[k], dec_out.get(k.replace("B_in", "B_out"), False)) for k in dec_in]
        neg = [(k.split("::")[0], dec_lat[k]) for k in dec_lat]
        Jci, Dci = jboot(pos, neg)
        return {"caught": f"{c}/{len(dec_in)}", "named": f"{named}/{len(dec_in)}", "srcout": f"{so}/{len(dec_out)}", "fa": f"{fa}/{len(dec_lat)}",
                "fa_pct": round(100 * fa / len(dec_lat), 1), "fa_CI": wilson(fa, len(dec_lat)),
                "J": round(c / len(dec_in) - fa / len(dec_lat), 3), "J_CI": Jci, "Delta": round(c / len(dec_in) - so / len(dec_out), 3), "Delta_CI": Dci}
    t1 = {k: r["rej"] for k, r in din.items()}, {k: r["rej"] for k, r in dout.items()}, {k: r["rej"] for k, r in lat.items()}
    out = {"top1_pairwise": block(*t1, sum(1 for r in din.values() if r["rej"] and r["top1_is_B"]))}
    out["top1_is_source_srcin"] = f"{sum(r['top1_is_B'] for r in din.values())}/{len(din)}"
    for name, ref in (("claimnov", cn), ("coi", coi)):
        d = [{k: ref[s][k]["rej"] for k in keys} for s, keys in (("disg_in", din), ("disg_out", dout), ("later", lat))]
        out[name] = block(*d, sum(1 for k in din if ref["disg_in"][k]["rej"] and ref["disg_in"][k].get("named")))
        out[f"mcnemar_top1_vs_{name}"] = {"catch": mcn(t1[0], d[0]), "srcout": mcn(t1[1], d[1]), "fa": mcn(t1[2], d[2])}
    y2[sub] = out
H = {k: r for (s, k), r in Y.items() if s == "hyb"}
if H:
    tp = {k: not r["rej"] for k, r in H.items()}; cp = {k: not cn["hyb"][k]["rej"] for k in H}; op = {k: not coi["hyb"][k]["rej"] for k in H}
    y2["hybrids"] = {"n": len(H), "top1_is_B": sum(r["top1_is_B"] for r in H.values()),
                     "top1_pass": sum(tp.values()), "top1_pass_and_top1_is_B": sum(1 for k, r in H.items() if tp[k] and r["top1_is_B"]),
                     "top1_reject_names_B": sum(1 for k, r in H.items() if r["rej"] and r["top1_is_B"]),
                     "top1_reject_top1_not_B": sum(1 for k, r in H.items() if r["rej"] and not r["top1_is_B"]),
                     "claimnov_pass": sum(cp.values()), "coi_pass": sum(op.values()),
                     "mcnemar_pass_top1_vs_claimnov": mcn(tp, cp), "mcnemar_pass_top1_vs_coi": mcn(tp, op),
                     "pass_by_top1_is_B": {"B": f"{sum(1 for k, r in H.items() if tp[k] and r['top1_is_B'])}/{sum(r['top1_is_B'] for r in H.values())}",
                                           "notB": f"{sum(1 for k, r in H.items() if tp[k] and not r['top1_is_B'])}/{sum(not r['top1_is_B'] for r in H.values())}"},
                     "claimnov_pass_by_top1_is_B": {"B": f"{sum(1 for k, r in H.items() if cp[k] and r['top1_is_B'])}", "notB": f"{sum(1 for k, r in H.items() if cp[k] and not r['top1_is_B'])}"}}
y2["parse_errors"] = sum(r["parse_error"] for r in Y.values()); y2["n_rows"] = len(Y)
y2["per_call"] = {"usd_mean": round(np.mean([r["meta"]["usd"] for r in Y.values()]), 5), "latency_median_s": round(float(np.median([r["meta"]["latency_s"] for r in Y.values()])), 2)}
y2["fa_by_anchor"] = dict(collections.Counter(k.split("::")[0][:24] for (s, k), r in Y.items() if s == "later" and r["rej"]))
S["Y2"] = y2
C = [json.loads(l) for l in open(f"{OUT}/r5_calls.jsonl")]
sp = collections.defaultdict(lambda: {"calls": 0, "in": 0, "out": 0, "usd": 0.0, "usd_guard": 0.0})
for r in C:
    b = sp[f"{r['exp']}|{r['key']}"]; b["calls"] += 1; b["in"] += r["in"]; b["out"] += r["out"]; b["usd"] += r["usd"]; b["usd_guard"] += r["usd_guard"]
S["spend"] = {k: {**v, "usd": round(v["usd"], 3), "usd_guard": round(v["usd_guard"], 3)} for k, v in sorted(sp.items())}
S["spend_total"] = {"calls": len(C), "usd": round(sum(r["usd"] for r in C), 3), "usd_guard": round(sum(r["usd_guard"] for r in C), 3)}
S["endpoints"] = {k: dict(collections.Counter(r["endpoint"] for r in C if r["key"] == k)) for k in sorted({r["key"] for r in C})}
json.dump(S, open(f"{OUT}/r5_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in S.items() if k != "Y1"}, indent=1)[:6000])
print(json.dumps({a: {k: v for k, v in S["Y1"][a].items() if k not in ("rejections",)} for a in ("claimnov", "coi", "none")}, indent=1)[:5000])
