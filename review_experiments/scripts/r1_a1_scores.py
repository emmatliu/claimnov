import json
from r0_common import *

DROP = {60, 83, 86, 87, 93}
DEP, _ = final_rows(DEPLOYED_FILES, pick="first")
I = json.load(open(f"{OUT}/r0_items.json"))["items"]
FIELD = {"ra": "rating", "scholareval": "score"}


def auc(pos, neg):
    s = 0.0
    for p in pos:
        for q in neg:
            s += 1.0 if p < q else (0.5 if p == q else 0.0)
    return s / (len(pos) * len(neg))


res = {}
for g, fld in FIELD.items():
    din = {k: v for k, v in I[g]["disg_in"].items() if seed_of(k) not in DROP}
    dout = {k: v for k, v in I[g]["disg_out"].items() if seed_of(k) not in DROP}
    lat = I[g]["later"]
    sc = {}
    mism = 0
    for name, d in (("in", din), ("out", dout), ("later", lat)):
        sc[name] = {}
        for k in d:
            r = DEP[(k, g)]
            sc[name][k] = float(r[fld])
            if (r["novel"] in (False, "False")) != d[k]["rej"]:
                mism += 1
    pos, out_, neg = list(sc["in"].values()), list(sc["out"].values()), list(sc["later"].values())
    ths = sorted(set(pos + out_ + neg))
    ths = [min(ths) - 1] + ths
    rows = []
    for th in ths:
        c = sum(x <= th for x in pos) / len(pos); so = sum(x <= th for x in out_) / len(out_); fa = sum(x <= th for x in neg) / len(neg)
        rows.append({"th": th, "catch": round(c, 4), "src_out": round(so, 4), "FA": round(fa, 4), "J": round(c - fa, 4), "Delta": round(c - so, 4),
                     "caught_n": sum(x <= th for x in pos), "src_out_n": sum(x <= th for x in out_), "FA_n": sum(x <= th for x in neg)})
    bj = max(rows, key=lambda r: r["J"]); bd = max(rows, key=lambda r: r["Delta"])
    native_th = 3.0 if g == "ra" else 0.0
    nat = next(r for r in rows if r["th"] == max(t for t in ths if t <= native_th))
    res[g] = {"n": {"in": len(pos), "out": len(out_), "later": len(neg)}, "verdict_vs_rule_mismatches": mism,
              "AUC_srcin_vs_later": round(auc(pos, neg), 4), "AUC_srcin_vs_srcout": round(auc(pos, out_), 4),
              "score_dist": {n: dict(collections.Counter(round(x, 2) if g == "ra" else round(x) for x in v.values()).most_common(12)) for n, v in
                             (("in", sc["in"]), ("out", sc["out"]), ("later", sc["later"]))},
              "native_rule": nat, "best_J": bj, "best_Delta": bd, "all_thresholds": rows if g == "ra" else None}
    print(g, json.dumps({k: v for k, v in res[g].items() if k not in ("all_thresholds",)}, indent=0)[:1500])
dump("r1_a1_scores.json", res)
