import csv, json, math, os, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__))
K = json.load(open(os.path.join(HERE, "key", "key.json"))); ITEMS = K["items"]

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))
def fmt(k, n):
    if n == 0: return f"{k}/0"
    lo, hi = wilson(k, n); return f"{k}/{n} = {100*k/n:.1f}% [{100*lo:.1f}, {100*hi:.1f}]"
def kappa(a, b):
    n = len(a)
    if n == 0: return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa = sum(a) / n; pb = sum(b) / n; pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)
yn = lambda s: str(s or "").strip().lower().startswith("y")

files = sys.argv[1:] or [os.path.join(HERE, f"labels_{p}.csv") for p in "ABC"]
rows = [r for fp in files if os.path.exists(fp) for r in csv.DictReader(open(fp))]
by = collections.defaultdict(list)
for r in rows:
    if r.get("item_id") in ITEMS and str(r.get("done_by_listed_paper", "")).strip():
        by[r["item_id"]].append(r)
H = {}
for iid, rs in by.items():
    v = [yn(r["done_by_listed_paper"]) for r in rs]
    done = sum(v) * 2 >= len(v)
    nums = [str(r.get("paper_number") or "").strip().lstrip("Pp") for r in rs if yn(r["done_by_listed_paper"])]
    nums = [int(x) for x in nums if x.isdigit()]
    c2 = [yn(r.get("contribution_established")) or yn(r["done_by_listed_paper"]) for r in rs]
    H[iid] = {"done": done, "papers": set(nums), "contrib": sum(c2) * 2 >= len(c2) and any(c2), "n_annot": len(rs), "votes": v}
out = {"n_items_labeled": len(H), "n_items": len(ITEMS)}
st = collections.defaultdict(lambda: [0, 0]); gp = collections.defaultdict(lambda: [0, 0])
for iid, h in H.items():
    it = ITEMS[iid]; st[it["stratum"]][0] += not h["done"]; st[it["stratum"]][1] += 1
    if it["population_group"]: gp[it["population_group"]][0] += not h["done"]; gp[it["population_group"]][1] += 1
out["confirmed_novel_by_stratum"] = {s: fmt(*v) for s, v in sorted(st.items())}
out["confirmed_novel_by_population_group"] = {g: fmt(*v) for g, v in sorted(gp.items())}
PG = K["population_groups"]
if all(gp[g][1] for g in PG):
    est = sum(PG[g] * gp[g][0] / gp[g][1] for g in PG) / sum(PG.values())
    out["weighted_confirmed_novel_rate_all_400"] = round(est, 3)
lat = [i for i in H if ITEMS[i]["set"] == "later"]
out["confirmed_novel_later_unweighted"] = fmt(sum(not H[i]["done"] for i in lat), len(lat))
out["contribution_established_rate"] = {s: fmt(sum(H[i]["contrib"] for i in H if ITEMS[i]["stratum"] == s), sum(ITEMS[i]["stratum"] == s for i in H)) for s in st}
checks = sorted({c for i in ITEMS.values() for c in i["checks"]})
fa = {}
for c in checks:
    conf = [i for i in lat if not H[i]["done"] and c in ITEMS[i]["checks"]]
    done_ = [i for i in lat if H[i]["done"] and c in ITEMS[i]["checks"]]
    rej_done = [i for i in done_ if ITEMS[i]["checks"][c]["rej"]]
    same = 0
    for i in rej_done:
        pn = ITEMS[i]["checks"][c].get("paper_num")
        pn = pn if isinstance(pn, list) else [pn]
        same += any(isinstance(x, int) and x in H[i]["papers"] for x in pn)
    fa[c] = {"FA_on_confirmed_novel": fmt(sum(ITEMS[i]["checks"][c]["rej"] for i in conf), len(conf)),
             "rejections_on_human_done": f"{len(rej_done)}/{len(done_)}", "of_which_same_paper_as_humans": same,
             "FA_on_confirmed_novel_uniform_stratum_only": fmt(sum(ITEMS[i]["checks"][c]["rej"] for i in conf if ITEMS[i]["stratum"] == "later_uniform"),
                                                                 sum(ITEMS[i]["stratum"] == "later_uniform" for i in conf))}
out["checks_on_later"] = fa
e12 = [i for i in H if ITEMS[i]["set"] == "e12"]
out["e12"] = {"human_done": fmt(sum(H[i]["done"] for i in e12), len(e12)),
              "human_done_same_paper_as_sorter": sum(H[i]["done"] and ITEMS[i]["opus_sorter"]["match_num"] in H[i]["papers"] for i in e12),
              "check_catch_on_human_done": {c: fmt(sum(ITEMS[i]["checks"][c]["rej"] for i in e12 if H[i]["done"] and c in ITEMS[i]["checks"]),
                                                   sum(H[i]["done"] and c in ITEMS[i]["checks"] for i in e12)) for c in sorted({c for i in e12 for c in ITEMS[i]["checks"]})},
              "check_rej_on_human_novel": {c: fmt(sum(ITEMS[i]["checks"][c]["rej"] for i in e12 if not H[i]["done"] and c in ITEMS[i]["checks"]),
                                                  sum((not H[i]["done"]) and c in ITEMS[i]["checks"] for i in e12)) for c in sorted({c for i in e12 for c in ITEMS[i]["checks"]})}}
hum = [H[i]["done"] for i in H]
sorter = [ITEMS[i]["opus_sorter"]["bucket"] == "a" for i in H]
out["agreement"] = {
    "opus_claim_labeler_(later items labeled novel)": fmt(sum(not H[i]["done"] for i in lat), len(lat)),
    "opus_whole_idea_sorter_(bucket a = done)": {"agree": fmt(sum(a == b for a, b in zip(hum, sorter)), len(hum)), "kappa": kappa(hum, sorter)},
}
g5 = [(H[i]["done"], bool(ITEMS[i]["gpt5"]["pool_contained"])) for i in H if ITEMS[i].get("gpt5") and ITEMS[i]["gpt5"].get("pool_contained") is not None]
if g5:
    out["agreement"]["gpt5_pool_contained"] = {"agree": fmt(sum(a == b for a, b in g5), len(g5)), "kappa": kappa([a for a, _ in g5], [b for _, b in g5])}
pairs = [(h["votes"][0], h["votes"][1]) for h in H.values() if len(h["votes"]) >= 2]
if pairs:
    out["inter_annotator"] = {"n": len(pairs), "agree": fmt(sum(a == b for a, b in pairs), len(pairs)), "kappa": kappa([a for a, _ in pairs], [b for _, b in pairs])}
json.dump(out, open(os.path.join(HERE, "audit_scores.json"), "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
