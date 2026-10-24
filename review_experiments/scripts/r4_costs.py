import json, collections
R = [json.loads(l) for l in open("review_experiments/out/r4_calls.jsonl")]
by = collections.defaultdict(lambda: {"calls": 0, "in": 0, "out": 0, "usd": 0.0, "guard": 0.0, "eps": collections.Counter()})
for r in R:
    b = by[(r["exp"], r["key"])]; b["calls"] += 1; b["in"] += r["in"]; b["out"] += r["out"]; b["usd"] += r["usd"]; b["guard"] += r["usd_guard"]; b["eps"][r["endpoint"]] += 1
tot = {"calls": len(R), "usd": round(sum(r["usd"] for r in R), 2), "guard": round(sum(r["usd_guard"] for r in R), 2)}
out = {f"{e}|{k}": {"calls": v["calls"], "in_tok": v["in"], "out_tok": v["out"], "usd_list": round(v["usd"], 3), "usd_guard": round(v["guard"], 3), "endpoints": dict(v["eps"])} for (e, k), v in sorted(by.items())}
eps = collections.Counter(r["endpoint"] for r in R if r["key"] == "sonnet")
json.dump({"total": tot, "by": out, "sonnet_endpoint_share": dict(eps)}, open("review_experiments/out/r4_costs.json", "w"), indent=1)
for k, v in out.items(): print(k, v["calls"], v["in_tok"], v["out_tok"], v["usd_list"], v["usd_guard"])
print(tot, dict(eps))
