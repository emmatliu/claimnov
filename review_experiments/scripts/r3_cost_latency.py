import json, os
import numpy as np
OUT = "review_experiments/out"
L = lambda fp: [json.loads(l) for l in open(fp)] if os.path.exists(fp) else []
def summ(rows):
    m = [r["meta"] for r in rows if not r.get("error") and r.get("meta")]
    if not m: return None
    i = np.array([x["in"] for x in m]); o = np.array([x["out"] for x in m]); lat = np.array([x["latency_s"] for x in m])
    usd = i / 1e6 * 3 + o / 1e6 * 15
    return {"n_calls": len(m), "in_tokens_mean": round(float(i.mean())), "out_tokens_mean": round(float(o.mean())), "usd_per_call_mean": round(float(usd.mean()), 4),
            "latency_s_median": round(float(np.median(lat)), 2), "latency_s_p90": round(float(np.percentile(lat, 90)), 2), "latency_s_mean": round(float(lat.mean()), 2)}
res = {}
ctx = L(f"{OUT}/r3_ctx.jsonl")
res["ClaimNov+ctx verdict, all sets"] = summ(ctx)
for s in ("disg", "later", "claim", "hyb", "e12"):
    res[f"ClaimNov+ctx verdict, {s}"] = summ([r for r in ctx if r.get("set") == s])
res["ClaimNov shipped verdict (D3 repeats, exact later+disguise prompts)"] = summ(L(f"{OUT}/r3_rep1.jsonl") + L(f"{OUT}/r3_rep2.jsonl"))
res["CoI-Agent final check call (D3)"] = summ(L(f"{OUT}/r3_coi_rerun.jsonl"))
json.dump(res, open(f"{OUT}/r3_cost_latency.json", "w"), indent=1)
print(json.dumps(res, indent=1))
