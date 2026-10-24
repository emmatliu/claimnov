import json, os, sys, threading, traceback
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import llm, gates, panel
from harness.sandbox import Pool
from harness.module_gate import module_gate, _ensure_key
from run_positive import MATCH_PROMPT
_ensure_key()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/results/smuggle_full.jsonl"
HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
llm.PRICES[HAIKU] = (1.00, 5.00)
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"

done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        r = json.loads(l)
        done.add((r["item"], r["gate"]))
out = open(OUT, "a")
_wlock = threading.Lock()

def veto(model, itxt, pool, tag):
    for p in pool.search(itxt, k=8):
        ans = llm.claude(model, [{"role": "user", "content": MATCH_PROMPT.format(
            idea=itxt, ptitle=p["title"], pabstract=p["abstract"])}],
            max_tokens=1500, cache_tag=tag)
        if ans.strip().lower().startswith("yes"):
            return {"novel": False, "evidence": p["title"][:80], "hit_B": bool(p.get("is_B"))}
    return {"novel": True, "evidence": None, "hit_B": False}

dz = json.load(open(f"{HERE}/data/disguises_v2.json"))
units = []
for d in dz:
    pd = json.load(open(f"{HERE}/data/pools/seed{d['seed_idx']}.json"))
    pool_in = pd["pool"]
    pool_out = [p for p in pd["pool"] if not p.get("is_B")]
    for cond, recs in (("B_in", pool_in), ("B_out", pool_out)):
        units.append((f"seed{d['seed_idx']}_r{d['rung']}_{cond}", d, cond, recs))
if SMOKE:
    units = [u for u in units if u[0] == "seed0_r3_B_in"]
print(f"{len(units)} units x 10 gates", flush=True)

def run_unit(u):
    try:
        key, d, cond, recs = u
        idea = {"title": d["title"], "description": d["description"]}
        itxt = f"{d['title']}: {d['description']}"
        pool = Pool(recs)
        t = f"sm_{key}"
        cells = [("v1", lambda: gates.v1_gate(idea, pool, tag=t)),
                 ("coi", lambda: gates.coi_gate(idea, pool, tag=t)),
                 ("si", lambda: gates.si_gate(idea, pool, tag=t)),
                 ("arc", lambda: gates.arc_gate(idea, pool, tag=t)),
                 ("module", lambda: module_gate(idea, recs)),
                 ("ra", lambda: panel.ra_gate(idea, pool, tag=f"_{t}")),
                 ("scholareval", lambda: panel.scholareval_gate(idea, pool, tag=f"_{t}")),
                 ("v2_reflection", lambda: panel.v2_reflection_gate(idea, pool, tag=f"_{t}")),
                 ("veto_sonnet", lambda: veto(llm.SONNET_46, itxt, pool, f"vs_{t}")),
                 ("veto_haiku", lambda: veto(HAIKU, itxt, pool, f"vh_{t}"))]
        for name, fn in cells:
            if (key, name) in done:
                continue
            try:
                res = fn()
            except Exception as e:
                res = {"novel": None, "error": str(e)[:150]}
            rec = {"item": key, "gate": name, "rung": d["rung"], "cond": cond,
                   **{k: v for k, v in res.items() if k != "trace"}}
            with _wlock:
                out.write(json.dumps(rec) + "\n"); out.flush()
        return key
    except Exception as e:
        print(f"UNIT {u[0]} failed: {str(e)[:100]}", flush=True)
        return u[0]

with ThreadPoolExecutor(max_workers=3) as ex:
    for k in ex.map(run_unit, units):
        pass

rows = [json.loads(l) for l in open(OUT)]
from collections import defaultdict
agg = defaultdict(lambda: [0, 0])
for r in rows:
    if r.get("novel") is not None:
        k = (r["gate"], r["rung"], r["cond"])
        agg[k][1] += 1
        agg[k][0] += (r["novel"] is False)
print("\n=== catch by gate x rung (B_in | B_out) ===")
gs = sorted({g for g, _, _ in agg})
for g in gs:
    line = f"{g:>14}:"
    for rung in (1, 2, 3, 4):
        ci, ni = agg[(g, rung, "B_in")]
        co, no = agg[(g, rung, "B_out")]
        line += f"  r{rung} {ci}/{ni}|{co}/{no}"
    print(line)
print("cost:", llm.cost())
