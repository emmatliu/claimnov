import json, os, sys, threading, traceback
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for line in open(f"{ROOT}/.env"):
    if "=" in line and not line.startswith("#"):
        k, v = line.strip().split("=", 1); os.environ.setdefault(k, v)
sys.path.insert(0, f"{ROOT}/eval"); sys.path.insert(0, f"{ROOT}/src"); os.chdir(f"{ROOT}/eval")
from harness import llm, gates, panel
from harness.sandbox import Pool
from harness.module_gate import module_gate
from novelty import pipeline, retrieve, gate
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("NOVELTY_OUT", f"{HERE}/results/smuggle_v2.jsonl")
done = set()
for _f in {OUT, f"{HERE}/results/smuggle_v2.jsonl", f"{HERE}/results/smuggle_v2_api.jsonl"}:
    if os.path.exists(_f):
        for l in open(_f):
            r = json.loads(l); done.add((r["item"], r["gate"]))
out = open(OUT, "a"); lock = threading.Lock()
def chat(tag):
    if os.environ.get("NOVELTY_BACKEND", "api") == "bedrock":
        return lambda pr: llm.claude(llm.SONNET_46, [{"role": "user", "content": pr}], max_tokens=1500, cache_tag=tag)
    return lambda pr: llm.claude_api("claude-sonnet-4-6", [{"role": "user", "content": pr}], max_tokens=1500, cache_tag=tag, temperature=0.0)
dz_all = [d for d in json.load(open(f"{HERE}/data/disguises_v3.json")) if d["seed_idx"] >= 10]
ok = {}
for d in dz_all:
    ok.setdefault(d["seed_idx"], set()).add(d["rung"]) if (d["rung"] <= 2 or d.get("verified_same")) else None
complete = sorted(s for s, r in ok.items() if r == {1, 2, 3, 4})
chosen = set(complete[:90])
dz = [d for d in dz_all if d["seed_idx"] in chosen]
print(f"sources with all four rungs verified: {len(complete)}; using {len(chosen)} -> {len(dz)} disguises", flush=True)
units = []
for d in dz:
    fp = f"{HERE}/data/pools/seed{d['seed_idx']}.json"
    if not os.path.exists(fp): continue
    pd = json.load(open(fp)); pool_in = pd["pool"]; pool_out = [p for p in pool_in if not p.get("is_B")]
    for cond, recs in (("B_in", pool_in), ("B_out", pool_out)):
        units.append((f"seed{d['seed_idx']}_r{d['rung']}_{cond}", d, cond, recs))
if os.environ.get("NOVELTY_REVERSE"): units.reverse()
print(f"{len(units)} units", flush=True)
def run_unit(u):
    key, d, cond, recs = u
    idea = {"title": d["title"], "description": d["description"]}
    itxt = f"{d['title']}: {d['description']}"; t = f"sm2_{key}"
    pool = Pool(recs); B = next((p for p in recs if p.get("is_B")), None)
    def claimnov(mode):
        target = {"id": key, "title": d["title"], "abstract": d["description"]}
        fac = retrieve.build_pool([{"id": p["id"], "title": p["title"], "abstract": p["abstract"]} for p in recs])
        rep = pipeline.score_paper(target, fac)
        if mode == "profile":
            res = gate.profile_veto(target, recs, rep, chat=chat(f"vsq_{t}"))
        else:
            top8 = [p["id"] for p in pool.search(itxt, k=8)]
            prof = [c["judged"][0]["paper_id"] for c in rep["claims"] if c.get("judged")]
            res = gate.candidate_central_veto(target, recs, rep, fac, top8 + [p for p in prof if p not in top8], chat=chat(f"vsz_{t}"))
        ev = res.get("evidence") or {}
        return {"novel": not res["veto"], "evidence": ev.get("title"), "hit_B": bool(B and ev.get("id") == B["id"]),
                "B_in_profile": bool(B and B["id"] in (res.get("candidates") or []))}
    cells = [("v1", lambda: gates.v1_gate(idea, pool, tag=t)),
             ("coi", lambda: gates.coi_gate(idea, pool, tag=t)),
             ("si", lambda: gates.si_gate(idea, pool, tag=t)),
             ("arc", lambda: gates.arc_gate(idea, pool, tag=t)),
             ("ra", lambda: panel.ra_gate(idea, pool, tag=f"_{t}")),
             ("scholareval", lambda: panel.scholareval_gate(idea, pool, tag=f"_{t}")),
             ("v2_reflection", lambda: panel.v2_reflection_gate(idea, pool, tag=f"_{t}")),
             ("claimnov", lambda: claimnov("profile")),
             ("graded_gate", lambda: module_gate(idea, recs)),
             ("central_rule", lambda: claimnov("central"))]
    for name, fn in cells:
        if (key, name) in done: continue
        try:
            res = fn()
        except Exception as e:
            res = {"novel": None, "error": str(e)[:200]}
        rec = {"item": key, "gate": name, "rung": d["rung"], "cond": cond, **{k: v for k, v in res.items() if k != "trace"}}
        with lock:
            out.write(json.dumps(rec) + "\n"); out.flush()
with ThreadPoolExecutor(max_workers=int(os.environ.get("NOVELTY_WORKERS", "6"))) as ex:
    list(ex.map(run_unit, units))
print("DONE smuggle_v2 cost", llm.cost(), flush=True)
