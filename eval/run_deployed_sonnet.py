import json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for line in open(f"{ROOT}/.env"):
    if "=" in line and not line.startswith("#"):
        k, v = line.strip().split("=", 1); os.environ.setdefault(k, v)
sys.path.insert(0, f"{ROOT}/eval"); os.chdir(f"{ROOT}/eval")
from harness import llm, gates, panel
from harness.sandbox import Pool
HERE = os.path.dirname(os.path.abspath(__file__))

SONNET = llm.SONNET_46
_orig_chat = llm.chat
def chat(model, messages, max_tokens=2000, temperature=0.0, cache_tag=""):
    if model != SONNET:
        return _orig_chat(model, messages, max_tokens=max_tokens, temperature=temperature, cache_tag=cache_tag)
    system = "\n".join(m["content"] for m in messages if m["role"] == "system") or None
    msgs = [m for m in messages if m["role"] != "system"]
    if os.environ.get("NOVELTY_BACKEND") == "bedrock":
        os.environ["BEDROCK_TEMP0"] = "1"
        return llm.claude(SONNET, msgs, max_tokens=max_tokens, system=system, cache_tag=cache_tag)
    return llm.claude_api("claude-sonnet-4-6", msgs, max_tokens=max_tokens, system=system, cache_tag=cache_tag, temperature=0.0)
llm.chat = chat; gates.llm.chat = chat; panel.llm.chat = chat
gates.JUDGE = SONNET

OUT = f"{HERE}/results/" + os.environ.get("NOVELTY_OUT_NAME", "deployed_sonnet.jsonl")
done = set()
for _f in {OUT, f"{HERE}/results/deployed_sonnet.jsonl", f"{HERE}/results/deployed_sonnet_panel.jsonl",
           f"{HERE}/results/deployed_sonnet_bedrock.jsonl", f"{HERE}/results/deployed_sonnet_panel_bedrock.jsonl",
           f"{HERE}/results/deployed_sonnet_rev.jsonl", f"{HERE}/results/deployed_sonnet_panel_rev.jsonl"}:
    if os.path.exists(_f):
        for l in open(_f):
            r = json.loads(l)
            if r.get("novel") is not None: done.add((r["item"], r["gate"]))
out = open(OUT, "a"); lock = threading.Lock()

units = []
_all = json.load(open(f"{HERE}/data/disguises_v3.json"))
if os.environ.get("NOVELTY_FULL"):
    ok = {}
    for d in _all:
        if d["seed_idx"] >= 10 and (d["rung"] <= 2 or d.get("verified_same")): ok.setdefault(d["seed_idx"], set()).add(d["rung"])
    chosen = set(range(10)) | set(sorted(s for s, rr in ok.items() if rr == {1, 2, 3, 4})[:90])
    dz = [d for d in _all if d["seed_idx"] in chosen]
else:
    dz = [d for d in _all if 10 <= d["seed_idx"] < 22]
for d in dz:
    fp = f"{HERE}/data/pools/seed{d['seed_idx']}.json"
    if not os.path.exists(fp): continue
    pool_in = json.load(open(fp))["pool"]; pool_out = [p for p in pool_in if not p.get("is_B")]
    for cond, recs in (("B_in", pool_in), ("B_out", pool_out)):
        units.append((f"seed{d['seed_idx']}_r{d['rung']}_{cond}", {"title": d["title"], "description": d["description"]}, recs, cond, d["rung"]))
SET = sorted(json.load(open(f"{HERE}/data/later_realized_set.json")))
SET = SET if os.environ.get("NOVELTY_FULL") else SET[:50]
pools = {}
for item in SET:
    slug, idx = item.split("::"); idx = int(idx)
    adir = next(p for p in (f"{HERE}/data/anchors/{slug}", f"{HERE}/data/anchors_full/{slug}") if os.path.exists(f"{p}/ideas.json"))
    if slug not in pools:
        pools[slug] = json.load(open(f"{adir}/pool_preT.json"))[:400]
    idea = next(i for i in json.load(open(f"{adir}/ideas.json")) if i["idx"] == idx)
    units.append((item, {"title": idea["title"], "description": idea["description"]}, pools[slug], "set", None))
if os.environ.get("NOVELTY_REVERSE"): units.reverse()
print(f"{len(units)} units", flush=True)

def run_unit(u):
    key, idea, recs, cond, rung = u
    pool = Pool(recs); t = f"sn_{key}"
    if os.environ.get("NOVELTY_PANEL"):
        cells = [("ra", lambda: panel.ra_gate(idea, pool, tag=f"_{t}", model=SONNET)),
                 ("scholareval", lambda: panel.scholareval_gate(idea, pool, tag=f"_{t}", model=SONNET)),
                 ("v2_reflection", lambda: panel.v2_reflection_gate(idea, pool, tag=f"_{t}", model=SONNET))]
    else:
        cells = [("v1", lambda: gates.v1_gate(idea, pool, tag=t)),
                 ("coi", lambda: gates.coi_gate(idea, pool, tag=t)),
                 ("si", lambda: gates.si_gate(idea, pool, tag=t))]
    for name, fn in cells:
        if (key, name) in done: continue
        try: res = fn()
        except Exception as e: res = {"novel": None, "error": str(e)[:200]}
        rec = {"item": key, "gate": name, "cond": cond, "rung": rung, "judge": "sonnet-4.6-api",
               **{k: v for k, v in res.items() if k != "trace"}}
        with lock: out.write(json.dumps(rec) + "\n"); out.flush()
with ThreadPoolExecutor(max_workers=int(os.environ.get("NOVELTY_WORKERS", "4"))) as ex:
    list(ex.map(run_unit, units))
print("done; cost", llm.cost(), flush=True)
