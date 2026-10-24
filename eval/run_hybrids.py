import json, os, sys, threading, difflib, re
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for line in open(f"{ROOT}/.env"):
    if "=" in line and not line.startswith("#"):
        k, v = line.strip().split("=", 1); os.environ.setdefault(k, v)
sys.path.insert(0, f"{ROOT}/eval"); sys.path.insert(0, f"{ROOT}/src"); os.chdir(f"{ROOT}/eval")
from harness import llm, gates, panel
from harness.sandbox import Pool
from novelty import pipeline, retrieve, gate
HERE = os.path.dirname(os.path.abspath(__file__))
SONNET = llm.SONNET_46
_orig_chat = llm.chat
def chat(model, messages, max_tokens=2000, temperature=0.0, cache_tag=""):
    if model != SONNET:
        return _orig_chat(model, messages, max_tokens=max_tokens, temperature=temperature, cache_tag=cache_tag)
    system = "\n".join(m["content"] for m in messages if m["role"] == "system") or None
    msgs = [m for m in messages if m["role"] != "system"]
    return llm.claude_api("claude-sonnet-4-6", msgs, max_tokens=max_tokens, system=system, cache_tag=cache_tag, temperature=0.0)
llm.chat = chat; gates.llm.chat = chat; panel.llm.chat = chat; gates.JUDGE = SONNET
def verdict_chat(tag):
    return lambda pr: llm.claude_api("claude-sonnet-4-6", [{"role": "user", "content": pr}], max_tokens=1500, cache_tag=tag, temperature=0.0)

OUT = f"{HERE}/results/hybrids_eval.jsonl"
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        r = json.loads(l)
        if r.get("novel") is not None: done.add((r["item"], r["gate"]))
out = open(OUT, "a"); lock = threading.Lock()
H = json.load(open(f"{HERE}/data/hybrids.json"))
norm = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())[:40]
pools = {}
def run(h):
    item = h["item"]; slug = h["anchor"]
    adir = next(p for p in (f"{HERE}/data/anchors/{slug}", f"{HERE}/data/anchors_full/{slug}") if os.path.exists(f"{p}/ideas.json"))
    if slug not in pools: pools[slug] = json.load(open(f"{adir}/pool_preT.json"))[:400]
    recs = pools[slug]; pool = Pool(recs)
    idea = {"title": h["title"], "description": h["description"]}; t = f"hyb_{item}"
    def claimnov():
        target = {"id": item, "title": h["title"], "abstract": h["description"]}
        fac = retrieve.build_pool([{"id": p["id"], "title": p["title"], "abstract": p["abstract"]} for p in recs])
        rep = pipeline.score_paper(target, fac)
        res = gate.profile_veto(target, recs, rep, chat=verdict_chat(f"vsq_{t}"))
        claims = rep.get("claims", [])
        sim = [difflib.SequenceMatcher(None, c.get("claim", "").lower(), h["spliced_claim"].lower()).ratio() for c in claims]
        si = max(range(len(claims)), key=lambda i: sim[i]) if claims else None
        in_profile = any(j.get("paper_id") == h["B_id"] for c in claims for j in c.get("judged", [])[:5])
        on_spliced = si is not None and any(j.get("paper_id") == h["B_id"] for j in claims[si].get("judged", [])[:5])
        lvl = max([j.get("entailment_level", 0) for c in claims for j in c.get("judged", [])[:5] if j.get("paper_id") == h["B_id"]] or [0])
        ev = res.get("evidence") or {}
        return {"novel": not res["veto"], "evidence": ev.get("title"), "named_B": in_profile, "B_on_spliced_claim": on_spliced,
                "B_level": lvl, "verdict_names_B": ev.get("id") == h["B_id"], "n_claims": len(claims)}
    cells = [("v1", lambda: gates.v1_gate(idea, pool, tag=t)), ("coi", lambda: gates.coi_gate(idea, pool, tag=t)),
             ("si", lambda: gates.si_gate(idea, pool, tag=t)), ("arc", lambda: gates.arc_gate(idea, pool, tag=t)),
             ("ra", lambda: panel.ra_gate(idea, pool, tag=f"_{t}", model=SONNET)),
             ("scholareval", lambda: panel.scholareval_gate(idea, pool, tag=f"_{t}", model=SONNET)),
             ("v2_reflection", lambda: panel.v2_reflection_gate(idea, pool, tag=f"_{t}", model=SONNET)),
             ("claimnov", claimnov)]
    for name, fn in cells:
        if (item, name) in done: continue
        try: res = fn()
        except Exception as e: res = {"novel": None, "error": str(e)[:200]}
        if name != "claimnov" and res.get("novel") is not None and "named_B" not in res:
            ev = res.get("evidence"); evs = ev if isinstance(ev, list) else [ev]
            res["named_B"] = (not res["novel"]) and any(e and norm(e) == norm(h["B_title"]) for e in evs)
        rec = {"item": item, "gate": name, "B_title": h["B_title"], **{k: v for k, v in res.items() if k != "trace"}}
        with lock: out.write(json.dumps(rec) + "\n"); out.flush()
with ThreadPoolExecutor(max_workers=int(os.environ.get("NOVELTY_WORKERS", "6"))) as ex:
    list(ex.map(run, H))
print("done", len(H), "hybrids; cost", llm.cost(), flush=True)
