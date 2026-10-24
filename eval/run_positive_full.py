import json, os, sys, threading, traceback
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import llm, gates, panel
from harness.sandbox import Pool
from harness.module_gate import module_gate, _ensure_key
from run_positive import MATCH_PROMPT
_ensure_key()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/results/positive_full.jsonl"
ANCHOR_ROOTS = [f"{HERE}/data/anchors", f"{HERE}/data/anchors_full"]

done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        r = json.loads(l)
        done.add((r["item"], r["gate"]))
out = open(OUT, "a")
_wlock = threading.Lock()

def veto_sonnet(itxt, pool, tag):
    for p in pool.search(itxt, k=8):
        ans = llm.claude(llm.SONNET_46, [{"role": "user", "content": MATCH_PROMPT.format(
            idea=itxt, ptitle=p["title"], pabstract=p["abstract"])}],
            max_tokens=1500, cache_tag=tag)
        if ans.strip().lower().startswith("yes"):
            return {"novel": False, "evidence": p["title"][:80]}
    return {"novel": True, "evidence": None}

units = []
for root in ANCHOR_ROOTS:
    for slug in sorted(os.listdir(root)):
        adir = f"{root}/{slug}"
        if not os.path.isdir(adir) or slug == "cut" or not os.path.exists(f"{adir}/ideas.json"):
            continue
        units.append((slug, adir))
print(f"{len(units)} anchors", flush=True)

def run_anchor(args):
    slug, adir = args
    ideas = json.load(open(f"{adir}/ideas.json"))
    pool_recs = json.load(open(f"{adir}/pool_preT.json"))[:400]
    pool = Pool(pool_recs)

    def run_idea(idea):
        key = f"{slug}::{idea['idx']}"
        itxt = f"{idea['title']}: {idea['description']}"
        io = {"title": idea["title"], "description": idea["description"]}
        cells = [("v1", lambda: gates.v1_gate(io, pool, tag=f"pf_{slug}_{idea['idx']}")),
                 ("coi", lambda: gates.coi_gate(io, pool, tag=f"pf_{slug}_{idea['idx']}")),
                 ("si", lambda: gates.si_gate(io, pool, tag=f"pf_{slug}_{idea['idx']}")),
                 ("arc", lambda: gates.arc_gate(io, pool, tag=f"pf_{slug}_{idea['idx']}")),
                 ("module", lambda: module_gate(io, pool_recs)),
                 ("ra", lambda: panel.ra_gate(io, pool, tag=f"_pf_{slug}_{idea['idx']}")),
                 ("scholareval", lambda: panel.scholareval_gate(io, pool, tag=f"_pf_{slug}_{idea['idx']}")),
                 ("v2_reflection", lambda: panel.v2_reflection_gate(io, pool, tag=f"_pf_{slug}_{idea['idx']}")),
                 ("veto_sonnet", lambda: veto_sonnet(itxt, pool, f"vs_pf_{slug}_{idea['idx']}"))]
        skip = set(os.environ.get("SKIP_GATES", "").split(","))
        for name, fn in cells:
            if (key, name) in done or name in skip:
                continue
            try:
                res = fn()
            except Exception as e:
                res = {"novel": None, "error": str(e)[:200]}
            rec = {"item": key, "gate": name,
                   **{k: v for k, v in res.items() if k != "trace"}}
            with _wlock:
                out.write(json.dumps(rec) + "\n")
                out.flush()

    with ThreadPoolExecutor(max_workers=6) as inner:
        list(inner.map(run_idea, ideas))
    return slug

with ThreadPoolExecutor(max_workers=3) as ex:
    for s in ex.map(run_anchor, units):
        print(f"anchor done: {s}", flush=True)

rows = [json.loads(l) for l in open(OUT)]
from collections import Counter
errs = Counter(r["gate"] for r in rows if r.get("novel") is None)
n = len({r["item"] for r in rows})
print(f"\ncells {len(rows)}, ideas touched {n}/3000, errors {dict(errs)}")
print("cost:", llm.cost())
