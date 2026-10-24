import json, os, sys, threading, traceback
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import llm, gates, panel
from harness.sandbox import Pool
from harness.module_gate import module_gate, _ensure_key
from run_positive import MATCH_PROMPT
from make_disguises import R1, parse
_ensure_key()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/results/negative_full.jsonl"
HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
llm.PRICES[HAIKU] = (1.00, 5.00)
VETO_TIERS = {"veto_haiku": HAIKU, "veto_sonnet": llm.SONNET_46, "veto_opus": llm.OPUS_46}

done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        r = json.loads(l)
        done.add((r["item"], r["gate"]))
out = open(OUT, "a")
_wlock = threading.Lock()

idx = json.load(open(f"{HERE}/data/n2_full_index.json"))
pairs = [json.loads(l) for l in
         open("eval/data/claim_disputed/testset_clean.jsonl")]
items = sorted(int(k) for k, v in idx.items() if v["status"] == "ok")
print(f"{len(items)} pairs; {len(done)} cells already done", flush=True)


def veto(model, itxt, pool, tag):
    for p in pool.search(itxt, k=8):
        ans = llm.claude(model, [{"role": "user", "content": MATCH_PROMPT.format(
            idea=itxt, ptitle=p["title"], pabstract=p["abstract"])}],
            max_tokens=1500, cache_tag=tag)
        if ans.strip().lower().startswith("yes"):
            return {"novel": False, "evidence": p["title"][:80], "hit_B": bool(p.get("is_B"))}
    return {"novel": True, "evidence": None, "hit_B": False}


def run_pair(i):
    try:
        return _run_pair(i)
    except Exception as e:
        print(f"PAIR {i} failed (will rerun on resume): {str(e)[:100]}", flush=True)
        return i


def _run_pair(i):
    r = pairs[i]
    pd = json.load(open(f"{HERE}/data/pools_full/n2f_{i}.json"))
    d = parse(llm.claude(llm.SONNET_46, [{"role": "user", "content":
        R1.format(abstract=r["A_abstract"])}], max_tokens=3000, cache_tag=f"n2f_ideaify{i}"))
    idea = {"title": d["title"], "description": d["description"]}
    itxt = f"{d['title']}: {d['description']}"
    pool = Pool(pd["pool"])
    key = f"n2f_{i}"
    cells = [("v1", lambda: gates.v1_gate(idea, pool, tag=f"nf{i}")),
             ("coi", lambda: gates.coi_gate(idea, pool, tag=f"nf{i}")),
             ("si", lambda: gates.si_gate(idea, pool, tag=f"nf{i}")),
             ("arc", lambda: gates.arc_gate(idea, pool, tag=f"nf{i}")),
             ("module", lambda: module_gate(idea, pd["pool"])),
             ("ra", lambda: panel.ra_gate(idea, pool, tag=f"_nf{i}")),
             ("scholareval", lambda: panel.scholareval_gate(idea, pool, tag=f"_nf{i}")),
             ("v2_reflection", lambda: panel.v2_reflection_gate(idea, pool, tag=f"_nf{i}"))]
    cells += [(name, (lambda m=model, t=f"{name[5:]}_nf{i}": veto(m, itxt, pool, t)))
              for name, model in VETO_TIERS.items()]
    skip = set(os.environ.get("SKIP_GATES", "").split(","))
    for name, fn in cells:
        if (key, name) in done or name in skip:
            continue
        try:
            res = fn()
        except Exception as e:
            res = {"novel": None, "error": str(e)[:200]}
            traceback.print_exc()
        rec = {"item": key, "gate": name, "dimension": r["novelty_dimension"],
               "B_provenance": idx[str(i)].get("source", ""),
               **{k: v for k, v in res.items() if k != "trace"},
               "trace": res.get("trace")}
        with _wlock:
            out.write(json.dumps(rec) + "\n")
            out.flush()
    return i


with ThreadPoolExecutor(max_workers=2) as ex:
    for n, i in enumerate(ex.map(run_pair, items)):
        if (n + 1) % 10 == 0:
            print(f"progress: {n+1}/{len(items)} pairs | cost {llm.cost()}", flush=True)

rows = [json.loads(l) for l in open(OUT)]
from collections import Counter, defaultdict
by = defaultdict(list)
for r in rows:
    by[r["gate"]].append(r)
print("\n=== FULL NEGATIVE ARM (N2, n=%d) ===" % len(items))
for g in sorted(by):
    rs = {r["item"]: r for r in by[g]}.values()
    catch = sum(1 for r in rs if r.get("novel") is False)
    hitb = sum(1 for r in rs if r.get("hit_B"))
    err = sum(1 for r in rs if r.get("novel") is None)
    print(f"{g:>14}: catch {catch}/{len(rs)}" + (f" (named B: {hitb})" if hitb else "") +
          (f" errors {err}" if err else ""))
print("cost:", llm.cost())
