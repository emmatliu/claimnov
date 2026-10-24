import json, os, sys, itertools
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("R5_EXP", "y1review")
import r5_common as R
def format_plan_json(d):
    return "".join(f"{k}: {v}\n" for k, v in d.items() if k != "score")
def prompt(i1, i2):
    p = "You are a reviewer specialized in Natural Language Processing and Large Language Models. You are given two project summaries. One of them is accepted by a top AI conference (like ICLR or ACL) and the other one is rejected. Your task is to identify the one that has been accepted.\n"
    p += "The two project proposals are:\n\n"
    p += "paper 1:\n" + format_plan_json(i1) + "\n\n"
    p += "paper 2:\n" + format_plan_json(i2) + "\n\n"
    p += "Now decide which one is the accepted idea. Directly return a number 1 or 2 and nothing else.\n"
    return p
L = [json.loads(l) for l in open(f"{R.OUT}/r5_y1_loop.jsonl")]
by = {(r["anchor"], r["idx"], r["arm"]): r for r in L}
seeds = sorted({(r["anchor"], r["idx"]) for r in L})
jobs = []
for s in seeds:
    for a, b in (("claimnov", "none"), ("coi", "none"), ("claimnov", "coi")):
        ra, rb = by.get(s + (a,)), by.get(s + (b,))
        if not (ra and rb and ra["committed"] and rb["committed"]): continue
        ia = {"title": ra["final"]["title"], "description": ra["final"]["description"]}
        ib = {"title": rb["final"]["title"], "description": rb["final"]["description"]}
        if ia == ib: continue
        for order in ("ab", "ba"):
            jobs.append((s, a, b, order, ia, ib))
print(len(jobs), "judge calls", flush=True)
def run(j):
    s, a, b, order, ia, ib = j
    p = prompt(ia, ib) if order == "ab" else prompt(ib, ia)
    txt, meta = R.call("gptoss", p, 3000, f"r5rev_{s[0]}_{s[1]}_{a}_{b}_{order}")
    t = txt.strip(); pick = 1 if t == "1" else 2 if t == "2" else None
    win = None if pick is None else (a if (pick == 1) == (order == "ab") else b)
    return {"anchor": s[0], "idx": s[1], "a": a, "b": b, "order": order, "raw": txt[:200], "pick": pick, "winner": win,
            "meta": {k: meta.get(k) for k in ("endpoint", "in", "out", "usd")}}
with ThreadPoolExecutor(4) as ex, open(f"{R.OUT}/r5_y1_review.jsonl", "w") as fo:
    for r in ex.map(run, jobs): fo.write(json.dumps(r) + "\n")
print(R.summary())
