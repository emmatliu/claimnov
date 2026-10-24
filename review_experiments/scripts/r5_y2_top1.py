import json, os, sys, re
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("R5_EXP", "y2")
import r5_common as R
from rank_bm25 import BM25Okapi
PILOT = "./eval"
STOP = set("""a an the and or of to in on for with by from as at is are was were be been being this that these those it its
we our us their they them which who whom whose what when where how than then there here such into onto over under
can could may might will would should shall do does did not no nor but if so also both each either neither via per
more most less least very much many few new novel propose proposed proposes approach method methods paper work using use
used based show shows shown""".split())
toks = lambda s: [t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(t) > 1 and t not in STOP]
PR = open(f"{R.EXP}/prompts/r5_top1_pairwise.txt").read()
DROP = {60, 83, 86, 87, 93}
EXACT = {(r["set"], r["key"]) for r in map(json.loads, open(f"{R.OUT}/r3_prompts.jsonl")) if r["src"] == "exact"}
def adir(s): return next(p for p in (f"{PILOT}/data/anchors/{s}", f"{PILOT}/data/anchors_full/{s}") if os.path.exists(f"{p}/ideas.json"))
_ap = {}
def apool(s):
    if s not in _ap: _ap[s] = json.load(open(f"{adir(s)}/pool_preT.json"))[:400]
    return _ap[s]
def units():
    U = []
    for d in json.load(open(f"{PILOT}/data/disguises_v3.json")):
        for cond in ("B_in", "B_out"):
            k = f"seed{d['seed_idx']}_r{d['rung']}_{cond}"
            U.append(dict(set="disg_" + cond[2:], key=k, cluster=d["seed_idx"], exact=("disg", k) in EXACT, seed=d["seed_idx"], cond=cond,
                          idea=f"{d['title']}: {d['description']}"))
    ideas = {}
    for it in json.load(open(f"{PILOT}/data/later_realized_set.json")):
        s, i = it.split("::")
        if s not in ideas: ideas[s] = {x["idx"]: x for x in json.load(open(f"{adir(s)}/ideas.json"))}
        x = ideas[s][int(i)]
        U.append(dict(set="later", key=it, cluster=s, exact=("later", it) in EXACT, anchor=s, idea=f"{x['title']}: {x['description']}"))
    for h in json.load(open(f"{PILOT}/data/hybrids.json")):
        U.append(dict(set="hyb", key=h["item"], cluster=h["anchor"], exact=True, anchor=h["anchor"], B=h["B_id"], idea=f"{h['title']}: {h['description']}"))
    return U
KEEP = json.load(open(f"{R.OUT}/r0_items.json"))["items"]["claimnov"]
def pool_of(u):
    if u["set"].startswith("disg"):
        p = json.load(open(f"{PILOT}/data/pools/seed{u['seed']}.json"))["pool"]
        u["B"] = next(x["id"] for x in p if x.get("is_B"))
        return p if u["cond"] == "B_in" else [x for x in p if not x.get("is_B")]
    return apool(u["anchor"])
OUTF = f"{R.OUT}/r5_y2_top1.jsonl"
done = {(r["set"], r["key"]) for r in map(json.loads, open(OUTF))} if os.path.exists(OUTF) else set()
def run(u):
    if (u["set"], u["key"]) in done: return None
    pool = pool_of(u)
    sc = BM25Okapi([toks(f"{x['title']}. {x.get('abstract') or ''}") for x in pool]).get_scores(toks(u["idea"]))
    top = pool[max(range(len(pool)), key=lambda i: sc[i])]
    try:
        txt, meta = R.call("sonnet", PR.format(idea=u["idea"], title=top["title"], abstract=top.get("abstract") or ""), 600, f"r5y2_{u['set']}_{u['key']}")
    except R.Budget: return None
    try: j = json.loads(re.search(r"\{.*\}", txt, re.S).group(0)); pe = False
    except Exception: j = {"decision": "novel"}; pe = True
    rej = str(j.get("decision", "")).lower().startswith("not")
    return {"set": u["set"], "key": u["key"], "cluster": u["cluster"], "exact": u["exact"], "rej": rej, "parse_error": pe,
            "top1_id": top["id"], "top1_title": top["title"], "top1_is_B": top["id"] == u.get("B"), "n_pool": len(pool),
            "why": j.get("why"), "meta": {k: meta.get(k) for k in ("endpoint", "in", "out", "usd", "latency_s")}}
if __name__ == "__main__":
    full = os.environ.get("SET") == "full"
    U = [u for u in units() if (u["exact"] or full) and not (u["set"].startswith("disg") and u["seed"] in DROP)
         and (u["set"] != "disg_out" or True)]
    U = [u for u in U if (u["set"] != "later" or u["key"] in KEEP["later"]) and (not u["set"].startswith("disg") or u["key"].replace("B_out", "B_in") in KEEP["disg_in"])]
    U.sort(key=lambda u: not u["exact"])
    print(len(U), {s: sum(1 for u in U if u["set"] == s) for s in ("disg_in", "disg_out", "later", "hyb")}, flush=True)
    with open(OUTF, "a") as fo, ThreadPoolExecutor(int(os.environ.get("WORKERS", "6"))) as ex:
        for r in ex.map(run, U):
            if r: fo.write(json.dumps(r) + "\n"); fo.flush()
    print(R.summary(), flush=True)
