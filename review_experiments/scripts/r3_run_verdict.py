import json, os, sys, re, threading, random
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3_common as R
from r0_common import PILOT, anchor_dir
MODE = sys.argv[1]; SETS = sys.argv[2:] or ["disg", "later", "claim", "hyb", "e12"]
INS = "\n\nReturn JSON only:"
OUTF = f"{R.OUT}/r3_{MODE}.jsonl"
done = {(json.loads(l)["set"], json.loads(l)["key"]) for l in open(OUTF) if not json.loads(l).get("error")} if os.path.exists(OUTF) else set()
P = [json.loads(l) for l in open(f"{R.OUT}/r3_prompts.jsonl")]
U = [p for p in P if p["set"] in SETS and p.get("prompt") and (p["set"], p["key"]) not in done]
if MODE.startswith("rep"):
    U = [p for p in U if p["src"] == "exact" and p["set"] in ("later", "disg")]
if os.environ.get("ORDER") == "shuffle":
    random.Random(0).shuffle(U)
U = U[:int(os.environ.get("LIMIT", 10**9))]
_ap = {}
def corpus(p):
    if p["set"] in ("later", "e12", "hyb"):
        slug = p["key"].split("::")[0]
        if slug not in _ap: _ap[slug] = json.load(open(f"{anchor_dir(slug)}/pool_preT.json"))[:400]
        return _ap[slug], None
    if p["set"] == "claim":
        recs = json.load(open(f"{PILOT}/data/pools_full/{p['key']}.json"))["pool"]
        return recs, next((x["id"] for x in recs if x.get("is_B")), None)
    s = int(re.match(r"seed(\d+)_", p["key"]).group(1)); pin = json.load(open(f"{PILOT}/data/pools/seed{s}.json"))["pool"]
    B = next(x["id"] for x in pin if x.get("is_B"))
    return (pin, B) if p["key"].endswith("B_in") else ([x for x in pin if not x.get("is_B")], None)
HYB = {h["item"]: h["B_id"] for h in json.load(open(f"{PILOT}/data/hybrids.json"))}
out = open(OUTF, "a"); lock = threading.Lock()

def run(p):
    recs, B = corpus(p)
    if p["set"] == "hyb": B = HYB[p["key"]]
    if MODE == "ctx":
        assert p["prompt"].count(INS) == 1
        pr = p["prompt"].replace(INS, "\n\nNEAREST PRIOR PAPERS (title and abstract):\n" + p["top8_block"] + INS)
        tag = f"r3ctx_{p['set']}_{p['key']}"
    else:
        pr = p["prompt"]; tag = f"r3d3{MODE}_{p['tag']}"
    try:
        txt, meta = R.sonnet(pr, 1500, tag)
        v = R.parse_verdict(txt, recs)
        rec = {"set": p["set"], "key": p["key"], "src": p["src"], "profile_k": p.get("profile_k"), "B_id": B,
               "names_B": bool(B and v["anticipating_id"] == B), "in_profile_ids": None, **v,
               "anticipating_in_top8": bool(v["anticipating_id"] and v["anticipating_id"] in p["shown_ids"]),
               "text": txt, "meta": meta}
        if MODE.startswith("rep"):
            rec["identical_to_paper_text"] = (txt == p.get("paper_verdict_text"))
    except R.Budget as e:
        return
    except Exception as e:
        rec = {"set": p["set"], "key": p["key"], "error": str(e)[:300]}
    with lock:
        out.write(json.dumps(rec) + "\n"); out.flush()

print(MODE, len(U), "to run; prior spend", round(R._PRIOR, 2), flush=True)
try:
    with ThreadPoolExecutor(max_workers=int(os.environ.get("WORKERS", "6"))) as ex:
        list(ex.map(run, U))
finally:
    R.log_cost(f"r3_run_verdict {MODE} {' '.join(SETS)}")
    print("DONE", MODE, R.cost, flush=True)
