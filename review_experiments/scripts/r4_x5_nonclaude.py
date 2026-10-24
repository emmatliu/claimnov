import json, os, sys, re, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("R4_EXP", "x5")
import r4_common as R
PILOT = "./eval"
MK = os.environ.get("MODEL", "gptoss"); MAXTOK = 6000 if MK == "gptoss" else 1500
P = [json.loads(l) for l in open(f"{R.OUT}/r3_prompts.jsonl")]
P = [p for p in P if p["set"] in ("disg", "later") and p.get("prompt")]
R1 = {(r["set"], r["key"]): r for r in map(json.loads, open(f"{R.OUT}/r1_nonclaude_profile_{'gpt-oss-120b-1' if MK == 'gptoss' else 'meta'}.jsonl"))}
_ap = {}
def corpus(p):
    if p["set"] == "later":
        s = p["key"].split("::")[0]
        if s not in _ap:
            d = next(x for x in (f"{PILOT}/data/anchors/{s}", f"{PILOT}/data/anchors_full/{s}") if os.path.exists(f"{x}/ideas.json"))
            _ap[s] = json.load(open(f"{d}/pool_preT.json"))[:400]
        return _ap[s], None
    s = int(re.match(r"seed(\d+)_", p["key"]).group(1)); pin = json.load(open(f"{PILOT}/data/pools/seed{s}.json"))["pool"]
    B = next(x["id"] for x in pin if x.get("is_B"))
    return (pin, B) if p["key"].endswith("B_in") else ([x for x in pin if not x.get("is_B")], B)
OUTF = f"{R.OUT}/r4_x5_profile_{MK}.jsonl"
done = {(json.loads(l)["set"], json.loads(l)["key"]) for l in open(OUTF) if not json.loads(l).get("error")} if os.path.exists(OUTF) else set()
fo = open(OUTF, "a"); lock = threading.Lock()
def one(p):
    if (p["set"], p["key"]) in done: return
    recs, B = corpus(p)
    if p["src"] == "exact" and (p["set"], p["key"]) in R1:
        r = R1[(p["set"], p["key"])]
        rec = {"set": p["set"], "key": p["key"], "src": "exact", "from": "r1_E7", "rej": r["rej"], "anticipating_id": r["anticipating_id"],
               "names_B": r["names_B"], "parse_error": r["parse_error"]}
    else:
        try:
            txt, meta = R.call(MK, p["prompt"], MAXTOK, f"r4x5_{p['set']}_{p['key']}")
            v = R.parse_verdict(txt, recs)
            rec = {"set": p["set"], "key": p["key"], "src": p["src"], "from": "r4", "rej": v["rej"], "anticipating_id": v["anticipating_id"] if v["rej"] else None,
                   "names_B": bool(v["rej"] and B and p["key"].endswith("B_in") and v["anticipating_id"] == B), "parse_error": v["parse_error"], "meta": meta}
        except R.Budget: return
        except Exception as e: rec = {"set": p["set"], "key": p["key"], "error": str(e)[:300]}
    with lock: fo.write(json.dumps(rec) + "\n"); fo.flush()
try:
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "8"))) as ex: list(ex.map(one, P))
finally: print(R.summary(), flush=True)
