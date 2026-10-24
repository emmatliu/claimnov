import json, os, sys, re, random, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("R4_EXP", "x1_jo_ctrl")
import r4_common as R
JO = open(f"{R.EXP}/prompts/e1_judge_only.txt").read()
P = {p["key"]: p for p in map(json.loads, open(f"{R.OUT}/r3_prompts.jsonl")) if p["set"] == "later"}
PR = {r["key"]: r for r in json.load(open(f"{R.OUT}/r4_x1_prompts.json"))}
keys = sorted(P); random.Random(7).shuffle(keys); keys = keys[:int(os.environ.get("N", "150"))]
OUTF = f"{R.OUT}/r4_x1_jo_control.jsonl"
done = {json.loads(l)["key"] for l in open(OUTF) if not json.loads(l).get("error")} if os.path.exists(OUTF) else set()
fo = open(OUTF, "a"); lock = threading.Lock()
def one(k):
    if k in done or not P[k]["jo_prompt_verified"]: return
    idea = PR[k]["prompt_base"].split("IDEA:\n", 1)[1].split("\n\nNOVELTY PROFILE:\n", 1)[0]
    pr = JO.format(idea=idea, papers=P[k]["top8_block"])
    try:
        txt, meta = R.call("sonnet", pr, 1500, f"r4x1joctrl_{k}")
        try: j = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
        except Exception: j = {}
        rec = {"key": k, "rej": str(j.get("decision", "")).lower().startswith("not"), "anticipating": j.get("anticipating_paper"), "meta": meta}
    except R.Budget: return
    except Exception as e: rec = {"key": k, "error": str(e)[:200]}
    with lock: fo.write(json.dumps(rec) + "\n"); fo.flush()
try:
    with ThreadPoolExecutor(8) as ex: list(ex.map(one, keys))
finally: print(R.summary(), flush=True)
