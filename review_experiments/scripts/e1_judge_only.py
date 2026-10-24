import json, os, sys, re, threading
from concurrent.futures import ThreadPoolExecutor
import common as C
from harness.sandbox import Pool
llm = C.llm
K = int(sys.argv[1]); SETS = sys.argv[2:] or ["disg", "later", "claim", "hyb"]
CAP = float(os.environ.get("CAP", "40"))
PROMPT = open(f"{C.EXP}/prompts/e1_judge_only.txt").read()
OUT = f"{C.OUTD}/e1_judge_only_k{K}.jsonl"
done = {(json.loads(l)["set"], json.loads(l)["key"]) for l in open(OUT)} if os.path.exists(OUT) else set()
out = open(OUT, "a"); lock = threading.Lock()
units = []
for s in SETS:
    units += {"disg": C.disguise_units, "later": C.later_units, "claim": C.claim_units, "hyb": C.hybrid_units}[s]()
units = [u for u in units if (u["set"], u["key"]) not in done][:int(os.environ.get("LIMIT", 10**9))]
print(f"{len(units)} units to run (K={K})", flush=True)
_pools = {}


def run(u):
    if llm.cost()["usd"] > CAP:
        return
    pk = id(u["recs"])
    with lock:
        if pk not in _pools:
            _pools[pk] = Pool(u["recs"])
    pool = _pools[pk]
    idea = f"{u['idea']['title']}: {u['idea']['description']}"
    top = pool.search(idea, k=K)
    papers = "\n".join(f"- {p['title']}: {(p.get('abstract') or '')[:700]}" for p in top)
    prompt = PROMPT.format(idea=idea, papers=papers)
    txt = C.sonnet([{"role": "user", "content": prompt}], 1500, f"e1jo_k{K}_{u['set']}_{u['key']}")
    try:
        j = json.loads(re.search(r"\{.*\}", txt, re.S).group(0)); perr = False
    except Exception:
        j = {"decision": "novel", "anticipating_paper": None}; perr = True
    veto = str(j.get("decision", "")).lower().startswith("not")
    ap = j.get("anticipating_paper")
    m = next((r for r in u["recs"] if ap and r["title"].lower()[:60] == str(ap).lower()[:60]), None)
    if m is None and ap:
        m = next((r for r in u["recs"] if C.title_match(r["title"], ap)), None)
    rec = {"key": u["key"], "set": u["set"], "backend": C.BACKEND, "cond": u["cond"], "novel": not veto, "parse_error": perr,
           "anticipating_paper": ap, "anticipating_id": m["id"] if m else None,
           "names_B": bool(u["B_id"] and m and m["id"] == u["B_id"]),
           "B_in_shown": bool(u["B_id"] and any(p["id"] == u["B_id"] for p in top)),
           "shown_ids": [p["id"] for p in top], "established_claims": j.get("established_claims"),
           "why": j.get("why"), "meta": u["meta"]}
    with lock:
        out.write(json.dumps(rec) + "\n"); out.flush()


with ThreadPoolExecutor(max_workers=int(os.environ.get("W", "8"))) as ex:
    def safe(u):
        try:
            run(u)
        except Exception as e:
            print("ERR", u["set"], u["key"], str(e)[:150], flush=True)
    list(ex.map(safe, units))
print("DONE e1 K", K, "cost", llm.cost(), flush=True)
