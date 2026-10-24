import json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor
import common as C, profile_util as PU
from novelty import gate, llm as NL
llm = C.llm
SETS = sys.argv[1:] or ["disg", "later"]
RUNS = [int(x) for x in os.environ.get("RUNS", "1,2").split(",")]
CAP = float(os.environ.get("CAP", "45"))
OUT = f"{C.OUTD}/e3_claimnov_runs.jsonl"
done = {(json.loads(l)["set"], json.loads(l)["key"]) for l in open(OUT)} if os.path.exists(OUT) else set()
out = open(OUT, "a"); lock = threading.Lock()
units = []
for s in SETS:
    units += {"disg": C.disguise_units, "later": C.later_units}[s]()
units = [u for u in units if (u["set"], u["key"]) not in done][:int(os.environ.get("LIMIT", 10**9))]
print(f"{len(units)} units; runs {RUNS}", flush=True)


def summarize(res, u):
    ev = res.get("evidence") or {}
    return {"novel": not res["veto"], "anticipating_id": ev.get("id"), "anticipating_title": ev.get("title"),
            "names_B": bool(u["B_id"] and ev.get("id") == u["B_id"])}


def run(u):
    if llm.cost()["usd"] > CAP:
        return
    target, rep, prompt = PU.profile_prompt(u)
    src = PU.old_call_cached(u, prompt)
    rec = {"key": u["key"], "set": u["set"], "cond": u["cond"], "prompt_matches_paper": src, "backend": C.BACKEND, "meta": u["meta"]}
    if src:
        fp = C.ant_cache_path("claude-sonnet-4-6", prompt, 1500, u["old_tag"]) if src == "api" else \
            C.bedrock_cache_path(llm.SONNET_46, prompt, 1500, u["old_tag"])
        old_txt = json.load(open(fp))["text"]
        rec["run0"] = summarize(gate.profile_veto(target, u["recs"], rep, chat=lambda pr: old_txt), u)
    for r in RUNS:
        tag = f"e3r{r}_{u['old_tag']}"
        res = gate.profile_veto(target, u["recs"], rep, chat=lambda pr, tag=tag: C.sonnet([{"role": "user", "content": pr}], 1500, tag))
        rec[f"run{r}"] = summarize(res, u)
    with lock:
        out.write(json.dumps(rec) + "\n"); out.flush()


with ThreadPoolExecutor(max_workers=int(os.environ.get("W", "6"))) as ex:
    def safe(u):
        try:
            run(u)
        except Exception as e:
            print("ERR", u["set"], u["key"], str(e)[:150], flush=True)
    list(ex.map(safe, units))
print("DONE e3 cost", llm.cost(), "novelty-llm", NL.cost, flush=True)
