import json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor
import common as C
from harness import gates
from harness.sandbox import Pool
llm = C.llm
SONNET = llm.SONNET_46
CAP = float(os.environ.get("CAP", "30"))
_orig = llm.chat


def chat(model, messages, max_tokens=2000, temperature=0.0, cache_tag=""):
    if model != SONNET:
        return _orig(model, messages, max_tokens=max_tokens, temperature=temperature, cache_tag=cache_tag)
    system = "\n".join(m["content"] for m in messages if m["role"] == "system") or None
    msgs = [m for m in messages if m["role"] != "system"]
    return C.sonnet(msgs, max_tokens, cache_tag, system=system)


llm.chat = chat; gates.llm.chat = chat; gates.JUDGE = SONNET
OUT = f"{C.OUTD}/e3_coi_rerun.jsonl"
done = {(json.loads(l)["set"], json.loads(l)["key"]) for l in open(OUT) if json.loads(l).get("novel") is not None} if os.path.exists(OUT) else set()
out = open(OUT, "a"); lock = threading.Lock()
units = [u for u in C.disguise_units() + C.later_units() if (u["set"], u["key"]) not in done][:int(os.environ.get("LIMIT", 10**9))]
print(f"{len(units)} units", flush=True)
_pools = {}


def run(u):
    if llm.cost()["usd"] > CAP:
        return
    pk = id(u["recs"])
    with lock:
        if pk not in _pools:
            _pools[pk] = Pool(u["recs"])
    try:
        res = gates.coi_gate(u["idea"], _pools[pk], tag=f"e3coi_{u['key']}")
    except Exception as e:
        res = {"novel": None, "error": str(e)[:200]}
    ev = res.get("evidence")
    rec = {"key": u["key"], "set": u["set"], "cond": u["cond"], "novel": res.get("novel"), "evidence": ev,
           "names_B": bool(u["B_title"] and res.get("novel") is False and C.title_match(ev, u["B_title"])),
           "error": res.get("error"), "backend": C.BACKEND, "meta": u["meta"]}
    with lock:
        out.write(json.dumps(rec) + "\n"); out.flush()


with ThreadPoolExecutor(max_workers=int(os.environ.get("W", "6"))) as ex:
    list(ex.map(run, units))
print("DONE coi cost", llm.cost(), flush=True)
