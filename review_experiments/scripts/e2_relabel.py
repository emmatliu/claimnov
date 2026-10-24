import json, os, re, sys, hashlib, threading
from concurrent.futures import ThreadPoolExecutor
import common as C
from harness.sandbox import Pool
llm = C.llm
MODEL = os.environ.get("E2_MODEL", "gpt-5")
EFFORT = os.environ.get("E2_EFFORT", "medium")
CAP = float(os.environ.get("CAP", "30"))
PA = open(f"{C.EXP}/prompts/e2_pool_prior.txt").read()
PF = open(f"{C.EXP}/prompts/e2_followup_realized.txt").read()
OUT = f"{C.OUTD}/e2_relabel.jsonl"
done = {json.loads(l)["key"] for l in open(OUT)} if os.path.exists(OUT) else set()
out = open(OUT, "a"); lock = threading.Lock()
usage = {"in": 0, "out": 0, "reasoning": 0}


def gpt(prompt, tag):
    h = hashlib.sha256(json.dumps(["e2", MODEL, EFFORT, prompt, tag]).encode()).hexdigest()[:24]
    fp = os.path.join(llm.CACHE, f"c_{h}.json")
    if os.path.exists(fp):
        return json.load(open(fp))["text"]
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
               "max_completion_tokens": 16000, "reasoning_effort": EFFORT}
    for _ in range(3):
        r = llm._post("chat/completions", payload)
        u = r.get("usage", {})
        llm._track(MODEL, u)
        with lock:
            usage["in"] += u.get("prompt_tokens", 0); usage["out"] += u.get("completion_tokens", 0)
            usage["reasoning"] += (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        text = r["choices"][0]["message"]["content"] or ""
        if text.strip():
            llm._atomic_dump({"text": text, "model": r.get("model")}, fp)
            return text
    return ""


def pj(t):
    try:
        return json.loads(re.search(r"\{.*\}", t, re.S).group(0))
    except Exception:
        return None


units = [u for u in C.later_units() if u["key"] not in done][:int(os.environ.get("LIMIT", 10**9))]
_pools, _F = {}, {}
for u in units:
    a = u["meta"]["anchor"]
    if a not in _pools:
        _pools[a] = Pool(u["recs"])
        _F[a] = Pool(json.load(open(f"{C.anchor_dir(a)}/F.json")))
print(f"{len(units)} units; model {MODEL} effort {EFFORT}", flush=True)


def fmt(ps):
    return "\n\n".join(f"[{i}] Title: {p['title']}\nAbstract: {p.get('abstract') or ''}" for i, p in enumerate(ps, 1))


def run(u):
    if llm.cost()["usd"] > CAP:
        return
    a = u["meta"]["anchor"]; idea = f"{u['idea']['title']}: {u['idea']['description']}"
    top20 = _pools[a].search(idea, k=20); topF = _F[a].search(idea, k=8)
    ta = gpt(PA.format(idea=idea, papers=fmt(top20)), f"e2a_{u['key']}")
    tf = gpt(PF.format(idea=idea, papers=fmt(topF)), f"e2f_{u['key']}")
    ja, jf = pj(ta), pj(tf)
    rec = {"key": u["key"], "anchor": a, "model": MODEL, "effort": EFFORT,
           "pool_parse_ok": ja is not None, "F_parse_ok": jf is not None,
           "pool_contained": (str((ja or {}).get("already_contained", "")).lower().startswith("y")) if ja else None,
           "pool_paper": (ja or {}).get("paper"), "pool_why": (ja or {}).get("why"),
           "main_contribution": (ja or {}).get("main_contribution"),
           "F_realized": (str((jf or {}).get("realized", "")).lower().startswith("y")) if jf else None,
           "F_paper": (jf or {}).get("paper"), "F_contribution": (jf or {}).get("realized_contribution"),
           "F_why": (jf or {}).get("why"), "contributions": (jf or {}).get("contributions"),
           "top20_ids": [p["id"] for p in top20], "topF_ids": [p["id"] for p in topF]}
    with lock:
        out.write(json.dumps(rec) + "\n"); out.flush()


with ThreadPoolExecutor(max_workers=int(os.environ.get("W", "8"))) as ex:
    list(ex.map(run, units))
print("DONE e2 cost", llm.cost(), "usage", usage, flush=True)
