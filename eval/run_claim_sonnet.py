import json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for line in open(f"{ROOT}/.env"):
    if "=" in line and not line.startswith("#"):
        k, v = line.strip().split("=", 1); os.environ.setdefault(k, v)
sys.path.insert(0, f"{ROOT}/eval"); os.chdir(f"{ROOT}/eval")
from harness import llm, gates, panel
from harness.sandbox import Pool
from make_disguises import R1, parse
HERE = os.path.dirname(os.path.abspath(__file__))
SONNET = llm.SONNET_46
_orig_chat = llm.chat
def chat(model, messages, max_tokens=2000, temperature=0.0, cache_tag=""):
    if model != SONNET:
        return _orig_chat(model, messages, max_tokens=max_tokens, temperature=temperature, cache_tag=cache_tag)
    system = "\n".join(m["content"] for m in messages if m["role"] == "system") or None
    msgs = [m for m in messages if m["role"] != "system"]
    if os.environ.get("NOVELTY_BACKEND") == "bedrock":
        os.environ["BEDROCK_TEMP0"] = "1"
        return llm.claude(SONNET, msgs, max_tokens=max_tokens, system=system, cache_tag=cache_tag)
    return llm.claude_api("claude-sonnet-4-6", msgs, max_tokens=max_tokens, system=system, cache_tag=cache_tag, temperature=0.0)
llm.chat = chat; gates.llm.chat = chat; panel.llm.chat = chat; gates.JUDGE = SONNET

OUT = f"{HERE}/results/claim_sonnet.jsonl"
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        r = json.loads(l)
        if r.get("novel") is not None: done.add((r["item"], r["gate"]))
out = open(OUT, "a"); lock = threading.Lock()
idx = json.load(open(f"{HERE}/data/n2_full_index.json"))
pairs = [json.loads(l) for l in open("eval/data/claim_disputed/testset_clean.jsonl")]
items = sorted(int(k) for k, v in idx.items() if v["status"] == "ok")
print(f"{len(items)} pairs", flush=True)

def run_pair(i):
    r = pairs[i]; pd = json.load(open(f"{HERE}/data/pools_full/n2f_{i}.json"))
    d = parse(llm.claude(llm.SONNET_46, [{"role": "user", "content": R1.format(abstract=r["A_abstract"])}], max_tokens=3000, cache_tag=f"n2f_ideaify{i}"))
    idea = {"title": d["title"], "description": d["description"]}
    pool = Pool(pd["pool"]); key = f"n2f_{i}"; B = next((p for p in pd["pool"] if p.get("is_B")), None)
    cells = [("v1", lambda: gates.v1_gate(idea, pool, tag=f"snf{i}")),
             ("coi", lambda: gates.coi_gate(idea, pool, tag=f"snf{i}")),
             ("si", lambda: gates.si_gate(idea, pool, tag=f"snf{i}")),
             ("ra", lambda: panel.ra_gate(idea, pool, tag=f"_snf{i}", model=SONNET)),
             ("scholareval", lambda: panel.scholareval_gate(idea, pool, tag=f"_snf{i}", model=SONNET)),
             ("v2_reflection", lambda: panel.v2_reflection_gate(idea, pool, tag=f"_snf{i}", model=SONNET))]
    for name, fn in cells:
        if (key, name) in done: continue
        try: res = fn()
        except Exception as e: res = {"novel": None, "error": str(e)[:200]}
        rec = {"item": key, "gate": name, "dimension": r["novelty_dimension"], "judge": "sonnet-4.6",
               "B_title": B["title"] if B else None, **{k: v for k, v in res.items() if k != "trace"}}
        with lock: out.write(json.dumps(rec) + "\n"); out.flush()
with ThreadPoolExecutor(max_workers=int(os.environ.get("NOVELTY_WORKERS", "12"))) as ex:
    list(ex.map(run_pair, items))
print("done; cost", llm.cost(), flush=True)
