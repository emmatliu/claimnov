import json, os, sys, threading, random, hashlib
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import r3_common as R
from harness import gates
from harness.sandbox import Pool
import harness.sandbox as SB
llm = C.llm
RUN = int(sys.argv[1]); NLATER = int(os.environ.get("NLATER", "150"))

def cache_only_embed(texts, model="text-embedding-3-small"):
    out = []
    for t in texts:
        fp = os.path.join(llm.CACHE, f"e_{hashlib.sha256((model + '::' + t).encode()).hexdigest()[:24]}.json")
        if not os.path.exists(fp): raise RuntimeError("embedding cache miss")
        out.append(json.load(open(fp)))
    return out
llm.embed = cache_only_embed; SB.llm.embed = cache_only_embed
SON = llm.SONNET_46
box = threading.local()

def paper_cached(msgs, max_tokens, tag, system):
    for fp in (C.bedrock_cache_path(SON, msgs[0]["content"], max_tokens, tag, system), C.ant_cache_path("claude-sonnet-4-6", msgs[0]["content"], max_tokens, tag, system)):
        if len(msgs) == 1 and os.path.exists(fp): return json.load(open(fp))["text"]
    return None

def chat(model, messages, max_tokens=2000, temperature=0.0, cache_tag=""):
    system = "\n".join(m["content"] for m in messages if m["role"] == "system") or None
    msgs = [m for m in messages if m["role"] != "system"]
    if cache_tag.startswith("coiq"):
        t = paper_cached(msgs, max_tokens, cache_tag, system)
        if t is None: raise RuntimeError("CoI query call not cached")
        return t
    box.paper_check = paper_cached(msgs, max_tokens, cache_tag, system)
    txt, meta = R.sonnet(msgs[0]["content"], max_tokens, f"r3coi{RUN}_{cache_tag}", system=system)
    box.text, box.meta = txt, meta
    return txt
gates.llm.chat = chat; gates.JUDGE = SON

P = {(json.loads(l)["set"], json.loads(l)["key"]) : json.loads(l) for l in open(f"{R.OUT}/r3_prompts.jsonl")}
ex_disg = sorted(k for (s, k), p in P.items() if s == "disg" and p["src"] == "exact")
ex_later = sorted(k for (s, k), p in P.items() if s == "later" and p["src"] == "exact")
random.Random(3).shuffle(ex_later); ex_later = sorted(ex_later[:NLATER])
U = [u for u in C.disguise_units() if u["key"] in set(ex_disg)] + [u for u in C.later_units() if u["key"] in set(ex_later)]
OUTF = f"{R.OUT}/r3_coi_rerun.jsonl"
done = {(json.loads(l)["run"], json.loads(l)["key"]) for l in open(OUTF) if not json.loads(l).get("error")} if os.path.exists(OUTF) else set()
U = [u for u in U if (RUN, u["key"]) not in done][:int(os.environ.get("LIMIT", 10**9))]
out = open(OUTF, "a"); lock = threading.Lock(); _pools = {}

def run(u):
    pk = id(u["recs"])
    with lock:
        if pk not in _pools: _pools[pk] = Pool(u["recs"])
    box.paper_check = None; box.text = None; box.meta = None
    try:
        res = gates.coi_gate(u["idea"], _pools[pk], tag=f"sn_{u['key']}")
        ev = res.get("evidence")
        rec = {"run": RUN, "key": u["key"], "set": u["set"], "cond": u["cond"], "rej": not res["novel"], "evidence": ev,
               "names_B": bool(u["B_title"] and not res["novel"] and C.title_match(ev, u["B_title"])),
               "check_prompt_is_papers": box.paper_check is not None,
               "identical_to_paper_text": (box.text == box.paper_check) if box.paper_check else None, "meta": box.meta}
    except R.Budget:
        return
    except Exception as e:
        rec = {"run": RUN, "key": u["key"], "set": u["set"], "error": str(e)[:200]}
    with lock:
        out.write(json.dumps(rec) + "\n"); out.flush()

print("coi run", RUN, len(U), "units", flush=True)
try:
    with ThreadPoolExecutor(max_workers=int(os.environ.get("WORKERS", "6"))) as ex:
        list(ex.map(run, U))
finally:
    R.log_cost(f"r3_coi_rerun run{RUN}")
    print("DONE", R.cost, flush=True)
