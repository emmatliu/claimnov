import hashlib, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import boto3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r0_common import OUT, PILOT, PCACHE

MODE = os.environ.get("MODE", "judgeonly")
MODEL = os.environ.get("MODEL", "openai.gpt-oss-120b-1:0")
PRICE = {"openai.gpt-oss-120b-1:0": (0.15, 0.60), "us.meta.llama4-maverick-17b-instruct-v1:0": (0.24, 0.97)}[MODEL]
CAP = float(os.environ.get("CAP", "8"))
MAXTOK = int(os.environ.get("MAXTOK", "6000"))
CDIR = f"{OUT}/r1_cache"; os.makedirs(CDIR, exist_ok=True)
br = boto3.client("bedrock-runtime", region_name="us-west-2")
lock = threading.Lock(); cost = {"in": 0, "out": 0, "calls": 0, "usd": 0.0}
mshort = MODEL.split(".")[1].split(":")[0]
OUTF = f"{OUT}/r1_nonclaude_{MODE}_{mshort}.jsonl"


def call(prompt, tag):
    h = hashlib.sha256(json.dumps([MODEL, prompt, MAXTOK, tag]).encode()).hexdigest()[:24]
    fp = f"{CDIR}/c_{h}.json"
    if os.path.exists(fp):
        return json.load(open(fp))["text"]
    for i in range(8):
        try:
            r = br.converse(modelId=MODEL, messages=[{"role": "user", "content": [{"text": prompt}]}],
                            inferenceConfig={"maxTokens": MAXTOK, "temperature": 0.0})
            break
        except Exception as e:
            if i == 7: raise
            time.sleep(min(60, 2 ** i))
    u = r.get("usage", {})
    with lock:
        cost["in"] += u.get("inputTokens", 0); cost["out"] += u.get("outputTokens", 0); cost["calls"] += 1
        cost["usd"] = cost["in"] / 1e6 * PRICE[0] + cost["out"] / 1e6 * PRICE[1]
    txt = "".join(b.get("text", "") for b in r["output"]["message"]["content"] if "text" in b)
    if txt.strip():
        json.dump({"text": txt, "stop": r.get("stopReason")}, open(fp, "w"))
    return txt


def parse(txt, recs):
    try:
        j = json.loads(re.search(r"\{.*\}", txt, re.S).group(0)); perr = False
    except Exception:
        j = {"decision": "novel", "anticipating_paper": None}; perr = True
    veto = str(j.get("decision", "")).lower().startswith("not")
    ev = None
    if veto:
        title = j.get("anticipating_paper") or ""
        c = next((x for x in recs if x["title"].lower()[:60] == str(title).lower()[:60]), None)
        ev = c["id"] if c else None
    return veto, ev, perr, j.get("anticipating_paper")


def recs_for(st, key):
    if st == "disg":
        s = int(re.match(r"seed(\d+)_", key).group(1))
        pin = json.load(open(f"{PILOT}/data/pools/seed{s}.json"))["pool"]
        return pin if key.endswith("B_in") else [p for p in pin if not p.get("is_B")]
    slug = key.split("::")[0]
    ad = next(p for p in (f"{PILOT}/data/anchors/{slug}", f"{PILOT}/data/anchors_full/{slug}") if os.path.exists(f"{p}/ideas.json"))
    return json.load(open(f"{ad}/pool_preT.json"))[:400]


def idea_text(st, key):
    if st == "disg":
        s, rg = re.match(r"seed(\d+)_r(\d)_", key).groups()
        d = next(d for d in DZ if d["seed_idx"] == int(s) and d["rung"] == int(rg))
        return f"{d['title']}: {d['description']}"
    if st == "hyb":
        h = HY[key]; return f"{h['title']}: {h['description']}"
    slug, idx = key.split("::")
    ad = next(p for p in (f"{PILOT}/data/anchors/{slug}", f"{PILOT}/data/anchors_full/{slug}") if os.path.exists(f"{p}/ideas.json"))
    i = next(i for i in json.load(open(f"{ad}/ideas.json")) if i["idx"] == int(idx))
    return f"{i['title']}: {i['description']}"


DZ = json.load(open(f"{PILOT}/data/disguises_v3.json"))
HY = {h["item"]: h for h in json.load(open(f"{PILOT}/data/hybrids.json"))}
DROP = {60, 83, 86, 87, 93}
units = []
if MODE == "profile":
    for l in open(f"{OUT}/r1_profile_prompts.jsonl"):
        r = json.loads(l)
        if r["status"] == "byte_identical":
            units.append({"set": r["set"], "key": r["key"], "prompt": r["prompt"], "paper_text": r["paper_verdict_text"]})
else:
    E1 = os.path.dirname(OUT) + "/prompts/e1_judge_only.txt"
    T = open(E1).read()
    for l in open(f"{OUT}/e1_judge_only_k8.jsonl"):
        r = json.loads(l)
        if r["set"] == "claim": continue
        if r["set"] == "disg" and int(re.match(r"seed(\d+)_", r["key"]).group(1)) in DROP: continue
        recs = recs_for(r["set"], r["key"]); by = {p["id"]: p for p in recs}
        papers = "\n".join(f"- {by[i]['title']}: {(by[i].get('abstract') or '')[:700]}" for i in r["shown_ids"])
        prompt = T.format(idea=idea_text(r["set"], r["key"]), papers=papers)
        msgs = [{"role": "user", "content": prompt}]
        tag = f"e1jo_k8_{r['set']}_{r['key']}"
        ok = any(os.path.exists(f"{PCACHE}/c_{hashlib.sha256(json.dumps(k).encode()).hexdigest()[:24]}.json") for k in
                 (["antapi", "claude-sonnet-4-6", None, msgs, 1500, tag], ["bedrock", "us.anthropic.claude-sonnet-4-6", None, msgs, 1500, tag]))
        units.append({"set": r["set"], "key": r["key"], "prompt": prompt, "byte_identical_to_e1": ok,
                      "sonnet_novel": r["novel"], "sonnet_names_B": r["names_B"]})
done = {(json.loads(l)["set"], json.loads(l)["key"]) for l in open(OUTF)} if os.path.exists(OUTF) else set()
units = [u for u in units if (u["set"], u["key"]) not in done][:int(os.environ.get("LIMIT", 10**9))]
print(MODE, MODEL, len(units), "units", flush=True)
fo = open(OUTF, "a")


def run(u):
    if cost["usd"] > CAP: return
    recs = recs_for(u["set"], u["key"])
    txt = call(u["prompt"], f"r1nc_{MODE}_{u['set']}_{u['key']}")
    veto, ev, perr, ap = parse(txt, recs)
    B = next((p["id"] for p in recs if p.get("is_B")), None) if u["set"] == "disg" else (HY[u["key"]]["B_id"] if u["set"] == "hyb" else None)
    rec = {"set": u["set"], "key": u["key"], "model": MODEL, "rej": veto, "anticipating_id": ev, "anticipating_paper": ap,
           "names_B": bool(veto and B and ev == B), "parse_error": perr, "empty": not txt.strip()}
    if MODE == "profile":
        sv, sev, sperr, _ = parse(u["paper_text"], recs)
        rec.update(sonnet_rej=sv, sonnet_names_B=bool(sv and B and sev == B))
    else:
        rec.update(sonnet_rej=not u["sonnet_novel"], sonnet_names_B=u["sonnet_names_B"], byte_identical_to_e1=u["byte_identical_to_e1"])
    with lock:
        fo.write(json.dumps(rec) + "\n"); fo.flush()


with ThreadPoolExecutor(max_workers=int(os.environ.get("W", "8"))) as ex:
    def safe(u):
        try: run(u)
        except Exception as e: print("ERR", u["set"], u["key"], str(e)[:200], flush=True)
    list(ex.map(safe, units))
print("DONE", MODE, MODEL, cost, flush=True)
json.dump(cost, open(f"{OUT}/r1_nonclaude_{MODE}_{mshort}_cost_lastrun.json", "w"))
