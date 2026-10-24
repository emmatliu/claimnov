import json, os, re, sys, hashlib, threading
from concurrent.futures import ThreadPoolExecutor
import boto3
import common as C
from rank_bm25 import BM25Okapi
sys.argv = [sys.argv[0]]
src = open(f"{C.EXP}/scripts/r1_opennovelty.py").read()
P_EXT = open(f"{C.EXP}/prompts/r1_opennovelty_extract.txt").read()
P_JUD = open(f"{C.EXP}/prompts/r1_opennovelty_judge.txt").read()
ns = {"re": re}
exec(src[src.index("STOP = set("):src.index("_bm = {}")], ns)
toks = ns["toks"]
br = boto3.client("bedrock-runtime", region_name="us-east-2")
lock = threading.Lock()


def ntok(text):
    for i in range(6):
        try:
            return br.count_tokens(modelId="anthropic.claude-sonnet-4-6",
                                   input={"converse": {"messages": [{"role": "user", "content": [{"text": text}]}]}})["inputTokens"]
        except Exception:
            import time; time.sleep(2 ** i)
    raise RuntimeError("count_tokens failed")


def cached(prompt, mt, tag):
    msgs = [{"role": "user", "content": prompt}]
    h = hashlib.sha256(json.dumps(["bedrock", C.llm.SONNET_46, None, msgs, mt, tag]).encode()).hexdigest()[:24]
    fp = os.path.join(C.llm.CACHE, f"c_{h}.json")
    return json.load(open(fp))["text"] if os.path.exists(fp) else None


rows = [json.loads(l) for l in open(f"{C.OUTD}/r1_opennovelty.jsonl")]
ok = {}
for r in rows:
    if not r.get("error"):
        ok.setdefault((r["set"], r["key"]), r)
U = {}
for s, f in (("disg", C.disguise_units), ("later", C.later_units), ("claim", C.claim_units), ("hyb", C.hybrid_units)):
    for u in f():
        U[(s, u["key"])] = u
seen_ext = set(); calls = []
order = sorted(ok.values(), key=lambda r: (r["backend"] != "bedrock-us-west-2", r["cond"] == "B_out"))
for r in order:
    u = U[(r["set"], r["key"])]
    idea = f"{u['idea']['title']}: {u['idea']['description']}"
    base = u["key"].replace("_B_in", "").replace("_B_out", "")
    etag = f"r1on_ext_{u['set']}_{base}"
    if etag not in seen_ext:
        seen_ext.add(etag)
        pe = P_EXT.format(idea=idea)
        calls.append((r["backend"], "extract", pe, cached(pe, 1200, etag)))
    if r.get("n_contrib"):
        bm = BM25Okapi([toks(f"{x['title']}. {x.get('abstract') or ''}") for x in u["recs"]])
        ids, plist, clines = {}, [], []
        for ci, c in enumerate(r["contributions"], 1):
            q = toks(" ".join(str(c.get(f, "")) for f in ("name", "claim", "query")))
            sc = bm.get_scores(q) if q and u["recs"] else []
            order_ = sorted(range(len(u["recs"])), key=lambda i: -sc[i])[:10]
            pids = []
            for i in order_:
                x = u["recs"][i]
                if x["id"] not in ids:
                    ids[x["id"]] = f"P{len(ids) + 1}"; plist.append(x)
                pids.append(ids[x["id"]])
            clines.append(f"Contribution {ci}: {c.get('name', '')}\n  Claim: {c.get('claim', '')}\n  Candidates: {', '.join(pids) or '(none)'}")
        papers = "\n".join(f"[{ids[x['id']]}] {x['title']}\n    {(x.get('abstract') or '')[:1500]}" for x in plist)
        pj = P_JUD.format(core_task=r["core_task"], idea=idea, contributions="\n\n".join(clines), papers=papers or "(none)")
        calls.append((r["backend"], "judge", pj, cached(pj, 2000, f"r1on_jud_{u['set']}_{u['key']}")))
print(len(calls), "calls; missing cached responses:", sum(1 for c in calls if c[3] is None), flush=True)
base0 = ntok("x")
res = {}


def work(c):
    be, kind, p, t = c
    a = ntok(p) - base0 + 1; b = (ntok(t) - base0 + 1) if t else 0
    with lock:
        d = res.setdefault(be, {"calls": 0, "in": 0, "out": 0, "extract": 0, "judge": 0, "missing_resp": 0})
        d["calls"] += 1; d["in"] += a; d["out"] += b; d[kind] += 1; d["missing_resp"] += t is None


with ThreadPoolExecutor(8) as ex:
    list(ex.map(work, calls))
for be, d in res.items():
    d["usd"] = round(d["in"] / 1e6 * 3 + d["out"] / 1e6 * 15, 2)
print(json.dumps(res, indent=1))
json.dump(res, open(f"{C.OUTD}/r1_cost_e5.json", "w"), indent=1)
