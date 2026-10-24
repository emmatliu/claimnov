import json, os, sys, re, threading, hashlib, collections
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("R4_EXP", "x1_on")
import r4_common as R
from rank_bm25 import BM25Okapi
PILOT = "./eval"; PCACHE = f"{PILOT}/cache"
STOP = set("""a an the and or of to in on for with by from as at is are was were be been being this that these those it its
we our us their they them which who whom whose what when where how than then there here such into onto over under
can could may might will would should shall do does did not no nor but if so also both each either neither via per
more most less least very much many few new novel propose proposed proposes approach method methods paper work using use
used based show shows shown""".split())
def toks(s): return [t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(t) > 1 and t not in STOP]
P_JUD = open(f"{R.EXP}/prompts/r1_opennovelty_judge.txt").read()
A = json.load(open(f"{R.OUT}/r4_x1_anchors.json"))
ON = {r["key"]: r for r in map(json.loads, open(f"{R.OUT}/r1_opennovelty.jsonl")) if r["set"] == "later" and not r.get("error")}
def adir(slug): return next(p for p in (f"{PILOT}/data/anchors/{slug}", f"{PILOT}/data/anchors_full/{slug}") if os.path.exists(f"{p}/ideas.json"))
_ideas, _pools, _bm = {}, {}, {}
def idea_of(k):
    s, i = k.split("::")
    if s not in _ideas: _ideas[s] = {x["idx"]: x for x in json.load(open(f"{adir(s)}/ideas.json"))}
    return _ideas[s][int(i)]
def pool(s, with_anchor):
    if (s, with_anchor) not in _pools:
        p = json.load(open(f"{adir(s)}/pool_preT.json"))[:400]
        _pools[(s, with_anchor)] = p + [A[s]] if with_anchor else p
        _bm[(s, with_anchor)] = BM25Okapi([toks(f"{r['title']}. {r.get('abstract') or ''}") for r in _pools[(s, with_anchor)]])
    return _pools[(s, with_anchor)], _bm[(s, with_anchor)]

def build(k, with_anchor):
    r = ON[k]; s = k.split("::")[0]; recs, bm = pool(s, with_anchor)
    idea = f"{idea_of(k)['title']}: {idea_of(k)['description']}"
    ids, plist, clines, cand = {}, [], [], []
    for ci, c in enumerate(r["contributions"], 1):
        q = toks(" ".join(str(c.get(f, "")) for f in ("name", "claim", "query")))
        sc = bm.get_scores(q)
        order = sorted(range(len(recs)), key=lambda i: -sc[i])[:10]
        pids = []
        for i in order:
            x = recs[i]
            if x["id"] not in ids: ids[x["id"]] = f"P{len(ids) + 1}"; plist.append(x)
            pids.append(ids[x["id"]])
        cand.append([recs[i]["id"] for i in order])
        clines.append(f"Contribution {ci}: {c.get('name', '')}\n  Claim: {c.get('claim', '')}\n  Candidates: {', '.join(pids) or '(none)'}")
    papers = "\n".join(f"[{ids[x['id']]}] {x['title']}\n    {(x.get('abstract') or '')[:1500]}" for x in plist)
    pr = P_JUD.format(core_task=r.get("core_task") or "", idea=idea, contributions="\n\n".join(clines), papers=papers or "(none)")
    return pr, cand, ids, plist

def cached_r1(prompt, tag):
    msgs = [{"role": "user", "content": prompt}]
    for key in (["bedrock", "us.anthropic.claude-sonnet-4-6", None, msgs, 2000, tag], ["antapi", "claude-sonnet-4-6", None, msgs, 2000, tag]):
        fp = f"{PCACHE}/c_{hashlib.sha256(json.dumps(key).encode()).hexdigest()[:24]}.json"
        if os.path.exists(fp): return True
    return False

def parse_json(txt):
    try: return json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
    except json.JSONDecodeError: return json.JSONDecoder().raw_decode(txt[txt.index("{"):])[0]

OUTF = f"{R.OUT}/r4_x1_on.jsonl"
done = {json.loads(l)["key"] for l in open(OUTF) if not json.loads(l).get("error")} if os.path.exists(OUTF) else set()
fo = open(OUTF, "a"); lock = threading.Lock(); cnt = collections.Counter()
jobs = []
for k in sorted(ON):
    if not ON[k]["contributions"]: continue
    s = k.split("::")[0]
    pr0, cand0, _, _ = build(k, False)
    ok = cached_r1(pr0, f"r1on_jud_later_{k}")
    cnt["rebuild_byte_identical"] += ok
    pr1, cand1, ids1, plist1 = build(k, True)
    hit = [A[s]["id"] in c for c in cand1]
    cnt["lists_changed_any"] += any(set(a) != set(b) for a, b in zip(cand0, [[x for x in c if x != A[s]["id"]] for c in cand1])) or any(hit)
    if any(hit): cnt["anchor_in_some_top10"] += 1
    if any(hit) and k not in done: jobs.append((k, pr1, cand1, ids1, plist1, hit))
print(dict(cnt), len(jobs), "to run", flush=True)

def one(j):
    k, pr, cand, ids, plist, hit = j; s = k.split("::")[0]
    try:
        txt, meta = R.call("sonnet", pr, 2000, f"r4x1on_{k}")
        jv = parse_json(txt).get("contributions") or []
        inv = {v: kk for kk, v in ids.items()}; verd = []
        for ci in range(len(cand)):
            x = next((y for y in jv if isinstance(y, dict) and y.get("index") == ci + 1), jv[ci] if ci < len(jv) and isinstance(jv[ci], dict) else {})
            v = str(x.get("verdict", "")).strip().lower(); pid = x.get("refuting_paper_id")
            rid = inv.get(str(pid).strip().strip("[]")) if pid else None
            verd.append({"verdict": v if v in ("can_refute", "cannot_refute", "unclear") else "PARSE:" + v, "refuting_id": rid,
                         "refuting_title": x.get("refuting_paper_title"), "anchor_in_top10": hit[ci]})
        vs = [v["verdict"] for v in verd]
        rec = {"key": k, "anchor": s, "verdicts": verd, "reject_all": bool(vs) and all(v == "can_refute" for v in vs),
               "reject_any": any(v == "can_refute" for v in vs),
               "cites_anchor_any": any(v["refuting_id"] == A[s]["id"] for v in verd),
               "cites_anchor_can_refute": any(v["refuting_id"] == A[s]["id"] and v["verdict"] == "can_refute" for v in verd),
               "text": txt, "meta": meta}
    except R.Budget: return
    except Exception as e: rec = {"key": k, "error": str(e)[:300]}
    with lock: fo.write(json.dumps(rec) + "\n"); fo.flush()
if "--dry" not in sys.argv:
    try:
        with ThreadPoolExecutor(int(os.environ.get("WORKERS", "6"))) as ex: list(ex.map(one, jobs))
    finally: print(R.summary(), flush=True)
