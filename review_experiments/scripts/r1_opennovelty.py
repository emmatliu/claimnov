import json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
import common as C
from rank_bm25 import BM25Okapi

llm = C.llm
SETS = sys.argv[1:] or ["disg", "later", "claim", "hyb"]
CAP = float(os.environ.get("CAP", "40"))
ABS = int(os.environ.get("ABS", "1500"))
TOPN = 10
DROP_SEEDS = {60, 83, 86, 87, 93}
P_EXT = open(f"{C.EXP}/prompts/r1_opennovelty_extract.txt").read()
P_JUD = open(f"{C.EXP}/prompts/r1_opennovelty_judge.txt").read()
OUT = f"{C.OUTD}/r1_opennovelty.jsonl"
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        r = json.loads(l)
        if not r.get("error"):
            done.add((r["set"], r["key"]))
out = open(OUT, "a"); lock = threading.Lock()

STOP = set("""a an the and or of to in on for with by from as at is are was were be been being this that these those it its
we our us their they them which who whom whose what when where how than then there here such into onto over under
can could may might will would should shall do does did not no nor but if so also both each either neither via per
more most less least very much many few new novel propose proposed proposes approach method methods paper work using use
used based show shows shown""".split())


def toks(s):
    return [t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(t) > 1 and t not in STOP]


_bm = {}


def bm25_for(u):
    k = (u["set"], u["cond"], u["meta"].get("seed"), u["meta"].get("anchor"), u["key"] if u["set"] == "claim" else None)
    with lock:
        if k not in _bm:
            _bm[k] = BM25Okapi([toks(f"{r['title']}. {r.get('abstract') or ''}") for r in u["recs"]])
        return _bm[k]


def parse_json(txt):
    try:
        return json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
    except json.JSONDecodeError:
        return json.JSONDecoder().raw_decode(txt[txt.index("{"):])[0]


def run(u):
    if llm.cost()["usd"] > CAP:
        return
    idea = f"{u['idea']['title']}: {u['idea']['description']}"
    base = u["key"].replace("_B_in", "").replace("_B_out", "")
    rec = {"key": u["key"], "set": u["set"], "cond": u["cond"], "backend": C.BACKEND, "meta": u["meta"], "B_id": u["B_id"]}
    try:
        etxt = C.sonnet([{"role": "user", "content": P_EXT.format(idea=idea)}], 1200, f"r1on_ext_{u['set']}_{base}")
        ex = parse_json(etxt)
        contribs = [c for c in (ex.get("contributions") or []) if isinstance(c, dict)][:3]
        core = ex.get("core_task") or ""
    except Exception as e:
        rec.update(error=f"extract: {str(e)[:200]}")
        with lock:
            out.write(json.dumps(rec) + "\n"); out.flush()
        return
    bm = bm25_for(u)
    ids, plist, clines, cand_ids = {}, [], [], []
    for ci, c in enumerate(contribs, 1):
        q = toks(" ".join(str(c.get(f, "")) for f in ("name", "claim", "query")))
        sc = bm.get_scores(q) if q and u["recs"] else []
        order = sorted(range(len(u["recs"])), key=lambda i: -sc[i])[:TOPN]
        pids = []
        for i in order:
            r = u["recs"][i]
            if r["id"] not in ids:
                ids[r["id"]] = f"P{len(ids) + 1}"
                plist.append(r)
            pids.append(ids[r["id"]])
        cand_ids.append([u["recs"][i]["id"] for i in order])
        clines.append(f"Contribution {ci}: {c.get('name', '')}\n  Claim: {c.get('claim', '')}\n  Candidates: {', '.join(pids) or '(none)'}")
    papers = "\n".join(f"[{ids[r['id']]}] {r['title']}\n    {(r.get('abstract') or '')[:ABS]}" for r in plist)
    rec.update(core_task=core, contributions=contribs, cand_ids=cand_ids)
    if not contribs:
        rec.update(verdicts=[], n_contrib=0)
    else:
        prompt = P_JUD.format(core_task=core, idea=idea, contributions="\n\n".join(clines), papers=papers or "(none)")
        try:
            jtxt = C.sonnet([{"role": "user", "content": prompt}], 2000, f"r1on_jud_{u['set']}_{u['key']}")
            jv = parse_json(jtxt).get("contributions") or []
        except Exception as e:
            rec.update(error=f"judge: {str(e)[:200]}")
            with lock:
                out.write(json.dumps(rec) + "\n"); out.flush()
            return
        inv = {v: k for k, v in ids.items()}
        verd = []
        for ci in range(len(contribs)):
            j = next((x for x in jv if isinstance(x, dict) and x.get("index") == ci + 1), jv[ci] if ci < len(jv) and isinstance(jv[ci], dict) else {})
            v = str(j.get("verdict", "")).strip().lower()
            pid = j.get("refuting_paper_id")
            rid = inv.get(str(pid).strip().strip("[]")) if pid else None
            if rid is None and j.get("refuting_paper_title"):
                m = next((r for r in plist if C.title_match(r["title"], j["refuting_paper_title"])), None)
                rid = m["id"] if m else None
            verd.append({"verdict": v if v in ("can_refute", "cannot_refute", "unclear") else "PARSE:" + v,
                         "refuting_id": rid, "refuting_title": j.get("refuting_paper_title"),
                         "evidence": j.get("evidence"), "reason": j.get("reason"),
                         "refuter_in_candidates": bool(rid and rid in cand_ids[ci])})
        rec.update(verdicts=verd, n_contrib=len(contribs), n_papers_shown=len(plist))
    vs = [v["verdict"] for v in rec["verdicts"]]
    rec["reject_all"] = bool(vs) and all(v == "can_refute" for v in vs)
    rec["reject_any"] = any(v == "can_refute" for v in vs)
    with lock:
        out.write(json.dumps(rec) + "\n"); out.flush()


units = []
for s in SETS:
    us = {"disg": C.disguise_units, "later": C.later_units, "claim": C.claim_units, "hyb": C.hybrid_units}[s]()
    if s == "disg":
        us = [x for x in us if x["meta"]["seed"] not in DROP_SEEDS]
    units += us
units.sort(key=lambda u: u["cond"] == "B_out")
units = [u for u in units if (u["set"], u["key"]) not in done][:int(os.environ.get("LIMIT", 10**9))]
print(f"{len(units)} units to run", flush=True)
with ThreadPoolExecutor(max_workers=int(os.environ.get("W", "8"))) as ex:
    def safe(u):
        try:
            run(u)
        except Exception as e:
            print("ERR", u["set"], u["key"], str(e)[:200], flush=True)
    list(ex.map(safe, units))
print("DONE r1_opennovelty cost", llm.cost(), flush=True)
