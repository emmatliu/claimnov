import json, os, re, glob, threading
from concurrent.futures import ThreadPoolExecutor
import common as C
from harness.sandbox import Pool
llm = C.llm
_orig = llm.embed
def cache_only_embed(texts, model="text-embedding-3-small"):
    out = []
    for t in texts:
        h = C.hashlib.sha256((model + "::" + t).encode()).hexdigest()[:24]
        fp = os.path.join(llm.CACHE, f"e_{h}.json")
        if not os.path.exists(fp): raise RuntimeError("embedding cache miss")
        out.append(json.load(open(fp)))
    return out
llm.embed = cache_only_embed
import harness.sandbox as SB
SB.llm.embed = cache_only_embed
CAP = float(os.environ.get("CAP", "8"))
PROMPT = open(f"{C.EXP}/prompts/e1_judge_only.txt").read()
OUTF = f"{C.OUTD}/r2_e12_judgeonly.jsonl"
done = {json.loads(l)["key"] for l in open(OUTF) if not json.loads(l).get("error")} if os.path.exists(OUTF) else set()
LATER = set(json.load(open(f"{C.P}/data/later_realized_set.json")))
units = []
for root in ("anchors", "anchors_full"):
    for adir in sorted(glob.glob(f"{C.P}/data/{root}/*")):
        slug = os.path.basename(adir)
        if slug == "cut" or not os.path.exists(f"{adir}/ideas.json"): continue
        sf = f"{adir}/sorted_full.json" if os.path.exists(f"{adir}/sorted_full.json") else f"{adir}/sorted.json"
        if not os.path.exists(sf): continue
        ideas = {i["idx"]: i for i in json.load(open(f"{adir}/ideas.json"))}
        for s in json.load(open(sf)):
            k = f"{slug}::{s['idx']}"
            if s["bucket"] == "a" and k not in LATER and k not in done:
                units.append((k, slug, ideas[s["idx"]], s["match"]))
out = open(OUTF, "a"); lock = threading.Lock(); _pools = {}
def run(u):
    if llm.cost()["usd"] > CAP: return
    k, slug, idea, match = u
    with lock:
        if slug not in _pools: _pools[slug] = (C.anchor_pool(slug), Pool(C.anchor_pool(slug)))
    recs, pool = _pools[slug]
    itxt = f"{idea['title']}: {idea['description']}"
    try:
        top = pool.search(itxt, k=8)
        papers = "\n".join(f"- {p['title']}: {(p.get('abstract') or '')[:700]}" for p in top)
        txt = C.sonnet([{"role": "user", "content": PROMPT.format(idea=itxt, papers=papers)}], 1500, f"r2e12jo_{k}")
        try: j = json.loads(re.search(r"\{.*\}", txt, re.S).group(0)); pe = False
        except Exception: j = {"decision": "novel"}; pe = True
        ap = j.get("anticipating_paper")
        rec = {"key": k, "novel": not str(j.get("decision", "")).lower().startswith("not"), "parse_error": pe, "anticipating_paper": ap,
               "names_match": bool(ap and C.title_match(ap, match)), "match_shown": any(C.title_match(p["title"], match) for p in top),
               "match": match, "backend": C.BACKEND}
    except Exception as e:
        rec = {"key": k, "error": str(e)[:200]}
    with lock:
        out.write(json.dumps(rec) + "\n"); out.flush()
print(len(units), "units", flush=True)
with ThreadPoolExecutor(max_workers=int(os.environ.get("WORKERS", "4"))) as ex:
    list(ex.map(run, units))
print("DONE cost", llm.cost(), flush=True)
