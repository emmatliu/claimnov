import json, os, sys, glob, random, re, threading
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for line in open(f"{ROOT}/.env"):
    if "=" in line and not line.startswith("#"):
        k, v = line.strip().split("=", 1); os.environ.setdefault(k, v)
sys.path.insert(0, f"{ROOT}/eval"); sys.path.insert(0, f"{ROOT}/src"); os.chdir(f"{ROOT}/eval")
from harness import llm
from run_positive import cos
from novelty import extract
ARM = sys.argv[1]
INDEP = os.environ.get("CLAIM_SORT_INDEP") == "1"
OUT = f"{ROOT}/eval/results/claim_sort_{ARM}{'_indep' if INDEP else ''}.jsonl"
EXTRACT_PROMPT = """Write the "Our contributions are:" list that this proposal's authors would put in their introduction, one contribution per line. Output nothing else.

PROPOSAL:
{idea}"""
PROMPT = """You are labeling a proposed research IDEA at the level of its individual contribution claims.

IDEA:
{idea}

IDEA'S CONTRIBUTION CLAIMS:
{claims}

CANDIDATE PAPERS ({side}):
{papers}

For EACH claim, decide whether some candidate paper already establishes it: the same thing is done or shown for that claim (same core problem and same core approach for that claim), not merely related work or work the claim builds on. If yes, name the paper.

Return JSON only:
{{"claims": [{{"i": <claim index>, "established": "yes"|"no", "by": "<exact paper title or null>", "why": "<one sentence>"}}]}}"""
def ask(prompt, tag):
    if os.environ.get("NOVELTY_BACKEND", "api") == "bedrock":
        return llm.claude(llm.OPUS_46, [{"role": "user", "content": prompt}], max_tokens=900, cache_tag=tag)
    return llm.claude_api(llm.OPUS_46_API, [{"role": "user", "content": prompt}], max_tokens=900, cache_tag=tag, temperature=0.0)
done = set()
if os.path.exists(OUT):
    for l in open(OUT): done.add(json.loads(l)["item"])
out = open(OUT, "a"); lock = threading.Lock()
units = []
for root in ("data/anchors", "data/anchors_full"):
    for adir in sorted(glob.glob(f"{root}/*")):
        slug = os.path.basename(adir)
        if slug == "cut" or not os.path.exists(f"{adir}/ideas.json"): continue
        sf = f"{adir}/sorted_full.json" if os.path.exists(f"{adir}/sorted_full.json") else f"{adir}/sorted.json"
        if not os.path.exists(sf): continue
        buckets = {s["idx"]: s["bucket"] for s in json.load(open(sf))}
        ideas = json.load(open(f"{adir}/ideas.json"))
        for idea in ideas:
            units.append((slug, adir, idea, buckets.get(idea["idx"])))
want = set(os.environ.get("CLAIM_SORT_BUCKETS", "abc"))
units = [u for u in units if u[3] in want]
if ARM == "pilot":
    random.Random(0).shuffle(units)
    per = {"a": 34, "b": 33, "c": 33}; pick = []
    for u in units:
        if per.get(u[3], 0) > 0: pick.append(u); per[u[3]] -= 1
    units = pick
print(f"{ARM}: {len(units)} ideas, {len(done)} done", flush=True)
_anchor_cache = {}; _anchor_lock = threading.Lock()
def anchor_data(adir):
    with _anchor_lock:
        if adir not in _anchor_cache:
            pool = json.load(open(f"{adir}/pool_preT.json"))[:400]; F = json.load(open(f"{adir}/F.json"))
            _anchor_cache[adir] = (pool, llm.embed([f"{r['title']}. {r['abstract']}" for r in pool]),
                                   F, llm.embed([f"{r['title']}. {r['abstract']}" for r in F]))
        return _anchor_cache[adir]
def run(u):
    slug, adir, idea, bucket = u
    item = f"{slug}::{idea['idx']}"
    if item in done: return
    pool, pv, F, fv = anchor_data(adir)
    iv = llm.embed([f"{idea['title']}. {idea['description']}"])[0]
    c0 = llm.cost()["calls"]
    if INDEP:
        raw = ask(EXTRACT_PROMPT.format(idea=f"{idea['title']}: {idea['description']}"), f"cs_extract_{item}")
        lines = [re.sub(r"^[\s\-\*\d\.\)]+", "", l).strip() for l in raw.splitlines()]
        lines = [l for l in lines if len(l) > 15 and not l.lower().startswith("our contributions")]
        claims = [{"claim": l, "dimension": None} for l in lines]
    else:
        claims = extract.extract_claims({"id": item, "title": idea["title"], "abstract": idea["description"]})
    ctext = "\n".join(f"{i+1}. " + (f"[{c.get('dimension')}] " if c.get("dimension") else "") + c.get("claim", "") for i, c in enumerate(claims)) or "(none)"
    rec = {"item": item, "bucket": bucket, "n_claims": len(claims), "claims": [{"i": i+1, "dimension": c.get("dimension"), "claim": c.get("claim")} for i, c in enumerate(claims)]}
    for side, recs, vecs, tag in (("pre", pool, pv, "cs_pre"), ("post", F, fv, "cs_post")):
        top = sorted(((cos(iv, v), r) for v, r in zip(vecs, recs)), key=lambda x: -x[0])[:8]
        ptext = "\n".join(f"- {r['title']}: {(r.get('abstract') or '')[:900]}" for _, r in top)
        side_label = "published before the cutoff" if side == "pre" else "published after the cutoff"
        outp = ask(PROMPT.format(idea=f"{idea['title']}: {idea['description']}", claims=ctext, papers=ptext, side=side_label), f"{tag}_{item}")
        try:
            j = json.loads(re.search(r"\{.*\}", outp, re.S).group(0)).get("claims", [])
        except Exception:
            j = []
        rec[side] = j
    rec["new_calls"] = llm.cost()["calls"] - c0
    with lock:
        out.write(json.dumps(rec) + "\n"); out.flush()
with ThreadPoolExecutor(max_workers=int(os.environ.get("NOVELTY_WORKERS", "8"))) as ex:
    list(ex.map(run, units))
print("DONE", ARM, "cost", llm.cost(), flush=True)
