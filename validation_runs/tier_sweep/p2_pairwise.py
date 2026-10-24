import hashlib, json, os, random, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.abspath(os.path.join(os.path.dirname(HERE), "tier_sweep_data"))
os.environ.setdefault("NOVELTY_DATA", DATA)
os.environ.setdefault("LOCALAPPDATA", os.path.join(DATA, "cache"))
for line in open(".env"):
    if "=" in line and line.split("=")[0].strip() in ("OPENAI_API_KEY", "S2_API_KEY"):
        k, v = line.strip().split("=", 1)
        os.environ.setdefault(k, v)
sys.path.insert(0, os.path.join(ROOT, "legacy", "claim_validation"))
sys.path.insert(0, os.path.join(ROOT, "legacy", "claim-level_retrieval_200"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import claim_retrieval as pm
import entail_eval as ee
from sweep import install_judge, _bedrock_json
from novelty import extract as nx
from novelty import llm as nllm

install_judge("4o-mini")
from openai import OpenAI
client = OpenAI()

SONNET = "us.anthropic.claude-sonnet-4-6"
VETO_C = """Does the PRIOR WORK below already substantially establish what the CLAIM proposes — same core contribution, not just topic overlap? Answer with one word, "yes" or "no".

CLAIM (from a paper under review): {claim}

PRIOR WORK: {title}
{abstract}"""

def same(pid, a, b):
    k = "p2same::" + hashlib.md5(f"{pid}|{a}|{b}".encode()).hexdigest()[:16]
    if k in pm._cache:
        return pm._cache[k]
    r = client.chat.completions.create(model="gpt-4o-mini", temperature=0,
        response_format={"type": "json_object"}, max_tokens=20,
        messages=[{"role": "system", "content": "Do these two statements point at the same contribution of the same paper, even at different granularity? Return JSON {\"same\": true|false}."},
                  {"role": "user", "content": f"A: {a}\nB: {b}"}])
    j = json.loads(r.choices[0].message.content).get("same", False)
    with pm._cache_lock:
        pm._cache[k] = j; pm._save_cache()
    return j

def binary_count(claim_text, cands):
    n = 0
    for c in cands[:5]:
        k = "p2veto::" + hashlib.md5((claim_text + "||" + c["_title"]).encode()).hexdigest()[:16]
        if k in pm._cache:
            ans = pm._cache[k]
        else:
            j = _bedrock_json(SONNET, "You judge whether prior work anticipates a claim. Answer strictly.",
                              VETO_C.format(claim=claim_text, title=c["_title"], abstract=(c.get("_abs") or "")[:1200])
                              + '\nReturn JSON {"answer": "yes"|"no"}.')
            ans = str(j.get("answer", "no")).lower().startswith("y")
            with pm._cache_lock:
                pm._cache[k] = ans; pm._save_cache()
        n += bool(ans)
    return n

tells = [json.loads(l) for l in open(os.path.join(ROOT, "claim_disputed_mining/positive_tells/outputs/positive_tells.jsonl"))]
_ids = {t["id"] for t in tells}
_abs = {}
with open(os.path.join(DATA, "iclr_novelty_validation_data", "all_reviews.jsonl")) as f:
    for line in f:
        r = json.loads(line)
        if r["id"] in _ids:
            _abs[r["id"]] = r.get("abstract", "")
for t in tells:
    t["abstract"] = _abs.get(t["id"], "")
pools = json.load(open(os.path.join(HERE, "outputs", "p2_full_pools.json")))
bodies_dir = os.path.join(ROOT, "eval", "p2", "bodies")
OUT = os.path.join(HERE, "outputs", "p2_pairwise.json")
state = json.load(open(OUT)) if os.path.exists(OUT) else {"papers": {}}

for t in tells:
    pid = t["id"]
    if pid in state["papers"]:
        continue
    keys = pools.get(pid, [])
    if len(keys) < 5:
        state["papers"][pid] = {"status": "thin_pool"}
        continue
    ab = t.get("abstract")
    if not ab:
        state["papers"][pid] = {"status": "no_abstract"}
        continue
    claims = nx.extract_claims({"id": pid, "title": t["title"], "abstract": ab})
    praised_i, supplemented = None, False
    for i, c in enumerate(claims):
        if same(pid, t["claim_span"], c["claim"]):
            praised_i = i
            break
    if praised_i is None:
        bfp = os.path.join(bodies_dir, pid + ".json")
        body = json.load(open(bfp)) if os.path.exists(bfp) else {}
        btxt = (body.get("intro") or body.get("intro_pdf") or "")[:2500]
        if btxt:
            extra = nx.extract_claims({"id": pid + "_body", "title": t["title"], "abstract": btxt})
            for c in extra:
                if same(pid, t["claim_span"], c["claim"]):
                    c["supplemented"] = True
                    claims.append(c)
                    praised_i = len(claims) - 1
                    supplemented = True
                    break
    if praised_i is None or len(claims) < 2:
        state["papers"][pid] = {"status": "no_praised_match" if praised_i is None else "single_claim"}
        json.dump(state, open(OUT, "w"))
        continue
    pool_facets = []
    for k in keys:
        abrec = pm._cache.get("REF::" + k) or {}
        for f in pm._cache.get("RF::" + k, []):
            pool_facets.append({**f, "_title": abrec.get("title", ""), "_abs": abrec.get("abstract", "")})
    rows = []
    for i, c in enumerate(claims):
        qf = {"attribute": c["attribute"], "value": c["value"], "dimension": c["dimension"].capitalize()}
        dim_pool = [f for f in pool_facets if (f.get("dimension") or "").lower() == c["dimension"]] or pool_facets
        qtext = f"{c['attribute']}: {c['value']} | {t['title']}"
        ptexts = [(f["_title"] + " | " + f.get("attribute", "") + ": " + f.get("value", "")) for f in dim_pool]
        E = ee.embed(client, [qtext] + ptexts)
        sims = [pm.cos(E[0], e) for e in E[1:]]
        order = sorted(range(len(dim_pool)), key=lambda j: -sims[j])[:5]
        cands = [dim_pool[j] for j in order]
        scores = [ee.entail(client, qf, cands[j], pm._cache, q_title=t["title"],
                            c_title=cands[j]["_title"], q_abs=ab[:700],
                            c_abs=(cands[j].get("_abs") or "")[:700])["score"]
                  for j in range(len(cands))]
        rows.append({"i": i, "claim": c["claim"][:120], "dimension": c["dimension"],
                     "praised": i == praised_i, "supplemented": c.get("supplemented", False),
                     "graded_novelty": round(1 - (max(scores, default=1) - 1) / 4, 3),
                     "binary_anticipation": binary_count(c["claim"], cands)})
    state["papers"][pid] = {"status": "ok", "supplemented": supplemented, "claims": rows}
    json.dump(state, open(OUT, "w"))
    print(f"{pid}: {len(claims)} claims, praised_i={praised_i}", flush=True)

ok = {pid: v for pid, v in state["papers"].items() if v.get("status") == "ok"}
def winrate(key, higher_better):
    per_paper = []
    for v in ok.values():
        pr = next(c for c in v["claims"] if c["praised"])
        sibs = [c for c in v["claims"] if not c["praised"]]
        if not sibs:
            continue
        w = 0.0
        for s in sibs:
            a, b = pr[key], s[key]
            w += 0.5 if a == b else (1.0 if (a > b) == higher_better else 0.0)
        per_paper.append(w / len(sibs))
    rng = random.Random(0)
    boots = []
    for _ in range(2000):
        samp = [per_paper[rng.randrange(len(per_paper))] for _ in per_paper]
        boots.append(sum(samp) / len(samp))
    boots.sort()
    m = sum(per_paper) / len(per_paper)
    return m, boots[50], boots[1949], len(per_paper)

from collections import Counter
print("statuses:", Counter(v.get("status") for v in state["papers"].values()))
for key, hb, name in (("graded_novelty", True, "graded (4o-mini)"),
                      ("binary_anticipation", False, "binary-contrastive (Sonnet 4.6)")):
    m, lo, hi, n = winrate(key, hb)
    print(f"{name}: win rate {m:.3f} [{lo:.3f}, {hi:.3f}] over {n} papers (chance 0.5)")
print("cost tokens:", pm.cost)
