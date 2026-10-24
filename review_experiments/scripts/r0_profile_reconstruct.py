import ast, json, hashlib
from r0_common import *

ncache = json.load(open(NCACHE))
src = open(f"{PILOT}/make_disguises.py").read()
R1 = next(ast.literal_eval(n.value) for n in ast.parse(src).body
          if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "R1")
pairs = [json.loads(l) for l in open("eval/data/claim_disputed/testset_clean.jsonl")]
items = json.load(open(f"{OUT}/r0_items.json"))["items"]["claimnov"]["claim"]

def _key(*parts):
    return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:16]
def claims_of(title, abstract):
    return ncache.get("claims4::" + _key(title, abstract))
def ctx(f):
    parts = [f"Facet — attribute: {f.get('attribute', '')}; value: {f.get('value', '')}"]
    if f.get("paper_abstract"):
        parts.append(f"Abstract: {f['paper_abstract'][:1200]}")
    return "\n".join(parts)
def level(t, c):
    ck = "entail::" + hashlib.md5((t["claim"] + "||" + c["claim"] + "||" + ctx(t) + "||" + ctx(c)).encode()).hexdigest()[:16]
    return ncache.get(ck)

def parse(text):
    m = re.search(r"\{.*\}", text, re.S); return json.loads(m.group(0))

out = {}; miss = collections.Counter()
for k in sorted(items, key=lambda s: int(s.split("_")[1])):
    i = int(k.split("_")[1])
    txt = cached_bedrock(SONNET_46_BEDROCK, [{"role": "user", "content": R1.format(abstract=pairs[i]["A_abstract"])}], 3000, f"n2f_ideaify{i}")
    if txt is None: miss["ideaify"] += 1; continue
    d = parse(txt)
    tclaims = claims_of(d["title"], d["description"])
    if tclaims is None: miss["target_claims"] += 1; continue
    pool = json.load(open(f"{PILOT}/data/pools_full/{k}.json"))["pool"]
    facets = []
    for p in pool:
        cl = claims_of(p["title"], p["abstract"])
        if cl is None:
            if p.get("abstract"): miss["pool_claims"] += 1
            cl = []
        for c in cl:
            facets.append({**c, "paper_id": p["id"], "paper_title": p["title"], "paper_abstract": p["abstract"]})
    B = next(p for p in pool if p.get("is_B"))
    rows = []
    for ci, c in enumerate(tclaims):
        t = {**c, "paper_abstract": d["description"]}
        same = [f for f in facets if f["dimension"] == c["dimension"]]
        npapers = len({f["paper_id"] for f in same})
        best = {}
        for f in same:
            lv = level(t, f)
            if lv is not None:
                best[f["paper_id"]] = max(best.get(f["paper_id"], 0), lv)
        rows.append({"i": ci, "dimension": c["dimension"], "claim": c["claim"], "n_same_dim_papers": npapers,
                     "exact_topk": npapers <= 10, "n_cached_papers": len(best),
                     "levels": best, "B_level": best.get(B["id"]), "B_same_dim_facet": any(f["paper_id"] == B["id"] for f in same)})
    out[k] = {"title": d["title"], "B_id": B["id"], "claims": rows, "candidates": items[k]["candidates"],
              "disputed_dim": pairs[i]["novelty_dimension"].lower()}
dump("r0_claim_profiles.json", {"profiles": out, "misses": miss})
print(len(out), dict(miss))
