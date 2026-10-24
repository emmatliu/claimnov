import json, os, sys, time, urllib.request, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import llm

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://api.semanticscholar.org/graph/v1"
T = "2026-03-01"
for line in open(".env"):
    if line.startswith("S2_API_KEY"): KEY = line.split("=",1)[1].strip()
HEAD = {"x-api-key": KEY, "User-Agent": "eval/0.1"}

def get(url, tries=5):
    import urllib.error
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEAD), timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            time.sleep(min(20, 2**i))
        except Exception: time.sleep(min(20, 2**i))
    return None

def candidates_for(s2_id):
    cands = {}
    d = get(f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{s2_id}?fields=title,abstract,publicationDate&limit=100")
    for p in ((d or {}).get("recommendedPapers") or []):
        if p.get("abstract") and p.get("publicationDate") and p["publicationDate"] < T:
            cands[p["paperId"]] = {"id": p["paperId"], "title": p["title"],
                                   "abstract": p["abstract"], "date": p["publicationDate"]}
    time.sleep(0.4)
    d = get(f"{API}/paper/{s2_id}/references?fields=title,abstract,publicationDate,paperId&limit=100")
    for c in ((d or {}).get("data") or []):
        p = c.get("citedPaper") or {}
        if p.get("abstract") and p.get("publicationDate") and p["publicationDate"] < T:
            cands[p["paperId"]] = {"id": p["paperId"], "title": p["title"],
                                   "abstract": p["abstract"], "date": p["publicationDate"]}
    time.sleep(0.4)
    return list(cands.values())

def rank_by_idea(cands, idea_text, k):
    import math
    vecs = llm.embed([f"{c['title']}. {c['abstract']}" for c in cands])
    qv = llm.embed([idea_text])[0]
    def cos(a, b):
        return sum(x*y for x, y in zip(a, b)) / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b)) + 1e-9)
    scored = sorted(((cos(qv, v), c) for v, c in zip(vecs, cands)), key=lambda x: -x[0])
    return [c for _, c in scored[:k]], [round(s, 4) for s, _ in scored[:k]]

os.makedirs(f"{HERE}/data/pools", exist_ok=True)
seeds = json.load(open(f"{HERE}/data/seeds.json"))["seeds"]
disguises = json.load(open(f"{HERE}/data/disguises.json"))

for i, s in enumerate(seeds):
    fp = f"{HERE}/data/pools/seed{i}.json"
    if os.path.exists(fp): continue
    cands = candidates_for(s["s2_id"])
    idea = next(d for d in disguises if d["seed_idx"] == i and d["rung"] == 1)
    top, sims = rank_by_idea(cands, f"{idea['title']}. {idea['description']}", 19)
    B = {"id": "B_" + s["s2_id"], "title": s["title"], "abstract": s["abstract"], "date": s["pub_date"], "is_B": True}
    json.dump({"pool": [B] + top, "n_candidates": len(cands), "sims": sims},
              open(fp, "w"))
    print(f"seed {i}: {len(cands)} candidates -> pool 20", flush=True)

n2 = json.load(open(f"{HERE}/data/n2_pairs.json"))
for i, r in enumerate(n2):
    fp = f"{HERE}/data/pools/n2_{i}.json"
    if os.path.exists(fp): continue
    d = get(f"{API}/paper/ARXIV:{r['B_arxiv']}?fields=paperId")
    time.sleep(0.4)
    if not d: print(f"n2 {i}: no S2 id", flush=True); continue
    cands = [c for c in candidates_for(d["paperId"]) if c["title"] != r["B_title"]]
    top, sims = rank_by_idea(cands, f"{r['A_title']}. {r['A_abstract']}", 19)
    B = {"id": "B_" + r["B_arxiv"], "title": r["B_title"], "abstract": r["B_abstract"], "is_B": True}
    json.dump({"pool": [B] + top, "n_candidates": len(cands), "sims": sims}, open(fp, "w"))
    print(f"n2 {i}: pool 20", flush=True)
print("cost:", llm.cost(), flush=True)
