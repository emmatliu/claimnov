import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_pools import candidates_for, rank_by_idea
HERE = os.path.dirname(os.path.abspath(__file__))
seeds = json.load(open(f"{HERE}/data/seeds_v2.json"))["seeds"]
disguises = json.load(open(f"{HERE}/data/disguises_v3.json"))
for i, s in enumerate(seeds):
    if i < 10: continue
    fp = f"{HERE}/data/pools/seed{i}.json"
    if os.path.exists(fp): continue
    idea = next((d for d in disguises if d["seed_idx"] == i and d["rung"] == 1), None)
    if not idea: continue
    cands = candidates_for(s["s2_id"])
    pool, sims = rank_by_idea(cands, f"{idea['title']}: {idea['description']}", 19)
    B = {"id": "B_" + s["s2_id"], "title": s["title"], "abstract": s["abstract"], "date": s["pub_date"], "is_B": True}
    json.dump({"pool": [B] + pool, "n_candidates": len(cands), "sims": sims}, open(fp, "w"))
    print(f"seed {i}: {len(cands)} candidates -> pool {1+len(pool)}", flush=True)
print("DONE pools")
