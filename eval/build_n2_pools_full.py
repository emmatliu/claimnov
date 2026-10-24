import json, os, re, sys, time, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_pools import get, candidates_for, rank_by_idea, API

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{HERE}/data/pools_full", exist_ok=True)
pairs = [json.loads(l) for l in
         open("eval/data/claim_disputed/testset_clean.jsonl")]
pilot = json.load(open(f"{HERE}/data/n2_pairs.json"))
pilot_by_title = {p["A_title"]: j for j, p in enumerate(pilot)}

INDEX = f"{HERE}/data/n2_full_index.json"
index = json.load(open(INDEX)) if os.path.exists(INDEX) else {}

def resolve_B(r):
    m = re.search(r"(\d{4}\.\d{4,5})", r.get("B_identifier") or "")
    if m:
        d = get(f"{API}/paper/ARXIV:{m.group(1)}?fields=paperId,title,abstract")
        time.sleep(0.35)
        if d and d.get("abstract"):
            return d
    title = re.sub(r"\([^)]*\d{4}[^)]*\)", "", r["B_referenced_work"]).strip()
    q = urllib.parse.quote(title[:300])
    d = get(f"{API}/paper/search/match?query={q}&fields=paperId,title,abstract")
    time.sleep(0.35)
    if d and d.get("data") and d["data"][0].get("abstract"):
        return d["data"][0]
    return None

built = sum(1 for v in index.values() if v.get("status") == "ok")
for i, r in enumerate(pairs):
    key = str(i)
    if index.get(key, {}).get("status") in ("ok", "unresolved_B", "thin_pool"):
        continue
    fp = f"{HERE}/data/pools_full/n2f_{i}.json"
    if r["A_title"] in pilot_by_title:
        j = pilot_by_title[r["A_title"]]
        pd = json.load(open(f"{HERE}/data/pools/n2_{j}.json"))
        json.dump(pd, open(fp, "w"))
        index[key] = {"status": "ok", "source": f"pilot n2_{j}", "A_id": r["A_id"],
                      "dimension": r["novelty_dimension"]}
    else:
        b = resolve_B(r)
        if not b:
            index[key] = {"status": "unresolved_B", "A_id": r["A_id"],
                          "B": r["B_referenced_work"][:80]}
            json.dump(index, open(INDEX, "w"), indent=0)
            print(f"{i}: unresolved B ({r['B_referenced_work'][:50]})", flush=True)
            continue
        cands = [c for c in candidates_for(b["paperId"]) if c["title"] != b["title"]]
        if len(cands) < 10:
            index[key] = {"status": "thin_pool", "A_id": r["A_id"], "n_cands": len(cands)}
            json.dump(index, open(INDEX, "w"), indent=0)
            print(f"{i}: thin pool ({len(cands)})", flush=True)
            continue
        top, sims = rank_by_idea(cands, f"{r['A_title']}. {r['A_abstract']}", 19)
        B = {"id": "B_" + b["paperId"], "title": b["title"], "abstract": b["abstract"],
             "is_B": True}
        json.dump({"pool": [B] + top, "n_candidates": len(cands), "sims": sims},
                  open(fp, "w"))
        index[key] = {"status": "ok", "source": "built", "A_id": r["A_id"],
                      "dimension": r["novelty_dimension"], "B_s2": b["paperId"]}
    built += 1
    json.dump(index, open(INDEX, "w"), indent=0)
    if built % 10 == 0:
        print(f"progress: {built} pools, pair {i}/200", flush=True)

from collections import Counter
print("\nfinal:", Counter(v["status"] for v in index.values()))
print("dimensions of ok pairs:",
      Counter(v.get("dimension") for v in index.values() if v["status"] == "ok"))
