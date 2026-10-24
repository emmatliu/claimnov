import json, os, re, sys, time, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_pools import get, candidates_for, rank_by_idea, API, T
from harness import llm

HERE = os.path.dirname(os.path.abspath(__file__))
pairs = [json.loads(l) for l in open("eval/data/claim_disputed/testset_clean.jsonl")]
INDEX = f"{HERE}/data/n2_full_index.json"
index = json.load(open(INDEX))

def jac(a, b):
    A, B = set(re.findall(r"[a-z0-9]+", a.lower())), set(re.findall(r"[a-z0-9]+", b.lower()))
    return len(A & B) / max(1, len(A | B))

def citing_candidates(pid):
    out, offset = {}, 0
    for _ in range(3):
        d = get(f"{API}/paper/{pid}/citations?fields=title,abstract,publicationDate,paperId&limit=1000&offset={offset}")
        if not d:
            break
        for c in ((d or {}).get("data") or []):
            p = c.get("citingPaper") or {}
            if p.get("abstract") and p.get("publicationDate") and p["publicationDate"] < T:
                out[p["paperId"]] = {"id": p["paperId"], "title": p["title"],
                                     "abstract": p["abstract"], "date": p["publicationDate"]}
        if d.get("next"):
            offset = d["next"]
        else:
            break
        time.sleep(0.3)
    return out

def build(i, r, b, route):
    cands = {c["id"]: c for c in candidates_for(b["paperId"]) if c["title"] != b["title"]}
    if len(cands) < 10:
        return False
    top, sims = rank_by_idea(list(cands.values()), f"{r['A_title']}. {r['A_abstract']}", 19)
    B = {"id": "B_" + b["paperId"], "title": b["title"], "abstract": b["abstract"], "is_B": True}
    json.dump({"pool": [B] + top, "n_candidates": len(cands), "sims": sims},
              open(f"{HERE}/data/pools_full/n2f_{i}.json", "w"))
    index[str(i)] = {"status": "ok", "source": f"rescue:{route}", "A_id": r["A_id"],
                     "dimension": r["novelty_dimension"], "B_s2": b["paperId"]}
    json.dump(index, open(INDEX, "w"), indent=0)
    return True

def s2_by_title(title, guard=0.6):
    q = urllib.parse.quote(title[:300])
    d = get(f"{API}/paper/search/match?query={q}&fields=paperId,title,abstract")
    time.sleep(0.3)
    hit = ((d or {}).get("data") or [None])[0]
    if hit and hit.get("abstract") and jac(hit["title"], title) >= guard:
        return hit
    d = get(f"{API}/paper/search?query={q}&fields=paperId,title,abstract&limit=3")
    time.sleep(0.3)
    for h in ((d or {}).get("data") or []):
        if h.get("abstract") and jac(h["title"], title) >= guard:
            return h
    return None

fixed = []
GUESS = """A peer reviewer disputed the novelty of paper A by pointing at prior work. Identify the prior work's most likely EXACT paper title.

Paper A: {A_title}
Reviewer's reference to the prior work: {B_ref}
Reviewer's comment: {snippet}

If the reference denotes a single identifiable paper, return JSON {{"title": "<exact title>"}}; if it is a concept/family of methods rather than one paper, return {{"title": null}}."""
for k, v in list(index.items()):
    if v["status"] != "unresolved_B":
        continue
    i = int(k)
    r = pairs[i]
    out = llm.claude(llm.SONNET_46, [{"role": "user", "content": GUESS.format(
        A_title=r["A_title"], B_ref=r["B_referenced_work"],
        snippet=r["reviewer_snippet"][:400])}], max_tokens=200, cache_tag=f"bguess{i}")
    m = re.search(r'\{.*\}', out, re.S)
    try:
        title = json.loads(m.group(0)).get("title") if m else None
    except Exception:
        title = None
    if not title:
        continue
    h = s2_by_title(title, guard=0.6)
    if h and build(i, r, h, "llm_title_guess"):
        fixed.append((i, "unresolved->ok"))
        print(f"{i}: resolved via LLM guess -> {h['title'][:60]}", flush=True)

from collections import Counter
print(f"\nrescued {len(fixed)}: {fixed}")
print("final:", Counter(v["status"] for v in index.values()))
