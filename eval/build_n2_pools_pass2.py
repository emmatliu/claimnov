import json, os, re, sys, time, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_pools import get, candidates_for, rank_by_idea, API

HERE = os.path.dirname(os.path.abspath(__file__))
RB = "claim_disputed_mining"
pairs = [json.loads(l) for l in open(f"{RB}/result/testset_clean.jsonl")]
INDEX = f"{HERE}/data/n2_full_index.json"
index = json.load(open(INDEX))

rel = {}
for l in open(f"{RB}/outputs/relevance_all.jsonl"):
    r = json.loads(l)
    rel[(r["A_id"], (r.get("referenced_work_B") or "").strip())] = r

def jac(a, b):
    A, B = set(re.findall(r"[a-z0-9]+", a.lower())), set(re.findall(r"[a-z0-9]+", b.lower()))
    return len(A & B) / max(1, len(A | B))

def by_id(pid):
    d = get(f"{API}/paper/{pid}?fields=paperId,title,abstract")
    time.sleep(0.35)
    return d if d and d.get("abstract") else None

def by_title(title, guard=None):
    q = urllib.parse.quote(title[:300])
    d = get(f"{API}/paper/search/match?query={q}&fields=paperId,title,abstract")
    time.sleep(0.35)
    hit = (d or {}).get("data") or []
    if hit and hit[0].get("abstract") and (guard is None or jac(hit[0]["title"], title) >= guard):
        return hit[0]
    d = get(f"{API}/paper/search?query={q}&fields=paperId,title,abstract&limit=3")
    time.sleep(0.35)
    for h in ((d or {}).get("data") or []):
        if h.get("abstract") and jac(h["title"], title) >= (guard or 0.55):
            return h
    return None

def ref_by_author(r):
    m = re.match(r"\s*([A-Z][\w'-]+)(?:,| and| et al\.?,?)\s*(?:et al\.?,?\s*)?(\d{4})?", r["B_referenced_work"])
    if not m:
        return None
    surname, year = m.group(1).lower(), m.group(2)
    rp = None
    for y in ("2022", "2023"):
        c = f"{RB}/iclr_novelty_validation_data/refs/{y}/{r['A_id']}.json"
        if os.path.exists(c):
            rp = c; break
    if not rp:
        return None
    hits = []
    for e in json.load(open(rp)):
        auth = (e.get("authors") or [""])[0].lower()
        if surname in auth and (not year or str(e.get("year", "")) == year):
            hits.append(e)
    if len(hits) == 1:
        return by_title(hits[0]["title"], guard=0.6)
    return None

def resolve2(r):
    blob = (r["B_referenced_work"] or "") + " " + (r.get("B_identifier") or "")
    m = re.search(r"(\d{4}\.\d{4,5})", blob)
    if m:
        d = get(f"{API}/paper/ARXIV:{m.group(1)}?fields=paperId,title,abstract")
        time.sleep(0.35)
        if d and d.get("abstract"):
            return d, "arxiv_anywhere"
    rr = rel.get((r["A_id"], r["B_referenced_work"].strip()))
    if rr:
        if rr.get("B_resolved_id"):
            d = by_id(rr["B_resolved_id"])
            if d:
                return d, "relevance_resolved_id"
        mr = rr.get("B_matched_ref")
        if isinstance(mr, dict) and mr.get("title"):
            d = by_title(mr["title"], guard=0.6)
            if d:
                return d, "relevance_matched_ref"
        elif isinstance(mr, str) and mr:
            d = by_title(mr, guard=0.6)
            if d:
                return d, "relevance_matched_ref"
    d = ref_by_author(r)
    if d:
        return d, "refs_author_year"
    if " et al" not in r["B_referenced_work"] and len(r["B_referenced_work"].split()) >= 4:
        d = by_title(re.sub(r"\([^)]*\)", "", r["B_referenced_work"]).strip(), guard=0.5)
        if d:
            return d, "title_search"
    return None, None

fixed = 0
for k, v in sorted(index.items(), key=lambda x: int(x[0])):
    if v["status"] != "unresolved_B":
        continue
    i = int(k)
    r = pairs[i]
    b, route = resolve2(r)
    if not b:
        continue
    cands = [c for c in candidates_for(b["paperId"]) if c["title"] != b["title"]]
    if len(cands) < 10:
        index[k] = {"status": "thin_pool", "A_id": r["A_id"], "n_cands": len(cands)}
    else:
        top, sims = rank_by_idea(cands, f"{r['A_title']}. {r['A_abstract']}", 19)
        B = {"id": "B_" + b["paperId"], "title": b["title"], "abstract": b["abstract"], "is_B": True}
        json.dump({"pool": [B] + top, "n_candidates": len(cands), "sims": sims},
                  open(f"{HERE}/data/pools_full/n2f_{i}.json", "w"))
        index[k] = {"status": "ok", "source": f"pass2:{route}", "A_id": r["A_id"],
                    "dimension": r["novelty_dimension"], "B_s2": b["paperId"]}
        fixed += 1
    json.dump(index, open(INDEX, "w"), indent=0)
    print(f"{i}: {index[k]['status']} ({route})", flush=True)

from collections import Counter
print(f"\npass2 fixed {fixed}")
print("final:", Counter(v["status"] for v in index.values()))
print("ok dims:", Counter(v.get("dimension") for v in index.values() if v["status"] == "ok"))
