import csv, json, math, os, random, sys, time, urllib.parse, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import llm

A = "../data/annotation"
os.makedirs(A, exist_ok=True)
rng = random.Random(11)
for line in open(".env"):
    if line.startswith("S2_API_KEY"): S2KEY = line.split("=", 1)[1].strip()

def s2(url):
    import urllib.error
    for i in range(4):
        try:
            req = urllib.request.Request(url, headers={"x-api-key": S2KEY, "User-Agent": "anno"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            time.sleep(min(15, 2 ** i))
        except Exception: time.sleep(min(15, 2 ** i))
    return None

def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

import re
pairs = [json.loads(l) for l in open("eval/data/claim_disputed/testset_clean.jsonl")]
sample = rng.sample(pairs, 100)
cache_fp = f"{A}/.b_abstracts_cache.json"
bcache = json.load(open(cache_fp)) if os.path.exists(cache_fp) else {}
rows = []
for r in sample:
    key = r["B_identifier"]
    if key not in bcache:
        m = re.search(r"arXiv:(\d{4}\.\d{4,5})", key or "")
        hit = None
        if m:
            d = s2(f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{m.group(1)}?fields=abstract")
            hit = (d or {}).get("abstract")
        if not hit:
            q = urllib.parse.quote(r["B_referenced_work"][:300])
            d = s2(f"https://api.semanticscholar.org/graph/v1/paper/search/match?query={q}&fields=title,abstract")
            if d and d.get("data"): hit = d["data"][0].get("abstract")
        bcache[key] = hit or ""
        json.dump(bcache, open(cache_fp, "w"))
        time.sleep(0.3)
    rows.append({
        "pair_id": r["A_id"],
        "A_title": r["A_title"],
        "A_abstract": (r["A_abstract"] or "")[:700],
        "A_link": r.get("A_pdf_url", ""),
        "reviewer_snippet": r["reviewer_snippet"],
        "B_title": r["B_referenced_work"],
        "B_abstract": (bcache[key] or "(unresolved — use identifier)")[:700],
        "B_identifier": r.get("B_identifier", ""),
        "q1_genuine_dispute_yes_no": "",
        "q2_B_correct_yes_no": "",
        "q3_dispute_type": "",
        "low_confidence_flag": "",
        "notes": "",
    })
for who in ("annotator1", "annotator2"):
    write_csv(f"{A}/n2_audit_100_{who}.csv", rows)
print(f"N2 audit: 100 rows, B abstracts resolved for {sum(1 for r in rows if not r['B_abstract'].startswith('(unresolved'))}")

arows = []
for slug in sorted(os.listdir("data/anchors")):
    adir = f"data/anchors/{slug}"
    ideas = {i["idx"]: i for i in json.load(open(f"{adir}/ideas.json"))}
    pool = {p["title"]: p for p in json.load(open(f"{adir}/pool_preT.json"))}
    a_s = [s for s in json.load(open(f"{adir}/sorted.json")) if s["bucket"] == "a" and s.get("match")]
    rng.shuffle(a_s)
    for s in a_s[:7]:
        idea = ideas[s["idx"]]
        m = pool.get(s["match"], {})
        arows.append({"item_id": f"{slug[:12]}_{s['idx']}", "idea_title": idea["title"],
                      "idea_description": idea["description"], "matched_paper_title": s["match"],
                      "matched_paper_abstract": m.get("abstract", ""), "does_match_yes_no": "",
                      "low_confidence_flag": "", "notes": ""})
rng.shuffle(arows); arows = arows[:20]
write_csv(f"{A}/a_match_audit_20.csv", arows)
print(f"(a) audit: {len(arows)} rows")

seeds = json.load(open("data/seeds.json"))["seeds"]
drows = []
for d in json.load(open("data/disguises.json")):
    s_ = seeds[d["seed_idx"]]
    drows.append({"item_id": f"seed{d['seed_idx']}_r{d['rung']}", "rung": d["rung"],
                  "source_title": s_["title"], "source_abstract": s_["abstract"][:800],
                  "disguised_idea_title": d["title"], "disguised_idea_description": d["description"],
                  "q1_same_contribution_yes_no": "", "q2_anything_beyond_source_yes_no": "",
                  "low_confidence_flag": "", "notes": ""})
write_csv(f"{A}/manipulation_dryrun_20.csv", drows)
print(f"manipulation dry run: {len(drows)} rows")

def cos(a, b):
    return sum(x*y for x, y in zip(a, b)) / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b)) + 1e-9)
crows = []
for slug in sorted(os.listdir("data/anchors")):
    adir = f"data/anchors/{slug}"
    ideas = {i["idx"]: i for i in json.load(open(f"{adir}/ideas.json"))}
    cs = [s for s in json.load(open(f"{adir}/sorted.json")) if s["bucket"] == "c"]
    rng.shuffle(cs)
    take = cs[:5]
    if not take: continue
    pool = json.load(open(f"{adir}/pool_preT.json"))[:400]
    F = json.load(open(f"{adir}/F.json"))
    pv = llm.embed([f"{r['title']}. {r['abstract']}" for r in pool])
    fv = llm.embed([f"{r['title']}. {r['abstract']}" for r in F])
    for s in take:
        idea = ideas[s["idx"]]
        iv = llm.embed([f"{idea['title']}. {idea['description']}"])[0]
        near_p = sorted(zip(pv, pool), key=lambda x: -cos(iv, x[0]))[:3]
        near_f = sorted(zip(fv, F), key=lambda x: -cos(iv, x[0]))[:3]
        nm = " ||| ".join(f"[pre-T] {r['title']}" for _, r in near_p) + " ||| " + \
             " ||| ".join(f"[post-T] {r['title']}" for _, r in near_f)
        crows.append({"item_id": f"{slug[:12]}_{s['idx']}", "idea_title": idea["title"],
                      "idea_description": idea["description"], "nearest_candidates": nm,
                      "any_candidate_anticipates_yes_no": "", "which_candidate": "",
                      "low_confidence_flag": "", "notes": ""})
write_csv(f"{A}/c_bucket_protocol_15.csv", crows[:15])
print(f"(c) protocol pilot: {min(len(crows),15)} rows")
print("cost:", llm.cost())
