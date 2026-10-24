import json, os, re, sys, time, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_pools import get, API

HERE = os.path.dirname(os.path.abspath(__file__))
SCREEN = f"{HERE}/../data/anchor_screening"
OUT = f"{HERE}/data/anchors_full"
T = "2026-03-01"
MAX_NEW = 4
MIN_CITES = 60
BAD = re.compile(r"benchmark|dataset|survey|: a comprehensive evaluation", re.I)

tiers = json.load(open(f"{SCREEN}/neurips2025_accepts_tiers.json"))
results = json.load(open(f"{SCREEN}/F_results.json"))
screened = {x["title"] for x in results}
existing = {x["title"] for x in results}

def rank(s):
    return {"Oral": 0, "Spotlight": 1}.get(s, 2)
todo = sorted([t for t in tiers if t["title"] not in screened and not BAD.search(t["title"])],
              key=lambda t: rank(t["status"]))

def count_F(pid):
    n, offset = 0, 0
    while True:
        d = get(f"{API}/paper/{pid}/citations?fields=publicationDate&limit=1000&offset={offset}")
        if not d:
            break
        for c in ((d or {}).get("data") or []):
            cp = c.get("citingPaper") or {}
            if cp.get("publicationDate") and cp["publicationDate"] >= T:
                n += 1
        if d.get("next"):
            offset = d["next"]
        else:
            break
        time.sleep(0.3)
    return n

def build_anchor(a):
    slug = re.sub(r"[^a-z0-9]+", "_", a["title"].lower())[:40].strip("_")
    os.makedirs(f"{OUT}/{slug}", exist_ok=True)
    json.dump(a, open(f"{OUT}/{slug}/anchor.json", "w"), indent=1)
    F, pre_cite, offset = [], [], 0
    while True:
        d = get(f"{API}/paper/{a['paperId']}/citations?fields=title,abstract,publicationDate,paperId&limit=1000&offset={offset}")
        if not d:
            break
        for x in ((d or {}).get("data") or []):
            cp = x.get("citingPaper") or {}
            if not (cp.get("abstract") and cp.get("publicationDate")):
                continue
            (F if cp["publicationDate"] >= T else pre_cite).append(
                {"id": cp["paperId"], "title": cp["title"], "abstract": cp["abstract"],
                 "date": cp["publicationDate"]})
        if d.get("next"):
            offset = d["next"]
        else:
            break
        time.sleep(0.3)
    refs = []
    d = get(f"{API}/paper/{a['paperId']}/references?fields=title,abstract,publicationDate,paperId&limit=1000")
    for x in ((d or {}).get("data") or []):
        cp = x.get("citedPaper") or {}
        if cp.get("abstract") and cp.get("publicationDate") and cp["publicationDate"] < T:
            refs.append({"id": cp["paperId"], "title": cp["title"], "abstract": cp["abstract"],
                         "date": cp["publicationDate"]})
    pool = {p["id"]: p for p in refs + pre_cite}
    json.dump(list(pool.values()), open(f"{OUT}/{slug}/pool_preT.json", "w"))
    json.dump(F, open(f"{OUT}/{slug}/F.json", "w"))
    return slug, len(pool), len(F)

found = 0
for i, t in enumerate(todo):
    if found >= MAX_NEW:
        break
    q = urllib.parse.quote(t["title"][:250])
    d = get(f"{API}/paper/search/match?query={q}&fields=paperId,title,abstract,citationCount,publicationDate")
    time.sleep(0.3)
    hit = ((d or {}).get("data") or [None])[0]
    if not hit or (hit.get("citationCount") or 0) < MIN_CITES:
        results.append({"title": t["title"], "status": t["status"],
                        "citationCount": (hit or {}).get("citationCount", 0),
                        "F_postT": None, "screen": "topup_skip_lowcite"})
        continue
    f = count_F(hit["paperId"])
    results.append({"title": t["title"], "status": t["status"],
                    "citationCount": hit["citationCount"], "F_postT": f,
                    "pubDate": hit.get("publicationDate"), "screen": "topup"})
    json.dump(results, open(f"{SCREEN}/F_results.json", "w"), indent=0)
    if f >= 50:
        slug, np_, nf = build_anchor(hit)
        found += 1
        sel = json.load(open(f"{OUT}/SELECTION.json"))
        sel["selected"].append({"title": t["title"], "F_postT": f, "status": t["status"],
                                "citations": hit["citationCount"], "source": "topup"})
        json.dump(sel, open(f"{OUT}/SELECTION.json", "w"), indent=1)
        print(f"FOUND #{found}: {slug} (F={f}, pool={np_})", flush=True)
    if i % 25 == 0:
        print(f"...screened {i}/{len(todo)}, found {found}", flush=True)

print(f"done: {found} new anchors", flush=True)
