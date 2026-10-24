import json, os, re, sys, time, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_pools import get, API

HERE = os.path.dirname(os.path.abspath(__file__))
T = "2026-03-01"
OUT = f"{HERE}/data/anchors_full"
os.makedirs(OUT, exist_ok=True)

PILOT = {"Large Language Diffusion Models",
         "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?",
         "Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion"}
BAD = re.compile(r"benchmark|dataset|survey|: a comprehensive evaluation", re.I)

cands = json.load(open(f"{HERE}/../data/anchor_screening/F_results.json"))
sel = [c for c in cands if c["F_postT"] >= 50 and c["title"] not in PILOT
       and not BAD.search(c["title"])]
sel = sorted(sel, key=lambda c: -c["F_postT"])[:27]
json.dump({"rule": "F_postT>=50, non-benchmark title, top-27 by F_postT (PROPOSED - anon to veto/swap)",
           "selected": [{"title": c["title"], "F_postT": c["F_postT"],
                         "status": c["status"], "citations": c["citationCount"]} for c in sel]},
          open(f"{OUT}/SELECTION.json", "w"), indent=1)
print(f"selected {len(sel)} anchors (F_postT {sel[-1]['F_postT']}-{sel[0]['F_postT']})", flush=True)

for c in sel:
    q = urllib.parse.quote(c["title"])
    d = get(f"{API}/paper/search/match?query={q}&fields=title,abstract,publicationDate,paperId")
    time.sleep(0.35)
    if not d or not d.get("data"):
        print(f"SKIP no match: {c['title'][:50]}", flush=True)
        continue
    a = d["data"][0]
    slug = re.sub(r"[^a-z0-9]+", "_", a["title"].lower())[:40].strip("_")
    if os.path.exists(f"{OUT}/{slug}/F.json"):
        continue
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
        time.sleep(0.35)
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
    print(f"{slug}: pool {len(pool)}, F {len(F)}", flush=True)
    time.sleep(0.35)
print("DONE", flush=True)
