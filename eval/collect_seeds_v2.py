import json, os, random, re, time, urllib.request, urllib.parse
HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://api.semanticscholar.org/graph/v1"; T = "2026-03-01"
for line in open(f"{HERE}/../.env"):
    if line.startswith("S2_API_KEY"): KEY = line.split("=", 1)[1].strip()
HEADERS = {"User-Agent": "pilot-collect/0.2", "x-api-key": KEY}
def get(url, tries=6):
    import urllib.error
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r: return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            time.sleep(min(30, 2 ** i))
        except Exception: time.sleep(min(30, 2 ** i))
    return None
def norm(t): return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()
old = json.load(open(f"{HERE}/data/seeds_v2.json"))["seeds"] if os.path.exists(f"{HERE}/data/seeds_v2.json") else json.load(open(f"{HERE}/data/seeds.json"))["seeds"]
have = {norm(s["title"]) for s in old}
acc = json.load(open("review_experiments/out/scratch/icml2026_accepts.json"))
spots = [p for p in acc if p["venue"] == "ICML 2026 spotlight" and norm(p["title"]) not in have]
random.Random(1).shuffle(spots)
TARGET = int(os.environ.get("SEED_TARGET", "100"))
seeds = list(old); tried = 0
for p in spots:
    if len(seeds) >= TARGET: break
    tried += 1
    d = get(f"{API}/paper/search/match?query={urllib.parse.quote(p['title'])}&fields=title,abstract,publicationDate,externalIds,citationCount,paperId")
    time.sleep(0.35)
    if not d or not d.get("data"): continue
    hit = d["data"][0]
    if norm(hit["title"]) != norm(p["title"]): continue
    pd_, arx, abst = hit.get("publicationDate"), (hit.get("externalIds") or {}).get("ArXiv"), hit.get("abstract")
    if not (pd_ and arx and abst) or pd_ < T: continue
    seeds.append({"title": hit["title"], "abstract": abst, "arxiv": arx, "pub_date": pd_, "s2_id": hit["paperId"],
                  "openreview": f"https://openreview.net/forum?id={p['id']}", "tier": "Spotlight", "venue": "ICML 2026"})
    print(f"seed {len(seeds)-1}: {hit['title'][:60]} ({pd_})", flush=True)
json.dump({"seeds": seeds, "spotlights_tried": tried}, open(f"{HERE}/data/seeds_v2.json", "w"), indent=1)
print(f"seeds: {len(seeds)} ({tried} spotlights tried)", flush=True)
