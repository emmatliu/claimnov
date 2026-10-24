import json, os, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.abspath(os.path.join(os.path.dirname(HERE), "tier_sweep_data"))
os.environ.setdefault("NOVELTY_DATA", DATA)
os.environ.setdefault("LOCALAPPDATA", os.path.join(DATA, "cache"))
for line in open(".env"):
    if "=" in line and line.split("=")[0].strip() in ("OPENAI_API_KEY", "S2_API_KEY"):
        k, v = line.strip().split("=", 1)
        os.environ.setdefault(k, v)
sys.path.insert(0, os.path.join(ROOT, "legacy", "claim_validation"))
sys.path.insert(0, os.path.join(ROOT, "legacy", "claim-level_retrieval_200"))
import claim_retrieval as pm
import realpool_eval as rp

tells = [json.loads(l) for l in
         open(os.path.join(ROOT, "claim_disputed_mining/positive_tells/outputs/positive_tells.jsonl"))]
OUT = os.path.join(HERE, "outputs", "p2_full_pools.json")
pools = json.load(open(OUT)) if os.path.exists(OUT) else {}

done = 0
for t in tells:
    pid = t["id"]
    if pid in pools:
        continue
    rpath = None
    for y in ("2022", "2023"):
        c = os.path.join(DATA, "iclr_novelty_validation_data", "refs", y, pid + ".json")
        if os.path.exists(c):
            rpath = c
            break
    keys = []
    if rpath:
        refs = json.load(open(rpath))[:20]
        with ThreadPoolExecutor(max_workers=8) as ex:
            for ab, key in ex.map(rp.resolve_ref, refs):
                if ab:
                    pm.extract_facets("RF::" + key, ab["title"], ab["abstract"])
                    keys.append(key)
    pools[pid] = keys
    done += 1
    json.dump(pools, open(OUT, "w"))
    if done % 20 == 0:
        print(f"...{done} papers, last {pid}: {len(keys)} refs", flush=True)
n_ok = sum(1 for v in pools.values() if len(v) >= 5)
print(f"pools: {len(pools)} papers, {n_ok} with >=5 resolved refs")
