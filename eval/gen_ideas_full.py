import json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import llm
from run_positive import GEN_PROMPT, parse_json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = f"{HERE}/data/anchors_full"
N_IDEAS = 100

slugs = sorted(d for d in os.listdir(ROOT) if os.path.isdir(f"{ROOT}/{d}") and d != "cut")
print(f"{len(slugs)} anchors", flush=True)
for slug in slugs:
    adir = f"{ROOT}/{slug}"
    fp = f"{adir}/ideas.json"
    if os.path.exists(fp):
        continue
    anchor = json.load(open(f"{adir}/anchor.json"))

    def _gen(i):
        d = parse_json(llm.claude(llm.SONNET_46, [{"role": "user", "content": GEN_PROMPT.format(
            month_year="February 2026", title=anchor["title"], abstract=anchor["abstract"], i=i + 1)}],
            max_tokens=3000, cache_tag=f"genf_{slug}_{i}"))
        return {"idx": i, "title": d["title"], "description": d["description"]}

    with ThreadPoolExecutor(max_workers=4) as ex:
        ideas = list(ex.map(_gen, range(N_IDEAS)))
    json.dump(ideas, open(fp, "w"))
    print(f"{slug}: {len(ideas)} ideas", flush=True)
print("DONE | cost:", llm.cost())
