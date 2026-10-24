import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import llm

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = llm.SONNET_46

R1 = """You are drafting a research idea proposal. Below is the abstract of a completed research paper. Recast it as a NEW research idea someone is proposing to work on: write a proposal title and a ~200-word description in proposal voice (future tense, "we propose / we will"). Keep the technical content as close to the original as possible — same contribution, same method, same expected findings; only the framing changes from "we did" to "we propose".

Abstract:
{abstract}

Return JSON only: {{"title": "...", "description": "..."}}"""

R2 = """You are drafting a research idea proposal. Below is the abstract of a completed research paper. Recast it as a NEW research idea someone is proposing to work on, as a deep paraphrase: preserve every technical claim, the method, and the expected findings, but rewrite from scratch in entirely different vocabulary and sentence structure — no phrase over 4 words may match the original. Technical proper nouns and dataset names may remain. Write a proposal title (different wording from the original title) and a ~200-word description in proposal voice.

Abstract:
{abstract}

Return JSON only: {{"title": "...", "description": "..."}}"""

def parse(text):
    import re
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0))

seeds = json.load(open(f"{HERE}/data/seeds.json"))["seeds"]
out = []
for i, s in enumerate(seeds):
    for rung, tmpl in ((1, R1), (2, R2)):
        d = parse(llm.claude(GEN, [{"role": "user", "content": tmpl.format(abstract=s["abstract"])}],
                             max_tokens=3000, cache_tag=f"disg{i}r{rung}"))
        out.append({"seed_idx": i, "seed_title": s["title"], "seed_arxiv": s["arxiv"],
                    "rung": rung, "title": d["title"], "description": d["description"]})
        print(f"seed {i} rung {rung}: {d['title'][:60]}", flush=True)
json.dump(out, open(f"{HERE}/data/disguises.json", "w"), indent=1)
print("cost:", llm.cost(), flush=True)
