import csv, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import llm
from make_disguises import parse

GEN = llm.SONNET_46

R3 = """You are drafting a research idea proposal. Below is the abstract of a completed research paper. Recast it as a NEW research idea proposal that makes the SAME core technical contribution but is presented completely differently:
- motivate it from a different angle (different application domain or research concern than the original opens with)
- reorder the contributions; do not mirror the original's sentence structure or presentation order
- do not reuse the original's method name; either leave the method unnamed or give it a neutral descriptive name
- do not reuse the original's examples or illustrative scenarios
- keep every substantive technical claim: the same method, the same key mechanism, the same expected findings (numbers may be stated approximately)
~200 words, proposal voice (future tense). Title must not echo the original title.

Abstract:
{abstract}

Return JSON only: {{"title": "...", "description": "..."}}"""

R4 = """You are drafting a research idea proposal. Below is the abstract of a completed research paper. Write a proposal that a hurried reader would take for an unrelated new idea, but that an expert doing a careful side-by-side comparison would judge to be substantially the same work:
- start from a different theoretical or practical starting point and let the core mechanism "emerge" as the proposed approach
- rename or leave unnamed every method/system; drop or generalize all specific numbers ("substantial improvement", not percentages)
- restructure completely: different opening problem, different order, different emphasis (e.g., foreground what the original treats as secondary)
- you may add plausible framing (motivation, evaluation plans) but must NOT add or change any substantive technical contribution
- the same core problem must be solved by the same core mechanism with the same expected outcome
~200 words, proposal voice. Title must appear unrelated to the original.

Abstract:
{abstract}

Return JSON only: {{"title": "...", "description": "..."}}"""

VERIFY = """Do these two texts describe substantially the same core research contribution — same core problem solved by the same core mechanism? Answer "yes" or "no", then one sentence.

TEXT A (paper abstract):
{abstract}

TEXT B (proposal):
{description}"""


def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def overlap(src, dis, n=6):
    A = set(zip(*[tok(src)[i:] for i in range(n)]))
    B = set(zip(*[tok(dis)[i:] for i in range(n)]))
    return len(A & B) / max(1, len(B))


HERE = os.path.dirname(os.path.abspath(__file__))
seeds = json.load(open(f"{HERE}/data/seeds.json"))["seeds"]
out = json.load(open(f"{HERE}/data/disguises.json"))

for i, s in enumerate(seeds):
    for rung, tmpl in ((3, R3), (4, R4)):
        for attempt in range(3):
            d = parse(llm.claude(GEN, [{"role": "user", "content": tmpl.format(abstract=s["abstract"])}],
                                 max_tokens=3000, cache_tag=f"disg2_{i}r{rung}_a{attempt}"))
            ov = overlap(s["abstract"], d["description"])
            v = llm.claude(GEN, [{"role": "user", "content": VERIFY.format(
                abstract=s["abstract"], description=d["description"])}],
                max_tokens=200, cache_tag=f"disg2v_{i}r{rung}_a{attempt}")
            same = v.strip().lower().startswith("yes")
            if ov <= 0.02 and same:
                break
            print(f"  seed{i} r{rung} attempt {attempt}: overlap={ov:.2f} same={same} -> retry", flush=True)
        out.append({"seed_idx": i, "seed_title": s["title"], "seed_arxiv": s["arxiv"],
                    "rung": rung, "title": d["title"], "description": d["description"],
                    "overlap6": round(ov, 3), "verified_same": same})
        print(f"seed {i} rung {rung}: ov={ov:.2f} same={same} | {d['title'][:55]}", flush=True)

json.dump(out, open(f"{HERE}/data/disguises_v2.json", "w"), indent=1)

rows = []
for d in sorted(out, key=lambda x: (x["seed_idx"], x["rung"])):
    if d["rung"] == 1:
        continue
    s = seeds[d["seed_idx"]]
    rows.append({"item_id": f"seed{d['seed_idx']}_r{d['rung']}", "rung": d["rung"],
                 "source_title": s["title"], "source_abstract": s["abstract"],
                 "disguised_idea_title": d["title"], "disguised_idea_description": d["description"],
                 "q1_same_contribution_yes_no": "", "q2_anything_beyond_source_yes_no": "",
                 "low_confidence_flag": "", "notes": ""})
with open(f"{HERE}/annotation/manipulation_dryrun_v2.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"\nwrote data/disguises_v2.json ({len(out)} disguises) and "
      f"annotation/manipulation_dryrun_v2.csv ({len(rows)} rows)")
print("cost:", llm.cost())
