import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import llm
from make_disguises import R1, R2, parse
import re
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


def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def overlap(src, dis, n=6):
    A = set(zip(*[tok(src)[i:] for i in range(n)]))
    B = set(zip(*[tok(dis)[i:] for i in range(n)]))
    return len(A & B) / max(1, len(B))


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/data/disguises_v3.json"
seeds = json.load(open(f"{HERE}/data/seeds_v2.json"))["seeds"]
out = json.load(open(OUT)) if os.path.exists(OUT) else json.load(open(f"{HERE}/data/disguises_v2.json"))
done = {(d["seed_idx"], d["rung"]) for d in out}
BEDROCK = os.environ.get("NOVELTY_BACKEND", "api") == "bedrock"
def gen(prompt, tag):
    if BEDROCK:
        return llm.claude(llm.SONNET_46, [{"role": "user", "content": prompt}], max_tokens=3000, cache_tag=tag)
    return llm.claude_api("claude-sonnet-4-6", [{"role": "user", "content": prompt}], max_tokens=3000, cache_tag=tag, temperature=1.0)
for i, s in enumerate(seeds):
    if i < 10: continue
    for rung, tmpl in ((1, R1), (2, R2), (3, R3), (4, R4)):
        if (i, rung) in done: continue
        for attempt in range(3):
            d = parse(gen(tmpl.format(abstract=s["abstract"]), f"disg3_{i}r{rung}_a{attempt}"))
            ov = overlap(s["abstract"], d["description"])
            if rung <= 2:
                same = True; break
            vp = VERIFY.format(abstract=s["abstract"], description=d["description"])
            v = (llm.claude(llm.SONNET_46, [{"role": "user", "content": vp}], max_tokens=200, cache_tag=f"disg3v_{i}r{rung}_a{attempt}") if BEDROCK
                 else llm.claude_api("claude-sonnet-4-6", [{"role": "user", "content": vp}], max_tokens=200, cache_tag=f"disg3v_{i}r{rung}_a{attempt}", temperature=0.0))
            same = re.sub(r"[^a-z]", "", v.strip().lower())[:3] == "yes"
            if ov <= 0.02 and same: break
        out.append({"seed_idx": i, "seed_title": s["title"], "seed_arxiv": s["arxiv"], "rung": rung,
                    "title": d["title"], "description": d["description"], "overlap6": round(ov, 3), "verified_same": same})
        json.dump(out, open(OUT, "w"), indent=1)
        print(f"seed {i} rung {rung}: ov={ov:.2f} same={same} | {d['title'][:55]}", flush=True)
print("DONE disguises", len(out), "cost", llm.cost())
