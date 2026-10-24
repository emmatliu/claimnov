import json, os, sys, random, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for line in open(f"{ROOT}/.env"):
    if "=" in line and not line.startswith("#"):
        k, v = line.strip().split("=", 1); os.environ.setdefault(k, v)
sys.path.insert(0, f"{ROOT}/eval"); os.chdir(f"{ROOT}/eval")
from harness import llm
from harness.sandbox import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/data/hybrids.json"

SPLICE = """You are editing a research idea proposal. Below are (1) a NOVEL proposal and (2) the abstract of a PRIOR paper.

Rewrite the proposal so that it additionally claims, as one of its own contributions and in proposal voice, the core contribution of the prior paper (the same problem solved by the same mechanism). Keep every original contribution of the proposal intact and keep its overall framing; integrate the new contribution naturally as a component or step of the plan, in roughly one to three sentences. Do not cite or name the prior paper, its method name, or its authors. Keep roughly the original length plus the added sentences.

PROPOSAL:
{idea}

PRIOR PAPER ABSTRACT:
{abstract}

Return JSON only: {{"title": "<keep or lightly adapt the original title>", "description": "<rewritten proposal>", "spliced_claim": "<one sentence stating the contribution you added, in the proposal's own words>"}}"""

VERIFY = """TEXT A is a research proposal. TEXT B is a prior paper abstract. Does TEXT A, among its contributions, propose to do substantially what TEXT B did (same core problem, same core mechanism) as one of its own contributions? Answer "yes" or "no", then one sentence.

TEXT A:
{description}

TEXT B:
{abstract}"""

def ask(prompt, tag, max_tokens=1800):
    return llm.claude_api("claude-sonnet-4-6", [{"role": "user", "content": prompt}], max_tokens=max_tokens, cache_tag=tag, temperature=0.0)

def parse_json(s):
    m = re.search(r"\{.*\}", s, re.S)
    return json.loads(m.group(0)) if m else None

rng = random.Random(11)
SET = sorted(json.load(open(f"{HERE}/data/later_realized_set.json")))
by_anchor = {}
for it in SET: by_anchor.setdefault(it.split("::")[0], []).append(it)
picks = []
pools = {a: rng.sample(v, len(v)) for a, v in by_anchor.items()}
while len(picks) < 100:
    for a in sorted(pools):
        if pools[a] and len(picks) < 100: picks.append(pools[a].pop())
done = {h["item"]: h for h in json.load(open(OUT))} if os.path.exists(OUT) else {}
out = list(done.values())
for item in picks:
    if item in done: continue
    slug, idx = item.split("::"); idx = int(idx)
    adir = next(p for p in (f"{HERE}/data/anchors/{slug}", f"{HERE}/data/anchors_full/{slug}") if os.path.exists(f"{p}/ideas.json"))
    idea = next(i for i in json.load(open(f"{adir}/ideas.json")) if i["idx"] == idx)
    recs = json.load(open(f"{adir}/pool_preT.json"))[:400]
    pool = Pool(recs)
    itxt = f"{idea['title']}: {idea['description']}"
    cands = pool.search(itxt, k=15)[4:]
    rng.shuffle(cands)
    rec = None
    for B in cands:
        j = parse_json(ask(SPLICE.format(idea=itxt, abstract=B["abstract"]), f"hyb_splice_{item}_{B['id'][:8]}"))
        if not j or not j.get("description") or not j.get("spliced_claim"): continue
        v = ask(VERIFY.format(description=j["description"], abstract=B["abstract"]), f"hyb_verify_{item}_{B['id'][:8]}", 200)
        if re.sub(r"[^a-z]", "", v.strip().lower())[:3] != "yes": continue
        rec = {"item": item, "anchor": slug, "orig_idx": idx, "orig_title": idea["title"], "orig_description": idea["description"],
               "B_id": B["id"], "B_title": B["title"], "B_abstract": B["abstract"], "spliced_claim": j["spliced_claim"],
               "title": j["title"], "description": j["description"], "verified": True}
        break
    if rec:
        out.append(rec); done[item] = rec
        json.dump(out, open(OUT, "w"), indent=1)
        print(f"{len(out):3d} {item} <- {B['title'][:60]}", flush=True)
    else:
        print(f"    {item}: no verified splice", flush=True)
print("DONE", len(out), "cost", llm.cost())
