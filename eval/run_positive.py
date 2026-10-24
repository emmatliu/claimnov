import json, math, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import llm, gates
from harness.sandbox import Pool
from harness.module_gate import module_gate

HERE = os.path.dirname(os.path.abspath(__file__))
import os as _os
SORTER_MODEL_NAME = _os.environ.get("SORTER", "opus46")
N_IDEAS = 100
WORKERS = int(os.environ.get("PILOT_WORKERS", "8"))

GEN_PROMPT = """You are an ambitious ML researcher in {month_year}. Below is a recent influential paper (the anchor). Propose ONE new research idea that builds on, extends, applies, or is inspired by this line of work. The idea must be concrete and executable as a single paper: state the problem, the proposed method, and the planned experiments, in ~200 words, plus a title.

Anchor paper:
Title: {title}
Abstract: {abstract}

This is idea #{i} of many diverse ideas being collected — make it distinct in focus (vary: sub-problem, technique, domain, evaluation angle).

Return JSON only: {{"title": "...", "description": "..."}}"""

MATCH_PROMPT = """Does the following PAPER do (substantially) what the following IDEA proposes? Answer at the level of research contribution: same core problem AND same core approach = yes; superficial topic overlap = no.

IDEA:
{idea}

PAPER:
Title: {ptitle}
Abstract: {pabstract}

Answer with one word, "yes" or "no", then a one-sentence justification."""

SOUND_PROMPT = """Rate the technical soundness of this proposed research idea on a 1-10 scale (10 = clearly well-posed, feasible, methodologically solid; 1 = incoherent or infeasible). Judge soundness ONLY — not novelty, not impact.

IDEA:
{idea}

Return JSON only: {{"score": <1-10>, "reason": "..."}}"""

def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0))

def cos(a, b):
    return sum(x*y for x, y in zip(a, b)) / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b)) + 1e-9)

results = {}
for slug in sorted(os.listdir(f"{HERE}/data/anchors")):
    adir = f"{HERE}/data/anchors/{slug}"
    anchor = json.load(open(f"{adir}/anchor.json"))
    pool_recs = json.load(open(f"{adir}/pool_preT.json"))
    F = json.load(open(f"{adir}/F.json"))
    print(f"== {slug}: pool {len(pool_recs)}, F {len(F)}", flush=True)

    ideas_fp = f"{adir}/ideas.json"
    if os.path.exists(ideas_fp):
        ideas = json.load(open(ideas_fp))
    else:
        def _gen(i):
            d = parse_json(llm.claude(llm.SONNET_46, [{"role": "user", "content": GEN_PROMPT.format(
                month_year="February 2026", title=anchor["title"], abstract=anchor["abstract"], i=i + 1)}],
                max_tokens=3000, cache_tag=f"gen_{slug}_{i}"))
            if i % 20 == 0: print(f"  gen {i}", flush=True)
            return {"idx": i, "title": d["title"], "description": d["description"]}
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            ideas = list(ex.map(_gen, range(N_IDEAS)))
        json.dump(ideas, open(ideas_fp, "w"))

    pool100 = pool_recs[:400]
    sp = Pool(pool100)

    arms_fp = f"{adir}/arms.json"
    if os.path.exists(arms_fp):
        arms = json.load(open(arms_fp))
    else:
        def _arm(idea):
            it = {"title": idea["title"], "description": idea["description"]}
            coi = gates.coi_gate(it, sp, topic=anchor["title"], tag=f"_{slug}_{idea['idx']}")
            near = sp.search(f"{it['title']}. {it['description']}", k=50)
            mod = module_gate(it, near)
            if idea["idx"] % 20 == 0: print(f"  arms {idea['idx']}", flush=True)
            return {"idx": idea["idx"], "coi_novel": coi["novel"], "coi_ev": coi.get("evidence"),
                    "mod_novel": mod["novel"], "mod_scores": mod["trace"]}
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            arms = list(ex.map(_arm, ideas))
        json.dump(arms, open(arms_fp, "w"))

    sort_fp = f"{adir}/sorted.json"
    if os.path.exists(sort_fp):
        srt = json.load(open(sort_fp))
    else:
        by_idx = {a["idx"]: a for a in arms}
        approved = [i for i in ideas if by_idx[i["idx"]]["coi_novel"] or by_idx[i["idx"]]["mod_novel"]]
        pool_vecs = llm.embed([f"{r['title']}. {r['abstract']}" for r in pool100])
        F_vecs = llm.embed([f"{r['title']}. {r['abstract']}" for r in F])
        def _sort_one(idea):
            iv = llm.embed([f"{idea['title']}. {idea['description']}"])[0]
            itxt = f"{idea['title']}: {idea['description']}"
            def best_match(records, vecs, topm=8):
                scored = sorted(((cos(iv, v), r) for v, r in zip(vecs, records)), key=lambda x: -x[0])[:topm]
                for s, r in scored:
                    ans = llm.claude(llm.OPUS_46 if SORTER_MODEL_NAME == "opus46" else llm.SONNET_46, [{"role": "user", "content": MATCH_PROMPT.format(
                        idea=itxt, ptitle=r["title"], pabstract=r["abstract"])}],
                        max_tokens=2000, cache_tag=f"match_{slug}_{idea['idx']}_{r['id'][:8]}")
                    if ans.strip().lower().startswith("yes"):
                        return r["title"], round(s, 3)
                return None, None
            a_match, _ = best_match(pool100, pool_vecs)
            b_match, _ = (None, None) if a_match else best_match(F, F_vecs)
            bucket = "a" if a_match else ("b" if b_match else "c")
            sound = None
            if bucket == "c":
                sound = parse_json(llm.claude(llm.OPUS_46 if SORTER_MODEL_NAME == "opus46" else llm.SONNET_46, [{"role": "user", "content": SOUND_PROMPT.format(idea=itxt)}],
                                            max_tokens=1500, cache_tag=f"sound_{slug}_{idea['idx']}"))["score"]
            print(f"  sort {idea['idx']}: {bucket}", flush=True)
            return {"idx": idea["idx"], "bucket": bucket, "match": a_match or b_match, "sound": sound}
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            srt = list(ex.map(_sort_one, approved))
        json.dump(srt, open(sort_fp, "w"))
    results[slug] = {"n_ideas": len(ideas)}
print("TOTAL COST:", llm.cost(), flush=True)
