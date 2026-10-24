import json, os, sys, threading, glob
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for line in open(f"{ROOT}/.env"):
    if "=" in line and not line.startswith("#"):
        k, v = line.strip().split("=", 1); os.environ.setdefault(k, v)
sys.path.insert(0, f"{ROOT}/eval"); sys.path.insert(0, f"{ROOT}/src")
os.chdir(f"{ROOT}/eval")
from harness import llm
from make_disguises import R1, parse
from novelty import pipeline, retrieve, gate
ARM = sys.argv[1]
ABL = os.environ.get("NOVELTY_ABLATION", "")
OUT = f"{ROOT}/eval/results/coupled_veto_{ARM}{('_' + ABL) if ABL else ''}.jsonl"
done = set()
if os.path.exists(OUT):
    for l in open(OUT): done.add(json.loads(l)["item"])
out = open(OUT, "a"); lock = threading.Lock()

def chat_with_tag(tag):
    def chat(prompt):
        if os.environ.get("NOVELTY_BACKEND") == "api":
            return llm.claude_api("claude-sonnet-4-6", [{"role": "user", "content": prompt}], max_tokens=1500, cache_tag=tag, temperature=0.0)
        return llm.claude(llm.SONNET_46, [{"role": "user", "content": prompt}], max_tokens=1500, cache_tag=tag)
    return chat

def run(item, idea, recs, tag, extra):
    if item in done: return
    target = {"id": "idea", "title": idea["title"], "abstract": idea["description"]}
    pool = retrieve.build_pool([{"id": r["id"], "title": r["title"], "abstract": r["abstract"]} for r in recs])
    c0 = llm.cost()["calls"]
    rep = pipeline.score_paper(target, pool)
    if os.environ.get("NOVELTY_CANDCLAIM") == "1":
        from harness.sandbox import Pool
        top8 = [p["id"] for p in Pool(recs).search(f"{idea['title']}: {idea['description']}", k=8)]
        prof = [c["judged"][0]["paper_id"] for c in rep["claims"] if c.get("judged")]
        cands = top8 + [p for p in prof if p not in top8]
        show = os.environ.get("NOVELTY_SHOWLEVELS", "1") == "1"
        if os.environ.get("NOVELTY_CENTRAL") == "1":
            res = gate.candidate_central_veto(target, recs, rep, pool, cands, chat=chat_with_tag("vsz_" + tag))
        else:
          res = gate.candidate_claim_veto(target, recs, rep, pool, cands, chat=chat_with_tag((("vsc2_" if os.environ.get("NOVELTY_CALIB") == "1" else "vsk_") if show else "vsn_") + tag), show_levels=show)
    elif os.environ.get("NOVELTY_PROFILE") == "1":
        res = gate.profile_veto(target, recs, rep, chat=chat_with_tag("vsq_" + tag))
    elif os.environ.get("NOVELTY_INFORMED") == "1":
        from harness.sandbox import Pool
        fb = Pool(recs).search(f"{idea['title']}: {idea['description']}", k=8)
        res = gate.claim_informed_veto(target, recs, rep, chat=chat_with_tag("vsc_" + tag), fallback=fb)
    else:
        res = gate.coupled_veto(target, recs, rep, chat=chat_with_tag(tag))
    B = next((r for r in recs if r.get("is_B")), None)
    rec = {"item": item, "novel": not res["veto"], "evidence": (res["evidence"] or {}).get("title"),
           "evidence_levels": (res["evidence"] or {}).get("per_claim") or (res["evidence"] or {}).get("established_claims"), "justification": (res["evidence"] or {}).get("justification"), "checked": res["checked"],
           "candidates": res["candidates"], "B_id": B["id"] if B else None,
           "B_rank": (res["candidates"].index(B["id"]) + 1) if B and B["id"] in res["candidates"] else None,
           "new_calls": llm.cost()["calls"] - c0, "n_claims": rep["n_claims"], **extra}
    with lock:
        out.write(json.dumps(rec) + "\n"); out.flush()

units = []
if ARM == "neg":
    rows = [json.loads(l) for l in open("eval/data/claim_disputed/testset_clean.jsonl")]
    items = sorted({json.loads(l)["item"] for l in open("results/negative_full.jsonl")})
    for it in items:
        i = int(it.split("_")[1]); r = rows[i]
        d = parse(llm.claude(llm.SONNET_46, [{"role": "user", "content": R1.format(abstract=r["A_abstract"])}], max_tokens=3000, cache_tag=f"n2f_ideaify{i}"))
        recs = json.load(open(f"data/pools_full/n2f_{i}.json"))["pool"]
        units.append((it, {"title": d["title"], "description": d["description"]}, recs, f"sonnet_nf{i}", {}))
elif ARM == "smuggle":
    dz = json.load(open("data/disguises_v2.json"))
    for d in dz:
        pd = json.load(open(f"data/pools/seed{d['seed_idx']}.json"))
        for cond, recs in (("B_in", pd["pool"]), ("B_out", [p for p in pd["pool"] if not p.get("is_B")])):
            key = f"seed{d['seed_idx']}_r{d['rung']}_{cond}"
            units.append((key, {"title": d["title"], "description": d["description"]}, recs, f"vs_sm_{key}", {"rung": d["rung"], "cond": cond}))
elif ARM in ("fa", "ac", "set"):
    for root in ("data/anchors", "data/anchors_full"):
        for adir in sorted(glob.glob(f"{root}/*")):
            slug = os.path.basename(adir)
            if slug == "cut" or not os.path.exists(f"{adir}/ideas.json"): continue
            sf = f"{adir}/sorted_full.json" if os.path.exists(f"{adir}/sorted_full.json") else f"{adir}/sorted.json"
            if not os.path.exists(sf): continue
            if ARM == "set":
                SETITEMS = set(json.load(open(os.environ["NOVELTY_SET_FILE"])))
                b_idx = {s["idx"] for s in json.load(open(sf)) if f"{slug}::{s['idx']}" in SETITEMS}
            else:
                want = {"b"} if ARM == "fa" else {"a", "c"}
                b_idx = {s["idx"] for s in json.load(open(sf)) if s["bucket"] in want}
            ideas = {i["idx"]: i for i in json.load(open(f"{adir}/ideas.json"))}
            recs = json.load(open(f"{adir}/pool_preT.json"))[:400]
            for idx in sorted(b_idx):
                bucket = next(x["bucket"] for x in json.load(open(sf)) if x["idx"] == idx)
                units.append((f"{slug}::{idx}", ideas[idx], recs, f"vs_pf_{slug}_{idx}", {"anchor": slug, "bucket": bucket}))
print(f"{ARM}: {len(units)} units, {len(done)} done", flush=True)
with ThreadPoolExecutor(max_workers=int(os.environ.get("NOVELTY_WORKERS", "4"))) as ex:
    list(ex.map(lambda u: run(*u), units))
print("DONE", ARM, "cost", llm.cost(), flush=True)
