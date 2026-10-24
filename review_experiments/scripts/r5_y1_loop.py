import json, os, sys, re, time, random, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("R5_EXP", "y1")
import r5_common as R
import r4_shipped as S
S.R = R
from rank_bm25 import BM25Okapi
config, gate, entail, extract = S.config, S.gate, S.entail, S.extract
sys.path.insert(0, S.PILOT)
from harness import gates as G
import ast as _ast
GEN_PROMPT = next(n.value.value for n in _ast.parse(open(f"{S.PILOT}/run_positive.py").read()).body
                  if isinstance(n, _ast.Assign) and getattr(n.targets[0], "id", "") == "GEN_PROMPT")

STOP = set("""a an the and or of to in on for with by from as at is are was were be been being this that these those it its
we our us their they them which who whom whose what when where how than then there here such into onto over under
can could may might will would should shall do does did not no nor but if so also both each either neither via per
more most less least very much many few new novel propose proposed proposes approach method methods paper work using use
used based show shows shown""".split())
toks = lambda s: [t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(t) > 1 and t not in STOP]
STOPF = set("a an and are as at be by for from has have in is it of on or that the this to was were will with we our can into using use via".split())
tokf = lambda s: [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOPF]

LR = json.load(open(f"{S.PILOT}/data/later_realized_set.json"))
ANCH = sorted({x.split("::")[0] for x in LR})
def psize(s): return len(json.load(open(f"{S.anchor_dir(s)}/pool_preT.json"))[:400])
CAND = sorted(s for s in ANCH if 150 <= psize(s) <= 220)
PICK = random.Random(5).sample(CAND, 2)
NIDEA = int(os.environ.get("NIDEA", "15"))
MAXREG = 3


class Acc:
    def __init__(self): self.calls = []
    def add(self, key, meta): self.calls.append({"key": key, **{k: meta.get(k) for k in ("in", "out", "usd", "latency_s", "endpoint", "cached")}})
    def tot(self):
        return {"calls": len(self.calls), "in": sum(c["in"] or 0 for c in self.calls), "out": sum(c["out"] or 0 for c in self.calls),
                "usd": round(sum(c["usd"] or 0 for c in self.calls), 5), "sum_latency_s": round(sum(c["latency_s"] or 0 for c in self.calls), 2),
                "by_model": {k: sum(1 for c in self.calls if c["key"] == k) for k in {c["key"] for c in self.calls}}}


def call(acc, key, prompt, mt, tag, system=None):
    txt, meta = R.call(key, prompt, mt, tag, system=system); acc.add(key, meta); return txt


_anc = {}
def anchor(slug):
    if slug not in _anc:
        d = S.anchor_dir(slug)
        pool = json.load(open(f"{d}/pool_preT.json"))[:400]
        fs = [{**c, "paper_id": p["id"], "paper_title": p["title"], "paper_abstract": p["abstract"]} for p in pool for c in (S.claims_of(p["title"], p["abstract"]) or [])]
        bm = BM25Okapi([toks(f"{p['title']}. {p.get('abstract') or ''}") for p in pool])
        _anc[slug] = dict(a=json.load(open(f"{d}/anchor.json")), pool=pool, facets=fs, bm=bm,
                          ideas={i["idx"]: i for i in json.load(open(f"{d}/ideas.json"))})
    return _anc[slug]


ftext = lambda f: f"{f['attribute']}: {f['value']}" if f.get("value") else f["attribute"]
def bm25_topk(tc, fs):
    cands = [f for f in fs if f["dimension"] == tc["dimension"]]
    if not cands: return []
    k = config.top_k_for(len({f["paper_id"] for f in cands}))
    sc = BM25Okapi([tokf(ftext(f)) for f in cands], k1=1.5, b=0.75).get_scores(tokf(ftext(tc)))
    best = {}
    for f, s in zip(cands, sc):
        if f["paper_id"] not in best or s > best[f["paper_id"]][1]: best[f["paper_id"]] = (f, s)
    return [f for f, s in sorted(best.values(), key=lambda x: -x[1])[:k]]


def haiku_level(acc, t, c):
    txt = call(acc, "haiku", entail._prompt(t, c), 16, f"r5ent_{S.ekey(t, c)}")
    d = [ch for ch in txt if ch.isdigit()]
    return max(1, min(5, int(d[0]) if d else config.ENTAIL_MIN))


def check_claimnov(idea, A, acc, tag):
    cl = S.claims_of(idea["title"], idea["description"]); src = "cached_shipped"
    if cl is None:
        cl, meta = S.haiku_extract(idea["title"], idea["description"], tag="r5ext"); acc.add("haiku", meta); src = "haiku"
    claims = []
    for c in cl:
        t = {**c, "paper_abstract": idea["description"]}
        cands = bm25_topk(t, A["facets"])
        with ThreadPoolExecutor(6) as ex: lv = list(ex.map(lambda f: haiku_level(acc, t, f), cands))
        js = sorted([{"paper_id": f["paper_id"], "paper_title": f["paper_title"], "claim": f["claim"], "entailment_level": l} for f, l in zip(cands, lv)],
                    key=lambda x: -x["entailment_level"])
        claims.append({"claim": c["claim"], "dimension": c["dimension"], "judged": js})
    tgt = {"id": tag, "title": idea["title"], "abstract": idea["description"]}
    r = gate.profile_veto(tgt, A["pool"], {"claims": claims}, chat=lambda pr: call(acc, "sonnet", pr, 1500, f"r5y1cn_{tag}"))
    ev = r.get("evidence") or {}
    paper = next((p for p in A["pool"] if p["id"] == ev.get("id")), None) if r["veto"] else None
    return {"rej": bool(r["veto"]), "why": ev.get("justification"), "paper_title": (paper or {}).get("title") or ev.get("title"),
            "paper_abstract": (paper or {}).get("abstract"), "paper_id": ev.get("id"), "established_claims": ev.get("established_claims"),
            "n_claims": len(cl), "claims_src": src, "n_pairs": sum(len(c["judged"]) for c in claims),
            "max_level": max([j["entailment_level"] for c in claims for j in c["judged"][:5]] or [0])}


def check_coi(idea, A, acc, tag):
    blob = f"{idea['title']}: {idea['description']}"
    q = call(acc, "sonnet", G.COI_QUERY_PROMPT.format(idea=blob, topic="machine learning"), 2000, f"r5y1coiq_{tag}")
    try: queries = json.loads(G._coi_extract(q, "queries"))[:3]
    except Exception: queries = [blob]
    papers, seen = [], set()
    for qq in queries:
        sc = A["bm"].get_scores(toks(qq))
        for i in sorted(range(len(A["pool"])), key=lambda i: -sc[i])[:5]:
            p = A["pool"][i]
            if p["title"] not in seen: seen.add(p["title"]); papers.append(p)
    content = "".join(f"Paper {i}: Title:{p['title']}\n Abstract:{p['abstract']}\n" for i, p in enumerate(papers))
    text = call(acc, "sonnet", G.COI_CHECK_PROMPT.format(idea=blob, papers_content=content), 2000, f"r5y1coic_{tag}")
    rej = G._coi_extract(text, "similar") == "1"
    paper = None
    if rej:
        try: paper = papers[int(G._coi_extract(text, "similar_paper_id"))]
        except Exception: pass
    return {"rej": rej, "why": G._coi_extract(text, "think")[:1200] if rej else None, "paper_title": (paper or {}).get("title"),
            "paper_abstract": (paper or {}).get("abstract"), "paper_id": (paper or {}).get("id"), "queries": queries, "n_papers": len(papers)}


FB_HEAD = "\n\nYour previous attempt(s) at idea #{i} were rejected by an automated novelty check, which found that prior work already does them:"
FB_ITEM = "\n\nRejected attempt {r}:\nTitle: {t}\nDescription: {d}\nReason given by the check: {why}\nCited prior paper: {pt}\nAbstract of the cited paper: {pa}"
FB_TAIL = "\n\nPropose a different idea #{i} that is novel with respect to the cited prior work, under the same requirements as above.\n\nReturn JSON only: {{\"title\": \"...\", \"description\": \"...\"}}"


def regenerate(A, i, history, acc, tag):
    pr = GEN_PROMPT.format(month_year="February 2026", title=A["a"]["title"], abstract=A["a"]["abstract"], i=i + 1)
    pr += FB_HEAD.format(i=i + 1)
    for r, (idea, chk) in enumerate(history, 1):
        pr += FB_ITEM.format(r=r, t=idea["title"], d=idea["description"], why=(chk.get("why") or "(none given)").strip(),
                             pt=chk.get("paper_title") or "(not named)", pa=(chk.get("paper_abstract") or "(unavailable)").strip())
    pr += FB_TAIL.format(i=i + 1)
    txt = call(acc, "sonnet", pr, 3000, f"r5y1gen_{tag}")
    try: d = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
    except Exception: d = json.JSONDecoder().raw_decode(txt[txt.index("{"):])[0]
    return {"title": d["title"], "description": d["description"]}


OUTF = f"{R.OUT}/r5_y1_loop.jsonl"
done = {(r["anchor"], r["idx"], r["arm"]) for r in map(json.loads, open(OUTF))} if os.path.exists(OUTF) else set()
fo = open(OUTF, "a"); lock = threading.Lock()


def run(unit):
    slug, i, arm = unit
    if (slug, i, arm) in done: return
    A = anchor(slug); idea = dict(A["ideas"][i]); idea.pop("idx", None)
    rounds, hist = [], []
    t0 = time.time(); tot = Acc()
    try:
        if arm == "none":
            rec = {"committed": True, "n_regen": 0, "final": idea}
        else:
            chk = check_claimnov if arm == "claimnov" else check_coi
            committed = False
            for r in range(MAXREG + 1):
                ac = Acc(); tc = time.time()
                if r > 0:
                    idea = regenerate(A, i, hist, ac, f"{arm}_{slug}_{i}_r{r}")
                tg = time.time()
                res = chk(idea, A, ac, f"{slug}_{i}_r{r}" if r == 0 else f"{arm}_{slug}_{i}_r{r}")
                rounds.append({"round": r, "idea": idea, "check": res, "cost": ac.tot(), "wall_gen_s": round(tg - tc, 2), "wall_check_s": round(time.time() - tg, 2)})
                tot.calls += ac.calls
                if not res["rej"]: committed = True; break
                hist.append((idea, res))
            rec = {"committed": committed, "n_regen": len(rounds) - 1 if committed else MAXREG, "final": idea if committed else None}
        out = {"anchor": slug, "idx": i, "arm": arm, **rec, "rounds": rounds, "cost": tot.tot(), "wall_s": round(time.time() - t0, 2)}
        with lock: fo.write(json.dumps(out) + "\n"); fo.flush()
        print(slug[:20], i, arm, "commit" if rec["committed"] else "GIVEUP", "regen", rec["n_regen"], "$%.3f" % tot.tot()["usd"], flush=True)
    except R.Budget as e: print("BUDGET", e, flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print("ERR", slug, i, arm, str(e)[:300], flush=True)


if __name__ == "__main__":
    print("candidates", CAND, "\npick", PICK, flush=True)
    arms = sys.argv[1:] or ["none", "claimnov", "coi"]
    units = [(s, i, a) for a in arms for s in PICK for i in range(NIDEA)]
    try:
        with ThreadPoolExecutor(int(os.environ.get("WORKERS", "4"))) as ex: list(ex.map(run, units))
    finally: print(R.summary(), flush=True)
