import argparse, csv, hashlib, json, os, random, re, tempfile, threading, time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.environ.get("NOVELTY_DATA", os.path.dirname(HERE))
MENTIONS = os.path.join(PKG, "iclr_novelty_validation_data", "review_mentions.jsonl")
REVIEWS = os.path.join(PKG, "iclr_novelty_validation_data", "all_reviews.jsonl")
OUTDIR = os.path.join(HERE, "outputs")
os.makedirs(OUTDIR, exist_ok=True)
_CDIR = os.path.join(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()), "claim_validation_cache")
os.makedirs(_CDIR, exist_ok=True)
CACHE = os.path.join(_CDIR, "miner_cache.json")

MODEL = "gpt-4o-mini"
DIMS = ["Problem", "Methodology", "Resource", "Result"]

POS_PAT = re.compile(r"\b(novel|novelty|original|first to|first work|to (my|our) knowledge"
                     r"|new (task|benchmark|dataset|problem|direction)|innovative|unexplored)\b", re.I)
NEG_PAT = re.compile(r"\b(not novel|limited novelty|lacks? novelty|incremental"
                     r"|already (been )?(done|proposed|studied|explored)|prior work"
                     r"|previous work|similar to|overlaps? with)\b", re.I)
FIELDS = {"main_review", "strength_and_weaknesses", "summary_of_the_review",
          "clarity,_quality,_novelty_and_reproducibility"}

SYS = """You classify snippets from ICLR peer reviews. Decide whether the snippet contains a POSITIVE NOVELTY TELL: the reviewer, in their OWN assessing voice, attributes novelty/originality/firstness to a SPECIFIC identifiable contribution of the paper under review.

Return JSON:
{"is_positive_tell": bool,      // reviewer's own voice asserts novelty of a specific contribution
 "specific": bool,              // ties novelty to an identifiable contribution (a method, task, dataset, result, analysis), not the paper in general
 "hedged": bool,                // "somewhat novel", "novel but incremental", "not particularly novel", negated or damning-with-faint-praise
 "author_echo": bool,           // merely restates the authors' claim ("the authors claim/propose a novel...", paper-summary voice) rather than the reviewer asserting it
 "claim_span": str,             // the contribution asserted novel, quoted/condensed from the snippet (<= 40 words); "" if none
 "dimension": "Problem|Methodology|Resource|Result|None",  // Problem = new task/question/setting; Methodology = new method/architecture/algorithm/analysis technique; Resource = new dataset/benchmark/tool; Result = new finding/insight/phenomenon
 "confidence": float}           // 0-1, your confidence in is_positive_tell

Rules: negative or disputed novelty ("not novel", "limited novelty") -> is_positive_tell=false. Generic praise ("interesting and novel paper") without a named contribution -> specific=false. "To my knowledge, this is the first work to X" -> strong positive tell."""


def _load_cache():
    return json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}


_cache = _load_cache()
_cache_lock = threading.RLock()
cost = {"in": 0, "out": 0}


def _save_cache():
    with _cache_lock:
        tmp = CACHE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False)
        os.replace(tmp, CACHE)


def load_titles():
    t = {}
    with open(REVIEWS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t[r["id"]] = r.get("title", "")
    return t


def stage_prefilter():
    titles = load_titles()
    cands, neg_by_paper = [], defaultdict(list)
    with open(MENTIONS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            sig = r.get("signature", "")
            if "Reviewer" not in sig:
                continue
            s = r["snippet"]
            p, g = bool(POS_PAT.search(s)), bool(NEG_PAT.search(s))
            if g:
                neg_by_paper[r["id"]].append({"signature": sig, "snippet": s[:600]})
            if p and r.get("field") in FIELDS:
                cands.append({"id": r["id"], "year": r["year"], "signature": sig,
                              "field": r["field"], "snippet": s,
                              "title": titles.get(r["id"], "")})
    json.dump(neg_by_paper, open(os.path.join(OUTDIR, "neg_mentions_by_paper.json"), "w", encoding="utf-8"))
    with open(os.path.join(OUTDIR, "candidates.jsonl"), "w", encoding="utf-8") as f:
        for c in cands:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"prefilter: {len(cands)} candidates (reviewer-voice fields, pos-only); "
          f"{len(neg_by_paper)} papers have negative mentions")


def classify_one(client, c):
    key = "PT::" + hashlib.md5(c["snippet"][:800].encode()).hexdigest()[:14]
    if key in _cache:
        return {**c, "key": key, **_cache[key]}
    msg = [{"role": "system", "content": SYS},
           {"role": "user", "content": f"Paper title: {c['title']}\nReview field: {c['field']}\nSnippet: {c['snippet'][:1500]}"}]
    for a in range(4):
        try:
            r = client.chat.completions.create(model=MODEL, temperature=0,
                response_format={"type": "json_object"}, messages=msg, max_tokens=250)
            cost["in"] += r.usage.prompt_tokens; cost["out"] += r.usage.completion_tokens
            j = json.loads(r.choices[0].message.content)
            with _cache_lock:
                _cache[key] = j
                if len(_cache) % 50 == 0:
                    _save_cache()
            return {**c, "key": key, **j}
        except Exception:
            time.sleep(2 ** a)
    return {**c, "key": key, "is_positive_tell": False, "error": True}


def stage_classify(cost_cap):
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI
    client = OpenAI()
    cands = [json.loads(l) for l in open(os.path.join(OUTDIR, "candidates.jsonl"), encoding="utf-8")]
    out, stop = [], False
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, res in enumerate(ex.map(lambda c: classify_one(client, c), cands)):
            out.append(res)
            usd = (cost["in"] * .15 + cost["out"] * .60) / 1e6
            if i % 200 == 0:
                print(f"  {i}/{len(cands)} classified, ${usd:.2f}")
            if usd > cost_cap:
                print(f"COST CAP ${cost_cap} hit at {i} — stopping (resume-safe)"); stop = True; break
    _save_cache()
    with open(os.path.join(OUTDIR, "classified.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    usd = (cost["in"] * .15 + cost["out"] * .60) / 1e6
    kept = sum(1 for r in out if r.get("is_positive_tell"))
    print(f"classify: {len(out)} done ({'PARTIAL' if stop else 'complete'}), "
          f"{kept} positive, this-run cost ${usd:.2f}")


def stage_select(per_dim, seed=0):
    rows = [json.loads(l) for l in open(os.path.join(OUTDIR, "classified.jsonl"), encoding="utf-8")]
    neg_by_paper = json.load(open(os.path.join(OUTDIR, "neg_mentions_by_paper.json"), encoding="utf-8"))
    good = [r for r in rows if r.get("is_positive_tell") and r.get("specific")
            and not r.get("hedged") and not r.get("author_echo")
            and r.get("dimension") in DIMS
            and r.get("confidence", 0) >= (0.7 if r.get("dimension") == "Methodology" else 0.6)
            and len(r.get("claim_span", "")) > 15
            and len(r.get("snippet", "")) >= 80]
    best = {}
    for r in good:
        k = (r["id"], r["dimension"])
        if k not in best or r["confidence"] > best[k]["confidence"]:
            best[k] = r
    good = list(best.values())
    for r in good:
        negs = [n for n in neg_by_paper.get(r["id"], []) if n["signature"] != r["signature"]]
        r["paper_has_neg_mentions"] = len(negs)
    by_dim = defaultdict(list)
    for r in good:
        by_dim[r["dimension"]].append(r)
    rng = random.Random(seed)
    sel = []
    for d in DIMS:
        pool = sorted(by_dim[d], key=lambda r: (r["paper_has_neg_mentions"], -r["confidence"]))
        sel.extend(pool[:per_dim])
    with open(os.path.join(OUTDIR, "positive_tells.jsonl"), "w", encoding="utf-8") as f:
        for r in sel:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    audit = []
    for d in DIMS:
        pool = [r for r in sel if r["dimension"] == d]
        audit.extend(rng.sample(pool, min(13, len(pool))))
    rng.shuffle(audit)
    audit = audit[:50]
    with open(os.path.join(OUTDIR, "audit_sheet_50.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "paper_id", "title", "dimension", "claim_span", "snippet",
                    "verdict (1=genuine positive tell / 0=not)", "notes"])
        for i, r in enumerate(audit, 1):
            w.writerow([i, r["id"], r["title"], r["dimension"], r["claim_span"],
                        r["snippet"][:500], "", ""])
    print(f"select: {len(good)} usable tells -> {len(sel)} selected "
          f"({Counter(r['dimension'] for r in sel)})")
    print(f"papers w/ neg mentions among selected: "
          f"{sum(1 for r in sel if r['paper_has_neg_mentions'])}")
    print("wrote positive_tells.jsonl + audit_sheet_50.csv")


VERIFY_SYS = """Strict second-pass check of a candidate positive novelty tell from an ICLR review. KEEP only if the REVIEWER, in their own assessing voice, asserts that a specific contribution OF THE PAPER UNDER REVIEW is novel/original/first.

REJECT if any of (observed failure modes):
- novelty is attributed to ANOTHER cited work, not the paper under review
- the snippet merely reports the AUTHORS' claim of novelty ("is proposed as an original contribution", paper-summary voice)
- generic or future-utility praise ("exciting addition to benchmarks") without asserting novelty
- the framing is "application of an existing method to a new task" (application, not novelty)
- descriptive summary of results without the reviewer asserting novelty/firstness
- hedged or partially negated ("novel but...", "somewhat novel")

Return JSON {"keep": bool, "reason": "<10 words"}."""


def stage_verify():
    from openai import OpenAI
    client = OpenAI()
    rows = [json.loads(l) for l in open(os.path.join(OUTDIR, "positive_tells.jsonl"), encoding="utf-8")]
    kept = []
    for r in rows:
        key = "PV::" + r["key"]
        if key not in _cache:
            msg = [{"role": "system", "content": VERIFY_SYS},
                   {"role": "user", "content": f"Paper title: {r['title']}\nClaimed-novel contribution: {r['claim_span']}\nSnippet: {r['snippet'][:1200]}"}]
            for a in range(4):
                try:
                    resp = client.chat.completions.create(model=MODEL, temperature=0,
                        response_format={"type": "json_object"}, messages=msg, max_tokens=80)
                    cost["in"] += resp.usage.prompt_tokens; cost["out"] += resp.usage.completion_tokens
                    _cache[key] = json.loads(resp.choices[0].message.content)
                    break
                except Exception:
                    time.sleep(2 ** a)
        v = _cache.get(key, {})
        if v.get("keep"):
            kept.append(r)
    _save_cache()
    with open(os.path.join(OUTDIR, "positive_tells_verified.jsonl"), "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    usd = (cost["in"] * .15 + cost["out"] * .60) / 1e6
    print(f"verify: {len(kept)}/{len(rows)} kept ({Counter(r['dimension'] for r in kept)}), cost ${usd:.2f}")
    print("wrote positive_tells_verified.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["prefilter", "classify", "select", "verify"])
    ap.add_argument("--cost_cap", type=float, default=1.2)
    ap.add_argument("--per_dim", type=int, default=50)
    args = ap.parse_args()
    {"prefilter": stage_prefilter,
     "classify": lambda: stage_classify(args.cost_cap),
     "select": lambda: stage_select(args.per_dim),
     "verify": stage_verify}[args.stage]()
