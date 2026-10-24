import argparse, hashlib, json, os, re, threading, time, random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = next((d for d in (os.path.join(HERE, "iclr_novelty_validation_data"),
                         os.path.join(os.path.dirname(HERE), "iclr_novelty_validation_data"))
             if os.path.isdir(d)), os.path.join(HERE, "iclr_novelty_validation_data"))
MENTIONS = os.path.join(DATA, "review_mentions.jsonl")
CACHE = os.path.join(HERE, "outputs", "relevance_cache.jsonl")

MODEL, REASONING = "gpt-5-nano", "medium"
PRICE_IN, PRICE_OUT = 0.05, 0.40
WORKERS, RETRIES = 8, 6

REF = re.compile(r"(et al\.?|arxiv|\[\d+\]|\(\d{4}\)|\b\d{4}\b|proposed by|introduced by)", re.I)
ANTIC = re.compile(r"(same (idea|method|approach|mechanism|architecture|technique|model|"
    r"framework|result|finding|dataset|benchmark|objective|loss)|identical to|isn'?t this|"
    r"already (been )?(proposed|done|shown|studied|explored|introduced|reported|addressed|"
    r"known|exist|existing|established)|has (already )?been (proposed|done|shown|studied|explored)|"
    r"essentially (the same|identical)|(not|lacks|limited|little|marginal) novelt|nothing new|"
    r"overlap(s|ping)? with|very similar to|closely related to|reminiscent of|"
    r"improve(ment)?s? over|differ(ence|s)? (from|with)|how (is|does) (this|it) (differ|improve))", re.I)
RESOURCE = re.compile(r"(datas?et|benchmark|corpus|collection|repositor|toolkit|simulator|"
    r"annotation|test ?bed|data ?source)", re.I)
RES_ANTIC = re.compile(r"(larger|bigger|broader|smaller|more (comprehensive|diverse|extensive|"
    r"complete)|already (exist|availab|released|collect|introduc|propos|public)|existing|prior|"
    r"previous|compared (to|with)|similar to|same as|superset|subset|cover(s|ed|age)?|"
    r"contain(s|ed)?|as (large|big|broad)|better than|improve)", re.I)
PROB_EXTRA = re.compile(r"((problem|setting|task|scenario|question|framing|perspective|"
    r"formulation) (is |has |was |been )?.{0,30}(already|previously|prior|well[- ]?studied|"
    r"well[- ]?known|standard|not new|explored|addressed|posed)|studied in|explored in|"
    r"addressed in|posed in)", re.I)


def in_pool(s):
    return bool(REF.search(s) and (ANTIC.search(s)
                or (RESOURCE.search(s) and RES_ANTIC.search(s)) or PROB_EXTRA.search(s)))


SYSTEM = """You label sentences from ICLR peer reviews for a claim-retrieval validation \
study. We want cases where a reviewer points at a SPECIFIC prior work that is directly \
relevant to a CONTRIBUTION of the paper under review — to use (paper contribution -> that \
prior work) as retrieval ground truth.

A RELEVANCE CASE: the reviewer names a SPECIFIC, identifiable prior work B (a citation, \
author-year, or arXiv id) and ties it to a SPECIFIC contribution of the paper A — its \
method, released resource, empirical result, or problem framing — as something A's \
contribution overlaps with, resembles, is anticipated by, extends, or should be measured \
against. The strength of the complaint does NOT matter: both "isn't this the same as B?" \
and "this should be compared with B / is closely related to B" qualify.

Return is_relevant_case=false when:
- it is a bare citation / reference-list / bibliography line / URL with no reviewer point;
- it is a "cite B for completeness" / general topical-relatedness remark not tied to a
  specific contribution of A;
- B is only a dataset/tool the paper USES for experiments, not a prior work weighed
  against A's own contribution;
- it is praise, or a style/clarity/notation comment;
- there is no identifiable named prior work B.

DIMENSION = which contribution B is relevant to:
- Problem (gap/framing/setting) | Methodology (method/design/evaluation)
- Resource (released dataset/tool/benchmark) | Result (finding/number).
RELATION = anticipation (already done / same as) | comparison (compare with / improve over /
similar to) | overlap (partial) | other.

Return STRICT JSON:
{"is_relevant_case": bool, "dimension": "Problem|Methodology|Resource|Result|null",
 "relation": "anticipation|comparison|overlap|other|null",
 "referenced_work_B": "name + cite, or null", "has_identifiable_ref": bool,
 "confidence": 0.0-1.0}"""

FEWSHOT = [
 ("Isn't the proposed routing mechanism essentially the same as the one in Zhou et al. 2021?",
  {"is_relevant_case": True, "dimension": "Methodology", "relation": "anticipation",
   "referenced_work_B": "Zhou et al. 2021", "has_identifiable_ref": True, "confidence": 0.95}),
 ("The proposed approach is closely related to and should be compared against the method "
  "of Iscen et al.; how does it differ?",
  {"is_relevant_case": True, "dimension": "Methodology", "relation": "comparison",
   "referenced_work_B": "Iscen et al.", "has_identifiable_ref": True, "confidence": 0.85}),
 ("How does the proposed dataset improve over the DocRED benchmark (Yao et al., 2019)?",
  {"is_relevant_case": True, "dimension": "Resource", "relation": "comparison",
   "referenced_work_B": "DocRED (Yao et al., 2019)", "has_identifiable_ref": True, "confidence": 0.9}),
 ("The authors may want to cite Smith et al. 2020 for completeness.",
  {"is_relevant_case": False, "dimension": None, "relation": None,
   "referenced_work_B": "Smith et al. 2020", "has_identifiable_ref": True, "confidence": 0.85}),
 ("[3] Ma, Cheng, et al. Overcoming Forgetting. arXiv:1801.01423 (2018).",
  {"is_relevant_case": False, "dimension": None, "relation": None,
   "referenced_work_B": "Ma, Cheng et al. 2018", "has_identifiable_ref": True, "confidence": 0.95}),
]


def build(snippet):
    msgs = [{"role": "system", "content": SYSTEM}]
    for s, j in FEWSHOT:
        msgs += [{"role": "user", "content": s}, {"role": "assistant", "content": json.dumps(j)}]
    msgs.append({"role": "user", "content": snippet[:900]})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost_cap", type=float, default=5.0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(MENTIONS, encoding="utf-8")]
    pool = [r for r in rows if in_pool(r["snippet"])]
    for r in pool:
        r["_key"] = f'{r["id"]}::{r["source"]}::{hashlib.md5(r["snippet"].encode()).hexdigest()[:12]}'

    done = {}
    if os.path.exists(CACHE):
        for l in open(CACHE, encoding="utf-8"):
            o = json.loads(l)
            if isinstance(o.get("out"), dict) and "error" not in o["out"]:
                done[o["_key"]] = o["out"]
    todo = [r for r in pool if r["_key"] not in done]
    print(f"relevance pool={len(pool)}  todo={len(todo)} (cached {len(pool)-len(todo)})")

    client = OpenAI()
    lock = threading.Lock()
    cost = {"in": 0, "out": 0}
    cf = open(CACHE, "a", encoding="utf-8")

    def work(r):
        for attempt in range(RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=MODEL, reasoning_effort=REASONING,
                    response_format={"type": "json_object"},
                    messages=build(r["snippet"]), max_completion_tokens=2000)
                out = json.loads(resp.choices[0].message.content)
                with lock:
                    cost["in"] += resp.usage.prompt_tokens; cost["out"] += resp.usage.completion_tokens
                    cf.write(json.dumps({"_key": r["_key"], "id": r["id"], "out": out}) + "\n"); cf.flush()
                return r["_key"], out
            except Exception as e:
                if attempt == RETRIES - 1:
                    return r["_key"], {"error": str(e)[:120]}
                time.sleep(min(2 ** attempt + random.random(), 30))

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for i, f in enumerate(as_completed(futs)):
            k, out = f.result(); done[k] = out
            c = (cost["in"] * PRICE_IN + cost["out"] * PRICE_OUT) / 1e6
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(todo)}  ${c:.2f}", flush=True)
            if c > args.cost_cap:
                print(f"cost cap ${args.cost_cap} hit; stopping."); break
    cf.close()

    dim = defaultdict(lambda: defaultdict(int)); dim_papers = defaultdict(set)
    for r in pool:
        o = done.get(r["_key"], {})
        if not (isinstance(o, dict) and o.get("is_relevant_case")
                and o.get("has_identifiable_ref") and o.get("confidence", 0) >= 0.6):
            continue
        d = o.get("dimension") or "(none)"
        dim[d][o.get("relation") or "(none)"] += 1
        dim_papers[d].add(r["id"])
    total = (cost["in"] * PRICE_IN + cost["out"] * PRICE_OUT) / 1e6
    print("\nRELEVANCE cases (conf>=.6, identifiable B), by dimension x relation:")
    for d in ["Problem", "Methodology", "Resource", "Result"]:
        rels = dim.get(d, {})
        tot = sum(rels.values())
        print(f"  {d:12s}: {tot:4d} cases ({len(dim_papers.get(d,set()))} papers)  "
              + "  ".join(f"{k}={v}" for k, v in sorted(rels.items())))
    print(f"cost this run: ${total:.3f}")


if __name__ == "__main__":
    main()
