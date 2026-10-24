import argparse, hashlib, json, os, re, threading, time, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from citation_check import ids, ref_line, build as build_match, MODEL, REASONING, DATA

HERE = os.path.dirname(os.path.abspath(__file__))
ALLREV = os.path.join(DATA, "all_reviews.jsonl")
MENTIONS = os.path.join(DATA, "review_mentions.jsonl")
REL_ALL = os.path.join(HERE, "outputs", "relevance_all.jsonl")
CACHE = os.path.join(HERE, "outputs", "bracket_cache.jsonl")

PRICE_IN, PRICE_OUT = 0.05, 0.40
WORKERS, RETRIES = 8, 6
BRACKET = re.compile(r"\[\d+\]")

RESOLVE_SYS = """You are given an ICLR review and a sentence from it that refers to a \
prior work by a bracket marker like [N]. The review contains a numbered reference list \
(often at the end). Identify the actual paper that [N] refers to. If the marker cannot be \
resolved (no reference list, or N not listed), return resolved=false.

Return STRICT JSON:
{"resolved": bool, "title": str|null, "authors": str|null, "year": str|null,
 "arxiv_or_doi": str|null, "raw": "the full citation string, or null"}"""


def review_text(rev):
    parts = []
    for k, v in rev.items():
        if isinstance(v, str) and len(v) > 40:
            parts.append(v)
    return "\n".join(parts)


def find_review(paper, source, signature):
    notes = paper.get("reviews", []) if source == "review" else paper.get("comments", [])
    if signature:
        for n in notes:
            if n.get("signature") == signature:
                return n
    merged = {}
    for i, n in enumerate(notes):
        for k, v in n.items():
            if isinstance(v, str):
                merged[f"{k}_{i}"] = v
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost_cap", type=float, default=5.0)
    args = ap.parse_args()

    reviews = {json.loads(l)["id"]: json.loads(l) for l in open(ALLREV, encoding="utf-8")}
    sig = {}
    for l in open(MENTIONS, encoding="utf-8"):
        m = json.loads(l)
        sig[hashlib.md5(m["snippet"].encode()).hexdigest()[:12]] = (m.get("source"), m.get("signature"))

    refs_cache = {}

    def load_refs(rp):
        if rp not in refs_cache:
            fp = os.path.join(DATA, rp) if rp else None
            refs_cache[rp] = (json.load(open(fp, encoding="utf-8"))
                              if fp and os.path.exists(fp) else [])
        return refs_cache[rp]

    tasks = []
    for l in open(REL_ALL, encoding="utf-8"):
        r = json.loads(l)
        b = r.get("referenced_work_B") or ""
        if not BRACKET.search(b):
            continue
        h = hashlib.md5(r["snippet"].encode()).hexdigest()[:12]
        tasks.append({"key": f'{r["A_id"]}::{h}', "A_id": r["A_id"], "snippet": r["snippet"],
                      "B": b, "refs_path": r.get("A_refs_path", ""), "active": r.get("A_is_active"),
                      "shash": h})

    done = set()
    if os.path.exists(CACHE):
        for l in open(CACHE, encoding="utf-8"):
            o = json.loads(l)
            if "error" not in o.get("out", {}):
                done.add(o["_key"])
    todo = [t for t in tasks if t["key"] not in done]
    print(f"bracket cases={len(tasks)}  todo={len(todo)} (cached {len(tasks)-len(todo)})")

    client = OpenAI()
    lock = threading.Lock()
    cost = {"in": 0, "out": 0}
    cf = open(CACHE, "a", encoding="utf-8")

    def chat(msgs, cap=2000):
        resp = client.chat.completions.create(
            model=MODEL, reasoning_effort=REASONING, response_format={"type": "json_object"},
            messages=msgs, max_completion_tokens=cap)
        with lock:
            cost["in"] += resp.usage.prompt_tokens; cost["out"] += resp.usage.completion_tokens
        return json.loads(resp.choices[0].message.content)

    def work(t):
        for attempt in range(RETRIES):
            try:
                src, signature = sig.get(t["shash"], (None, None))
                paper = reviews.get(t["A_id"])
                rev = find_review(paper, src, signature) if paper else {}
                rtext = review_text(rev)[:8000]
                marker = BRACKET.search(t["B"]).group()
                res = chat([{"role": "system", "content": RESOLVE_SYS},
                            {"role": "user", "content":
                             f'Sentence: "{t["snippet"][:400]}"\nMarker to resolve: {marker}\n\n'
                             f'Full review:\n{rtext}\n\nResolve {marker}.'}])
                resolved_b = res.get("raw") if res.get("resolved") else None
                out = {"resolved_B": resolved_b, "resolve_conf": None}
                if not resolved_b:
                    out.update({"verdict": "indeterminate", "method": "bracket_unresolved",
                                "matched_ref": None})
                else:
                    refs = load_refs(t["refs_path"]) if t["active"] else []
                    bx, bd = ids((res.get("arxiv_or_doi") or "") + " " + (resolved_b or ""))
                    hit = None
                    for r in refs:
                        rx, rd = ids(r.get("raw", ""))
                        if (bx & rx) or (bd & rd):
                            hit = r; break
                    if hit:
                        out.update({"verdict": "cited", "method": "id_exact", "matched_ref": ref_line(hit)})
                    elif refs:
                        j = chat(build_match(resolved_b, refs))
                        matched = bool(j.get("matched"))
                        v = "cited" if matched else ("uncited" if (j.get("confidence") or 0) >= 0.5
                                                     else "indeterminate")
                        out.update({"verdict": v, "method": "llm",
                                    "matched_ref": j.get("matched_title") if matched else None})
                    else:
                        out.update({"verdict": "unknown", "method": "no_refs", "matched_ref": None})
                with lock:
                    cf.write(json.dumps({"_key": t["key"], "A_id": t["A_id"], "B_raw": t["B"],
                                         "out": out}) + "\n"); cf.flush()
                return t["key"], out
            except Exception as e:
                if attempt == RETRIES - 1:
                    return t["key"], {"error": str(e)[:120]}
                time.sleep(min(2 ** attempt + random.random(), 30))

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(work, t) for t in todo]
        for i, f in enumerate(as_completed(futs)):
            f.result()
            c = (cost["in"] * PRICE_IN + cost["out"] * PRICE_OUT) / 1e6
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(todo)}  ${c:.2f}", flush=True)
            if c > args.cost_cap:
                print("cost cap hit; stopping."); break
    cf.close()
    print(f"cost this run: ${(cost['in']*PRICE_IN + cost['out']*PRICE_OUT)/1e6:.3f}")


if __name__ == "__main__":
    main()
