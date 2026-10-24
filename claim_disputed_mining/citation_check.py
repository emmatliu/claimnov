import argparse, hashlib, json, os, re, threading, time, random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = next((d for d in (os.path.join(HERE, "iclr_novelty_validation_data"),
                         os.path.join(os.path.dirname(HERE), "iclr_novelty_validation_data"))
             if os.path.isdir(d)), os.path.join(HERE, "iclr_novelty_validation_data"))
ALL = os.path.join(HERE, "outputs", "relevance_all.jsonl")
CACHE = os.path.join(HERE, "outputs", "citation_cache.jsonl")

MODEL, REASONING = "gpt-5-nano", "medium"
PRICE_IN, PRICE_OUT = 0.05, 0.40
WORKERS, RETRIES = 8, 6
MAX_REFS = 80

ARXIV = re.compile(r"\d{4}\.\d{4,5}|[a-z\-]+/\d{7}", re.I)
DOI = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.I)


def ids(text):
    return (set(m.lower() for m in ARXIV.findall(text or "")),
            set(m.lower().rstrip(".") for m in DOI.findall(text or "")))


def ref_line(r):
    raw = (r.get("raw") or "").strip()
    if raw:
        return raw[:220]
    au = ", ".join((r.get("authors") or [])[:3])
    return f'{r.get("title","")}. {au}. {r.get("year","")}'[:220]


SYSTEM = """You decide whether reviewer reference B is THE SAME PAPER as one of the \
entries in paper A's bibliography.

Set matched=true ONLY with strong evidence of same-paper identity:
- the titles clearly denote the same work (a nickname like "DocRED"/"Graphormer" counts if
  the entry's title matches it), OR
- a distinctive author name (or multiple matching authors) + consistent year/topic.

A shared COMMON surname alone is NOT evidence — many different papers share "Zhang", "Wang", \
"Li", "Liu", "Chen", "Yu", "Yang", "Yao", "Xu", "Kim", "Lee", "Singh", "Kumar", "Smith", \
etc. Matching only on a common surname, or only on similar topic, must be matched=false. \
When the specific paper is not clearly present, or you are unsure, return matched=false — a \
false "matched=true" is worse than a miss.

If B is only a bare list marker ("[1]", "citation [2]") with no title/author, return \
matched=false.

Return STRICT JSON:
{"matched": bool, "ref_number": <int or null>, "matched_title": "<the entry, or null>",
 "confidence": 0.0-1.0, "reason": "<brief: what identity evidence, or why not>"}"""


def build(b, refs):
    lines = "\n".join(f"{i+1}. {ref_line(r)}" for i, r in enumerate(refs[:MAX_REFS]))
    u = f'Reviewer reference B:\n"{b}"\n\nPaper A bibliography:\n{lines}\n\n' \
        f'Is B one of these entries? Return the JSON.'
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": u}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost_cap", type=float, default=5.0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(ALL, encoding="utf-8")]
    refs_cache, tasks, seen = {}, [], set()
    for r in rows:
        b, rp, active = r.get("referenced_work_B"), r.get("A_refs_path"), r.get("A_is_active")
        if not (b and rp and active):
            continue
        k = f'{r["A_id"]}::{hashlib.md5(b.encode()).hexdigest()[:12]}'
        if k in seen:
            continue
        seen.add(k)
        if rp not in refs_cache:
            fp = os.path.join(DATA, rp)
            refs_cache[rp] = json.load(open(fp, encoding="utf-8")) if os.path.exists(fp) else []
        if refs_cache[rp]:
            tasks.append((k, r["A_id"], b, refs_cache[rp]))

    done = {}
    if os.path.exists(CACHE):
        for l in open(CACHE, encoding="utf-8"):
            o = json.loads(l)
            if isinstance(o.get("out"), dict) and "error" not in o["out"]:
                done[o["_key"]] = o["out"]

    todo = []
    cf = open(CACHE, "a", encoding="utf-8")
    id_hits = 0
    for k, aid, b, refs in tasks:
        if k in done:
            continue
        bx, bd = ids(b)
        hit = None
        if bx or bd:
            for r in refs:
                rx, rd = ids(r.get("raw", ""))
                if (bx & rx) or (bd & rd):
                    hit = r
                    break
        if hit:
            out = {"verdict": "cited", "method": "id_exact",
                   "matched_ref": ref_line(hit), "confidence": 1.0}
            cf.write(json.dumps({"_key": k, "A_id": aid, "B": b, "out": out}) + "\n")
            done[k] = out
            id_hits += 1
        else:
            todo.append((k, aid, b, refs))
    cf.flush()
    print(f"tasks={len(tasks)}  id_exact cited={id_hits}  cached={len(done)-id_hits}  llm todo={len(todo)}")

    client = OpenAI()
    lock = threading.Lock()
    cost = {"in": 0, "out": 0}

    def work(t):
        k, aid, b, refs = t
        for attempt in range(RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=MODEL, reasoning_effort=REASONING,
                    response_format={"type": "json_object"},
                    messages=build(b, refs), max_completion_tokens=2000)
                j = json.loads(resp.choices[0].message.content)
                matched = bool(j.get("matched"))
                out = {"verdict": "cited" if matched else "uncited", "method": "llm",
                       "matched_ref": j.get("matched_title") if matched else None,
                       "confidence": j.get("confidence"), "reason": j.get("reason")}
                with lock:
                    cost["in"] += resp.usage.prompt_tokens; cost["out"] += resp.usage.completion_tokens
                    cf.write(json.dumps({"_key": k, "A_id": aid, "B": b, "out": out}) + "\n"); cf.flush()
                return k, out
            except Exception as e:
                if attempt == RETRIES - 1:
                    return k, {"error": str(e)[:120]}
                time.sleep(min(2 ** attempt + random.random(), 30))

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(work, t) for t in todo]
        for i, f in enumerate(as_completed(futs)):
            k, out = f.result()
            if "error" not in out:
                done[k] = out
            c = (cost["in"] * PRICE_IN + cost["out"] * PRICE_OUT) / 1e6
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(todo)}  ${c:.2f}", flush=True)
            if c > args.cost_cap:
                print(f"cost cap ${args.cost_cap} hit; stopping."); break
    cf.close()

    verdicts = Counter(o.get("verdict") for o in done.values())
    methods = Counter(o.get("method") for o in done.values())
    total = (cost["in"] * PRICE_IN + cost["out"] * PRICE_OUT) / 1e6
    print(f"\nverdicts: {dict(verdicts)}")
    print(f"methods:  {dict(methods)}")
    print(f"cost this run: ${total:.3f}")


if __name__ == "__main__":
    main()
