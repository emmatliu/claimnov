import json, os, sys, re, threading
from concurrent.futures import ThreadPoolExecutor
import common as C
llm = C.llm
V = sys.argv[1]; CAP = float(os.environ.get("CAP", "9"))
Q_OLD = "Decide whether the IDEA is already done: is there a paper in the profile that does substantially what the idea proposes, same core problem AND same core approach?"
G_OLD = " Judge the idea as a whole: an anticipated peripheral claim does not make the idea non-novel, but an anticipated core contribution does, even if side claims are new. Superficial topic overlap or related prior work that the idea builds on = novel."
Q_NEW = "Would an expert reviewer say this idea's main contribution has already been published in one of the profiled papers?"
def reword(p):
    assert p.count(Q_OLD + G_OLD) == 1
    return p.replace(Q_OLD, Q_NEW) if V == "A" else p.replace(Q_OLD + G_OLD, Q_NEW)
OUTF = f"{C.OUTD}/r2_e10_reword_{V}.jsonl"
done = {(json.loads(l)["set"], json.loads(l)["key"]) for l in open(OUTF) if not json.loads(l).get("error")} if os.path.exists(OUTF) else set()
R = [json.loads(l) for l in open(f"{C.OUTD}/r1_profile_prompts.jsonl")]
U = [r for r in R if r["status"] == "byte_identical" and r["set"] in ("later", "disg") and (r["set"], r["key"]) not in done]
out = open(OUTF, "a"); lock = threading.Lock()
def pools(r):
    if r["set"] == "later":
        slug = r["key"].split("::")[0]; return json.load(open(f"{C.anchor_dir(slug)}/pool_preT.json"))[:400], None
    s = int(re.match(r"seed(\d+)_", r["key"]).group(1)); pin = json.load(open(f"{C.P}/data/pools/seed{s}.json"))["pool"]
    B = next(p for p in pin if p.get("is_B"))
    return (pin if r["key"].endswith("B_in") else [p for p in pin if not p.get("is_B")]), B
def parse(txt, corpus):
    try: j = json.loads(re.search(r"\{.*\}", txt, re.S).group(0)); pe = False
    except Exception: j = {"decision": "novel"}; pe = True
    veto = str(j.get("decision", "")).lower().startswith("not")
    t = str(j.get("anticipating_paper") or "")
    c = next((x for x in corpus if x["title"].lower()[:60] == t.lower()[:60]), None) if t else None
    return veto, (c["id"] if c else None), pe
def run(r):
    if llm.cost().get("usd", 0) > CAP: return
    corpus, B = pools(r)
    pr = reword(r["prompt"])
    try:
        txt = C.sonnet([{"role": "user", "content": pr}], 1500, f"r2rw{V}_{r['tag']}")
        veto, aid, pe = parse(txt, corpus)
        ov, oid, _ = parse(r["paper_verdict_text"], corpus)
        rec = {"set": r["set"], "key": r["key"], "rej": veto, "anticipating_id": aid, "parse_error": pe,
               "orig_rej": ov, "orig_anticipating_id": oid, "B_id": B["id"] if B else None, "text": txt, "backend": C.BACKEND}
    except Exception as e:
        rec = {"set": r["set"], "key": r["key"], "error": str(e)[:300]}
    with lock:
        out.write(json.dumps(rec) + "\n"); out.flush()
print(V, len(U), "to run", flush=True)
with ThreadPoolExecutor(max_workers=int(os.environ.get("WORKERS", "4"))) as ex:
    list(ex.map(run, U))
print("DONE", V, "cost", llm.cost(), flush=True)
