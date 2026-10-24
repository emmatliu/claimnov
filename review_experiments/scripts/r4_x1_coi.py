import json, os, sys, threading, random, hashlib
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("R4_EXP", "x1_coi")
import common as C
import r4_common as R
from harness import gates
from harness.sandbox import Pool
import harness.sandbox as SB
llm = C.llm
def cache_only_embed(texts, model="text-embedding-3-small"):
    out = []
    for t in texts:
        fp = os.path.join(llm.CACHE, f"e_{hashlib.sha256((model + '::' + t).encode()).hexdigest()[:24]}.json")
        if not os.path.exists(fp): raise RuntimeError("embedding cache miss")
        out.append(json.load(open(fp)))
    return out
llm.embed = cache_only_embed; SB.llm.embed = cache_only_embed
SON = llm.SONNET_46
def paper_cached(content, max_tokens, tag):
    msgs = [{"role": "user", "content": content}]
    for fp in (C.bedrock_cache_path(SON, content, max_tokens, tag), C.ant_cache_path("claude-sonnet-4-6", content, max_tokens, tag)):
        if os.path.exists(fp): return json.load(open(fp))["text"]
    return None
A = json.load(open(f"{R.OUT}/r4_x1_anchors.json"))
N = int(os.environ.get("NCOI", "120"))
U = C.later_units(); random.Random(4).shuffle(U); U = U[:N]
OUTF = f"{R.OUT}/r4_x1_coi.jsonl"
done = {json.loads(l)["key"] for l in open(OUTF) if not json.loads(l).get("error")} if os.path.exists(OUTF) else set()
fo = open(OUTF, "a"); lock = threading.Lock(); _pools = {}

def run(u):
    if u["key"] in done: return
    slug = u["meta"]["anchor"]; an = A[slug]
    with lock:
        if slug not in _pools: _pools[slug] = Pool(u["recs"])
    blob = f"{u['idea']['title']}: {u['idea']['description']}"
    try:
        qtext = paper_cached(gates.COI_QUERY_PROMPT.format(idea=blob, topic="machine learning"), 2000, f"coiqsn_{u['key']}")
        if qtext is None: raise RuntimeError("query call not cached")
        try: queries = json.loads(gates._coi_extract(qtext, "queries"))[:3]
        except Exception: queries = [blob]
        papers, seen = [], set()
        for q in queries:
            for p in _pools[slug].search(q, k=5):
                if p["title"] not in seen: seen.add(p["title"]); papers.append(p)
        content0 = "".join(f"Paper {i}: Title:{p['title']}\n Abstract:{p['abstract']}\n" for i, p in enumerate(papers))
        paper_txt = paper_cached(gates.COI_CHECK_PROMPT.format(idea=blob, papers_content=content0), 2000, f"coicsn_{u['key']}")
        papers1 = [{"title": an["title"], "abstract": an["abstract"]}] + papers
        content1 = "".join(f"Paper {i}: Title:{p['title']}\n Abstract:{p['abstract']}\n" for i, p in enumerate(papers1))
        txt, meta = R.call("sonnet", gates.COI_CHECK_PROMPT.format(idea=blob, papers_content=content1), 2000, f"r4x1coi_{u['key']}")
        sim = gates._coi_extract(txt, "similar"); rej = sim == "1"; ev = None
        try: ev = papers1[int(gates._coi_extract(txt, "similar_paper_id"))]["title"]
        except Exception: pass
        p_rej = (gates._coi_extract(paper_txt, "similar") == "1") if paper_txt else None
        rec = {"key": u["key"], "anchor": slug, "rej": rej, "evidence": ev if rej else None, "named_paper_any": ev,
               "cites_anchor": bool(rej and ev == an["title"]), "names_anchor_any": ev == an["title"],
               "paper_check_cached": paper_txt is not None, "paper_rej_from_cache": p_rej, "n_papers": len(papers1), "meta": meta}
    except R.Budget: return
    except Exception as e: rec = {"key": u["key"], "error": str(e)[:300]}
    with lock: fo.write(json.dumps(rec) + "\n"); fo.flush()
try:
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "8"))) as ex: list(ex.map(run, U))
finally:
    print(R.summary(), flush=True)
