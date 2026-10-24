import argparse, json, os, re, time, threading
from collections import defaultdict
import requests
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.environ.get("NOVELTY_DATA", os.path.dirname(HERE))
TESTSET = os.path.join(PKG, "from_miri_0719", "testset_clean.jsonl")
import tempfile
_CDIR = os.path.join(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()), "claim_retrieval_cache")
os.makedirs(_CDIR, exist_ok=True)
CACHE = os.path.join(_CDIR, "facet_cache.json")
OUT = os.path.join(HERE, "results", "pilot_results.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

EXTRACT_MODEL, EMBED_MODEL = "gpt-4o-mini", "text-embedding-3-small"
S2 = "https://api.semanticscholar.org/graph/v1/paper/search"
DIMS = ["Problem", "Methodology", "Resource", "Result"]

SCHEMA_HINT = ("Extract each headline contribution of the paper as a typed facet. "
    "Return JSON {\"facets\":[{\"dimension\":\"Problem|Methodology|Resource|Result\","
    "\"subcat\":\"PB1..RE3\",\"attribute\":\"the comparable axis (short noun phrase)\","
    "\"value\":\"what the paper asserts on that axis\"}]}. Attribute is the retrieval key.")

S2_BASE = "https://api.semanticscholar.org/graph/v1/paper"
client = OpenAI()
_cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
cost = {"in": 0, "out": 0, "emb": 0}
_LOCAL = None
_s2_last = [0.0]

_arxiv_lock = threading.Lock()
_s2_lock = threading.Lock()
_cache_lock = threading.RLock()
_oai_sem = threading.Semaphore(4)


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def load_local():
    import csv
    d = {}
    p = os.path.join(PKG, "iclr_novelty_validation_data", "papers.csv")
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["abstract"]:
                d[_norm(row["title"])] = {"title": row["title"], "abstract": row["abstract"]}
    return d


def s2_get(url, params):
    with _s2_lock:
        for a in range(3):
            wait = 1.2 - (time.time() - _s2_last[0])
            if wait > 0:
                time.sleep(wait)
            _s2_last[0] = time.time()
            try:
                _h = {}
                if "semanticscholar.org" in url and os.environ.get("S2_API_KEY"):
                    _h["x-api-key"] = os.environ["S2_API_KEY"]
                r = requests.get(url, params=params, timeout=25, headers=_h)
                if r.status_code == 429:
                    time.sleep(2 ** a + 1); continue
                return r.json()
            except Exception:
                time.sleep(2 ** a)
        return {}


def _arxiv_parse(xml):
    ent = re.search(r"<entry>(.*?)</entry>", xml, re.S)
    if not ent:
        return None
    e = ent.group(1)
    t = re.search(r"<title>(.*?)</title>", e, re.S)
    s = re.search(r"<summary>(.*?)</summary>", e, re.S)
    if not s:
        return None
    return {"title": re.sub(r"\s+", " ", t.group(1)).strip() if t else "",
            "abstract": re.sub(r"\s+", " ", s.group(1)).strip()}


_arxiv_last = [0.0]


def arxiv_get(params):
    with _arxiv_lock:
        for a in range(3):
            wait = 1.5 - (time.time() - _arxiv_last[0])
            if wait > 0:
                time.sleep(wait)
            _arxiv_last[0] = time.time()
            try:
                r = requests.get("https://export.arxiv.org/api/query", params=params, timeout=30,
                                 headers={"User-Agent": "novelty-validation/0.1 (research; mailto:you@example.org)"})
                if r.status_code == 200:
                    return r.text
            except Exception:
                time.sleep(2 ** a)
        return ""


def arxiv_by_id(aid):
    return _arxiv_parse(arxiv_get({"id_list": aid}))


def _title_match(q, got):
    qt = set(re.findall(r"[a-z0-9]+", q.lower()))
    gt = set(re.findall(r"[a-z0-9]+", got.lower()))
    if len(qt) < 3:
        return False
    return len(qt & gt) / len(qt) >= 0.6


def arxiv_by_title(title):
    r = _arxiv_parse(arxiv_get({"search_query": f'ti:"{title}"', "max_results": 1}))
    return r if (r and _title_match(title, r["title"])) else None


def clean_title(s):
    s = s.split(";")[0]
    m = re.search(r"\b(?:19|20)\d{2}[a-z]?\.\s+(.+)", s)
    if m and len(m.group(1)) > 10:
        s = m.group(1)
    s = re.sub(r"\((?:[^)]*\d{4}[^)]*)\)", "", s)
    s = re.sub(r"\bar[Xx]iv:\s*\S+", "", s)
    s = re.sub(r"^(?:[A-Z][A-Za-z\-]+|de|van|von)"
               r"(?:\s*(?:&|and)\s*[A-Z][A-Za-z\-]+)?"
               r"(?:\s+et al\.?)?,?\s*\d{4}[.,]?\s*", "", s)
    return s.strip(" .,")


def title_from_prov(prov):
    if not prov:
        return ""
    p = re.sub(r"^\s*(\[[^\]]*\]\s*)+", "", prov).strip()
    m = re.match(r"^([A-Z][A-Za-z.\- ]+(?:,[^.]*?)?(?:,? and [^.]*?)?)\.\s+(.+)", p)
    rest = m.group(2) if m else p
    rest = re.split(r"(?<=[.?])\s|\s+arXiv|\s+In\s|\s+https?://", rest)[0]
    return rest.strip(" .")


def arxiv_id_any(*fields):
    m = re.search(r"ar[Xx]iv:\s*([0-9]+\.[0-9]+)", " ".join(f or "" for f in fields))
    return m.group(1) if m else None


def crossref_by_query(text):
    if not text or len(_norm(text)) < 8:
        return None
    try:
        r = requests.get("https://api.crossref.org/works",
                         params={"query.bibliographic": text[:300], "rows": 1,
                                 "select": "title,abstract",
                                 "mailto": "you@example.org"}, timeout=25)
        items = r.json().get("message", {}).get("items", [])
        if items and items[0].get("abstract"):
            ab = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", items[0]["abstract"])).strip()
            ti = (items[0].get("title") or [""])[0]
            if ab and _title_match(text, ti):
                return {"title": ti, "abstract": ab}
    except Exception:
        pass
    return None


def _save_cache():
    with _cache_lock:
        tmp = CACHE + f".tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False)
        os.replace(tmp, CACHE)


def extract_facets(key, title, abstract):
    if key in _cache:
        return _cache[key]
    if not abstract:
        return []
    msg = [{"role": "system", "content": SCHEMA_HINT},
           {"role": "user", "content": f"Title: {title}\nAbstract: {abstract[:2500]}"}]
    for a in range(4):
        try:
            with _oai_sem:
                r = client.chat.completions.create(model=EXTRACT_MODEL, temperature=0,
                    response_format={"type": "json_object"}, messages=msg, max_tokens=800)
            cost["in"] += r.usage.prompt_tokens; cost["out"] += r.usage.completion_tokens
            f = json.loads(r.choices[0].message.content).get("facets", [])
            with _cache_lock:
                _cache[key] = f; _save_cache()
            return f
        except Exception:
            time.sleep(2 ** a)
    return []


def fetch_B_abstract(row):
    global _LOCAL
    ck = "B::" + row["A_id"]
    if _cache.get(ck):
        return _cache[ck]
    if _LOCAL is None:
        _LOCAL = load_local()
    B_title = row.get("B_referenced_work") or ""
    ident = row.get("B_identifier") or ""
    prov = row.get("B_provenance") or ""
    cands = []
    for t in (clean_title(B_title), title_from_prov(prov)):
        if t and len(_norm(t)) > 8 and _norm(t) not in [_norm(c) for c in cands]:
            cands.append(t)
    res = None
    src = None

    for t in [B_title] + cands:
        res = _LOCAL.get(_norm(t))
        if res:
            src = "local"; break

    if not res:
        aid = arxiv_id_any(ident, B_title, prov)
        if aid:
            res = arxiv_by_id(aid); src = "arxiv_id" if res else None

    for t in cands:
        if res:
            break
        res = arxiv_by_title(t); src = "arxiv_title" if res else None

    if not res:
        m = re.search(r"DOI:\s*(\S+)", ident, re.I)
        if m:
            j = s2_get(f"{S2_BASE}/DOI:{m.group(1).rstrip('.')}", {"fields": "title,abstract"})
            if j.get("abstract"):
                res = {"title": j.get("title") or (cands[0] if cands else B_title),
                       "abstract": j["abstract"]}; src = "doi"

    for t in cands:
        if res:
            break
        j = s2_get(S2, {"query": t[:200], "limit": 1, "fields": "title,abstract"})
        d = j.get("data", [])
        if d and d[0].get("abstract") and _title_match(t, d[0]["title"]):
            res = {"title": d[0]["title"], "abstract": d[0]["abstract"]}; src = "s2_title"

    if res:
        res["_src"] = src
    with _cache_lock:
        _cache[ck] = res; _save_cache()
    return res


def embed(texts):
    r = client.embeddings.create(model=EMBED_MODEL, input=texts)
    cost["emb"] += r.usage.prompt_tokens
    return [d.embedding for d in r.data]


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / ((sum(x * x for x in a) ** .5) * (sum(y * y for y in b) ** .5) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_dim", type=int, default=5)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(TESTSET, encoding="utf-8")]
    by = defaultdict(list)
    for r in rows:
        by[r["novelty_dimension"]].append(r)
    pilot = []
    for d in DIMS:
        pilot.extend(by[d][:args.per_dim])
    print(f"pilot: {len(pilot)} pairs ({args.per_dim}/dim)")

    Af, Bf, Bok = {}, {}, {}
    for r in pilot:
        Af[r["A_id"]] = extract_facets("A::" + r["A_id"], r["A_title"], r.get("A_abstract", ""))
        b = fetch_B_abstract(r)
        Bok[r["A_id"]] = bool(b)
        Bf[r["A_id"]] = extract_facets("Bf::" + r["A_id"], b["title"], b["abstract"]) if b else []

    allf = []
    for aid, fs in {**{("A", k): v for k, v in Af.items()}, **{("B", k): v for k, v in Bf.items()}}.items():
        for f in fs:
            f2 = dict(f); f2["_owner"] = aid[1]; f2["_side"] = aid[0]; allf.append(f2)
    attrs = [f.get("attribute", "") for f in allf]
    embs = embed(attrs) if attrs else []
    for f, e in zip(allf, embs):
        f["_emb"] = e

    MODES = ["strict", "relaxed"]
    hit = {m: defaultdict(lambda: defaultdict(int)) for m in MODES}
    usable = {m: defaultdict(int) for m in MODES}
    tot = defaultdict(int)
    trials = []
    for r in pilot:
        aid, dim = r["A_id"], r["novelty_dimension"]; tot[dim] += 1
        B_all = [f for f in allf if f["_side"] == "B" and f["_owner"] == aid]
        if not Bok[aid] or not B_all:
            trials.append({"A_id": aid, "dim": dim, "status": "no_B_resolved_or_facet"}); continue
        A_dim = [f for f in allf if f["_side"] == "A" and f["_owner"] == aid and f.get("dimension") == dim]
        qe = embed([r["reviewer_snippet"][:400]])[0]
        if A_dim:
            query = max(A_dim, key=lambda f: cos(qe, f["_emb"])); q = query["_emb"]; qattr = query.get("attribute")
        else:
            q = qe; qattr = "(snippet)"
        B_dim = [f for f in B_all if f.get("dimension") == dim]
        rec = {"A_id": aid, "dim": dim, "status": "ok", "query_attr": qattr,
               "B": r["B_referenced_work"][:60], "best_rank": {}, "pool": {}}
        for m in MODES:
            if m == "strict":
                if not B_dim:
                    continue
                pool = [f for f in allf if f.get("dimension") == dim and f["_owner"] != aid] + B_dim
            else:
                pool = [f for f in allf if f["_owner"] != aid] + B_all
            scored = sorted(pool, key=lambda f: -cos(q, f["_emb"]))
            ranks = [i for i, f in enumerate(scored) if f["_side"] == "B" and f["_owner"] == aid]
            best = min(ranks) if ranks else 10 ** 9
            usable[m][dim] += 1
            rec["best_rank"][m] = best; rec["pool"][m] = len(pool)
            for k in args.k:
                if best < k:
                    hit[m][dim][k] += 1
        trials.append(rec)

    total = (cost["in"] * .15 + cost["out"] * .60 + cost["emb"] * .02) / 1e6
    summary = {"k": args.k, "cost_usd": round(total, 3)}
    for m in MODES:
        per = {}
        for d in DIMS:
            if usable[m][d]:
                per[d] = {"usable": usable[m][d],
                          **{f"success@{k}": round(hit[m][d][k] / usable[m][d], 2) for k in args.k}}
        ov = sum(usable[m].values())
        summary[m] = {"per_dimension": per, "overall": ({"usable": ov,
            **{f"success@{k}": round(sum(hit[m][d][k] for d in DIMS) / ov, 2) for k in args.k}} if ov else {})}
    json.dump({"summary": summary, "trials": trials}, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=2))
    print("cost this run: $%.3f" % total, "| wrote", OUT)


if __name__ == "__main__":
    main()
