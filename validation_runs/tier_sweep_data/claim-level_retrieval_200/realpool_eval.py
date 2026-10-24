import argparse, hashlib, json, os, collections
from concurrent.futures import ThreadPoolExecutor
import claim_retrieval as pm
os.makedirs(os.path.join(pm.HERE, "results"), exist_ok=True)
from openai import OpenAI

client = OpenAI()
PKG = pm.PKG
DIMS, K = pm.DIMS, [1, 3, 5, 10]
MODEL = "text-embedding-3-large"
OUT = os.path.join(pm.HERE, "results", "realpool_results.json")


def norm(s):
    return pm._norm(s)


def ref_path(rp, active):
    base = os.path.basename(rp)
    year = rp.strip("./").split("/")[1] if "/" in rp.strip("./") else ""
    cands = [os.path.join(PKG, "iclr_novelty_validation_data", "refs", year, base),
             os.path.join(PKG, "iclr_novelty_validation_data", rp.replace("./", "")),
             os.path.join(PKG, "from_miri_0719", base)]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def resolve_ref(ref):
    raw, title = (ref.get("raw") or ""), (ref.get("title") or "")
    key = "REF::" + hashlib.md5((title + "|" + raw).encode()).hexdigest()[:12]
    if key in pm._cache:
        return pm._cache[key], key
    res, src = None, None
    aid = pm.arxiv_id_any(raw, title)
    if aid:
        res = pm.arxiv_by_id(aid); src = "arxiv_id" if res else None
    if not res:
        res = pm.crossref_by_query(raw or title); src = "crossref" if res else None
    if not res and len(norm(title)) > 8:
        j = pm.s2_get(pm.S2, {"query": title[:200], "limit": 1, "fields": "title,abstract"})
        d = j.get("data", [])
        if d and d[0].get("abstract") and pm._title_match(title, d[0]["title"]):
            res = {"title": d[0]["title"], "abstract": d[0]["abstract"]}; src = "s2_title"
    if not res and len(norm(title)) > 8:
        res = pm.arxiv_by_title(title); src = "arxiv_title" if res else None
    if res:
        res["_src"] = src
    with pm._cache_lock:
        pm._cache[key] = res; pm._save_cache()
    return res, key


def attr_val(f):
    return (f"{f.get('attribute','')}: {f.get('value','')}").strip() or " "


def embed_texts(texts):
    out = []
    for i in range(0, len(texts), 256):
        r = client.embeddings.create(model=MODEL, input=[t[:512] for t in texts[i:i + 256]])
        out += [d.embedding for d in r.data]
    return out


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / ((sum(x * x for x in a) ** .5) * (sum(y * y for y in b) ** .5) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_dim", type=int, default=5)
    ap.add_argument("--max_refs", type=int, default=25)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(pm.TESTSET, encoding="utf-8")]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["novelty_dimension"]].append(r)
    pilot = [r for d in DIMS for r in by[d][:args.per_dim]]

    hit = collections.defaultdict(lambda: collections.defaultdict(int))
    usable = collections.defaultdict(int)
    trials = []
    for r in pilot:
        aid, dim = r["A_id"], r["novelty_dimension"]
        b = pm.fetch_B_abstract(r)
        A_f = pm.extract_facets("A::" + aid, r["A_title"], r.get("A_abstract", ""))
        B_f = pm.extract_facets("Bf::" + aid, b["title"], b["abstract"]) if b else []
        if not b or not B_f:
            trials.append({"A_id": aid, "dim": dim, "status": "no_B"}); continue

        rp = ref_path(r.get("A_refs_path", ""), r.get("A_is_active", True))
        ref_facets = []
        if rp:
            refs = json.load(open(rp, encoding="utf-8"))
            bt = norm(b["title"])
            cand = [ref for ref in refs if norm(ref.get("title", "")) != bt][:args.max_refs]

            def _resolve_extract(ref):
                ab, key = resolve_ref(ref)
                if not ab:
                    return []
                return [{**f, "_side": "ref"}
                        for f in pm.extract_facets("RF::" + key, ab["title"], ab["abstract"])]

            with ThreadPoolExecutor(max_workers=8) as ex:
                for facs in ex.map(_resolve_extract, cand):
                    ref_facets.extend(facs)
        for f in B_f:
            ref_facets.append({**f, "_side": "B"})
        pool = ref_facets
        B_dim = [f for f in pool if f["_side"] == "B" and f.get("dimension") == dim]
        if not B_dim:
            trials.append({"A_id": aid, "dim": dim, "status": "no_B_dim_facet",
                           "n_refs": len(ref_facets)}); continue

        A_dim = [f for f in A_f if f.get("dimension") == dim]
        pool_dim = [f for f in pool if f.get("dimension") == dim]
        texts = [attr_val(f) for f in pool_dim] + [attr_val(f) for f in A_dim] + [r["reviewer_snippet"][:400]]
        E = embed_texts(texts)
        pe = E[:len(pool_dim)]; ae = E[len(pool_dim):len(pool_dim) + len(A_dim)]; se = E[-1]
        q = ae[max(range(len(A_dim)), key=lambda i: cos(se, ae[i]))] if A_dim else se
        order = sorted(range(len(pool_dim)), key=lambda i: -cos(q, pe[i]))
        ranks = [p for p, i in enumerate(order) if pool_dim[i]["_side"] == "B"]
        best = min(ranks) if ranks else 10 ** 9

        cited = r.get("B_already_cited_by_A", "unknown")
        cbucket = "cited" if cited == "cited" else ("uncited" if cited == "uncited" else "other")
        for bucket in ["overall", dim, cbucket]:
            usable[bucket] += 1
            for k in K:
                if best < k:
                    hit[bucket][k] += 1
        trials.append({"A_id": aid, "dim": dim, "status": "ok", "best_rank": best,
                       "pool_dim": len(pool_dim), "pool_total": len(pool),
                       "cited": cited, "B": r["B_referenced_work"][:50]})

    def tab(bucket):
        n = usable[bucket]
        return {"usable": n, **{f"@{k}": round(hit[bucket][k] / n, 2) if n else None for k in K}}

    summary = {"config": "3-large + attr+value + strict, pool = A_refs + B",
               "overall": tab("overall"),
               "per_dimension": {d: tab(d) for d in DIMS if usable[d]},
               "by_cited": {c: tab(c) for c in ["cited", "uncited", "other"] if usable[c]},
               "avg_pool_dim": round(sum(t.get("pool_dim", 0) for t in trials if t["status"] == "ok")
                                     / max(1, usable["overall"]), 1)}
    json.dump({"summary": summary, "trials": trials}, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
