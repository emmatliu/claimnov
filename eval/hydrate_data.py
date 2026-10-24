import json
import os
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data")
DST = os.path.join(ROOT, "eval", "data")
CACHE = os.path.join(ROOT, "eval", "cache", "abstracts.json")
S2_KEY = os.environ.get("S2_API_KEY")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
ARXIV = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")

cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def save_cache():
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(cache, open(CACHE, "w"))


def load(p):
    return json.load(open(p, encoding="utf-8"))


def dump(obj, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(obj, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def s2_key(pid):
    pid = pid[2:] if pid.startswith("B_") else pid
    if HEX40.match(pid):
        return pid
    if ARXIV.match(pid):
        return "ARXIV:" + pid
    return None


def s2_batch(keys):
    headers = {"x-api-key": S2_KEY} if S2_KEY else {}
    out = {}
    for i in range(0, len(keys), 400):
        chunk = keys[i:i + 400]
        for attempt in range(10):
            try:
                r = requests.post("https://api.semanticscholar.org/graph/v1/paper/batch",
                                  params={"fields": "title,abstract,externalIds"},
                                  json={"ids": chunk}, headers=headers, timeout=120)
                if r.status_code == 200:
                    for k, p in zip(chunk, r.json()):
                        out[k] = p
                    break
            except requests.RequestException:
                pass
            time.sleep(min(60, 3 * 2 ** attempt))
        else:
            sys.exit(f"Semantic Scholar batch failed for ids {chunk[0]}..; set S2_API_KEY or retry later")
        time.sleep(1.1)
        print(f"  Semantic Scholar: {min(i + 400, len(keys))}/{len(keys)}", flush=True)
    return out


def arxiv_abstract(aid):
    for attempt in range(5):
        try:
            r = requests.get("http://export.arxiv.org/api/query", params={"id_list": aid}, timeout=60)
            if r.status_code == 200:
                ns = {"a": "http://www.w3.org/2005/Atom"}
                e = ET.fromstring(r.text).find("a:entry", ns)
                s = e.find("a:summary", ns) if e is not None else None
                time.sleep(3)
                return " ".join(s.text.split()) if s is not None and s.text else None
        except (requests.RequestException, ET.ParseError):
            pass
        time.sleep(5 * (attempt + 1))
    return None


def s2_title_abstract(title):
    headers = {"x-api-key": S2_KEY} if S2_KEY else {}
    for attempt in range(6):
        try:
            r = requests.get("https://api.semanticscholar.org/graph/v1/paper/search/match",
                             params={"query": title, "fields": "title,abstract,externalIds"},
                             headers=headers, timeout=60)
            if r.status_code == 200:
                p = (r.json().get("data") or [{}])[0]
                ab = p.get("abstract")
                if not ab and (p.get("externalIds") or {}).get("ArXiv"):
                    ab = arxiv_abstract(p["externalIds"]["ArXiv"])
                time.sleep(1.1)
                return ab
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(min(60, 3 * 2 ** attempt))
    return None


_or_token = None


def openreview_note(nid):
    global _or_token
    user, pw = os.environ.get("OPENREVIEW_USERNAME"), os.environ.get("OPENREVIEW_PASSWORD")
    if not (user and pw):
        return {}
    if _or_token is None:
        try:
            _or_token = requests.post("https://api2.openreview.net/login", json={"id": user, "password": pw},
                                      timeout=60).json().get("token", "")
        except requests.RequestException:
            _or_token = ""
    headers = {"Authorization": f"Bearer {_or_token}"} if _or_token else {}
    for base in ("https://api2.openreview.net", "https://api.openreview.net"):
        for attempt in range(4):
            try:
                r = requests.get(f"{base}/notes", params={"id": nid}, headers=headers, timeout=60)
                if r.status_code == 200:
                    notes = r.json().get("notes") or []
                    if notes:
                        c = notes[0]["content"]
                        val = lambda k: (c.get(k) or {}).get("value") if isinstance(c.get(k), dict) else c.get(k)
                        return {"title": val("title"), "abstract": val("abstract")}
                    break
            except requests.RequestException:
                pass
            time.sleep(3 * (attempt + 1))
    return {}


def fetch_abstracts(ids):
    need = sorted({k for k in (s2_key(i) for i in ids) if k} - set(cache))
    if need:
        print(f"Fetching {len(need)} abstracts")
        got = s2_batch(need)
        for k in need:
            p = got.get(k) or {}
            ab = p.get("abstract")
            if not ab:
                ax = ((p.get("externalIds") or {}).get("ArXiv")) or (k[6:] if k.startswith("ARXIV:") else None)
                if ax:
                    ab = arxiv_abstract(ax)
            cache[k] = ab
        save_cache()


def abstract_of(pid):
    k = s2_key(pid)
    return cache.get(k) if k else None


def fill(items):
    for it in items:
        it["abstract"] = abstract_of(it["id"] if "id" in it else it.get("paperId", "")) or ""
    return items


def collect_ids():
    ids = []
    for sub in ("disguises/pools", "claim_disputed/pools"):
        for f in os.listdir(os.path.join(SRC, sub)):
            ids += [p["id"] for p in load(os.path.join(SRC, sub, f))["pool"]]
    for s in load(os.path.join(SRC, "disguises", "seeds_pointers.json"))["seeds"]:
        ids.append(s.get("s2_id") or s.get("arxiv"))
    for grp in ("anchors", "anchors_full"):
        base = os.path.join(SRC, "later_realized", grp)
        for slug in os.listdir(base):
            d = os.path.join(base, slug)
            ids.append(load(os.path.join(d, "anchor.json"))["paperId"])
            for f in ("pool_preT.json", "F.json"):
                if os.path.exists(os.path.join(d, f)):
                    ids += [p["id"] for p in load(os.path.join(d, f))]
    return [i for i in ids if i]


def hydrate_disguises():
    sp = load(os.path.join(SRC, "disguises", "seeds_pointers.json"))
    for s in sp["seeds"]:
        s["abstract"] = abstract_of(s.get("s2_id") or s.get("arxiv") or "") or ""
    dump(sp, os.path.join(DST, "seeds_v2.json"))
    dump({"seeds": sp["seeds"][:10], "spotlights_tried": sp.get("spotlights_tried", [])}, os.path.join(DST, "seeds.json"))
    for f in os.listdir(os.path.join(SRC, "disguises", "pools")):
        d = load(os.path.join(SRC, "disguises", "pools", f))
        fill(d["pool"])
        dump(d, os.path.join(DST, "pools", f))


def hydrate_claim_disputed():
    pairs = [json.loads(l) for l in open(os.path.join(SRC, "claim_disputed", "pairs.jsonl"), encoding="utf-8")]
    index, rows = {}, []
    print(f"Fetching {len(pairs)} ICLR submission abstracts")
    for n, p in enumerate(pairs):
        i = p["pair_index"]
        pool_f = os.path.join(SRC, "claim_disputed", "pools", f"n2f_{i}.json")
        B = None
        if os.path.exists(pool_f):
            d = load(pool_f)
            fill(d["pool"])
            dump(d, os.path.join(DST, "pools_full", f"n2f_{i}.json"))
            B = next((x for x in d["pool"] if x.get("is_B")), None)
        entry = {"status": p["status"], "A_id": p["A_id"], "dimension": p["novelty_dimension"]}
        if B and HEX40.match(B["id"][2:]):
            entry["B_s2"] = B["id"][2:]
        index[str(i)] = entry
        key = "openreview:" + p["A_id"]
        if not cache.get(key):
            cache[key] = s2_title_abstract(p["A_title"]) or openreview_note(p["A_id"]).get("abstract")
            if n % 20 == 0:
                save_cache()
        rows.append({
            "A_id": p["A_id"], "A_title": p["A_title"], "A_abstract": cache.get(key) or "",
            "A_year": str(p["A_year"]), "A_pdf_url": f"https://openreview.net/pdf?id={p['A_id']}",
            "novelty_dimension": p["novelty_dimension"], "relation": p["relation"],
            "B_referenced_work": B["title"] if B else "", "B_identifier": p["B_identifier"],
            "B_how_identified": p["B_how_identified"], "B_already_cited_by_A": p["B_already_cited_by_A"],
            "iclr_decision": p.get("iclr_decision"),
        })
    save_cache()
    dump(index, os.path.join(DST, "n2_full_index.json"))
    os.makedirs(os.path.join(DST, "claim_disputed"), exist_ok=True)
    with open(os.path.join(DST, "claim_disputed", "testset_clean.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def hydrate_later_realized():
    shutil.copy(os.path.join(SRC, "later_realized", "later_realized_set.json"), os.path.join(DST, "later_realized_set.json"))
    for grp in ("anchors", "anchors_full"):
        base = os.path.join(SRC, "later_realized", grp)
        for slug in os.listdir(base):
            s, t = os.path.join(base, slug), os.path.join(DST, grp, slug)
            os.makedirs(t, exist_ok=True)
            for f in os.listdir(s):
                d = load(os.path.join(s, f))
                if f == "anchor.json":
                    d["abstract"] = abstract_of(d["paperId"]) or ""
                elif f in ("pool_preT.json", "F.json"):
                    fill(d)
                dump(d, os.path.join(t, f))


def main():
    os.makedirs(DST, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "eval", "annotation"), exist_ok=True)
    fetch_abstracts(collect_ids())
    hydrate_disguises()
    hydrate_claim_disputed()
    hydrate_later_realized()
    missing = sum(1 for v in cache.values() if not v)
    print(f"Wrote {DST} ({missing} papers have no public abstract and are left empty)")
    print("Next: python eval/make_disguises.py && python eval/make_disguises_v2.py && "
          "python eval/make_disguises_v3.py && python eval/build_hybrids.py")


if __name__ == "__main__":
    main()
