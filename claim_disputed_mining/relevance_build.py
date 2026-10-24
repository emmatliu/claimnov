import argparse, hashlib, csv, json, os, re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = next((d for d in (os.path.join(HERE, "iclr_novelty_validation_data"),
                         os.path.join(os.path.dirname(HERE), "iclr_novelty_validation_data"))
             if os.path.isdir(d)), os.path.join(HERE, "iclr_novelty_validation_data"))
MENTIONS = os.path.join(DATA, "review_mentions.jsonl")
PAPERS = os.path.join(DATA, "papers.csv")
CACHE = os.path.join(HERE, "outputs", "relevance_cache.jsonl")
OUT_ALL = os.path.join(HERE, "outputs", "relevance_all.jsonl")
OUT_SEL = os.path.join(HERE, "outputs", "relevance_selected.jsonl")

REL_RANK = {"anticipation": 0, "comparison": 1, "overlap": 2, "other": 3, None: 4}
CITE_CACHE = os.path.join(HERE, "outputs", "citation_cache.jsonl")
BRACKET_CACHE = os.path.join(HERE, "outputs", "bracket_cache.jsonl")
AMB_CACHE = os.path.join(HERE, "outputs", "ambiguous_resolution_cache.jsonl")
BRACKET = re.compile(r"\[\d+\]")
STRONG_UNCITED = re.compile(
    r"not cited|is not cited|are not cited|do(es)? not cite|did not cite|"
    r"fail(s|ed)? to cite|with(out| no) citing|un-?cited|not referenced|not acknowledged|"
    r"neglect(s|ed)? to cite|omit(s|ted)? to (cite|reference)|"
    r"missing (from|in) (the )?(citation|references?|related work|bibliograph)|"
    r"(authors? )?should (also )?(be )?cite", re.I)


def key_of(snippet):
    return hashlib.md5(snippet.encode("utf-8")).hexdigest()[:12]


def _load(path):
    c = {}
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            o = json.loads(l)
            if isinstance(o.get("out"), dict) and "error" not in o["out"]:
                c[o["_key"]] = o["out"]
    return c


def resolve_case(cite, bracket, a_id, b, snippet, active):
    shash = hashlib.md5(snippet.encode()).hexdigest()[:12]
    if b and BRACKET.search(b):
        hit = bracket.get(f"{a_id}::{shash}", {})
        verdict = hit.get("verdict", "unknown")
        method, mref = hit.get("method"), hit.get("matched_ref")
        resolved = hit.get("resolved_B")
    elif b:
        hit = cite.get(f'{a_id}::{hashlib.md5(b.encode()).hexdigest()[:12]}',
                       {"verdict": "unknown"})
        verdict, method, mref, resolved = hit.get("verdict", "unknown"), hit.get("method"), \
            hit.get("matched_ref"), None
    else:
        return "unknown", None, None, None, False
    override = False
    if verdict == "cited" and method != "id_exact" and STRONG_UNCITED.search(snippet):
        verdict, override = "uncited", True
    return verdict, method, mref, resolved, override


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_dim", type=int, default=50)
    ap.add_argument("--min_conf", type=float, default=0.6)
    ap.add_argument("--active_first", action="store_true", default=True,
                    help="prioritize active papers (resolvable refs) over relation strength")
    ap.add_argument("--signal_first", dest="active_first", action="store_false",
                    help="prioritize relation strength over active status")
    ap.add_argument("--uncited_only", action="store_true",
                    help="exclude cases where the rigorous check found B already cited by A "
                         "(verdict=cited); keeps uncited/indeterminate/unknown")
    args = ap.parse_args()

    labels = {}
    for l in open(CACHE, encoding="utf-8"):
        o = json.loads(l)
        out = o.get("out")
        if not isinstance(out, dict) or "error" in out:
            continue
        if out.get("is_relevant_case") and out.get("has_identifiable_ref") \
                and out.get("confidence", 0) >= args.min_conf:
            labels[o["_key"].split("::")[-1]] = out

    meta = {}
    with open(PAPERS, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta[row["id"]] = {"title": row["title"], "abstract": row["abstract"],
                               "year": row["year"], "refs_path": row.get("refs_path", "")}

    allrev = {}
    _arv = os.path.join(DATA, "all_reviews.jsonl")
    if os.path.exists(_arv):
        for l in open(_arv, encoding="utf-8"):
            o = json.loads(l)
            allrev[o["id"]] = {"title": o.get("title"), "abstract": o.get("abstract")}

    cite, bracket, amb = _load(CITE_CACHE), _load(BRACKET_CACHE), _load(AMB_CACHE)

    kept, seen = [], set()
    for l in open(MENTIONS, encoding="utf-8"):
        m = json.loads(l)
        lab = labels.get(key_of(m["snippet"]))
        if not lab:
            continue
        uid = (m["id"], key_of(m["snippet"]))
        if uid in seen:
            continue
        seen.add(uid)
        active = m["id"] in meta
        rp = meta.get(m["id"], {}).get("refs_path", "")
        if not rp:
            for _yr in (m.get("year"), "2022", "2023"):
                _cand = os.path.join("refs", str(_yr), f'{m["id"]}.json')
                if _yr and os.path.exists(os.path.join(DATA, _cand)):
                    rp = "./" + _cand
                    break
        b_raw = lab.get("referenced_work_B")
        verdict, method, mref, resolved, override = resolve_case(
            cite, bracket, m["id"], b_raw, m["snippet"], active)
        a = amb.get(f'{m["id"]}::{key_of(m["snippet"])}', {})
        if a.get("resolved") and a.get("title"):
            concrete_B, b_orig, agentic = a["title"], b_raw, True
        else:
            concrete_B, b_orig, agentic = (resolved or b_raw), (b_raw if resolved else None), False
        kept.append({
            "A_id": m["id"], "year": m.get("year"), "source": m["source"],
            "field": m.get("field"), "snippet": m["snippet"],
            "dimension": lab.get("dimension"), "relation": lab.get("relation"),
            "referenced_work_B": concrete_B,
            "referenced_work_B_raw": b_orig,
            "B_resolved_id": a.get("arxiv_or_doi") if agentic else None,
            "B_agentic_resolved": agentic,
            "B_resolve_confidence": a.get("confidence") if agentic else None,
            "B_resolve_evidence": a.get("evidence") if agentic else None,
            "confidence": lab.get("confidence"),
            "A_title": meta.get(m["id"], {}).get("title") or allrev.get(m["id"], {}).get("title"),
            "A_abstract": meta.get(m["id"], {}).get("abstract") or allrev.get(m["id"], {}).get("abstract"),
            "A_refs_path": rp, "A_is_active": active,
            "A_pdf_url": f'https://openreview.net/pdf?id={m["id"]}',
            "B_in_refs": verdict,
            "B_cite_method": method,
            "B_matched_ref": mref,
            "B_cite_reviewer_override": override,
        })

    with open(OUT_ALL, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_dim = defaultdict(list)
    for r in kept:
        by_dim[r["dimension"]].append(r)

    drop = {"cited"} if args.uncited_only else set()

    EXCLUDE_CASES = {"5hLP5JY9S2d::0e37eec5954f"}
    FORCE_CASES = {"cUX2psP06OL::5358a50b9bbc"}
    def _ck(r):
        return f'{r["A_id"]}::{hashlib.md5(r["snippet"].encode("utf-8")).hexdigest()[:12]}'

    sel = []
    for dim in ["Problem", "Methodology", "Resource", "Result"]:
        pool = [r for r in by_dim.get(dim, [])
                if r["B_in_refs"] not in drop and _ck(r) not in EXCLUDE_CASES]
        if args.active_first:
            pool.sort(key=lambda r: (not r["A_is_active"], REL_RANK.get(r["relation"], 4),
                                     -(r["confidence"] or 0)))
        else:
            pool.sort(key=lambda r: (REL_RANK.get(r["relation"], 4), not r["A_is_active"],
                                     -(r["confidence"] or 0)))
        pool = [r for r in pool if _ck(r) in FORCE_CASES] + \
               [r for r in pool if _ck(r) not in FORCE_CASES]
        picked, papers = [], set()
        for r in pool:
            if r["A_id"] in papers:
                continue
            papers.add(r["A_id"]); picked.append(r)
            if len(picked) >= args.per_dim:
                break
        sel.extend(picked)

    with open(OUT_SEL, "w", encoding="utf-8") as f:
        for r in sel:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"kept relevance cases: {len(kept)}  ({len(set(r['A_id'] for r in kept))} distinct papers)")
    print(f"selected: {len(sel)} across {len(set(r['A_id'] for r in sel))} distinct papers\n")
    print("{:12s} {:>8s} {:>7s} | {:>4s} {:>4s} {:>4s} (selected relation mix)".format(
        "dim", "pool", "picked", "ant", "cmp", "ovl"))
    for dim in ["Problem", "Methodology", "Resource", "Result"]:
        pk = [r for r in sel if r["dimension"] == dim]
        rc = defaultdict(int)
        for r in pk:
            rc[r["relation"]] += 1
        print("{:12s} {:8d} {:7d} | {:4d} {:4d} {:4d}".format(
            dim, len(by_dim.get(dim, [])), len(pk),
            rc["anticipation"], rc["comparison"], rc["overlap"]))
    print(f"\nuncited_only={args.uncited_only}  (rigorous verdicts from citation_check.py)")
    print("B_in_refs breakdown over the full kept pool:")
    print("{:12s} | {:>6s} {:>7s} {:>13s} {:>8s}  -> avail if uncited-only".format(
        "dim", "cited", "uncited", "indeterminate", "unknown"))
    for dim in ["Problem", "Methodology", "Resource", "Result"]:
        c = defaultdict(int)
        for r in by_dim.get(dim, []):
            c[r["B_in_refs"]] += 1
        avail = c["uncited"] + c["indeterminate"] + c["unknown"]
        print("{:12s} | {:6d} {:7d} {:13d} {:8d}  -> {:d}".format(
            dim, c["cited"], c["uncited"], c["indeterminate"], c["unknown"], avail))
    print("\nwrote", OUT_ALL, "and", OUT_SEL)


if __name__ == "__main__":
    main()
