import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")


def how_identified(r):
    if r.get("B_agentic_resolved"):
        return "web_resolved"
    if r.get("B_in_refs") == "cited" and r.get("B_matched_ref"):
        return "cited_in_A"
    return "self_identified"


def verification(r, src):
    if src == "cited_in_A":
        return "cited_in_A_bibliography"
    if src == "self_identified":
        return "reviewer_gave_full_citation"
    ev = (r.get("B_resolve_evidence") or "").lower()
    verified = any(m in ev for m in ("review", "verified", "dblp", "id-verified"))
    return "review_thread_verified" if verified else "web_search_unverified"


def clean_row(r):
    src = how_identified(r)
    prov = r.get("B_matched_ref") or r.get("B_resolve_evidence") or ""
    return {
        "A_id": r.get("A_id"),
        "A_title": r.get("A_title"),
        "A_abstract": r.get("A_abstract"),
        "A_year": r.get("year"),
        "A_is_active": r.get("A_is_active"),
        "A_pdf_url": r.get("A_pdf_url") or "",
        "A_refs_path": r.get("A_refs_path") or "",
        "novelty_dimension": r.get("dimension"),
        "relation": r.get("relation"),
        "reviewer_snippet": r.get("snippet"),
        "relevance_confidence": r.get("confidence"),
        "B_referenced_work": r.get("referenced_work_B"),
        "B_identifier": r.get("B_resolved_id") or "",
        "B_how_identified": src,
        "B_verification": verification(r, src),
        "B_reviewer_wrote": r.get("referenced_work_B_raw") or r.get("referenced_work_B"),
        "B_already_cited_by_A": r.get("B_in_refs"),
        "B_provenance": prov,
    }


def export(src_name, dst_name):
    src = os.path.join(OUT, src_name)
    if not os.path.exists(src):
        return
    rows = [clean_row(json.loads(l)) for l in open(src, encoding="utf-8")]
    with open(os.path.join(OUT, dst_name), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    c = Counter(r["B_how_identified"] for r in rows)
    print(f"{dst_name}: {len(rows)} rows  ({dict(c)})")


if __name__ == "__main__":
    export("relevance_selected.jsonl", "testset_clean.jsonl")
    export("relevance_all.jsonl", "testset_all_clean.jsonl")
