import argparse
import io
import json
import os
import re
import sys
import time

import requests
from tqdm import tqdm


GROBID_URL = "http://localhost:8070"
GROBID_ENDPOINT = "/api/processFulltextDocument"

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_MATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/match"
S2_FIELDS = "title,externalIds"
S2_BATCH_SIZE = 500

ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"


ARXIV_NEW_RE = re.compile(
    r"(?:arxiv[:\s]*|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)
ARXIV_OLD_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf)/([a-z\-]+(?:\.[A-Z]{2})?/\d{7})",
    re.IGNORECASE,
)
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.IGNORECASE)
DOI_TRAIL_STRIP = ".,;:)]}>"


def extract_arxiv_id(raw: str) -> str:
    if not raw:
        return ""
    m = ARXIV_NEW_RE.search(raw)
    if m:
        return m.group(1)
    m = ARXIV_OLD_RE.search(raw)
    if m:
        return m.group(1)
    return ""


def extract_doi(raw: str) -> str:
    if not raw:
        return ""
    m = DOI_RE.search(raw)
    if not m:
        return ""
    doi = m.group(1)
    while doi and doi[-1] in DOI_TRAIL_STRIP:
        doi = doi[:-1]
    return doi


def check_grobid(grobid_url: str) -> bool:
    try:
        return requests.get(grobid_url.rstrip("/") + "/api/isalive", timeout=5).status_code == 200
    except requests.RequestException:
        return False


def grobid_fulltext(pdf_bytes: bytes, filename: str, grobid_url: str) -> str | None:
    url = grobid_url.rstrip("/") + GROBID_ENDPOINT
    try:
        resp = requests.post(
            url,
            files={"input": (filename, io.BytesIO(pdf_bytes), "application/pdf")},
            data={
                "consolidateHeader": "1",
                "consolidateCitations": "0",
                "includeRawCitations": "0",
                "segmentSentences": "0",
            },
            timeout=300,
        )
    except requests.RequestException as e:
        print(f"  [grobid] {filename}: {e}", file=sys.stderr)
        return None
    if resp.status_code != 200:
        print(f"  [grobid] {filename}: HTTP {resp.status_code}", file=sys.stderr)
        return None
    return resp.text


def download_arxiv_pdf(arxiv_id: str, session: requests.Session) -> bytes | None:
    url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
    try:
        r = session.get(url, timeout=60)
    except requests.RequestException as e:
        print(f"  [arxiv] {arxiv_id}: {e}", file=sys.stderr)
        return None
    if r.status_code != 200 or not r.content.startswith(b"%PDF"):
        print(f"  [arxiv] {arxiv_id}: HTTP {r.status_code}", file=sys.stderr)
        return None
    return r.content


def _s2_headers(api_key: str | None) -> dict:
    return {"x-api-key": api_key} if api_key else {}


def s2_batch_lookup(
    ids: list[str],
    api_key: str | None,
    session: requests.Session,
    sleep: float,
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not ids:
        return out
    for i in range(0, len(ids), S2_BATCH_SIZE):
        chunk = ids[i : i + S2_BATCH_SIZE]
        for attempt in range(4):
            try:
                r = session.post(
                    S2_BATCH_URL,
                    params={"fields": S2_FIELDS},
                    json={"ids": chunk},
                    headers=_s2_headers(api_key),
                    timeout=60,
                )
            except requests.RequestException as e:
                print(f"  [s2 batch] network: {e}", file=sys.stderr)
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            if r.status_code != 200:
                print(f"  [s2 batch] HTTP {r.status_code}: {r.text[:200]}", file=sys.stderr)
                break
            data = r.json()
            for req_id, entry in zip(chunk, data):
                if not entry:
                    continue
                ext = entry.get("externalIds") or {}
                out[req_id] = {
                    "s2_id": entry.get("paperId", ""),
                    "arxiv_id": ext.get("ArXiv", "") or "",
                    "title": entry.get("title", "") or "",
                }
            break
        time.sleep(sleep)
    return out


def s2_match_title(
    title: str,
    api_key: str | None,
    session: requests.Session,
) -> dict | None:
    if not title or len(title) < 5:
        return None
    params = {"query": title, "fields": S2_FIELDS}
    for attempt in range(4):
        try:
            r = session.get(S2_MATCH_URL, params=params, headers=_s2_headers(api_key), timeout=30)
        except requests.RequestException as e:
            print(f"  [s2 match] network: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            print(f"  [s2 match] HTTP {r.status_code}: {r.text[:200]}", file=sys.stderr)
            return None
        data = r.json().get("data", [])
        if not data:
            return None
        hit = data[0]
        ext = hit.get("externalIds") or {}
        return {
            "s2_id": hit.get("paperId", ""),
            "arxiv_id": ext.get("ArXiv", "") or "",
            "title": hit.get("title", "") or "",
        }
    return None


def iter_ref_files(refs_root: str, years: list[int], limit: int | None):
    for year in years:
        year_dir = os.path.join(refs_root, str(year))
        if not os.path.isdir(year_dir):
            continue
        files = sorted(f for f in os.listdir(year_dir) if f.endswith(".json"))
        if limit:
            files = files[:limit]
        for fn in files:
            yield year, os.path.join(year_dir, fn)


def collect_refs(refs_root: str, years: list[int], limit: int | None) -> list[dict]:
    rows = []
    for year, ref_file in iter_ref_files(refs_root, years, limit):
        parent_id = os.path.basename(ref_file).removesuffix(".json")
        with open(ref_file, encoding="utf-8") as f:
            refs = json.load(f)
        for ref in refs:
            title = (ref.get("title") or "").strip()
            raw = (ref.get("raw") or "").strip()
            arxiv_id = extract_arxiv_id(raw) or extract_arxiv_id(title)
            doi = "" if arxiv_id else extract_doi(raw)
            rows.append({
                "parent_paper_id": parent_id,
                "parent_year": year,
                "ref_title": title,
                "raw": raw,
                "arxiv_id": arxiv_id,
                "doi": doi,
                "s2_id": "",
                "method": "arxiv_regex" if arxiv_id else "",
            })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Resolve cited references to arXiv + Grobid full text"
    )
    parser.add_argument("--refs-dir", type=str, default="refs",
                        help="Dir containing refs/{year}/{id}.json from get_reviews.py")
    parser.add_argument("--out-dir", type=str, default=".",
                        help="Output root (cited/ is created inside)")
    parser.add_argument("--year", type=int, choices=[2022, 2023], default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N parent papers per year (debug)")
    parser.add_argument("--grobid-url", type=str, default=GROBID_URL)
    parser.add_argument("--s2-api-key", type=str, default=os.environ.get("S2_API_KEY"))
    parser.add_argument("--s2-sleep", type=float, default=1.1,
                        help="Seconds between S2 calls (default 1.1 for public limit)")
    parser.add_argument("--min-title-len", type=int, default=10)
    args = parser.parse_args()

    if not check_grobid(args.grobid_url):
        print(f"ERROR: Grobid not reachable at {args.grobid_url}.", file=sys.stderr)
        sys.exit(1)

    cited_dir = os.path.join(args.out_dir, "cited")
    tei_dir = os.path.join(cited_dir, "tei")
    os.makedirs(tei_dir, exist_ok=True)

    cache_path = os.path.join(cited_dir, "cache.json")
    resolved_path = os.path.join(cited_dir, "resolved.jsonl")

    cache: dict[str, dict] = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)

    session = requests.Session()
    years = [args.year] if args.year else [2022, 2023]

    rows = collect_refs(args.refs_dir, years, args.limit)
    print(f"Collected {len(rows)} references from {len(set(r['parent_paper_id'] for r in rows))} parent papers.")

    n_arxiv_regex = sum(1 for r in rows if r["arxiv_id"])
    n_doi_regex = sum(1 for r in rows if r["doi"])
    print(f"  Regex hits: arXiv={n_arxiv_regex}  DOI={n_doi_regex}  "
          f"leftover={len(rows) - n_arxiv_regex - n_doi_regex}")

    doi_rows = [r for r in rows if not r["arxiv_id"] and r["doi"]]
    doi_to_query = []
    for r in doi_rows:
        key = f"doi:{r['doi'].lower()}"
        if key not in cache:
            doi_to_query.append(r["doi"])
    doi_to_query = sorted(set(doi_to_query))
    print(f"  DOI batch queue: {len(doi_to_query)} new (after cache)")

    if doi_to_query:
        prefixed = [f"DOI:{d}" for d in doi_to_query]
        batch_hits = s2_batch_lookup(prefixed, args.s2_api_key, session, args.s2_sleep)
        for doi in doi_to_query:
            hit = batch_hits.get(f"DOI:{doi}") or {}
            cache[f"doi:{doi.lower()}"] = hit
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    for r in doi_rows:
        hit = cache.get(f"doi:{r['doi'].lower()}") or {}
        r["s2_id"] = hit.get("s2_id", "")
        if hit.get("arxiv_id"):
            r["arxiv_id"] = hit["arxiv_id"]
            r["method"] = "doi_batch"
        elif hit.get("s2_id"):
            r["method"] = "doi_batch_no_arxiv"

    title_rows = [
        r for r in rows
        if not r["arxiv_id"] and not r["doi"] and len(r["ref_title"]) >= args.min_title_len
    ]
    unique_titles = sorted({r["ref_title"].lower() for r in title_rows})
    to_query = [t for t in unique_titles if f"title:{t}" not in cache]
    print(f"  Title match queue: {len(to_query)} new (after cache, from {len(unique_titles)} unique)")

    for t in tqdm(to_query, desc="s2 title match"):
        orig = next(r["ref_title"] for r in title_rows if r["ref_title"].lower() == t)
        hit = s2_match_title(orig, args.s2_api_key, session) or {}
        cache[f"title:{t}"] = hit
        time.sleep(args.s2_sleep)
    if to_query:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    for r in title_rows:
        hit = cache.get(f"title:{r['ref_title'].lower()}") or {}
        r["s2_id"] = hit.get("s2_id", "")
        if hit.get("arxiv_id"):
            r["arxiv_id"] = hit["arxiv_id"]
            r["method"] = "title_match"
        elif hit.get("s2_id"):
            r["method"] = "title_match_no_arxiv"

    unique_arxiv = sorted({r["arxiv_id"] for r in rows if r["arxiv_id"]})
    print(f"  Unique arXiv ids to process: {len(unique_arxiv)}")

    arxiv_to_tei: dict[str, str] = {}
    arxiv_to_status: dict[str, str] = {}
    for aid in tqdm(unique_arxiv, desc="arxiv+grobid"):
        tei_path = os.path.join(tei_dir, f"{aid.replace('/', '_')}.tei.xml")
        if os.path.exists(tei_path) and os.path.getsize(tei_path) > 0:
            arxiv_to_tei[aid] = tei_path
            arxiv_to_status[aid] = "cached"
            continue
        pdf_bytes = download_arxiv_pdf(aid, session)
        if not pdf_bytes:
            arxiv_to_status[aid] = "pdf_failed"
            continue
        tei_xml = grobid_fulltext(pdf_bytes, f"{aid}.pdf", args.grobid_url)
        if not tei_xml:
            arxiv_to_status[aid] = "grobid_failed"
            continue
        with open(tei_path, "w", encoding="utf-8") as f:
            f.write(tei_xml)
        arxiv_to_tei[aid] = tei_path
        arxiv_to_status[aid] = "new"

    with open(resolved_path, "w", encoding="utf-8") as f:
        for r in rows:
            aid = r["arxiv_id"]
            f.write(json.dumps({
                "parent_paper_id": r["parent_paper_id"],
                "parent_year": r["parent_year"],
                "ref_title": r["ref_title"],
                "arxiv_id": aid,
                "doi": r.get("doi", ""),
                "s2_id": r.get("s2_id", ""),
                "resolve_method": r.get("method") or "unresolved",
                "tei_path": arxiv_to_tei.get(aid, "") if aid else "",
                "status": arxiv_to_status.get(aid, "") if aid else "no_id",
            }) + "\n")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    resolved = sum(1 for r in rows if r["arxiv_id"])
    tei_new = sum(1 for s in arxiv_to_status.values() if s == "new")
    print(f"\nRefs total:        {len(rows)}")
    print(f"Resolved to arXiv: {resolved}")
    print(f"Unique arXiv:      {len(unique_arxiv)}")
    print(f"New TEI this run:  {tei_new}")
    print(f"Resolved log:      {resolved_path}")


if __name__ == "__main__":
    main()
