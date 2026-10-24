import argparse
import io
import json
import os
import re
import sys
import time

import openreview
import pandas as pd
import requests
from lxml import etree
from tqdm import tqdm


VENUE_IDS = {
    2022: "ICLR.cc/2022/Conference",
    2023: "ICLR.cc/2023/Conference",
}

TECHNICAL_FIELD = "technical_novelty_and_significance"
EMPIRICAL_FIELD = "empirical_novelty_and_significance"

GROBID_URL = "http://localhost:8070"
GROBID_ENDPOINT = "/api/processFulltextDocument"

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def parse_score(value_str) -> int | None:
    if value_str is None:
        return None
    m = re.match(r"(\d+)", str(value_str).strip())
    return int(m.group(1)) if m else None


def get_client() -> openreview.Client:
    return openreview.Client(
        baseurl="https://api.openreview.net",
        username=os.environ.get("OPENREVIEW_USERNAME"),
        password=os.environ.get("OPENREVIEW_PASSWORD"),
    )


def check_grobid(grobid_url: str) -> bool:
    try:
        r = requests.get(grobid_url.rstrip("/") + "/api/isalive", timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


def fetch_pdf_bytes(client: openreview.Client, note_id: str) -> bytes | None:
    try:
        return client.get_pdf(note_id)
    except Exception as e:
        print(f"  [pdf] {note_id}: {e}", file=sys.stderr)
        return None


def grobid_fulltext(pdf_bytes: bytes, filename: str, grobid_url: str) -> str | None:
    url = grobid_url.rstrip("/") + GROBID_ENDPOINT
    try:
        resp = requests.post(
            url,
            files={"input": (filename, io.BytesIO(pdf_bytes), "application/pdf")},
            data={
                "consolidateHeader": "1",
                "consolidateCitations": "0",
                "includeRawCitations": "1",
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


def extract_references(tei_xml: str) -> list[dict]:
    try:
        root = etree.fromstring(tei_xml.encode("utf-8"))
    except etree.XMLSyntaxError:
        return []

    refs = []
    for bibl in root.iterfind(".//tei:listBibl/tei:biblStruct", TEI_NS):
        title_el = bibl.find(".//tei:title", TEI_NS)
        title = (title_el.text or "").strip() if title_el is not None else ""

        authors = []
        for pers in bibl.iterfind(".//tei:author/tei:persName", TEI_NS):
            fore = pers.find("tei:forename", TEI_NS)
            surn = pers.find("tei:surname", TEI_NS)
            name = " ".join(
                x.text for x in (fore, surn) if x is not None and x.text
            ).strip()
            if name:
                authors.append(name)

        year = ""
        date_el = bibl.find(".//tei:date[@when]", TEI_NS)
        if date_el is not None:
            year = (date_el.get("when") or "")[:4]

        raw_el = bibl.find("tei:note[@type='raw_reference']", TEI_NS)
        raw = (raw_el.text or "").strip() if raw_el is not None else ""

        refs.append({"title": title, "authors": authors, "year": year, "raw": raw})
    return refs


def process_year(
    client: openreview.Client,
    year: int,
    out_dir: str,
    do_grobid: bool,
    grobid_url: str,
    limit: int | None,
) -> list[dict]:
    venue_id = VENUE_IDS[year]
    print(f"\nFetching submissions for {venue_id}...")
    submissions = client.get_all_notes(
        invitation=f"{venue_id}/-/Blind_Submission",
        details="directReplies",
    )
    print(f"  Found {len(submissions)} submissions.")
    if limit:
        submissions = submissions[:limit]

    tei_dir = os.path.join(out_dir, "tei", str(year))
    refs_dir = os.path.join(out_dir, "refs", str(year))
    if do_grobid:
        os.makedirs(tei_dir, exist_ok=True)
        os.makedirs(refs_dir, exist_ok=True)

    records = []
    skipped_no_reviews = 0
    skipped_no_novelty = 0
    tei_ok = 0
    refs_total = 0

    for sub in tqdm(submissions, desc=f"Processing {year}"):
        content = sub.content
        title = content.get("title", "")
        abstract = content.get("abstract", "")

        reviews = [
            r for r in sub.details.get("directReplies", [])
            if r.get("invitation", "").endswith("Official_Review")
        ]
        if not reviews:
            skipped_no_reviews += 1
            continue

        tech_scores, emp_scores = [], []
        for rev in reviews:
            rc = rev.get("content", {})
            t = parse_score(rc.get(TECHNICAL_FIELD))
            e = parse_score(rc.get(EMPIRICAL_FIELD))
            if t is not None:
                tech_scores.append(t)
            if e is not None:
                emp_scores.append(e)

        if not tech_scores and not emp_scores:
            skipped_no_novelty += 1
            continue

        tei_path = os.path.join(tei_dir, f"{sub.id}.tei.xml")
        refs_path = os.path.join(refs_dir, f"{sub.id}.json")
        got_tei = False
        n_refs = 0

        if do_grobid:
            if os.path.exists(tei_path) and os.path.getsize(tei_path) > 0:
                with open(tei_path, encoding="utf-8") as f:
                    tei_xml = f.read()
                got_tei = True
            else:
                pdf_bytes = fetch_pdf_bytes(client, sub.id)
                tei_xml = None
                if pdf_bytes:
                    tei_xml = grobid_fulltext(pdf_bytes, f"{sub.id}.pdf", grobid_url)
                if tei_xml:
                    with open(tei_path, "w", encoding="utf-8") as f:
                        f.write(tei_xml)
                    got_tei = True
                    tei_ok += 1
                else:
                    time.sleep(0.5)

            if got_tei:
                refs = extract_references(tei_xml)
                n_refs = len(refs)
                refs_total += n_refs
                with open(refs_path, "w", encoding="utf-8") as f:
                    json.dump(refs, f, ensure_ascii=False, indent=2)

        records.append({
            "id": sub.id,
            "title": title,
            "abstract": abstract,
            "year": year,
            "technical_novelty": json.dumps(tech_scores),
            "empirical_novelty": json.dumps(emp_scores),
            "tei_path": tei_path if got_tei else "",
            "refs_path": refs_path if got_tei else "",
            "n_refs": n_refs,
        })

    print(f"  Extracted {len(records)} papers with novelty scores.")
    print(f"  Skipped {skipped_no_reviews} no-reviews, {skipped_no_novelty} no-novelty.")
    if do_grobid:
        print(f"  TEI generated: {tei_ok} new this run")
        if records:
            print(f"  Avg refs/paper: {refs_total / len(records):.1f}")

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Extract ICLR papers and parse their PDFs with Grobid (no PDFs on disk)"
    )
    parser.add_argument("--year", type=int, choices=[2022, 2023], default=None)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--output-csv", type=str, default="papers.csv")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Skip Grobid entirely — just novelty metadata")
    parser.add_argument("--grobid-url", type=str, default=GROBID_URL)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N submissions per year (debug)")
    args = parser.parse_args()

    do_grobid = not args.metadata_only
    if do_grobid and not check_grobid(args.grobid_url):
        print(f"ERROR: Grobid not reachable at {args.grobid_url}. "
              f"Start it with `./gradlew run` from grobid-0.9.0, "
              f"or pass --metadata-only.", file=sys.stderr)
        sys.exit(1)

    client = get_client()
    years = [args.year] if args.year else [2022, 2023]
    os.makedirs(args.out_dir, exist_ok=True)

    all_records = []
    for year in years:
        all_records.extend(process_year(
            client, year, args.out_dir,
            do_grobid=do_grobid, grobid_url=args.grobid_url, limit=args.limit,
        ))

    out_csv = os.path.join(args.out_dir, args.output_csv)
    df = pd.DataFrame(all_records)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} papers to {out_csv}")
    if len(df):
        print(f"Year distribution:\n{df['year'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
