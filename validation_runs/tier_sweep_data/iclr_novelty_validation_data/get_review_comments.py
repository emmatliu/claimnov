import argparse
import json
import os
import re
import sys

import openreview
import pandas as pd
from tqdm import tqdm

VENUE_IDS = {
    2022: "ICLR.cc/2022/Conference",
    2023: "ICLR.cc/2023/Conference",
}

SUBMISSION_INVITATIONS = {
    "Blind_Submission": "active",
    "Withdrawn_Submission": "withdrawn",
    "Desk_Rejected_Submission": "desk_rejected",
}

REVIEW_TEXT_FIELDS = [
    "summary_of_the_paper",
    "main_review",
    "strength_and_weaknesses",
    "clarity,_quality,_novelty_and_reproducibility",
    "summary_of_the_review",
]

MENTION_PATTERNS = re.compile(
    r"""(
        related\s+work | prior\s+work | previous\s+work | missing\s+(?:reference|citation)
        | not\s+cited | should\s+cite | fail(?:s|ed)?\s+to\s+cite | overlook
        | similar\s+(?:to|work|paper|approach) | already\s+(?:been\s+)?(?:done|proposed|studied)
        | see\s+\[ | et\s+al\.? | arxiv | \bcf\.\b
        | \(\d{4}\) | \[\d+\]
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def get_client() -> openreview.Client:
    return openreview.Client(
        baseurl="https://api.openreview.net",
        username=os.environ.get("OPENREVIEW_USERNAME"),
        password=os.environ.get("OPENREVIEW_PASSWORD"),
    )


def note_text_fields(content: dict) -> dict:
    out = {}
    for k in REVIEW_TEXT_FIELDS:
        v = content.get(k)
        if isinstance(v, str) and v.strip():
            out[k] = v.strip()
    return out


def comment_text(content: dict) -> str:
    for k in ("comment", "main_review", "review", "text"):
        v = content.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def short_sig(signatures) -> str:
    if not signatures:
        return ""
    s = signatures[0] if isinstance(signatures, list) else str(signatures)
    return s.split("/")[-1]


def process_year(client, year, keep_ids, limit):
    venue_id = VENUE_IDS[year]
    print(f"\nFetching submissions for {venue_id}...")

    submissions, status_by_id, seen = [], {}, set()
    for inv, status in SUBMISSION_INVITATIONS.items():
        try:
            notes = client.get_all_notes(
                invitation=f"{venue_id}/-/{inv}", details="directReplies")
        except Exception as e:
            print(f"  [{inv}] skipped: {str(e)[:80]}", file=sys.stderr)
            continue
        n_new = 0
        for s in notes:
            if s.id in seen:
                continue
            seen.add(s.id)
            status_by_id[s.id] = status
            submissions.append(s)
            n_new += 1
        print(f"  {inv}: {len(notes)} notes ({n_new} new) -> status='{status}'")
    print(f"  Total unique submissions: {len(submissions)}")
    if limit:
        submissions = submissions[:limit]

    records = []
    n_reviews = n_comments = 0
    for sub in tqdm(submissions, desc=f"Reviews {year}"):
        if keep_ids is not None and sub.id not in keep_ids:
            continue

        reviews, comments = [], []
        for r in sub.details.get("directReplies", []):
            inv = r.get("invitation", "")
            content = r.get("content", {})
            if inv.endswith("Official_Review"):
                fields = note_text_fields(content)
                if not fields:
                    continue
                reviews.append({
                    "signature": short_sig(r.get("signatures")),
                    "recommendation": str(content.get("recommendation", "")).strip(),
                    "confidence": str(content.get("confidence", "")).strip(),
                    "technical_novelty": str(
                        content.get("technical_novelty_and_significance", "")).strip(),
                    "empirical_novelty": str(
                        content.get("empirical_novelty_and_significance", "")).strip(),
                    **fields,
                })
                n_reviews += 1
            elif inv.endswith("Official_Comment"):
                txt = comment_text(content)
                if not txt:
                    continue
                comments.append({
                    "signature": short_sig(r.get("signatures")),
                    "title": str(content.get("title", "")).strip(),
                    "comment": txt,
                })
                n_comments += 1

        if not reviews and not comments:
            continue

        records.append({
            "id": sub.id,
            "year": year,
            "status": status_by_id.get(sub.id, "active"),
            "title": sub.content.get("title", ""),
            "abstract": sub.content.get("abstract", ""),
            "reviews": reviews,
            "comments": comments,
        })

    print(f"  Kept {len(records)} papers; {n_reviews} reviews, {n_comments} comments.")
    return records


def extract_mentions(record) -> list[dict]:
    mentions = []

    def scan(source_type, signature, field, text):
        for sent in re.split(r"(?<=[.!?])\s+", text):
            sent = sent.strip()
            if len(sent) < 20:
                continue
            if MENTION_PATTERNS.search(sent):
                mentions.append({
                    "id": record["id"],
                    "year": record["year"],
                    "source": source_type,
                    "signature": signature,
                    "field": field,
                    "snippet": sent,
                })

    for rev in record["reviews"]:
        for f in REVIEW_TEXT_FIELDS:
            if f in rev:
                scan("review", rev["signature"], f, rev[f])
    for com in record["comments"]:
        scan("comment", com["signature"], "comment", com["comment"])
    return mentions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, choices=[2022, 2023], default=None)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--papers-csv", type=str, default="",
                        help="Optional: restrict to ids in this CSV. Default '' "
                             "keeps EVERY submission (active + withdrawn + "
                             "desk-rejected).")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    keep_ids = None
    if args.papers_csv:
        pcsv = os.path.join(args.out_dir, args.papers_csv)
        if os.path.exists(pcsv):
            keep_ids = set(pd.read_csv(pcsv, usecols=["id"])["id"].astype(str))
            print(f"Restricting to {len(keep_ids)} ids from {args.papers_csv}.")
        else:
            print(f"WARN: {pcsv} not found; keeping all submissions.", file=sys.stderr)

    client = get_client()
    years = [args.year] if args.year else [2022, 2023]

    all_records = []
    for year in years:
        recs = process_year(client, year, keep_ids, args.limit)
        year_dir = os.path.join(args.out_dir, "reviews", str(year))
        os.makedirs(year_dir, exist_ok=True)
        for rec in recs:
            with open(os.path.join(year_dir, f"{rec['id']}.json"), "w",
                      encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
        all_records.extend(recs)

    flat_path = os.path.join(args.out_dir, "all_reviews.jsonl")
    with open(flat_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    mentions_path = os.path.join(args.out_dir, "review_mentions.jsonl")
    n_mentions = 0
    with open(mentions_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            for m in extract_mentions(rec):
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
                n_mentions += 1

    n_rev = sum(len(r["reviews"]) for r in all_records)
    n_com = sum(len(r["comments"]) for r in all_records)
    print(f"\nSaved {len(all_records)} papers "
          f"({n_rev} reviews, {n_com} comments) to reviews/ and {flat_path}")
    print(f"Surfaced {n_mentions} candidate 'related-work mention' snippets "
          f"to {mentions_path}")


if __name__ == "__main__":
    main()
