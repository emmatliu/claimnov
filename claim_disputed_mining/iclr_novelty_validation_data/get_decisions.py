import argparse
import os
import sys

import openreview
import pandas as pd
from tqdm import tqdm

VENUE_IDS = {
    2022: "ICLR.cc/2022/Conference",
    2023: "ICLR.cc/2023/Conference",
}


def get_client() -> openreview.Client:
    return openreview.Client(
        baseurl="https://api.openreview.net",
        username=os.environ.get("OPENREVIEW_USERNAME"),
        password=os.environ.get("OPENREVIEW_PASSWORD"),
    )


def normalize_tier(raw: str) -> str:
    if not raw:
        return "unknown"
    s = raw.strip().lower()
    if "reject" in s or s == "withdraw":
        return "reject"
    if "oral" in s:
        return "oral"
    if "spotlight" in s or "top-25" in s or "top 25" in s or "notable-top-25" in s:
        return "spotlight"
    if "top-5" in s or "top 5" in s or "notable-top-5" in s:
        return "oral"
    if "accept" in s or "poster" in s:
        return "poster"
    return "unknown"


def extract_decision(sub) -> str:
    for r in sub.details.get("directReplies", []):
        inv = r.get("invitation", "")
        if inv.endswith("/Decision") or inv.endswith("Decision"):
            content = r.get("content", {})
            for key in ("decision", "recommendation", "Decision"):
                val = content.get(key)
                if val:
                    return str(val).strip()
    return ""


def process_year(client, year, limit):
    venue_id = VENUE_IDS[year]
    print(f"\nFetching submissions for {venue_id}...")
    submissions = client.get_all_notes(
        invitation=f"{venue_id}/-/Blind_Submission",
        details="directReplies",
    )
    print(f"  Found {len(submissions)} submissions.")
    if limit:
        submissions = submissions[:limit]

    rows = []
    n_missing = 0
    for sub in tqdm(submissions, desc=f"Decisions {year}"):
        raw = extract_decision(sub)
        if not raw:
            n_missing += 1
        rows.append({
            "id": sub.id,
            "year": year,
            "decision_raw": raw,
            "tier": normalize_tier(raw),
        })
    print(f"  Decisions found: {len(rows) - n_missing}/{len(rows)} "
          f"({n_missing} missing).")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, choices=[2022, 2023], default=None)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--merge", action="store_true",
                        help="Also merge tier/decision into papers.csv on id.")
    args = parser.parse_args()

    client = get_client()
    years = [args.year] if args.year else [2022, 2023]

    rows = []
    for year in years:
        rows.extend(process_year(client, year, args.limit))

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, "decisions.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} decisions to {out_csv}")
    if len(df):
        print("Tier distribution:")
        print(df["tier"].value_counts().to_string())

    if args.merge:
        papers_path = os.path.join(args.out_dir, "papers.csv")
        if not os.path.exists(papers_path):
            print(f"WARN: {papers_path} not found; skipping merge.", file=sys.stderr)
            return
        papers = pd.read_csv(papers_path)
        merged = papers.merge(
            df[["id", "decision_raw", "tier"]], on="id", how="left"
        )
        merged.to_csv(papers_path, index=False)
        print(f"Merged tier/decision into {papers_path} "
              f"({merged['tier'].notna().sum()}/{len(merged)} matched).")


if __name__ == "__main__":
    main()
