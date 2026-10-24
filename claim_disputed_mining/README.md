# Mining the claim-disputed pairs

These scripts rebuild the 200 (A, B) pairs of the claim-disputed set from public
ICLR 2022-23 reviews on OpenReview. We do not redistribute review text; run these
scripts to regenerate it locally. Requires `openai`, `requests`, `openreview-py`
and `OPENAI_API_KEY`.

Layout: the scripts read `iclr_novelty_validation_data/` next to themselves.

0. In `iclr_novelty_validation_data/`, `get_reviews.py` pulls ICLR 2022-23
   submissions from OpenReview and writes `papers.csv` and parsed reference lists
   `refs/{year}/{id}.json` (via a local Grobid server), `get_decisions.py` adds the
   final ICLR decisions, and `fetch_cited.py` resolves cited papers to arXiv.
   Set `OPENREVIEW_USERNAME` / `OPENREVIEW_PASSWORD` (and optionally `S2_API_KEY`).
1. `iclr_novelty_validation_data/get_review_comments.py` pulls the text of every
   Official_Review and Official_Comment note and writes `all_reviews.jsonl` and
   `review_mentions.jsonl` (sentences that pass a citation/keyword regex prefilter).
   It expects `papers.csv` and `refs/{year}/{id}.json`, which come from step 0.
2. `relevance_classifier.py` labels each mention: is a specific prior work B
   relevant to a contribution of A, in which dimension, and with which relation
   (anticipation / comparison / overlap).
3. `resolve_brackets.py` resolves bracket references such as "[2]" to the
   reviewer's own reference list in the same thread.
4. `citation_check.py` decides whether B is already in A's reference list.
5. `relevance_build.py` selects the balanced 200-pair set (50 per dimension).
6. `clean_export.py` writes `outputs/testset_clean.jsonl`, the input to
   `eval/build_n2_pools_full.py`, which builds the 20-paper pools.

`parse_withdrawn_refs.py` builds reference lists for two withdrawn papers with a
local Grobid server. `positive_tells/positive_tell_miner.py` mines reviewer praise
of specific contributions, used by the within-paper ranking validation in
`validation_runs/tier_sweep/p2_pairwise.py`.
