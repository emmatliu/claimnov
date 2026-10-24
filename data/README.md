# ClaimNov: data

Generated ideas are included in full; third-party text (paper abstracts, review text) and proposals that restate published papers (disguises, hybrids) are replaced by pointers and the code that rebuilds them.

To run the evaluation code, first rebuild the full-text files it reads:

```bash
python eval/hydrate_data.py
python eval/make_disguises.py && python eval/make_disguises_v2.py && python eval/make_disguises_v3.py
python eval/build_hybrids.py
```

`hydrate_data.py` fetches the abstracts of every paper referenced here from Semantic Scholar, arXiv and OpenReview (set `S2_API_KEY` for faster fetching) and writes `eval/data/`. A few abstracts may differ slightly from the versions we used, e.g. a camera-ready rather than submitted ICLR abstract. The other scripts regenerate the disguised and hybrid proposals with Sonnet 4.6 (keys in `.env`); regenerated text will not match ours word for word.

- `disguises/`: for each of the 100 sources, pointers to the source paper (arXiv id, Semantic Scholar id) and to the 19 pool papers (Semantic Scholar ids and titles; abstracts via the Semantic Scholar API). The disguised proposals themselves are not included, since each one restates a published paper; the commands above rebuild them from the source abstracts.
- `claim_disputed/`: pointers for the 200 mined reviewer-disputed pairs, of which the 160 with `status` ok are evaluated (OpenReview id of A, identifier of B, relation type, disputed dimension, ICLR decision) and pool pointers; abstracts and review text via OpenReview and Semantic Scholar.
- `later_realized/`: per anchor, pointers to the anchor, its pre-cutoff pool, and its post-cutoff follow-up set, plus the 100 generated ideas (ours) and their labels; `later_realized_set.json` lists the 400 ideas in the later-realized set.
- Hybrid proposals are not included, since each one restates a published paper's contribution. `eval/build_hybrids.py` rebuilds them from the later-realized ideas and pools; the per-item verdict records are in `eval/results/hybrids_eval.jsonl`.
- `data/annotation/`: majority labels from the three-annotator audit.
- `data/annotation/later_realized_audit/`: the human audit of 60 generated ideas (Appendix F.1), with each packet's labels (one annotator per item), the hidden key of strata and check decisions, the scores from `score_audit.py`, and the annotator instructions. The packets themselves contain paper abstracts and are rebuilt by `review_experiments/scripts/r6_audit_build.py`.
- `eval/results/`: the per-item verdict records behind the paper's tables; `results/` holds the summary files, indexed in `results/README.md`.
