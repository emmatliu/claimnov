# Robustness and attribution experiments

Scripts behind the appendix analyses and the OpenNovelty-style, shared-ranking and
judge-only rows of the main tables. Run from the repository root;
outputs and per-call caches go to `review_experiments/out/`. Model calls use
temperature 0 through the Anthropic API, Amazon Bedrock or OpenAI (keys in `.env`).
Scripts marked "no API" only read per-item records and caches.

The claim-disputed scripts read `eval/data/claim_disputed/testset_clean.jsonl`,
which `eval/hydrate_data.py` builds from `data/claim_disputed/` (see `data/README.md`).

| prefix | analyses |
|---|---|
| `r0_*`, `e1`-`e3`, `analyze.py` | per-item tables, intervals and McNemar tests, judge-only ablation, gpt-5 relabel, repeated runs |
| `r1_*` | OpenNovelty-style baseline, non-Claude judges (byte-identical subset), best thresholds of score-based checks |
| `r2_*` | attribution precision, reworded verdict questions, similarity thresholds (BM25, embeddings) |
| `r3_*` | clustered tests, repeated CoI-Agent runs, cost and latency |
| `r4_*` | anchor in the pool, attribution as retrieval and shared rankings, non-Claude judges on the full sets, all generated ideas |
| `r5_*` | in-the-loop pilot, retrieve-and-judge baseline, SciNCL/SPECTER2 thresholds |
| `r6_*` | shipped ClaimNov on ~200-paper pools and with the anchor in the pool |

BM25 is `rank_bm25`'s BM25Okapi with default parameters (k1 1.5, b 0.75) over
lowercased alphanumeric tokens with stopwords removed. Scientific embeddings are
SciNCL (`malteos/scincl`) and SPECTER2 (`allenai/specter2_base` with the
`allenai/specter2` proximity adapter).
