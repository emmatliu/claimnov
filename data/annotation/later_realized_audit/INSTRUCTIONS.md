# Annotator instructions

There are three packets, `packet_A.md`, `packet_B.md` and `packet_C.md`, with 20 research ideas each. Each annotator takes one packet, so every item is labeled once.

Each idea comes with the 20 most similar papers from the prior-work pool, all published before the cutoff, most similar first (P1–P20). Some items also list one or more later papers published after the cutoff, marked "F". These are for reference only. Do not count an F paper as prior work.

Work independently, do not search the web, and judge only from the idea text and the listed abstracts.

## Questions for each item

**Q1. Does any listed paper (P1–P20) already do substantially what the idea proposes, with the same core problem and the same core approach?**
- Answer `yes` or `no`.
- If `yes`, give the number of the paper that does it best (for example `7` for P7).
- Answer `no` for a paper that only shares the topic, that the idea builds on, or that does one side component of the idea.
- Answer `yes` only if the idea's main contribution is already there. A missing peripheral detail does not make the idea new.

**Q2 (only if Q1 = `no`). Is any single contribution of the idea clearly established by a listed paper, even though the idea as a whole is not?**
- If Q1 is `yes`, leave Q2 blank; a yes on Q1 already means a contribution is established.
- Otherwise answer `yes` or `no`.
- If `yes`, give the paper number. A few words naming the contribution are optional.

**Confidence.** Rate your Q1 answer `1` (guess), `2` (fairly sure) or `3` (certain).

## Recording answers

Fill in your packet's file (`labels_A.csv`, `labels_B.csv` or `labels_C.csv`), one row per item:

| column | value |
|---|---|
| `item_id` | already filled in |
| `annotator` | your initials |
| `done_by_listed_paper` | `yes` / `no` (Q1) |
| `paper_number` | the P number for Q1 = yes, otherwise blank |
| `contribution_established` | `yes` / `no` (Q2); blank when Q1 = `yes` |
| `contribution_text` | optional short paraphrase for Q2 = yes. Quote it if it contains commas |
| `contribution_paper_number` | the P number for Q2 = yes |
| `confidence` | `1`, `2` or `3` |
| `notes` | optional |

Expect about 3–4 minutes per item, so roughly 1–1.5 hours per packet. Skim the abstracts, most similar first; Q1 is usually settled within the first few papers.

## Scoring (for the authors)

Run `python score_audit.py`. It reads the three label files and `key.json`, the hidden mapping of strata, model labels and check decisions. Do not open `key.json` before annotating. It reports:
- the human-confirmed novel rate per stratum, and a stratum-weighted estimate for all 400 later-realized ideas;
- each check's false-alarm rate on the human-confirmed-novel subset;
- whether each check's rejections name the same paper the annotator named;
- agreement with the Opus and gpt-5 labels;
- for the 15 sorter-matched ideas, the human already-done rate and each check's recall on them.
