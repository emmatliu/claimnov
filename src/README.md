# ClaimNov module

```
idea + prior work
  → typed contribution claims                         novelty/extract.py
  → per claim, same-dimension prior claims            novelty/retrieve.py
  → graded coverage of each claim (1–5)               novelty/entail.py
  → aggregate novelty score N(P)                      novelty/aggregate.py
  → verdict judge over the claim profile              novelty/gate.py (profile_veto)
```

Requires `OPENAI_API_KEY` (extraction, coverage, embeddings) and
`ANTHROPIC_API_KEY` (verdict judge). Models and other settings are in
`novelty/config.py`. See the quickstart in the top-level README for library use.

Papers and ideas are dicts (or JSONL lines) of the form

```json
{"id": "p1", "title": "...", "abstract": "..."}
```

To write the claim profile and N(P) for many ideas at once:

```bash
python run_novelty.py --target ideas.jsonl --corpus prior_work.jsonl --out report.json
```
