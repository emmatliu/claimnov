# Results

`eval/results/` holds the per-item verdict records behind the paper's tables, one JSON
line per (check, item). `review_experiments/scripts/r0_build_items.py` and
`review_experiments/scripts/analyze.py` rebuild the main and appendix tables from
these records without model calls.

| file(s) in `eval/results/` | contents |
|---|---|
| `smuggle_full.jsonl`, `smuggle_v2.jsonl`, `smuggle_v2_api.jsonl` | existing checks on the disguised proposals (sources 0–9, 10–99) |
| `positive_full.jsonl`, `claim_sort_full_indep.jsonl` | existing checks on the generated ideas, and the labels that define the later-realized set |
| `negative_full.jsonl`, `claim_sonnet.jsonl` | existing checks on the claim-disputed pairs |
| `deployed_sonnet*.jsonl` | existing checks with Sonnet 4.6 as judge |
| `hybrids_eval.jsonl` | all checks on the hybrid proposals |
| `coupled_veto_*_profile*.jsonl` | ClaimNov (claim profile + verdict) on each set |
| `coupled_veto_*_candcentral.jsonl`, `coupled_veto_neg.jsonl` | ClaimNov ablations |

The summary files here give numbers reported in the paper:

| file | paper |
|---|---|
| `gate_audit/smuggle_n400_sonnet_judge.json` | Table 1 (disguises and later-realized set) |
| `gate_audit/claim_level_sonnet.json` | Table 2 (claim-disputed pairs) |
| `gate_audit/hybrids_sonnet_judge.json` | Table 3 (hybrid proposals) |
| `gate_audit/tokens_per_verdict.json` | cost comparison (Pipeline Survey appendix) |
| `label_quality/n2_audit_final.json` | annotator agreement (Label Validation appendix) |
| `module_validation/extraction_gepa.json` | claim extraction (Component Validations appendix) |
| `module_validation/retrieval_S1.json` | same-dimension retrieval (Component Validations appendix) |
| `module_validation/a2_context_vs_tier.json`, `entailment_tier_sweep.json`, `entailment_human_agreement.json`, `p2_pairwise_winrate.json` | graded entailment judge (Component Validations appendix) |
