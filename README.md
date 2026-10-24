# Do AI Scientists Know What's New?

Miri Liu, Xiayu Zhao, Encheng Xie, ChengXiang Zhai (University of Illinois Urbana-Champaign)

A major challenge facing AI scientific discovery pipelines is how to determine
when a generated idea is meaningfully novel. Ideally, novelty checks in these
pipelines must be able to reject non-novel ideas while *not* rejecting too many
novel ideas, and they should also be able to distinguish which parts of a
proposal are non-novel, as opposed to opaquely discarding entire proposals. We
design an analytical framework for evaluation of novelty checks with temporal
ground truth at both the proposal- and claim-level, using four datasets of 1,060
proposals built from published papers, reviewer disputes, and generated ideas
matching ones the field went on to publish. Auditing existing novelty checks with
this framework, we find that checks with high rejection rates on non-novel work
also tend to have high false alarms on ideas that are later realized by human
researchers, and observe that checks built to evaluate proposals as a whole can
point to claim-level prior art only by rejecting the whole proposal. We therefore
introduce **ClaimNov**, a fine-grained, interpretable claim-level novelty module.
ClaimNov catches 97% of disguised papers at a 1.5% false-alarm rate, and its
claim-level novelty profile cuts the LLM judge's false alarms from 11.8% to 1.5%.
Because its verdict rests on claim-level evidence, ClaimNov also keeps proposals
that build on established contributions, rejecting only 3% of ICLR-accepted
proposals with a reviewer-disputed contribution and keeping 91 of 100 novel
proposals with one spliced prior contribution, against 76 for the strongest
existing check.

## Quickstart

```bash
git clone https://github.com/emmatliu/claimnov && cd claimnov
pip install -r src/requirements.txt anthropic
export OPENAI_API_KEY=...  ANTHROPIC_API_KEY=...
```

```python
import sys; sys.path.insert(0, "src")
from novelty import pipeline, retrieve, gate

idea = {"id": "idea", "title": "...", "abstract": "..."}
prior_work = [{"id": "p1", "title": "...", "abstract": "..."}, ...]

pool = retrieve.build_pool(prior_work)
profile = pipeline.score_paper(idea, pool)
verdict = gate.profile_veto(idea, prior_work, profile)

print("N(P) =", profile["novelty"])
print("not novel" if verdict["veto"] else "novel", verdict["evidence"])
for c in profile["claims"]:
    print(c["dimension"], c["claim"], [j["entailment_level"] for j in c["judged"]])
```

Novelty is measured against the prior work you pass in. The evaluation
framework, datasets and per-item results behind the paper are in `eval/`,
`review_experiments/`, `data/` and `results/` (see the READMEs there).

## Citation

```bibtex
@article{liu2026claimnov,
  title  = {Do {AI} Scientists Know What's New?},
  author = {Liu, Miri and Zhao, Xiayu and Xie, Encheng and Zhai, ChengXiang},
  year   = {2026},
  note   = {arXiv preprint}
}
```
