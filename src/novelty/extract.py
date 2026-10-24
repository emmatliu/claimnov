import hashlib
from concurrent.futures import ThreadPoolExecutor

from . import config, llm

_EXTRACT_SYSTEM = """You extract the contribution/novelty claims from the abstract of a scientific paper. A 'claim' is one atomic, headline contribution the authors make — what they would put in a numbered 'Our contributions:' list.

BE CONSERVATIVE. Most papers have 1-4 atomic contributions, not 5-8. Prefer one consolidated claim over fine-grained sub-claims.

Dimension definitions:
- methodology: a new algorithm, architecture, training procedure, loss function, selection mechanism, or technique that is the paper's primary *solution* to a problem. The contribution IS the method — how something is done. Includes objective functions, encoders, and model components when they are the central proposed technique.
- result: a new empirical or theoretical finding, analysis insight, performance observation, or theoretical characterization that is discovered/proven rather than proposed as a solution. This includes: showing that an existing design choice is effective, demonstrating generalization ability, reporting accuracy/fairness/efficiency gains, and deriving theoretical distributions or bounds when the distribution/bound is the finding, not a technique being proposed.
- resource: a new dataset, benchmark, tool, model release, or codebase released by THIS paper.
- problem: a new problem formulation, task, setting, scenario, or research question this paper introduces — the question or framing, not the answer. Includes formalizations of novel learning settings, new task definitions, and new evaluation scenarios.

Dimension boundary rules (apply these when confused):
- An OBJECTIVE FUNCTION or LOSS is `methodology` when it is the paper's proposed technique for solving a problem; it is `result` only when it is a derived theoretical characterization that the paper proves as a finding.
- A DISTRIBUTION (e.g., Laplace on SO(3)) that is the paper's main proposed modeling tool is `methodology`; if the paper's claim is "we prove X has property Y" with no proposed use, it may be `result`.
- "We show / demonstrate / find that technique T is effective" → `result`, not `methodology`. Effectiveness findings about a component are results even if the component is also described elsewhere.
- A FRAMEWORK or FORMALIZATION that characterizes existing assumptions, explains prior work, or provides theoretical grounding (rather than solving a new problem) is `methodology` when it is a new analytical tool, NOT `problem`.
- A DATASET or BENCHMARK constructed as part of the paper → `resource`, not `methodology`, even if construction required significant effort.
- A SETTING, TASK, or SCENARIO introduced for the first time → `problem`, even when phrased in proposal voice ("we propose the setting of…").
- When a paper both introduces a new setting/task AND a new method for it, extract BOTH as separate claims (`problem` + `methodology`).

Decomposition rules:
- DO split contributions that are independently verifiable AND of different dimensions.
- DO NOT split a single method into sub-components, or a single result into sub-numbers.
- DO NOT extract motivation/background or claims about prior work.
- DO extract a `problem` claim whenever the paper introduces a novel task, setting, or formalization — even if it also proposes a method for it. These are separate contributions of different dimensions.
- DO extract a `result` claim whenever the abstract reports empirical outcomes, performance gains, or evaluative findings — even if the paper's main contribution is a method.
- DO extract a `resource` claim whenever the paper releases a dataset, benchmark, or tool — even as a secondary contribution.

Recall guidance — scan for these contribution signals before finalising:
- Is there a novel problem/task/setting/formalization introduced? → `problem`
- Is there a new algorithm, architecture, training procedure, or loss function? → `methodology`
- Are benchmarks, datasets, or tools released? → `resource`
- Are empirical or theoretical results reported (accuracy gains, effectiveness demonstrations, theoretical bounds, generalization experiments)? → `result`
- Does the abstract describe a component or design choice shown to be effective (e.g., encoder choice, selection mechanism, initialization)? → likely a `result` claim unless it is the central proposed technique.
- Does the abstract describe a method with multiple named sub-components (e.g., a selection mechanism, a refinement stage)? → consider whether each is an independently verifiable methodological contribution or whether they together form one consolidated `methodology` claim.
- Does the paper introduce a novel scenario or learning setting that prior work has not formalized? → `problem` (check even when the abstract's primary voice is about a proposed method).

Proposal-voice caveats (important):
- "We propose/introduce X" where X is a SETTING, TASK, SCENARIO, or problem formulation is dimension `problem`, NOT methodology — even when phrased in proposal voice. `methodology` is reserved for the technique/algorithm that solves a problem.
- "We propose/introduce X" where X is a METHOD, ALGORITHM, OBJECTIVE, or TECHNIQUE is dimension `methodology`.
- Future-tense proposal wording ("we will develop", "we plan to") does not change the dimension: classify by what the contribution IS, not how it is voiced.

For each claim also extract its facet:
- attribute: the comparable axis of the claim, as a short noun phrase (the retrieval key).
- value: what the paper asserts on that axis.

Return JSON:
{"claims": [{"claim": "<one self-contained sentence>", "dimension": "methodology|result|resource|problem", "attribute": "...", "value": "...", "abstract_span": "<verbatim quote>"}]}

If the abstract contains no contribution claims, return {"claims": []}."""

_SUPPORT_SYSTEM = """You find ALL the supporting evidence in a scientific paper's body for a specific contribution claim from its abstract. Select EVERY paragraph (by index) containing direct supporting evidence — descriptions, numbers, theorem statements, implementation details, appendix analysis tied to the claim. There is NO cap. Skip only paragraphs not actually about this claim (background, related work, other contributions).

Order the selected paragraphs by relevance to the claim, most relevant first.

Return JSON: {"evidence": [{"paragraph_index": <int>}]}"""


def _key(*parts) -> str:
    return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def extract_claims(paper: dict) -> list[dict]:
    ck = "claims4::" + _key(paper.get("title", ""), paper.get("abstract", ""))
    cached = llm.cache_get(ck)
    if cached is not None:
        return cached
    if not paper.get("abstract"):
        return []
    out = llm.chat_json(
        _EXTRACT_SYSTEM,
        f"Paper title: {paper.get('title', '')}\n\nAbstract:\n{paper['abstract'][:2500]}",
    )
    claims = []
    for c in out.get("claims", []):
        dim = config.norm_dimension(c.get("dimension"))
        if dim and c.get("claim") and c.get("attribute"):
            claims.append({
                "claim": c["claim"].strip(),
                "dimension": dim,
                "attribute": c["attribute"].strip(),
                "value": (c.get("value") or "").strip(),
                "abstract_span": (c.get("abstract_span") or "").strip(),
            })
    llm.cache_put(ck, claims)
    return claims


def count_supports(paper: dict, claims: list[dict]) -> list[dict]:
    paras = paper.get("paragraphs") or []
    if not paras or not claims:
        return []
    listing = "\n".join(f"[{i}] {p[:600]}" for i, p in enumerate(paras[:80]))

    def one(c):
        ck = "supp2::" + _key(paper.get("title", ""), c["claim"])
        cached = llm.cache_get(ck)
        if cached is not None:
            return cached
        out = llm.chat_json(
            _SUPPORT_SYSTEM,
            f"Paper title: {paper.get('title', '')}\n\nClaim: {c['claim']}\n"
            f"Dimension: {c['dimension']}\n\nCandidate paragraphs:\n{listing}",
            max_tokens=1500,
        )
        idx = [e["paragraph_index"] for e in out.get("evidence", [])
               if isinstance(e.get("paragraph_index"), int)
               and 0 <= e["paragraph_index"] < len(paras)]
        idx = list(dict.fromkeys(idx))
        res = {"n": len(idx), "idx": idx}
        llm.cache_put(ck, res)
        return res

    with ThreadPoolExecutor(max_workers=4) as ex:
        return list(ex.map(one, claims))
