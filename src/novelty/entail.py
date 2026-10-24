import hashlib
from concurrent.futures import ThreadPoolExecutor

from . import config, llm

_LEVELS = """Entailment levels:
1 = No entailment / unrelated. Claim 2 does not cover Claim 1.
2 = Weak overlap. Claim 2 shares a topic with Claim 1 but does not really support the same contribution.
3 = Partial entailment. Claim 2 covers part of Claim 1, but important method/result/resource details differ or are missing.
4 = Strong entailment. Claim 2 mostly covers Claim 1, with only minor missing details or wording differences.
5 = Full entailment. Claim 2 fully covers the substantive content of Claim 1."""

_NUMERIC_GUIDE = """When the claims assert numeric values: a prior claim anticipates a target only if it establishes the same kind of finding under comparable conditions. For performance/achievement claims on a shared benchmark and metric, a prior result at least as strong may fully entail the target. For measurements, effect sizes, estimates, or findings in different settings or populations, a different (even larger) prior number does NOT by itself entail the target — the direction, setting, or refined estimate may be exactly what is new."""


def _context(facet: dict) -> str:
    parts = [f"Facet — attribute: {facet.get('attribute', '')}; value: {facet.get('value', '')}"]
    if config.NO_JUDGE_CONTEXT:
        return "\n".join(parts)
    supp = facet.get("support_passages") or []
    if supp:
        for i, p in enumerate(supp[:config.SUPPORT_CTX_N]):
            parts.append(f"Support passage {i + 1}: {p[:config.SUPPORT_CTX_CHARS]}")
    elif facet.get("paper_abstract"):
        parts.append(f"Abstract: {facet['paper_abstract'][:1200]}")
    return "\n".join(parts)


def _prompt(target: dict, cand: dict) -> str:
    return f"""You are evaluating claim entailment for research-paper novelty analysis.

Your task is to decide the level to which Claim 2 entails or covers Claim 1.
The context blocks are only additional context. They are NOT the claims being evaluated directly.
Evaluate entailment between Claim 2 and Claim 1.
Use the 1-5 scale below.

{_LEVELS}

{_NUMERIC_GUIDE}

Claim 1 (target):
{target['claim']}

Additional context for Claim 1:
{_context(target)}

Claim 2 (prior work: {cand.get('paper_title', 'unknown')}):
{cand['claim']}

Additional context for Claim 2:
{_context(cand)}

Output ONLY one integer: 1, 2, 3, 4, or 5."""


def judge(target: dict, cand: dict) -> dict:
    ck = "entail::" + hashlib.md5(
        (target["claim"] + "||" + cand["claim"] + "||"
         + _context(target) + "||" + _context(cand)).encode()).hexdigest()[:16]
    cached = llm.cache_get(ck)
    if cached is None:
        raw = llm.chat_text(_prompt(target, cand), model=config.JUDGE_MODEL)
        digits = [c for c in raw if c.isdigit()]
        cached = int(digits[0]) if digits else config.ENTAIL_MIN
        llm.cache_put(ck, cached)
    return {"level": cached, "e": config.entail_to_unit(cached)}


def judge_all(target: dict, cands: list[dict], max_workers: int = 4) -> list[dict]:
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        verdicts = list(ex.map(lambda c: judge(target, c), cands))
    out = [{**{k: v for k, v in c.items() if not k.startswith("_emb")}, **v}
           for c, v in zip(cands, verdicts)]
    return sorted(out, key=lambda x: -x["e"])
