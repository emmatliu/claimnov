import os

NO_JUDGE_CONTEXT = os.environ.get("NOVELTY_NO_CONTEXT") == "1"
ANY_DIMENSION = os.environ.get("NOVELTY_ANY_DIM") == "1"
CALIBRATED_LEGEND = os.environ.get("NOVELTY_CALIB") == "1"

EXTRACT_MODEL = "gpt-4o-mini"

GATE_JUDGE_MODEL = "claude-sonnet-4-6"
GATE_TOP_K = 8
VERDICT = "profile"
JUDGE_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-large"

DIMENSIONS = ("problem", "methodology", "resource", "result")

TOP_K = 10
TOP_K_MIN, TOP_K_MAX, TOP_K_FRAC = 10, 40, 0.20
def top_k_for(n_candidates: int) -> int:
    return int(min(TOP_K_MAX, max(TOP_K_MIN, round(TOP_K_FRAC * n_candidates))))

ALPHA = 0.5
DIMENSION_WEIGHTS = {
    "problem": 0.4,
    "result": 0.3,
    "methodology": 0.2,
    "resource": 0.1,
}

ENTAIL_MIN, ENTAIL_MAX = 1, 5

STRONG_LEVEL = 4

SUPPORT_CTX_N = 2
SUPPORT_CTX_CHARS = 800


def entail_to_unit(level: int) -> float:
    level = max(ENTAIL_MIN, min(ENTAIL_MAX, int(level)))
    return (level - ENTAIL_MIN) / (ENTAIL_MAX - ENTAIL_MIN)


def norm_dimension(d: str) -> str:
    d = (d or "").strip().lower()
    return d if d in DIMENSIONS else ""
