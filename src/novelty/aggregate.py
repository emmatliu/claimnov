from . import config


def claim_anticipation(es: list[float], alpha: float = config.ALPHA,
                       k: int = config.TOP_K) -> float:
    top = sorted(es, reverse=True)[:k]
    if not top:
        return 0.0
    num = sum(alpha ** i * e for i, e in enumerate(top))
    den = sum(alpha ** i for i in range(len(top)))
    return num / den


def claim_signals(es: list[float], alpha: float = config.ALPHA,
                  k: int = config.TOP_K) -> dict:
    top = sorted(es, reverse=True)[:k]
    strong = config.entail_to_unit(config.STRONG_LEVEL)
    return {
        "anticipation": claim_anticipation(es, alpha=alpha, k=k),
        "max_e": top[0] if top else 0.0,
        "breadth": (sum(e >= strong for e in top) / len(top)) if top else 0.0,
    }


def paper_novelty(claims: list[dict],
                  weights: dict = config.DIMENSION_WEIGHTS) -> dict:
    if not claims:
        return {"novelty": None, "novelty_max": None, "novelty_breadth": None,
                "anticipation": None, "per_dimension": {}}

    supports = [c.get("n_supports") for c in claims]
    if any(s is None for s in supports) or sum(s or 0 for s in supports) == 0:
        imps = [1.0 / len(claims)] * len(claims)
    else:
        total = sum(supports)
        imps = [s / total for s in supports]

    signals = ("anticipation", "max_e", "breadth")
    A = dict.fromkeys(signals, 0.0)
    A_max = 0.0
    per_dim = {}
    for c, imp in zip(claims, imps):
        w = weights.get(c["dimension"], 0.0)
        A_max += w * imp
        for s in signals:
            A[s] += w * imp * c.get(s, 0.0)
        d = per_dim.setdefault(c["dimension"], {"claims": 0, "anticipation_mass": 0.0,
                                                "importance_mass": 0.0})
        d["claims"] += 1
        d["anticipation_mass"] += imp * c["anticipation"]
        d["importance_mass"] += imp

    if A_max == 0:
        return {"novelty": None, "novelty_max": None, "novelty_breadth": None,
                "anticipation": None, "per_dimension": per_dim}
    return {
        "novelty": round(1.0 - A["anticipation"] / A_max, 4),
        "novelty_max": round(1.0 - A["max_e"] / A_max, 4),
        "novelty_breadth": round(1.0 - A["breadth"] / A_max, 4),
        "anticipation": round(A["anticipation"] / A_max, 4),
        "per_dimension": {
            d: {**v, "mean_anticipation":
                round(v["anticipation_mass"] / v["importance_mass"], 4)
                if v["importance_mass"] else None}
            for d, v in per_dim.items()
        },
    }
