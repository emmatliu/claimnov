import json, os, re, math, hashlib, collections
import numpy as np

REPO = "."
PILOT = f"{REPO}/eval"
RES = f"{PILOT}/results"
EXP = f"{REPO}/review_experiments"
OUT = f"{EXP}/out"
PCACHE = f"{PILOT}/cache"
NCACHE = f"{REPO}/src/.cache/novelty_cache.json"
SONNET_46_BEDROCK = "us.anthropic.claude-sonnet-4-6"

norm = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())[:40]


def jl(name):
    return [json.loads(l) for l in open(f"{RES}/{name}.jsonl")]


def final_rows(files, pick="first", gate_key="gate"):
    by = collections.defaultdict(list)
    for f in files:
        for r in jl(f):
            r["_f"] = f
            by[(r["item"], r.get(gate_key))].append(r)
    out = {}
    for k, v in by.items():
        ok = [r for r in v if r.get("novel") is not None]
        out[k] = (ok[0] if pick == "first" else ok[-1]) if ok else v[-1]
    return out, by


DEPLOYED_FILES = ["deployed_sonnet", "deployed_sonnet_rev", "deployed_sonnet_bedrock",
                  "deployed_sonnet_panel", "deployed_sonnet_panel_rev", "deployed_sonnet_panel_bedrock"]
SMUGGLE_FILES = ["smuggle_full", "smuggle_v2", "smuggle_v2_api"]


def later_set():
    return sorted(json.load(open(f"{PILOT}/data/later_realized_set.json")))


def disguise_keys():
    dz_all = json.load(open(f"{PILOT}/data/disguises_v3.json"))
    ok = {}
    for d in dz_all:
        if d["seed_idx"] >= 10 and (d["rung"] <= 2 or d.get("verified_same")):
            ok.setdefault(d["seed_idx"], set()).add(d["rung"])
    complete = sorted(s for s, rr in ok.items() if rr == {1, 2, 3, 4})
    chosen = set(range(10)) | set(complete[:90])
    dz = [d for d in dz_all if d["seed_idx"] in chosen]
    return sorted(chosen), dz


def pool_B(seed):
    pd = json.load(open(f"{PILOT}/data/pools/seed{seed}.json"))
    return next((p for p in pd["pool"] if p.get("is_B")), None), pd["pool"]


def seed_of(item):
    return int(re.match(r"seed(\d+)_", item).group(1))


def anchor_of(item):
    return item.split("::")[0]


def anchor_dir(slug):
    return next(p for p in (f"{PILOT}/data/anchors/{slug}", f"{PILOT}/data/anchors_full/{slug}")
                if os.path.exists(f"{p}/ideas.json"))


def cached_embed(text, model="text-embedding-3-small"):
    h = hashlib.sha256((model + "::" + text).encode()).hexdigest()[:24]
    fp = f"{PCACHE}/e_{h}.json"
    return json.load(open(fp)) if os.path.exists(fp) else None


def cached_bedrock(model, messages, max_tokens, cache_tag, system=None):
    h = hashlib.sha256(json.dumps(["bedrock", model, system, messages, max_tokens, cache_tag]).encode()).hexdigest()[:24]
    fp = f"{PCACHE}/c_{h}.json"
    return json.load(open(fp))["text"] if os.path.exists(fp) else None


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def fmt_ci(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100*k/n:.1f}% [{100*lo:.1f}, {100*hi:.1f}]"


def mcnemar_exact(a, b):
    common = sorted(set(a) & set(b))
    a_only = sum(1 for i in common if a[i] and not b[i])
    b_only = sum(1 for i in common if b[i] and not a[i])
    both = sum(1 for i in common if a[i] and b[i])
    neither = len(common) - a_only - b_only - both
    n = a_only + b_only
    if n == 0:
        p = 1.0
    else:
        k = min(a_only, b_only)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n)
    return {"n_paired": len(common), "both": both, "a_only": a_only, "b_only": b_only,
            "neither": neither, "p_exact": p}


def dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{name}", "w") as f:
        json.dump(obj, f, indent=1, default=str)
