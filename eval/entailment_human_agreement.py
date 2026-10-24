import csv, json, os, re, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import llm

LEVEL_DESCRIPTION = """Entailment levels:
1 = No entailment / unrelated. Claim 2 does not cover Claim 1.
2 = Weak overlap. Claim 2 shares a topic with Claim 1 but does not really support the same contribution.
3 = Partial entailment. Claim 2 covers part of Claim 1, but important method/result/resource details differ or are missing.
4 = Strong entailment. Claim 2 mostly covers Claim 1, with only minor missing details or wording differences.
5 = Full entailment. Claim 2 fully covers the substantive content of Claim 1.
""".strip()

def build_claim_only_prompt(test_claim, reference_claim):
    return f"""You are evaluating claim entailment for research-paper novelty analysis.

Your task is to decide the level to which Claim 2 entails or covers Claim 1.
Use the 1-5 scale below.

{LEVEL_DESCRIPTION}

Claim 1:
{test_claim}

Claim 2:
{reference_claim}

Output ONLY one integer: 1, 2, 3, 4, or 5."""

def build_context_prompt(test_claim, reference_claim, test_support="", reference_support=""):
    return f"""You are evaluating claim entailment for research-paper novelty analysis.

Your task is to decide the level to which Claim 2 entails or covers Claim 1.
The support passages are only additional context. They are NOT the claims being evaluated directly.
Evaluate entailment between Claim 2 and Claim 1.
Use the 1-5 scale below.

{LEVEL_DESCRIPTION}

Claim 1:
{test_claim}

Additional context for Claim 1:
{test_support.strip() or "No additional context provided."}

Claim 2:
{reference_claim}

Additional context for Claim 2:
{reference_support.strip() or "No additional context provided."}

Output ONLY one integer: 1, 2, 3, 4, or 5."""

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SHEET = os.path.join(ROOT, "temp", "Entailment Annotation Sheet - {}.csv")
ANNOTATORS = ["anon", "annotator_c", "annotator_a"]
MODELS = {
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "haiku-4.5": ("anthropic", "claude-haiku-4-5-20251001"),
    "sonnet-4.6": ("anthropic", "claude-sonnet-4-6"),
    "opus-4.6": ("anthropic", "claude-opus-4-6"),
}

def load(name):
    rows = list(csv.DictReader(open(SHEET.format(name))))
    return {r["Example ID"]: r for r in rows if r["Example ID"].strip()}

def score(x):
    m = re.search(r"[1-5]", str(x))
    return int(m.group()) if m else None

def alpha_ordinal(units):
    vals = sorted({v for u in units for v in u})
    n_c = Counter(v for u in units for v in u if len(u) >= 2)
    n = sum(n_c.values())
    def delta(c, k):
        if c == k: return 0.0
        lo, hi = min(c, k), max(c, k)
        s = sum(n_c[g] for g in vals if lo <= g <= hi) - (n_c[lo] + n_c[hi]) / 2
        return s * s
    Do = 0.0
    for u in units:
        m = len(u)
        if m < 2: continue
        Do += sum(delta(c, k) for c in u for k in u) / (m - 1)
    Do /= n
    De = sum(n_c[c] * n_c[k] * delta(c, k) for c in vals for k in vals) / (n * (n - 1))
    return 1 - Do / De

def metrics(pred, gold):
    pairs = [(p, g) for p, g in zip(pred, gold) if p is not None]
    d = [abs(p - g) for p, g in pairs]
    return {"n": len(pairs), "alpha": round(alpha_ordinal([[p, g] for p, g in pairs]), 3),
            "mad": round(sum(d) / len(d), 3), "exact": round(sum(x == 0 for x in d) / len(d), 3),
            "within1": round(sum(x <= 1 for x in d) / len(d), 3)}

def ask(backend, model, prompt):
    msgs = [{"role": "user", "content": prompt}]
    if backend == "openai":
        return llm.chat(model, msgs, max_tokens=5, temperature=0.0)
    return llm.claude_api(model, msgs, max_tokens=5, temperature=0.0)

def main():
    sheets = {a: load(a) for a in ANNOTATORS}
    base = load("base")
    ids = [i for i in base if all(i in sheets[a] for a in ANNOTATORS)]
    human = {i: [score(sheets[a][i]["Coverage Score (1-5)"]) for a in ANNOTATORS] for i in ids}
    human = {i: [v for v in vs if v is not None] for i, vs in human.items()}
    out = {"n_rows": len(ids), "human": {}}
    out["human"]["alpha"] = round(alpha_ordinal([v for v in human.values() if len(v) >= 2]), 3)
    pair_d = [abs(a - b) for vs in human.values() for x, a in enumerate(vs) for b in vs[x + 1:]]
    out["human"]["pairwise_exact"] = round(sum(d == 0 for d in pair_d) / len(pair_d), 3)
    out["human"]["pairwise_within1"] = round(sum(d <= 1 for d in pair_d) / len(pair_d), 3)
    gold = {}
    for i, vs in human.items():
        c = Counter(vs).most_common(1)[0]
        if c[1] >= 2: gold[i] = c[0]
    out["n_majority"] = len(gold)
    gids = sorted(gold)
    prompts = {}
    for i in gids:
        r = base[i]
        prompts[("claim_only", i)] = build_claim_only_prompt(r["Test Claim"], r["Reference Claim"])
        prompts[("context", i)] = build_context_prompt(r["Test Claim"], r["Reference Claim"],
            "Paper title: " + r["Test Paper Title"], "Paper title: " + r["Reference Paper Title"])
    out["models"] = {}
    for name, (backend, model) in MODELS.items():
        out["models"][name] = {}
        for variant in ["claim_only", "context"]:
            with ThreadPoolExecutor(8) as ex:
                preds = list(ex.map(lambda i: score(ask(backend, model, prompts[(variant, i)])), gids))
            out["models"][name][variant] = metrics(preds, [gold[i] for i in gids])
            print(name, variant, out["models"][name][variant], flush=True)
    out["cost_usd"] = round(llm.cost()["usd"], 3)
    out["gold_distribution"] = dict(sorted(Counter(gold.values()).items()))
    json.dump(out, open(os.path.join(ROOT, "eval", "results", "entailment_human_agreement.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
