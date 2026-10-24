import hashlib
import json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.abspath(os.path.join(os.path.dirname(HERE), "tier_sweep_data"))
assert os.path.isdir(os.path.join(DATA, "from_miri_0719")), DATA
os.environ.setdefault("NOVELTY_DATA", DATA)
os.environ.setdefault("LOCALAPPDATA", os.path.join(DATA, "cache"))
os.makedirs(os.environ["LOCALAPPDATA"], exist_ok=True)
if not os.environ.get("OPENAI_API_KEY"):
    for line in open(".env"):
        if line.startswith("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()

sys.path.insert(0, os.path.join(ROOT, "legacy", "claim_validation"))
sys.path.insert(0, os.path.join(ROOT, "legacy", "claim-level_retrieval_200"))
import claim_retrieval as pm
import entail_eval as ee

OUTDIR = os.path.join(HERE, "outputs")
os.makedirs(OUTDIR, exist_ok=True)

MODELS = {"4o-mini": ("openai", "gpt-4o-mini"),
          "5nano": ("openai", "gpt-5-nano"),
          "5mini": ("openai", "gpt-5-mini"),
          "haiku45": ("bedrock", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
          "sonnet46": ("bedrock", "us.anthropic.claude-sonnet-4-6"),
          "opus46": ("bedrock", "us.anthropic.claude-opus-4-6-v1")}

_bedrock = None


def _bedrock_json(model_id, system, user):
    global _bedrock
    if _bedrock is None:
        import boto3
        _bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
    for a in range(6):
        try:
            r = _bedrock.converse(
                modelId=model_id, system=[{"text": system}],
                messages=[{"role": "user",
                           "content": [{"text": user + "\n\nReturn ONLY the JSON object."}]}],
                inferenceConfig={"maxTokens": 200})
            u = r.get("usage", {})
            pm.cost["in"] += u.get("inputTokens", 0)
            pm.cost["out"] += u.get("outputTokens", 0)
            txt = r["output"]["message"]["content"][0]["text"].strip()
            i, j = txt.find("{"), txt.rfind("}")
            return json.loads(txt[i:j + 1])
        except Exception:
            time.sleep(2 ** a)
    return {"score": 1, "rationale": "error"}


def install_judge(mtag):
    kind, model_id = MODELS[mtag]

    def entail(client, qf, cf, cache, q_title="", c_title="", q_abs="", c_abs=""):
        key = f"EN4::{mtag}::" + hashlib.md5(
            (ee.facet_text(qf) + "||" + ee.facet_text(cf) + "||"
             + q_title + "||" + c_title).encode()).hexdigest()[:16]
        if key in cache:
            return cache[key]
        u = f"TARGET paper: {q_title[:120]}\n"
        if q_abs:
            u += f"TARGET abstract excerpt: {q_abs[:700]}\n"
        u += (f"TARGET claim:\n  attribute: {qf.get('attribute','')}\n"
              f"  value: {qf.get('value','')}\n\n")
        u += f"PRIOR work: {c_title[:120]}\n"
        if c_abs:
            u += f"PRIOR abstract excerpt: {c_abs[:700]}\n"
        u += (f"PRIOR claim:\n  attribute: {cf.get('attribute','')}\n"
              f"  value: {cf.get('value','')}")
        if kind == "bedrock":
            j = _bedrock_json(model_id, ee.ENTAIL_SYS, u)
        else:
            msg = [{"role": "system", "content": ee.ENTAIL_SYS},
                   {"role": "user", "content": u}]
            j = None
            for a in range(4):
                try:
                    if model_id.startswith("gpt-5"):
                        r = client.chat.completions.create(
                            model=model_id, response_format={"type": "json_object"},
                            messages=msg, max_completion_tokens=2000,
                            extra_body={"reasoning_effort": "low"})
                    else:
                        r = client.chat.completions.create(
                            model=model_id, temperature=0,
                            response_format={"type": "json_object"}, messages=msg,
                            max_tokens=90)
                    pm.cost["in"] += r.usage.prompt_tokens
                    pm.cost["out"] += r.usage.completion_tokens
                    j = json.loads(r.choices[0].message.content)
                    break
                except Exception:
                    time.sleep(2 ** a)
            if j is None:
                j = {"score": 1, "rationale": "error"}
        j["score"] = max(1, min(5, int(j.get("score", 1))))
        with pm._cache_lock:
            cache[key] = j
            pm._save_cache()
        return j

    ee.entail = entail
    ee.score_instance.__globals__["entail"] = entail


CONFIGS = [("", "base", 5), ("_d4k5", "d4", 5), ("_d4k10", "d4", 10)]


def run(mtag):
    install_judge(mtag)
    from openai import OpenAI
    client = OpenAI()
    for tag, retrieval, k in CONFIGS:
        ee.RETRIEVAL, ee.TOPK = retrieval, k
        for which, fn in (("pos", ee.run_pos), ("neg", ee.run_neg)):
            p = os.path.join(OUTDIR, f"entail_{which}_{mtag}{tag}.jsonl")
            if os.path.exists(p):
                print(f"skip {p} (exists)")
                continue
            out = fn(client, per_dim=100)
            with open(p, "w", encoding="utf-8") as f:
                for r in out:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[{mtag}{tag}] {which}: n={len(out)} -> {p}", flush=True)
    print("cost tokens:", pm.cost)


def auc(pos, neg):
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg)) if pos and neg else float("nan")


def analyze():
    print(f"{'judge':>9} {'cfg':>7} {'n_pos':>5} {'n_neg':>5} {'AUC':>6} "
          f"{'gold_mean':>9} {'gold@topk':>9}")
    for mtag in MODELS:
        for tag, _, _ in CONFIGS:
            pp = os.path.join(OUTDIR, f"entail_pos_{mtag}{tag}.jsonl")
            np_ = os.path.join(OUTDIR, f"entail_neg_{mtag}{tag}.jsonl")
            if not (os.path.exists(pp) and os.path.exists(np_)):
                continue
            pos = [json.loads(l) for l in open(pp)]
            neg = [json.loads(l) for l in open(np_)]
            a = auc([r["novelty"] for r in pos], [r["novelty"] for r in neg])
            gs = [s for r in neg for s in r.get("gold_scores", [])]
            gr = sum(bool(r.get("gold_in_topk")) for r in neg) / max(1, len(neg))
            print(f"{mtag:>9} {tag or 'base':>7} {len(pos):>5} {len(neg):>5} "
                  f"{a:>6.3f} {sum(gs)/max(1,len(gs)):>9.2f} {gr:>9.2f}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        import build_instances
        build_instances.main()
    elif cmd == "run":
        run(sys.argv[2])
    elif cmd == "analyze":
        analyze()
