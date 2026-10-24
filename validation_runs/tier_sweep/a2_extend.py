import hashlib, json, os, statistics as st, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.abspath(os.path.join(os.path.dirname(HERE), "tier_sweep_data"))
assert os.path.isdir(os.path.join(DATA, "from_miri_0719")), DATA
os.environ.setdefault("NOVELTY_DATA", DATA)
os.environ.setdefault("LOCALAPPDATA", os.path.join(DATA, "cache"))
if not os.environ.get("OPENAI_API_KEY"):
    for line in open(".env"):
        if line.startswith("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
sys.path.insert(0, os.path.join(ROOT, "legacy", "claim_validation"))
sys.path.insert(0, os.path.join(ROOT, "legacy", "claim-level_retrieval_200"))
import claim_retrieval as pm
import entail_eval as ee
import calibrate_entailment as ce

SYS_V2 = ee.ENTAIL_SYS.replace(
    "You are given each paper's title, an abstract excerpt, and a typed (attribute, value) claim facet extracted from that abstract.",
    "You are given each paper's title plus a typed (attribute, value) claim facet extracted from its abstract.")
assert SYS_V2 != ee.ENTAIL_SYS, "v2 reconstruction needle not found"

VARIANTS = {
    "v2": ("openai", "gpt-4o-mini", False),
    "v3": ("openai", "gpt-4o-mini", True),
    "v3_4o": ("openai", "gpt-4o", True),
    "v3_haiku45": ("bedrock", "us.anthropic.claude-haiku-4-5-20251001-v1:0", True),
    "v3_sonnet46": ("bedrock", "us.anthropic.claude-sonnet-4-6", True),
    "v3_opus46": ("bedrock", "us.anthropic.claude-opus-4-6-v1", True),
    "v3_5nano": ("openai", "gpt-5-nano", True),
    "v3_5mini": ("openai", "gpt-5-mini", True),
}

_bedrock = None
def bedrock_json(model_id, system, user):
    global _bedrock
    if _bedrock is None:
        import boto3
        _bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
    for a in range(6):
        try:
            r = _bedrock.converse(modelId=model_id, system=[{"text": system}],
                messages=[{"role": "user", "content": [{"text": user + "\n\nReturn ONLY the JSON object."}]}],
                inferenceConfig={"maxTokens": 200})
            u = r.get("usage", {})
            pm.cost["in"] += u.get("inputTokens", 0); pm.cost["out"] += u.get("outputTokens", 0)
            t = r["output"]["message"]["content"][0]["text"]
            return json.loads(t[t.find("{"):t.rfind("}") + 1])
        except Exception:
            time.sleep(2 ** a)
    return {"score": 0, "rationale": "error"}

def judge(client, variant, qf, cf, q_title, c_title, q_abs="", c_abs=""):
    kind, model, use_abs = VARIANTS[variant]
    sys_p = ee.ENTAIL_SYS if use_abs else SYS_V2
    key = "CAL2::" + variant + "::" + hashlib.md5(
        (ee.facet_text(qf) + "||" + ee.facet_text(cf) + "||" + q_title + "||" + c_title).encode()).hexdigest()[:14]
    if key in pm._cache:
        return pm._cache[key]
    u = f"TARGET paper: {q_title[:120]}\n"
    if use_abs and q_abs:
        u += f"TARGET abstract excerpt: {q_abs[:700]}\n"
    u += f"TARGET claim:\n  attribute: {qf.get('attribute','')}\n  value: {qf.get('value','')}\n\n"
    u += f"PRIOR work: {c_title[:120]}\n"
    if use_abs and c_abs:
        u += f"PRIOR abstract excerpt: {c_abs[:700]}\n"
    u += f"PRIOR claim:\n  attribute: {cf.get('attribute','')}\n  value: {cf.get('value','')}"
    if kind == "bedrock":
        j = bedrock_json(model, sys_p, u)
    else:
        j = {"score": 0, "rationale": "error"}
        for a in range(4):
            try:
                if model.startswith("gpt-5"):
                    r = client.chat.completions.create(model=model,
                        response_format={"type": "json_object"},
                        messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": u}],
                        max_completion_tokens=2000, extra_body={"reasoning_effort": "low"})
                else:
                    r = client.chat.completions.create(model=model, temperature=0,
                        response_format={"type": "json_object"},
                        messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": u}],
                        max_tokens=90)
                pm.cost["in"] += r.usage.prompt_tokens; pm.cost["out"] += r.usage.completion_tokens
                j = json.loads(r.choices[0].message.content)
                break
            except Exception:
                time.sleep(2 ** a)
    j["score"] = max(0, min(5, int(j.get("score", 0))))
    with pm._cache_lock:
        pm._cache[key] = j; pm._save_cache()
    return j

def main():
    from openai import OpenAI
    client = OpenAI()
    cases = ce.build_cases(client, 30)
    print(f"built {len(cases)} cases ({sum(1 for c in cases if c['df'])} with distractor)")
    res = {}
    from concurrent.futures import ThreadPoolExecutor
    for v in VARIANTS:
        def _cell(c):
            g = judge(client, v, c["qf"], c["bf"], c["q_title"], c["b_title"], c["q_abs"], c["b_abs"])
            d = judge(client, v, c["qf"], c["df"], c["q_title"], c["d_title"], c["q_abs"], c["d_abs"]) if c["df"] else None
            return g, d
        with ThreadPoolExecutor(max_workers=8) as ex:
            cells = list(ex.map(_cell, cases))
        gold = [g["score"] for g, _ in cells if g["score"]]
        dist = [d["score"] for _, d in cells if d and d["score"]]
        res[v] = {"gold_mean": round(st.mean(gold), 2), "gold_n": len(gold),
                  "gold_ge4": round(sum(1 for x in gold if x >= 4) / len(gold), 2),
                  "distractor_mean": round(st.mean(dist), 2) if dist else None,
                  "separation": round(st.mean(gold) - st.mean(dist), 2) if dist else None}
        print(v, res[v], flush=True)
    out = os.path.join(HERE, "outputs", "a2_extended.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out, "| cost tokens:", pm.cost)

if __name__ == "__main__":
    main()
