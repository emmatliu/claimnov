import json, os, sys, csv, hashlib, re
ROOT = "."
EXP = f"{ROOT}/review_experiments"
OUTD = f"{EXP}/out"
for line in open(f"{ROOT}/.env"):
    if "=" in line and not line.startswith("#"):
        k, v = line.strip().split("=", 1); os.environ.setdefault(k, v)
os.environ.setdefault("ANT_PROMPT_CACHE", "0")
sys.path.insert(0, f"{ROOT}/eval"); sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, f"{EXP}/scripts")
os.chdir(f"{ROOT}/eval")
from harness import llm

P = f"{ROOT}/eval"
_anchor_pools = {}


def anchor_dir(slug):
    return next(p for p in (f"{P}/data/anchors/{slug}", f"{P}/data/anchors_full/{slug}") if os.path.exists(f"{p}/ideas.json"))


def anchor_pool(slug):
    if slug not in _anchor_pools:
        _anchor_pools[slug] = json.load(open(f"{anchor_dir(slug)}/pool_preT.json"))[:400]
    return _anchor_pools[slug]


def disguise_units():
    allz = json.load(open(f"{P}/data/disguises_v3.json"))
    ok = {}
    for d in allz:
        if d["seed_idx"] >= 10 and (d["rung"] <= 2 or d.get("verified_same")):
            ok.setdefault(d["seed_idx"], set()).add(d["rung"])
    chosen = set(range(10)) | set(sorted(s for s, rr in ok.items() if rr == {1, 2, 3, 4})[:90])
    dz = [d for d in allz if d["seed_idx"] in chosen]
    units = []
    for d in dz:
        pin = json.load(open(f"{P}/data/pools/seed{d['seed_idx']}.json"))["pool"]
        pout = [p for p in pin if not p.get("is_B")]
        B = next(p for p in pin if p.get("is_B"))
        for cond, recs in (("B_in", pin), ("B_out", pout)):
            key = f"seed{d['seed_idx']}_r{d['rung']}_{cond}"
            old = f"vsq_vs_sm_{key}" if d["seed_idx"] < 10 else f"vsq_sm2_{key}"
            units.append(dict(key=key, set="disg", cond=cond, idea={"title": d["title"], "description": d["description"]},
                              recs=recs, B_id=B["id"], B_title=B["title"], old_tag=old,
                              meta={"rung": d["rung"], "seed": d["seed_idx"]}))
    return units


def later_units():
    S = sorted(json.load(open(f"{P}/data/later_realized_set.json")))
    units, ideas_cache = [], {}
    for item in S:
        slug, idx = item.split("::"); idx = int(idx)
        if slug not in ideas_cache:
            ideas_cache[slug] = {i["idx"]: i for i in json.load(open(f"{anchor_dir(slug)}/ideas.json"))}
        idea = ideas_cache[slug][idx]
        units.append(dict(key=item, set="later", cond="set", idea={"title": idea["title"], "description": idea["description"]},
                          recs=anchor_pool(slug), B_id=None, B_title=None, old_tag=f"vsq_vs_pf_{slug}_{idx}",
                          meta={"anchor": slug, "idx": idx}))
    return units


def iclr_decisions():
    csv.field_size_limit(10**9)
    return {r["id"]: r["decision_raw"] for r in csv.DictReader(open(
        "claim_disputed_mining/iclr_novelty_validation_data/papers.csv"))}


def claim_units():
    from make_disguises import R1, parse
    pairs = [json.loads(l) for l in open("eval/data/claim_disputed/testset_clean.jsonl")]
    idx = json.load(open(f"{P}/data/n2_full_index.json"))
    items = sorted(int(k) for k, v in idx.items() if v["status"] == "ok")
    dec = iclr_decisions()
    units = []
    for i in items:
        r = pairs[i]
        txt = llm.claude(llm.SONNET_46, [{"role": "user", "content": R1.format(abstract=r["A_abstract"])}], max_tokens=3000, cache_tag=f"n2f_ideaify{i}")
        d = parse(txt)
        recs = json.load(open(f"{P}/data/pools_full/n2f_{i}.json"))["pool"]
        B = next((p for p in recs if p.get("is_B")), None)
        dr = dec.get(r["A_id"])
        units.append(dict(key=f"n2f_{i}", set="claim", cond="claim", idea={"title": d["title"], "description": d["description"]},
                          recs=recs, B_id=B["id"] if B else None, B_title=B["title"] if B else None,
                          old_tag=f"vsq_sonnet_nf{i}",
                          meta={"decision": None if dr is None else ("accept" if dr.startswith("Accept") else "reject"),
                                "dimension": r["novelty_dimension"]}))
    return units


def hybrid_units():
    H = json.load(open(f"{P}/data/hybrids.json"))
    return [dict(key=h["item"], set="hyb", cond="hyb", idea={"title": h["title"], "description": h["description"]},
                 recs=anchor_pool(h["anchor"]), B_id=h["B_id"], B_title=h["B_title"], old_tag=f"vsq_hyb_{h['item']}",
                 meta={"anchor": h["anchor"], "orig_idx": h["orig_idx"]}) for h in H]


def ant_cache_path(model, prompt, max_tokens, tag, system=None):
    msgs = [{"role": "user", "content": prompt}]
    h = hashlib.sha256(json.dumps(["antapi", model, system, msgs, max_tokens, tag]).encode()).hexdigest()[:24]
    return os.path.join(llm.CACHE, f"c_{h}.json")


def bedrock_cache_path(model, prompt, max_tokens, tag, system=None):
    msgs = [{"role": "user", "content": prompt}]
    h = hashlib.sha256(json.dumps(["bedrock", model, system, msgs, max_tokens, tag]).encode()).hexdigest()[:24]
    return os.path.join(llm.CACHE, f"c_{h}.json")


norm = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())[:60]


def title_match(a, b):
    return bool(a) and bool(b) and norm(a)[:50] == norm(b)[:50]


def load_jsonl(fp):
    return [json.loads(l) for l in open(fp)] if os.path.exists(fp) else []


BACKEND = os.environ.get("SONNET_BACKEND", "api")
if BACKEND == "bedrock":
    os.environ["BEDROCK_TEMP0"] = "1"
    import boto3
    REGION = os.environ.get("SONNET_REGION", "us-east-1")
    llm._bedrock = boto3.client("bedrock-runtime", region_name=REGION)
    BACKEND = f"bedrock-{REGION}"


def sonnet(messages, max_tokens, cache_tag, system=None):
    if BACKEND.startswith("bedrock"):
        t = llm.claude(llm.SONNET_46, messages, max_tokens=max_tokens, system=system, cache_tag=cache_tag)
    else:
        t = llm.claude_api("claude-sonnet-4-6", messages, max_tokens=max_tokens, system=system, cache_tag=cache_tag, temperature=0.0)
    if not (t or "").strip():
        raise RuntimeError("empty Sonnet response")
    return t
