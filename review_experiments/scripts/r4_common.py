import json, os, hashlib, threading, time, re
import boto3

EXP = "review_experiments"
OUT = f"{EXP}/out"
CDIR = f"{OUT}/r4_cache"; os.makedirs(CDIR, exist_ok=True)
CALLS = f"{OUT}/r4_calls.jsonl"
BUDGET = float(os.environ.get("R4_BUDGET", "45"))
EXPNAME = os.environ.get("R4_EXP", "r4")
EXPCAP = float(os.environ.get("R4_EXPCAP", "1e9"))

MODELS = {
    "sonnet": ("anthropic.claude-sonnet-4-6", 3.0, 15.0),
    "haiku": ("anthropic.claude-haiku-4-5-20251001-v1:0", 1.0, 5.0),
    "opus": ("anthropic.claude-opus-4-6-v1", 5.0, 25.0),
    "gptoss": ("openai.gpt-oss-120b-1:0", 0.15, 0.60),
    "maverick": ("us.meta.llama4-maverick-17b-instruct-v1:0", 0.24, 0.97),
}
DEFAULT_EPS = {
    "sonnet": "us-west-2,us-east-2,us-east-1,us-west-2@global,us-east-2@global,us-east-1@global",
    "haiku": "us-east-2,us-west-2,us-east-1,us-east-2@global,us-west-2@global",
    "opus": "us-west-2@global,us-east-2@global,us-east-1@global,us-west-2,us-east-2",
    "gptoss": "us-west-2",
    "maverick": "us-west-2,us-east-1,us-east-2",
}


def model_id(key, ep):
    base = MODELS[key][0]
    if key in ("gptoss", "maverick"): return base
    return ("global." if ep.endswith("@global") else "us.") + base


_lock = threading.Lock()
_state = {"ri": 0, "clients": {}, "cool": {}}
COOL = float(os.environ.get("R4_COOL", "900"))
cost = {"usd": 0.0, "calls": 0, "cache_hits": 0, "by": {}}


def _prior():
    if not os.path.exists(CALLS): return 0.0, 0.0
    tot = expt = 0.0
    for l in open(CALLS):
        r = json.loads(l); tot += r["usd_guard"]
        if r["exp"] == EXPNAME: expt += r["usd_guard"]
    return tot, expt


_PRIOR, _PRIOR_EXP = _prior()


def spent(): return _PRIOR + cost["usd"]


def _client(region):
    if region not in _state["clients"]:
        from botocore.config import Config
        _state["clients"][region] = boto3.client("bedrock-runtime", region_name=region,
                                                 config=Config(read_timeout=300, retries={"max_attempts": 2}))
    return _state["clients"][region]


def cache_path(key, prompt, max_tokens, tag, system=None):
    h = hashlib.sha256(json.dumps(["r4", key, system, prompt, max_tokens, tag]).encode()).hexdigest()[:24]
    return os.path.join(CDIR, f"c_{h}.json")


class Budget(Exception):
    pass


def call(key, prompt, max_tokens, tag, system=None, messages=None):
    pk = messages if messages is not None else prompt
    fp = cache_path(key, pk, max_tokens, tag, system)
    if os.path.exists(fp):
        with _lock: cost["cache_hits"] += 1
        d = json.load(open(fp)); return d["text"], {**d.get("meta", {}), "cached": True}
    if spent() > BUDGET: raise Budget(f"global budget {BUDGET} reached (spent {spent():.2f})")
    if _PRIOR_EXP + cost["by"].get("_exp", 0.0) > EXPCAP: raise Budget(f"experiment cap {EXPCAP} reached")
    msgs = messages if messages is not None else [{"role": "user", "content": prompt}]
    kw = {"messages": [{"role": m["role"], "content": [{"text": m["content"]}]} for m in msgs],
          "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0.0}}
    if system: kw["system"] = [{"text": system}]
    eps = [e for e in os.environ.get(f"EPS_{key.upper()}", DEFAULT_EPS[key]).split(",") if e]
    last = None
    for attempt in range(300):
        with _lock:
            now = time.time()
            alive = [r for r in eps if _state["cool"].get(r, 0) < now]
            if not alive:
                wait = min(_state["cool"].values()) - now
                if wait > float(os.environ.get("R4_MAXWAIT", "120")): raise RuntimeError(f"all endpoints exhausted for {key}: {str(last)[:200]}")
            else: wait = 0
            _state["ri"] += 1; ep = alive[_state["ri"] % len(alive)] if alive else None
        if wait > 0:
            time.sleep(min(wait, 30)); continue
        try:
            t0 = time.time()
            r = _client(ep.split("@")[0]).converse(modelId=model_id(key, ep), **kw)
            lat = time.time() - t0; break
        except Exception as e:
            last = str(e)
            if "per day" in last or "daily" in last.lower() or "AccessDenied" in last or "not authorized" in last:
                with _lock:
                    if _state["cool"].get(ep, 0) < time.time():
                        print(f"[r4] {time.strftime('%H:%M:%S')} {key}@{ep} unavailable: {last[:160]}; cooling {COOL}s", flush=True)
                    _state["cool"][ep] = time.time() + COOL
                continue
            if "ValidationException" in last and "max" in last.lower():
                raise
            time.sleep(min(30, 2 ** min(attempt, 5)))
    else:
        raise RuntimeError(f"bedrock failed: {str(last)[:200]}")
    u = r.get("usage", {}); pin, pout = MODELS[key][1:]
    usd = u.get("inputTokens", 0) / 1e6 * pin + u.get("outputTokens", 0) / 1e6 * pout
    guard = usd * (1.1 if key in ("sonnet", "haiku", "opus") and not ep.endswith("@global") else 1.0)
    text = "".join(b.get("text", "") for b in r["output"]["message"]["content"] if "text" in b)
    meta = {"model": model_id(key, ep), "endpoint": ep, "in": u.get("inputTokens", 0), "out": u.get("outputTokens", 0),
            "latency_s": round(lat, 2), "stop": r.get("stopReason"), "usd": round(usd, 6)}
    with _lock:
        cost["usd"] += guard; cost["calls"] += 1
        cost["by"]["_exp"] = cost["by"].get("_exp", 0.0) + guard
        b = cost["by"].setdefault(f"{key}@{ep}", {"calls": 0, "usd": 0.0}); b["calls"] += 1; b["usd"] += usd
        with open(CALLS, "a") as f:
            f.write(json.dumps({"exp": EXPNAME, "key": key, "tag": tag[:80], "time": time.strftime("%H:%M:%S"), **meta,
                                "usd_guard": round(guard, 6)}) + "\n")
    if not text.strip():
        raise RuntimeError("empty response")
    tmp = f"{fp}.{os.getpid()}.{threading.get_ident()}.tmp"
    json.dump({"text": text, "meta": meta}, open(tmp, "w")); os.replace(tmp, fp)
    return text, meta


def parse_verdict(txt, corpus):
    try:
        j = json.loads(re.search(r"\{.*\}", txt, re.S).group(0)); pe = False
    except Exception:
        j = {"decision": "novel"}; pe = True
    veto = str(j.get("decision", "")).lower().startswith("not")
    t = str(j.get("anticipating_paper") or "")
    c = next((x for x in corpus if x["title"].lower()[:60] == t.lower()[:60]), None) if t else None
    return {"rej": veto, "anticipating_id": c["id"] if c else None, "anticipating_title": t or None, "parse_error": pe,
            "established_claims": j.get("established_claims"), "why": j.get("why")}


def summary():
    return {"session_usd_guard": round(cost["usd"], 4), "calls": cost["calls"], "cache_hits": cost["cache_hits"],
            "by": {k: v for k, v in cost["by"].items() if k != "_exp"}, "round_total_usd_guard": round(spent(), 4)}
