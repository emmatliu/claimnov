import json, os, hashlib, threading, time, re
import boto3

EXP = "review_experiments"
OUT = f"{EXP}/out"
CDIR = f"{OUT}/r3_cache"; os.makedirs(CDIR, exist_ok=True)
COSTF = f"{OUT}/r3_costs.jsonl"
MODEL = "us.anthropic.claude-sonnet-4-6"
PIN, POUT = 3.0, 15.0
BUDGET = float(os.environ.get("R3_BUDGET", "60"))
REGIONS = [r for r in os.environ.get("SONNET_REGIONS", "us-west-2,us-east-2,us-east-1,us-west-2@global,us-east-2@global,us-east-1@global").split(",") if r]
def _model_of(ep): return "global.anthropic.claude-sonnet-4-6" if ep.endswith("@global") else MODEL
_lock = threading.Lock()
_state = {"ri": 0, "clients": {}, "cool": {}}
COOL = float(os.environ.get("R3_COOL", "600"))
cost = {"usd": 0.0, "calls": 0, "in": 0, "out": 0, "cache_hits": 0, "by_region": {}}


def prior_spend():
    if not os.path.exists(COSTF): return 0.0
    return sum(json.loads(l)["usd"] for l in open(COSTF))


_PRIOR = prior_spend()


def spent():
    return _PRIOR + cost["usd"]


def _client(region):
    if region not in _state["clients"]:
        _state["clients"][region] = boto3.client("bedrock-runtime", region_name=region)
    return _state["clients"][region]


def cache_path(prompt, max_tokens, tag, system=None):
    msgs = [{"role": "user", "content": prompt}]
    h = hashlib.sha256(json.dumps(["bedrock", MODEL, system, msgs, max_tokens, tag]).encode()).hexdigest()[:24]
    return os.path.join(CDIR, f"c_{h}.json")


class Budget(Exception):
    pass


def sonnet(prompt, max_tokens, tag, system=None):
    fp = cache_path(prompt, max_tokens, tag, system)
    if os.path.exists(fp):
        with _lock: cost["cache_hits"] += 1
        d = json.load(open(fp)); return d["text"], d.get("meta", {})
    if spent() > BUDGET:
        raise Budget(f"budget {BUDGET} reached")
    kw = {"modelId": MODEL, "messages": [{"role": "user", "content": [{"text": prompt}]}],
          "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0.0}}
    if system: kw["system"] = [{"text": system}]
    last = None
    for attempt in range(400):
        with _lock:
            now = time.time()
            alive = [r for r in REGIONS if _state["cool"].get(r, 0) < now]
            if not alive:
                wait = min(_state["cool"].values()) - now
                if wait > float(os.environ.get("R3_MAXWAIT", "1800")): raise RuntimeError("all regions exhausted (daily quota)")
            else: wait = 0
            _state["ri"] += 1; region = alive[_state["ri"] % len(alive)] if alive else None
        if wait > 0:
            time.sleep(min(wait, 60)); continue
        try:
            t0 = time.time()
            r = _client(region.split("@")[0]).converse(**{**kw, "modelId": _model_of(region)})
            lat = time.time() - t0
            break
        except Exception as e:
            last = str(e)
            if "per day" in last or "daily" in last.lower():
                with _lock:
                    if _state["cool"].get(region, 0) < time.time():
                        print(f"[r3] {time.strftime('%H:%M:%S')} region {region} quota: {last[:160]}; cooling {COOL}s", flush=True)
                    _state["cool"][region] = time.time() + COOL
                continue
            time.sleep(min(60, 2 ** min(attempt, 6)))
    else:
        raise RuntimeError(f"bedrock failed: {last[:200]}")
    u = r.get("usage", {})
    usd = u.get("inputTokens", 0) / 1e6 * PIN + u.get("outputTokens", 0) / 1e6 * POUT
    with _lock:
        cost["usd"] += usd; cost["calls"] += 1; cost["in"] += u.get("inputTokens", 0); cost["out"] += u.get("outputTokens", 0)
        br = cost["by_region"].setdefault(region, {"calls": 0, "usd": 0.0}); br["calls"] += 1; br["usd"] += usd
    text = "".join(b.get("text", "") for b in r["output"]["message"]["content"])
    meta = {"region": region, "in": u.get("inputTokens", 0), "out": u.get("outputTokens", 0), "latency_s": round(lat, 2),
            "stop": r.get("stopReason")}
    if not text.strip():
        raise RuntimeError("empty response")
    tmp = f"{fp}.{os.getpid()}.{threading.get_ident()}.tmp"
    json.dump({"text": text, "meta": meta}, open(tmp, "w")); os.replace(tmp, fp)
    return text, meta


def log_cost(script):
    if cost["calls"] == 0: return
    with open(COSTF, "a") as f:
        f.write(json.dumps({"script": script, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "usd": round(cost["usd"], 4),
                            "calls": cost["calls"], "in_tokens": cost["in"], "out_tokens": cost["out"],
                            "by_region": cost["by_region"], "cache_hits": cost["cache_hits"]}) + "\n")


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
