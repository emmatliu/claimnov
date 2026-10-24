import hashlib, json, os, threading, time, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)
_lock = threading.Lock()
_cost = {"usd": 0.0, "calls": 0}

PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}

ENV_FILES = [os.path.join(HERE, "..", ".env"), ".env"]

def _env(name):
    k = os.environ.get(name)
    if k: return k
    for fp in ENV_FILES:
        if not os.path.exists(fp): continue
        for line in open(fp):
            if line.startswith(name + "="):
                return line.strip().split("=", 1)[1]

def _key():
    return _env("OPENAI_API_KEY")

def _post(path, payload, tries=9):
    for i in range(tries):
        try:
            req = urllib.request.Request(f"https://api.openai.com/v1/{path}",
                data=json.dumps(payload).encode(),
                headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except Exception as e:
            if i == tries - 1: raise
            time.sleep(min(75, 2 ** i))

def _atomic_dump(obj, fp):
    tmp = f"{fp}.{os.getpid()}.{threading.get_ident()}.tmp"
    json.dump(obj, open(tmp, "w"))
    os.replace(tmp, fp)

def _track(model, usage):
    pin, pout = PRICES.get(model, (1.0, 5.0))
    with _lock:
        _cost["usd"] += usage.get("prompt_tokens", 0) / 1e6 * pin + usage.get("completion_tokens", 0) / 1e6 * pout
        _cost["calls"] += 1

def cost(): return dict(_cost)

def chat(model, messages, max_tokens=2000, temperature=0.0, cache_tag=""):
    h = hashlib.sha256(json.dumps([model, messages, max_tokens, temperature, cache_tag]).encode()).hexdigest()[:24]
    fp = os.path.join(CACHE, f"c_{h}.json")
    if os.path.exists(fp):
        return json.load(open(fp))["text"]
    payload = {"model": model, "messages": messages}
    if model.startswith("gpt-5"):
        payload["max_completion_tokens"] = max(max_tokens, 8000)
        payload["reasoning_effort"] = "low"
    else:
        payload.update({"max_tokens": max_tokens, "temperature": temperature})
    text = ""
    for attempt in range(3):
        r = _post("chat/completions", payload)
        _track(model, r.get("usage", {}))
        text = r["choices"][0]["message"]["content"] or ""
        if text.strip():
            break
        payload["max_completion_tokens"] = payload.get("max_completion_tokens", max_tokens) + 8000
    if text.strip():
        _atomic_dump({"text": text}, fp)
    return text

def embed(texts, model="text-embedding-3-small"):
    out, todo, idx = [None] * len(texts), [], []
    for i, t in enumerate(texts):
        h = hashlib.sha256((model + "::" + t).encode()).hexdigest()[:24]
        fp = os.path.join(CACHE, f"e_{h}.json")
        if os.path.exists(fp):
            out[i] = json.load(open(fp))
        else:
            todo.append(t); idx.append(i)
    for j in range(0, len(todo), 32):
        batch = todo[j:j+32]
        r = _post("embeddings", {"model": model, "input": [t[:8000] for t in batch]})
        _track(model, r.get("usage", {"prompt_tokens": r.get("usage", {}).get("total_tokens", 0)}))
        for k, d in enumerate(r["data"]):
            i = idx[j + k]
            out[i] = d["embedding"]
            h = hashlib.sha256((model + "::" + texts[i]).encode()).hexdigest()[:24]
            _atomic_dump(d["embedding"], os.path.join(CACHE, f"e_{h}.json"))
    return out

SONNET_46 = "us.anthropic.claude-sonnet-4-6"
OPUS_46 = "us.anthropic.claude-opus-4-6-v1"
PRICES[SONNET_46] = (3.00, 15.00)
PRICES[OPUS_46] = (5.00, 25.00)
PRICES["claude-sonnet-4-6"] = (3.00, 15.00)
PRICES["claude-opus-4-6"] = (5.00, 25.00)

_bedrock = None

def claude(model, messages, max_tokens=2000, system=None, cache_tag=""):
    global _bedrock
    h = hashlib.sha256(json.dumps(["bedrock", model, system, messages, max_tokens, cache_tag]).encode()).hexdigest()[:24]
    fp = os.path.join(CACHE, f"c_{h}.json")
    if os.path.exists(fp):
        return json.load(open(fp))["text"]
    if _bedrock is None:
        import boto3
        _bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
    kw = {"modelId": model,
          "messages": [{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages],
          "inferenceConfig": {"maxTokens": max_tokens, **({"temperature": 0.0} if os.environ.get("BEDROCK_TEMP0") == "1" else {})}}
    if system:
        kw["system"] = [{"text": system}]
    for i in range(6):
        try:
            r = _bedrock.converse(**kw)
            break
        except Exception:
            if i == 5: raise
            time.sleep(min(30, 2 ** i))
    u = r.get("usage", {})
    _track(model, {"prompt_tokens": u.get("inputTokens", 0), "completion_tokens": u.get("outputTokens", 0)})
    text = "".join(b.get("text", "") for b in r["output"]["message"]["content"])
    if text.strip():
        _atomic_dump({"text": text}, fp)
    return text


OPUS_46_API = "claude-opus-4-6"

def claude_api(model, messages, max_tokens=2000, system=None, cache_tag="", temperature=None):
    h = hashlib.sha256(json.dumps(["antapi", model, system, messages, max_tokens, cache_tag]).encode()).hexdigest()[:24]
    fp = os.path.join(CACHE, f"c_{h}.json")
    if os.path.exists(fp):
        return json.load(open(fp))["text"]
    key = _env("ANT_API_KEY")
    msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
    if os.environ.get("ANT_PROMPT_CACHE", "1") == "1" and msgs:
        last = msgs[-1]
        msgs[-1] = {"role": last["role"], "content": [{"type": "text", "text": last["content"],
                                                        "cache_control": {"type": "ephemeral"}}]}
    body = {"model": model, "max_tokens": max_tokens, "messages": msgs}
    if system:
        body["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    if temperature is not None:
        body["temperature"] = temperature
    for a in range(7):
        try:
            req = urllib.request.Request("https://api.anthropic.com/v1/messages",
                data=json.dumps(body).encode(),
                headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.load(r)
            u = d.get("usage", {})
            pin, pout = PRICES.get(model, (1.0, 5.0))
            with _lock:
                _cost["usd"] += (u.get("input_tokens", 0) * pin + u.get("cache_creation_input_tokens", 0) * pin * 1.25
                                 + u.get("cache_read_input_tokens", 0) * pin * 0.1 + u.get("output_tokens", 0) * pout) / 1e6
                _cost["calls"] += 1
                _cost["cache_read"] = _cost.get("cache_read", 0) + u.get("cache_read_input_tokens", 0)
                _cost["uncached_in"] = _cost.get("uncached_in", 0) + u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
            text = "".join(b.get("text", "") for b in d.get("content", []))
            if text.strip():
                _atomic_dump({"text": text}, fp)
            return text
        except urllib.error.HTTPError as e:
            if e.code in (429, 529, 500):
                time.sleep(min(75, 2 ** a + 1)); continue
            raise
        except Exception:
            time.sleep(min(75, 2 ** a))
    return ""
