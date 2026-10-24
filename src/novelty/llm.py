import json
import os
import threading
import time

from openai import OpenAI

from . import config

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.environ.get(
    "NOVELTY_CACHE", os.path.join(_HERE, "..", ".cache", "novelty_cache.json")
)
os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)

_client = None
_client_lock = threading.Lock()
_cache = json.load(open(_CACHE_PATH, encoding="utf-8")) if os.path.exists(_CACHE_PATH) else {}


def _get_client() -> OpenAI:
    global _client
    with _client_lock:
        if _client is None:
            _client = OpenAI()
        return _client
_cache_lock = threading.RLock()
_oai_sem = threading.Semaphore(int(os.environ.get("NOVELTY_OAI_CONCURRENCY", "4")))

cost = {"in": 0, "out": 0, "emb": 0}


def _save_cache():
    with _cache_lock:
        tmp = _CACHE_PATH + f".tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False)
        os.replace(tmp, _CACHE_PATH)


def cache_get(key):
    with _cache_lock:
        return _cache.get(key)


_put_count = 0

def cache_put(key, value):
    global _put_count
    with _cache_lock:
        _cache[key] = value
        _put_count += 1
        if _put_count % int(os.environ.get("NOVELTY_CACHE_FLUSH_EVERY", "25")) == 0:
            _save_cache()


import atexit
atexit.register(_save_cache)


_anthropic_client = None


def gate_chat(prompt: str, model: str, max_tokens: int = 200) -> str:
    key = f"gate::{model}::{prompt}"
    hit = cache_get(key)
    if hit is not None:
        return hit
    if model.startswith(("claude", "anthropic", "us.anthropic")):
        global _anthropic_client
        if _anthropic_client is None:
            try:
                import anthropic
            except ImportError as e:
                raise RuntimeError(
                    f"The verdict judge ({model!r}) needs the `anthropic` package: pip install anthropic.") from e
            _anthropic_client = anthropic.Anthropic()
        for attempt in range(4):
            try:
                r = _anthropic_client.messages.create(
                    model=model, max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}])
                text = "".join(b.text for b in r.content if b.type == "text").strip()
                cache_put(key, text)
                return text
            except Exception:
                time.sleep(2 ** attempt)
        return ""
    text = chat_text(prompt, model=model, max_tokens=max_tokens)
    cache_put(key, text)
    return text


def chat_json(system: str, user: str, model: str = config.EXTRACT_MODEL,
              max_tokens: int = 800) -> dict:
    msg = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for attempt in range(4):
        try:
            with _oai_sem:
                if model.startswith("gpt-5"):
                    r = _get_client().chat.completions.create(
                        model=model, max_completion_tokens=max(max_tokens, 2000),
                        response_format={"type": "json_object"}, messages=msg,
                        extra_body={"reasoning_effort": "low"})
                else:
                    r = _get_client().chat.completions.create(
                        model=model, temperature=0, max_tokens=max_tokens,
                        response_format={"type": "json_object"}, messages=msg)
            cost["in"] += r.usage.prompt_tokens
            cost["out"] += r.usage.completion_tokens
            return json.loads(r.choices[0].message.content)
        except Exception:
            time.sleep(2 ** attempt)
    return {}


def chat_text(prompt: str, model: str = config.JUDGE_MODEL, max_tokens: int = 8) -> str:
    for attempt in range(4):
        try:
            with _oai_sem:
                r = _get_client().chat.completions.create(
                    model=model, temperature=0, max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}])
            cost["in"] += r.usage.prompt_tokens
            cost["out"] += r.usage.completion_tokens
            return (r.choices[0].message.content or "").strip()
        except Exception:
            time.sleep(2 ** attempt)
    return ""


def embed(texts, model: str = config.EMBED_MODEL):
    out = []
    for i in range(0, len(texts), 128):
        batch = texts[i:i + 128]
        for attempt in range(4):
            try:
                with _oai_sem:
                    r = _get_client().embeddings.create(model=model, input=batch)
                cost["emb"] += r.usage.prompt_tokens
                out.extend(d.embedding for d in r.data)
                break
            except Exception:
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError("embedding batch failed after retries")
    return out


def cost_usd() -> float:
    return (cost["in"] * 0.15 + cost["out"] * 0.60 + cost["emb"] * 0.13) / 1e6
