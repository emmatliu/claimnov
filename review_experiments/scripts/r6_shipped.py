import os, sys, json, shutil, sqlite3, hashlib, threading, time
ROOT = "."
EXP = f"{ROOT}/review_experiments"
OUT = f"{EXP}/out"; CD = f"{OUT}/r6_cache"; os.makedirs(CD, exist_ok=True)
NC = f"{CD}/novelty_cache_r6.json"
if not os.path.exists(NC):
    shutil.copyfile(f"{ROOT}/src/.cache/novelty_cache.json", NC)
os.environ["NOVELTY_CACHE"] = NC
os.environ.setdefault("NOVELTY_OAI_CONCURRENCY", "8")
os.environ.setdefault("NOVELTY_CACHE_FLUSH_EVERY", "200")
for line in open(f"{ROOT}/.env"):
    if line.startswith("OPENAI_API_KEY="):
        os.environ.setdefault("OPENAI_API_KEY", line.strip().split("=", 1)[1])
sys.path.insert(0, f"{ROOT}/src")
from novelty import config, llm, extract, entail, retrieve, pipeline, gate
assert os.path.abspath(llm._CACHE_PATH) == os.path.abspath(NC), llm._CACHE_PATH

OAI_LOG = f"{OUT}/r6_oai_calls.jsonl"
OAI_BUDGET = float(os.environ.get("R6_OAI_BUDGET", "10"))
PRICE = {"gpt-4o-mini": (0.15, 0.60), "text-embedding-3-large": (0.13, 0.0)}
EXPNAME = os.environ.get("R6_EXP", "r6")
_olock = threading.Lock()
def _prior():
    if not os.path.exists(OAI_LOG): return 0.0
    return sum(json.loads(l)["usd"] for l in open(OAI_LOG))
OAI = {"usd": _prior(), "session_usd": 0.0, "calls": 0, "emb_hits": 0}


class OAIBudget(Exception):
    pass


def _log(model, kind, pin, pout, lat):
    p = PRICE.get(model, (5.0, 15.0))
    usd = pin / 1e6 * p[0] + pout / 1e6 * p[1]
    with _olock:
        OAI["usd"] += usd; OAI["session_usd"] += usd; OAI["calls"] += 1
        with open(OAI_LOG, "a") as f:
            f.write(json.dumps({"exp": EXPNAME, "kind": kind, "model": model, "in": pin, "out": pout, "usd": round(usd, 7),
                                "latency_s": round(lat, 2), "time": time.strftime("%H:%M:%S")}) + "\n")


def _guard():
    if OAI["usd"] > OAI_BUDGET: raise OAIBudget(f"OpenAI budget {OAI_BUDGET} reached ({OAI['usd']:.3f})")


class _Chat:
    def __init__(s, c): s.c = c
    def create(s, **kw):
        _guard(); t0 = time.time(); r = s.c.chat.completions.create(**kw)
        _log(kw["model"], "chat", r.usage.prompt_tokens, r.usage.completion_tokens, time.time() - t0); return r
class _Emb:
    def __init__(s, c): s.c = c
    def create(s, **kw):
        _guard(); t0 = time.time(); r = s.c.embeddings.create(**kw)
        _log(kw["model"], "embed", r.usage.prompt_tokens, 0, time.time() - t0); return r
class _Proxy:
    def __init__(s, c):
        s.chat = type("C", (), {})(); s.chat.completions = _Chat(c); s.embeddings = _Emb(c)
from openai import OpenAI
llm._client = _Proxy(OpenAI())


def chat_text(prompt, model=config.JUDGE_MODEL, max_tokens=8):
    last = None
    for attempt in range(10):
        try:
            with llm._oai_sem:
                r = llm._get_client().chat.completions.create(model=model, temperature=0, max_tokens=max_tokens,
                                                              messages=[{"role": "user", "content": prompt}])
            llm.cost["in"] += r.usage.prompt_tokens; llm.cost["out"] += r.usage.completion_tokens
            return (r.choices[0].message.content or "").strip()
        except OAIBudget: raise
        except Exception as e:
            last = e
            if "insufficient_quota" in str(e) or "invalid_api_key" in str(e): raise
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"chat_text failed: {str(last)[:200]}")


def chat_json(system, user, model=config.EXTRACT_MODEL, max_tokens=800):
    msg = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last = None
    for attempt in range(10):
        try:
            with llm._oai_sem:
                r = llm._get_client().chat.completions.create(model=model, temperature=0, max_tokens=max_tokens,
                                                              response_format={"type": "json_object"}, messages=msg)
            llm.cost["in"] += r.usage.prompt_tokens; llm.cost["out"] += r.usage.completion_tokens
            return json.loads(r.choices[0].message.content)
        except OAIBudget: raise
        except Exception as e:
            last = e
            if "insufficient_quota" in str(e) or "invalid_api_key" in str(e): raise
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"chat_json failed: {str(last)[:200]}")


llm.chat_text = chat_text; llm.chat_json = chat_json

_db = sqlite3.connect(f"{CD}/emb.sqlite", check_same_thread=False)
_db.execute("create table if not exists e (k text primary key, v blob)"); _db.commit()
_dlock = threading.Lock()
_shipped_embed = llm.embed
import numpy as np
def _ek(model, t): return hashlib.sha256(f"{model}::{t}".encode()).hexdigest()
def embed(texts, model=config.EMBED_MODEL):
    keys = [_ek(model, t) for t in texts]
    got = {}
    with _dlock:
        for i in range(0, len(keys), 500):
            ch = keys[i:i + 500]
            for k, v in _db.execute(f"select k, v from e where k in ({','.join('?' * len(ch))})", ch):
                got[k] = np.frombuffer(v, dtype=np.float64).tolist()
    miss = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in got))
    OAI["emb_hits"] += len(texts) - len(miss)
    if miss:
        vecs = _shipped_embed(miss, model=model)
        with _dlock:
            _db.executemany("insert or replace into e values (?, ?)",
                            [(_ek(model, t), np.asarray(v, dtype=np.float64).tobytes()) for t, v in zip(miss, vecs)])
            _db.commit()
        for t, v in zip(miss, vecs): got[_ek(model, t)] = list(v)
    return [got[k] for k in keys]
llm.embed = embed
retrieve.llm.embed = embed


import fcntl
def _save_merge():
    with llm._cache_lock, open(NC + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            disk = json.load(open(NC, encoding="utf-8"))
        except Exception:
            disk = {}
        disk.update(llm._cache); llm._cache.update(disk)
        tmp = NC + f".tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f: json.dump(llm._cache, f, ensure_ascii=False)
        os.replace(tmp, NC)
        fcntl.flock(lk, fcntl.LOCK_UN)
import atexit as _ax
_ax.unregister(llm._save_cache)
llm._save_cache = _save_merge
_ax.register(_save_merge)


def flush():
    _save_merge()


def summary():
    return {"oai_round_usd": round(OAI["usd"], 4), "oai_session_usd": round(OAI["session_usd"], 4),
            "oai_calls": OAI["calls"], "emb_cache_hits": OAI["emb_hits"]}
