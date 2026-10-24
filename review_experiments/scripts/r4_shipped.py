import sys, types, json, hashlib, re, os
ROOT = "."
sys.path.insert(0, f"{ROOT}/src")
_stub = types.ModuleType("novelty.llm")
def _no(*a, **k): raise RuntimeError("novelty.llm stub: no live calls")
for n in ("cache_get", "cache_put", "chat_json", "chat_text", "gate_chat", "embed"): setattr(_stub, n, _no)
sys.modules["novelty.llm"] = _stub
import novelty
novelty.llm = _stub
from novelty import config, entail, extract, gate
import r4_common as R

NCACHE = json.load(open(f"{ROOT}/src/.cache/novelty_cache.json"))
PILOT = f"{ROOT}/eval"; PCACHE = f"{PILOT}/cache"
def _key(*p): return hashlib.md5("|".join(str(x) for x in p).encode()).hexdigest()[:16]
def claims_of(t, a): return NCACHE.get("claims4::" + _key(t, a))
def ekey(t, c):
    return hashlib.md5((t["claim"] + "||" + c["claim"] + "||" + entail._context(t) + "||" + entail._context(c)).encode()).hexdigest()[:16]
def orig_level(t, c): return NCACHE.get("entail::" + ekey(t, c))
E11 = json.load(open(f"{R.OUT}/r2_e11_entail_cache.json")) if os.path.exists(f"{R.OUT}/r2_e11_entail_cache.json") else {}
HAIKU_E11 = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def anchor_dir(slug):
    return next(p for p in (f"{PILOT}/data/anchors/{slug}", f"{PILOT}/data/anchors_full/{slug}") if os.path.exists(f"{p}/ideas.json"))


def haiku_extract(title, abstract, tag="r4ext"):
    user = f"Paper title: {title}\n\nAbstract:\n{abstract[:2500]}"
    txt, meta = R.call("haiku", user, 800, f"{tag}_{_key(title, abstract)}", system=extract._EXTRACT_SYSTEM)
    try:
        out = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
    except Exception:
        try: out = json.JSONDecoder().raw_decode(txt[txt.index("{"):])[0]
        except Exception: out = {}
    claims = []
    for c in out.get("claims", []):
        dim = config.norm_dimension(c.get("dimension"))
        if dim and c.get("claim") and c.get("attribute"):
            claims.append({"claim": c["claim"].strip(), "dimension": dim, "attribute": c["attribute"].strip(),
                           "value": (c.get("value") or "").strip(), "abstract_span": (c.get("abstract_span") or "").strip()})
    return claims, meta


def haiku_level(t, c, tag="r4ent"):
    k = HAIKU_E11 + "::" + ekey(t, c)
    if k in E11: return E11[k], {"cached": True, "src": "r2_e11"}
    txt, meta = R.call("haiku", entail._prompt(t, c), 16, f"{tag}_{ekey(t, c)}")
    d = [ch for ch in txt if ch.isdigit()]
    lv = int(d[0]) if d else config.ENTAIL_MIN
    return max(1, min(5, lv)), meta


def build_profile_prompt(idea, claims_lists, by_title):
    lines, seen = [], []
    for i, (c, js) in enumerate(claims_lists, 1):
        lines.append(f"Claim {i} [{c.get('dimension', '?')}]: {c.get('claim', '')}")
        if not js: lines.append("    (no related existing claim retrieved)")
        for t, cl, lv in js:
            if t not in seen: seen.append(t)
            lines.append(f"    - [{t}] {cl[:300]}  -> coverage {lv}/5")
    papers = []
    for t in seen[:8]:
        c = by_title.get(t)
        if c: papers.append(f"- {c['title']}: {(c.get('abstract') or '')[:700]}")
    return gate._PROFILE_PROMPT.format(idea=idea, profile="\n".join(lines) or "(no claims extracted)", papers="\n".join(papers) or "(none)"), seen


def parse_profile_prompt(prompt, titles):
    idea = prompt.split("IDEA:\n", 1)[1].split("\n\nNOVELTY PROFILE:\n", 1)[0]
    body = prompt.split("NOVELTY PROFILE:\n", 1)[1].split("\n\nPAPERS IN THE PROFILE:\n", 1)[0]
    out = []
    tl = sorted(titles, key=len, reverse=True)
    for line in body.split("\n"):
        m = re.match(r"^Claim (\d+) \[([a-z?]+)\]: (.*)$", line)
        if m:
            out.append([m.group(2), m.group(3), []]); continue
        if line.startswith("    - ["):
            rest = line[len("    - ["):]
            lv = int(re.search(r"  -> coverage (\d)/5$", rest).group(1))
            rest = rest[: rest.rfind("  -> coverage ")]
            t = next((x for x in tl if rest.startswith(x + "] ")), None)
            if t is None: raise ValueError("title not resolved: " + rest[:80])
            out[-1][2].append((t, rest[len(t) + 2:], lv))
        elif line.strip() in ("(no related existing claim retrieved)", "(no claims extracted)"):
            pass
        elif line.strip():
            out[-1][1] += "\n" + line
    return idea, out
