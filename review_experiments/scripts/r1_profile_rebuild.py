import hashlib, json, os, re, sys, itertools, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r0_common import PILOT, RES, OUT, NCACHE, PCACHE, jl
sys.path.insert(0, "./src")
from novelty.gate import _PROFILE_PROMPT

SETS = sys.argv[1:] or ["disg", "later", "hyb"]
MAXLEAF = int(os.environ.get("MAXLEAF", "200000"))
MAXLIST = int(os.environ.get("MAXLIST", "4000"))
ncache = json.load(open(NCACHE))
SON_BR, SON_API = "us.anthropic.claude-sonnet-4-6", "claude-sonnet-4-6"


def _key(*p): return hashlib.md5("|".join(str(x) for x in p).encode()).hexdigest()[:16]
def claims_of(t, a): return ncache.get("claims4::" + _key(t, a))
def ctx(f):
    parts = [f"Facet — attribute: {f.get('attribute', '')}; value: {f.get('value', '')}"]
    if f.get("paper_abstract"): parts.append(f"Abstract: {f['paper_abstract'][:1200]}")
    return "\n".join(parts)
def level(t, c):
    return ncache.get("entail::" + hashlib.md5((t["claim"] + "||" + c["claim"] + "||" + ctx(t) + "||" + ctx(c)).encode()).hexdigest()[:16])
def ftext(f): return f"{f['attribute']}: {f['value']}" if f.get("value") else f["attribute"]
def tk(s): return set(re.findall(r"[a-z0-9]+", s.lower()))
def proxy(a, b):
    A, B = tk(ftext(a)), tk(ftext(b)); return len(A & B) / (len(A | B) or 1)


def cache_hit(prompt, tag):
    msgs = [{"role": "user", "content": prompt}]
    for kind, key in (("bedrock", ["bedrock", SON_BR, None, msgs, 1500, tag]), ("api", ["antapi", SON_API, None, msgs, 1500, tag])):
        fp = f"{PCACHE}/c_{hashlib.sha256(json.dumps(key).encode()).hexdigest()[:24]}.json"
        if os.path.exists(fp):
            return kind, json.load(open(fp))["text"]
    return None, None


def claim_lists(t, facets_same):
    byp = collections.defaultdict(list)
    for f in facets_same:
        lv = level(t, f)
        if lv is not None:
            byp[f["paper_id"]].append((f, lv))
    n_d = len({f["paper_id"] for f in facets_same})
    n = min(5, n_d)
    if n == 0:
        yield []; return
    multi = [p for p, v in byp.items() if len(v) > 1]
    for p in byp:
        byp[p].sort(key=lambda fv: -proxy(t, fv[0]))
    choices = [range(len(byp[p])) for p in multi]
    emitted = 0
    for ch in itertools.islice(itertools.product(*choices), 64):
        sel = {p: byp[p][0] for p in byp}
        for p, c in zip(multi, ch):
            sel[p] = byp[p][c]
        papers = sorted(sel, key=lambda p: (-sel[p][1], -proxy(t, sel[p][0])))
        if len(papers) < n:
            continue
        lv = {p: sel[p][1] for p in papers}

        def rec(prefix, used):
            nonlocal emitted
            if emitted >= MAXLIST: return
            if len(prefix) == n:
                emitted += 1
                yield [(p, sel[p][0], lv[p]) for p in prefix]; return
            last = lv[prefix[-1]] if prefix else 99
            for p in papers:
                if p in used or lv[p] > last: continue
                if any(q not in used and lv[q] > lv[p] for q in papers): continue
                yield from rec(prefix + [p], used | {p})
        yield from rec([], frozenset())


def build_prompt(idea, tclaims, lists, by_id):
    lines, seen = [], []
    for i, (c, js) in enumerate(zip(tclaims, lists), 1):
        lines.append(f"Claim {i} [{c.get('dimension', '?')}]: {c.get('claim', '')}")
        if not js:
            lines.append("    (no related existing claim retrieved)")
        for pid, f, lvl in js:
            if pid not in seen: seen.append(pid)
            lines.append(f"    - [{f.get('paper_title', '?')}] {f.get('claim', '')[:300]}  -> coverage {lvl}/5")
    papers = []
    for pid in seen[:8]:
        c = by_id.get(pid)
        if c: papers.append(f"- {c['title']}: {(c.get('abstract') or '')[:700]}")
    return _PROFILE_PROMPT.format(idea=idea, profile="\n".join(lines) or "(no claims extracted)", papers="\n".join(papers) or "(none)"), seen


def rebuild(u, cands=None):
    idea = f"{u['title']}: {u['description']}"
    tcl = claims_of(u["title"], u["description"])
    if tcl is None:
        return {"status": "no_target_claims"}
    facets = []
    for p in u["recs"]:
        for c in (claims_of(p["title"], p["abstract"]) or []):
            facets.append({**c, "paper_id": p["id"], "paper_title": p["title"], "paper_abstract": p["abstract"]})
    by_id = {r["id"]: r for r in u["recs"]}
    if cands is not None:
        return rebuild_constrained(u, idea, tcl, facets, by_id, cands)
    per_claim = []
    for c in tcl:
        t = {**c, "paper_abstract": u["description"]}
        per_claim.append(list(claim_lists(t, [f for f in facets if f["dimension"] == c["dimension"]])))
        if not per_claim[-1]:
            return {"status": "claim_without_lists"}
    leaves = 0
    def dfs(i, chosen, seen):
        nonlocal leaves
        if leaves >= MAXLEAF: return None
        if i == len(tcl):
            if cands is not None and seen != cands: return None
            leaves += 1
            pr, _ = build_prompt(idea, tcl, chosen, by_id)
            kind, txt = cache_hit(pr, u["tag"])
            return (pr, kind, txt) if kind else None
        for L in per_claim[i]:
            s2 = list(seen)
            ok = True
            for pid, _, _ in L:
                if pid not in s2:
                    s2.append(pid)
                    if cands is not None and (len(s2) > len(cands) or cands[len(s2) - 1] != pid):
                        ok = False; break
            if not ok: continue
            r = dfs(i + 1, chosen + [L], s2)
            if r: return r
        return None
    r = dfs(0, [], [])
    if r:
        return {"status": "byte_identical", "backend": r[1], "prompt": r[0], "paper_verdict_text": r[2], "leaves_tried": leaves}
    return {"status": "not_found", "leaves_tried": leaves, "n_lists": [len(x) for x in per_claim]}


def rebuild_constrained(u, idea, tcl, facets, by_id, cands):
    opts = []
    for c in tcl:
        t = {**c, "paper_abstract": u["description"]}
        same = [f for f in facets if f["dimension"] == c["dimension"]]
        byp = collections.defaultdict(list)
        for f in same:
            if f["paper_id"] in cands:
                lv = level(t, f)
                if lv is not None: byp[f["paper_id"]].append((f, lv))
        for p in byp: byp[p].sort(key=lambda fv: -proxy(t, fv[0]))
        opts.append((min(5, len({f["paper_id"] for f in same})), byp))
    leaves = 0
    def claim_gen(i, seen):
        n, byp = opts[i]
        def rec(L, s, last):
            if len(L) == n:
                yield L, s; return
            nxt = [cands[len(s)]] if len(s) < len(cands) else []
            for p in nxt + [q for q in s if q not in {x[0] for x in L}]:
                for f, lv in byp.get(p, []):
                    if lv > last: continue
                    yield from rec(L + [(p, f, lv)], s + [p] if p not in s else s, lv)
        yield from rec([], list(seen), 99)
    def dfs(i, chosen, seen):
        nonlocal leaves
        if leaves >= MAXLEAF: return None
        if i == len(tcl):
            if seen != cands: return None
            leaves += 1
            pr, _ = build_prompt(idea, tcl, chosen, by_id)
            kind, txt = cache_hit(pr, u["tag"])
            return (pr, kind, txt) if kind else None
        for L, s2 in claim_gen(i, seen):
            r = dfs(i + 1, chosen + [L], s2)
            if r: return r
            if leaves >= MAXLEAF: return None
        return None
    r = dfs(0, [], [])
    if r:
        return {"status": "byte_identical", "backend": r[1], "prompt": r[0], "paper_verdict_text": r[2], "leaves_tried": leaves}
    return {"status": "not_found", "leaves_tried": leaves, "n_slots": [o[0] for o in opts]}


def units():
    U = []
    if "disg" in SETS:
        dz = json.load(open(f"{PILOT}/data/disguises_v3.json"))
        cv = {r["item"]: r for r in jl("coupled_veto_smuggle_profile")}
        keys = json.load(open(f"{OUT}/r0_items.json"))["items"]["claimnov"]
        for d in dz:
            if d["seed_idx"] in (60, 83, 86, 87, 93): continue
            pin = json.load(open(f"{PILOT}/data/pools/seed{d['seed_idx']}.json"))["pool"]
            for cond, recs in (("B_in", pin), ("B_out", [p for p in pin if not p.get("is_B")])):
                key = f"seed{d['seed_idx']}_r{d['rung']}_{cond}"
                if key not in keys["disg_in" if cond == "B_in" else "disg_out"]: continue
                tag = f"vsq_vs_sm_{key}" if d["seed_idx"] < 10 else f"vsq_sm2_{key}"
                U.append(dict(set="disg", key=key, title=d["title"], description=d["description"], recs=recs, tag=tag,
                              cands=cv[key]["candidates"] if d["seed_idx"] < 10 and key in cv else None))
    if "later" in SETS:
        CN = json.load(open(f"{OUT}/r0_items.json"))["items"]["claimnov"]["later"]
        for item in sorted(CN):
            slug, idx = item.split("::")
            ad = next(p for p in (f"{PILOT}/data/anchors/{slug}", f"{PILOT}/data/anchors_full/{slug}") if os.path.exists(f"{p}/ideas.json"))
            idea = next(i for i in json.load(open(f"{ad}/ideas.json")) if i["idx"] == int(idx))
            U.append(dict(set="later", key=item, title=idea["title"], description=idea["description"],
                          recs=json.load(open(f"{ad}/pool_preT.json"))[:400], tag=f"vsq_vs_pf_{slug}_{idx}", cands=CN[item]["candidates"]))
    if "hyb" in SETS:
        for h in json.load(open(f"{PILOT}/data/hybrids.json")):
            ad = next(p for p in (f"{PILOT}/data/anchors/{h['anchor']}", f"{PILOT}/data/anchors_full/{h['anchor']}") if os.path.exists(f"{p}/ideas.json"))
            U.append(dict(set="hyb", key=h["item"], title=h["title"], description=h["description"],
                          recs=json.load(open(f"{ad}/pool_preT.json"))[:400], tag=f"vsq_hyb_{h['item']}", cands=None))
    return U


if __name__ == "__main__":
    OUTF = f"{OUT}/r1_profile_prompts.jsonl"
    done = {(json.loads(l)["set"], json.loads(l)["key"]) for l in open(OUTF)} if os.path.exists(OUTF) else set()
    fo = open(OUTF, "a")
    U = [u for u in units() if (u["set"], u["key"]) not in done][:int(os.environ.get("LIMIT", 10**9))]
    cnt = collections.Counter()
    for n, u in enumerate(U):
        r = rebuild(u, u["cands"])
        cnt[(u["set"], r["status"])] += 1
        fo.write(json.dumps({"set": u["set"], "key": u["key"], "tag": u["tag"], "constrained": u["cands"] is not None, **r}) + "\n"); fo.flush()
        if n % 25 == 0: print(n, dict(cnt), flush=True)
    print("DONE", dict(cnt))
