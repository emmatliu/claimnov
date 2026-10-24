import common as C
from novelty import pipeline, retrieve, gate

_fac_cache = {}


def facets(recs, key):
    if key not in _fac_cache:
        _fac_cache[key] = retrieve.build_pool([{"id": r["id"], "title": r["title"], "abstract": r["abstract"]} for r in recs])
    return _fac_cache[key]


def report(u):
    target = {"id": u["key"], "title": u["idea"]["title"], "abstract": u["idea"]["description"]}
    pk = (u["set"], u["key"]) if u["set"] in ("disg", "claim") else ("anchor", u["meta"]["anchor"])
    fac = facets(u["recs"], pk)
    return target, pipeline.score_paper(target, fac)


def profile_prompt(u):
    target, rep = report(u)
    box = {}

    def cap(pr):
        box["p"] = pr
        return '{"decision": "novel"}'
    gate.profile_veto(target, u["recs"], rep, chat=cap)
    return target, rep, box.get("p")


def old_call_cached(u, prompt):
    import os
    a = C.ant_cache_path("claude-sonnet-4-6", prompt, 1500, u["old_tag"])
    b = C.bedrock_cache_path(C.llm.SONNET_46, prompt, 1500, u["old_tag"])
    return "api" if os.path.exists(a) else ("bedrock" if os.path.exists(b) else None)
