import json, os, collections
import common as C
R = f"{C.P}/results"


def _jl(f):
    return C.load_jsonl(f"{R}/{f}")


def _first(rows, keyf):
    d = {}
    for r in rows:
        if r.get("novel") is None:
            continue
        d.setdefault(keyf(r), r)
    return d


def claimnov():
    out = {}
    disg = {(u["key"]): u for u in C.disguise_units()}
    for r in _jl("coupled_veto_smuggle_profile.jsonl"):
        u = disg.get(r["item"])
        if u:
            out[("disg", r["item"])] = {"novel": r["novel"], "named": (not r["novel"]) and C.title_match(r["evidence"], u["B_title"]), "evidence": r["evidence"]}
    for f in ("smuggle_v2.jsonl", "smuggle_v2_api.jsonl"):
        for r in _jl(f):
            if r["gate"] == "claimnov" and r.get("novel") is not None and r["item"] in disg:
                out.setdefault(("disg", r["item"]), {"novel": r["novel"], "named": (not r["novel"]) and bool(r.get("hit_B")), "evidence": r.get("evidence")})
    S = set(json.load(open(f"{C.P}/data/later_realized_set.json")))
    for f in ("coupled_veto_fa_profile_k.jsonl", "coupled_veto_ac_profile_k.jsonl", "coupled_veto_set_profile_k.jsonl"):
        for r in _jl(f):
            if r["item"] in S:
                out.setdefault(("later", r["item"]), {"novel": r["novel"], "named": False, "evidence": r["evidence"]})
    for r in _jl("coupled_veto_neg_profile.jsonl"):
        out[("claim", r["item"])] = {"novel": r["novel"], "named": r.get("B_rank") is not None, "evidence": r["evidence"]}
    for r in _jl("hybrids_eval.jsonl"):
        if r["gate"] == "claimnov" and r.get("novel") is not None:
            out.setdefault(("hyb", r["item"]), {"novel": r["novel"], "named": bool(r.get("named_B")), "evidence": r.get("evidence"),
                                                "verdict_names_B": r.get("verdict_names_B")})
    return out


DEPLOYED_FILES = ["deployed_sonnet.jsonl", "deployed_sonnet_rev.jsonl", "deployed_sonnet_bedrock.jsonl",
                  "deployed_sonnet_panel.jsonl", "deployed_sonnet_panel_rev.jsonl", "deployed_sonnet_panel_bedrock.jsonl"]


def baselines():
    disg = {u["key"]: u for u in C.disguise_units()}
    S = set(json.load(open(f"{C.P}/data/later_realized_set.json")))
    g = collections.defaultdict(dict)
    rows = []
    for f in DEPLOYED_FILES:
        rows += _jl(f)
    for r in rows:
        if r.get("novel") is None:
            continue
        if r["item"] in disg:
            k = ("disg", r["item"]); u = disg[r["item"]]
            ev = r.get("evidence"); evs = ev if isinstance(ev, list) else [ev]
            named = (not r["novel"]) and any(C.title_match(e, u["B_title"]) for e in evs if e)
        elif r["item"] in S:
            k = ("later", r["item"]); named = False
        else:
            continue
        g[r["gate"]].setdefault(k, {"novel": r["novel"], "named": named, "evidence": r.get("evidence")})
    for f in ("smuggle_full.jsonl", "smuggle_v2.jsonl", "smuggle_v2_api.jsonl"):
        for r in _jl(f):
            if r["gate"] == "arc" and r.get("novel") is not None and r["item"] in disg:
                ev = r.get("evidence"); evs = ev if isinstance(ev, list) else [ev]
                g["arc"].setdefault(("disg", r["item"]), {"novel": r["novel"], "named": (not r["novel"]) and any(C.title_match(e, disg[r["item"]]["B_title"]) for e in evs if e)})
    for r in _jl("claim_sonnet.jsonl"):
        if r.get("novel") is None:
            continue
        ev = r.get("evidence"); evs = ev if isinstance(ev, list) else [ev]
        g[r["gate"]].setdefault(("claim", r["item"]), {"novel": r["novel"], "named": (not r["novel"]) and any(C.title_match(e, r.get("B_title")) for e in evs if e)})
    for r in _jl("hybrids_eval.jsonl"):
        if r["gate"] != "claimnov" and r.get("novel") is not None:
            g[r["gate"]].setdefault(("hyb", r["item"]), {"novel": r["novel"], "named": bool(r.get("named_B"))})
    return dict(g)


def whole(d):
    def rej(cond):
        ks = [k for k in d if k[0] == "disg" and k[1].endswith(cond)]
        return sum(not d[k]["novel"] for k in ks), len(ks)
    ci, ni = rej("B_in"); co, no = rej("B_out")
    named = sum(d[k]["named"] for k in d if k[0] == "disg" and k[1].endswith("B_in") and not d[k]["novel"])
    lk = [k for k in d if k[0] == "later"]
    fa = sum(not d[k]["novel"] for k in lk)
    res = {"caught": ci, "n_in": ni, "named": named, "src_out": co, "n_out": no, "fa": fa, "n_later": len(lk)}
    if ni and lk:
        res["FA%"] = round(100 * fa / len(lk), 2); res["J"] = round(ci / ni - fa / len(lk), 3)
    if ni and no:
        res["Delta"] = round(ci / ni - co / no, 3)
    return res
