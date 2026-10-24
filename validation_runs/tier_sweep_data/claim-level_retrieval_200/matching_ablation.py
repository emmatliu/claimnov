import json, os, collections
import claim_retrieval as pm
os.makedirs(os.path.join(pm.HERE, "results"), exist_ok=True)
from openai import OpenAI

client = OpenAI()
DIMS, K = pm.DIMS, [1, 3, 5, 10]
VARIANTS = ["attr", "attr_val", "value"]
MODELS = ["text-embedding-3-small", "text-embedding-3-large"]

pilot, allf, Bok = [], [], {}


def _prepare():
    global pilot, allf, Bok
    rows = [json.loads(l) for l in open(pm.TESTSET, encoding="utf-8")]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["novelty_dimension"]].append(r)
    pilot = [r for d in DIMS for r in by[d][:5]]
    Af, Bf = {}, {}
    for r in pilot:
        Af[r["A_id"]] = pm.extract_facets("A::" + r["A_id"], r["A_title"], r.get("A_abstract", ""))
        b = pm.fetch_B_abstract(r); Bok[r["A_id"]] = bool(b)
        Bf[r["A_id"]] = pm.extract_facets("Bf::" + r["A_id"], b["title"], b["abstract"]) if b else []
    allf = []
    for r in pilot:
        for f in Af[r["A_id"]]:
            allf.append({**f, "_owner": r["A_id"], "_side": "A"})
        for f in Bf[r["A_id"]]:
            allf.append({**f, "_owner": r["A_id"], "_side": "B"})


def text_of(f, variant):
    a, v = (f.get("attribute") or ""), (f.get("value") or "")
    return {"attr": a, "value": v}.get(variant, f"{a}: {v}") or " "


def embed_texts(texts, model):
    out = []
    for i in range(0, len(texts), 256):
        r = client.embeddings.create(model=model, input=[t[:512] for t in texts[i:i + 256]])
        out += [d.embedding for d in r.data]
    return out


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / ((sum(x * x for x in a) ** .5) * (sum(y * y for y in b) ** .5) + 1e-9)


def run(model, variant):
    fe = embed_texts([text_of(f, variant) for f in allf], model)
    se = embed_texts([r["reviewer_snippet"][:400] for r in pilot], model)
    hit = {m: collections.defaultdict(int) for m in ["strict", "relaxed"]}
    usable = {m: 0 for m in ["strict", "relaxed"]}
    for j, r in enumerate(pilot):
        aid, dim = r["A_id"], r["novelty_dimension"]
        Bidx = [i for i, f in enumerate(allf) if f["_side"] == "B" and f["_owner"] == aid]
        if not Bok[aid] or not Bidx:
            continue
        Adim = [i for i, f in enumerate(allf) if f["_side"] == "A" and f["_owner"] == aid and f.get("dimension") == dim]
        q = fe[max(Adim, key=lambda i: cos(se[j], fe[i]))] if Adim else se[j]
        Bdim = [i for i in Bidx if allf[i].get("dimension") == dim]
        for m in ["strict", "relaxed"]:
            if m == "strict":
                if not Bdim:
                    continue
                pool = [i for i, f in enumerate(allf) if f.get("dimension") == dim and f["_owner"] != aid] + Bdim
            else:
                pool = [i for i, f in enumerate(allf) if f["_owner"] != aid] + Bidx
            order = sorted(pool, key=lambda i: -cos(q, fe[i]))
            ranks = [p for p, i in enumerate(order) if allf[i]["_side"] == "B" and allf[i]["_owner"] == aid]
            best = min(ranks) if ranks else 10 ** 9
            usable[m] += 1
            for k in K:
                if best < k:
                    hit[m][k] += 1
    return {m: {"usable": usable[m], **{f"@{k}": round(hit[m][k] / usable[m], 2) if usable[m] else None for k in K}}
            for m in ["strict", "relaxed"]}


def main():
    _prepare()
    print(f"pilot {len(pilot)} pairs, {len(allf)} facets\n")
    print(f"{'model':22s} {'variant':9s} {'mode':8s} {'n':>3s} " + " ".join(f'@{k:<3d}' for k in K))
    print("-" * 70)
    out = {}
    for model in MODELS:
        for variant in VARIANTS:
            res = run(model, variant)
            out[f"{model}|{variant}"] = res
            for m in ["strict", "relaxed"]:
                v = res[m]
                print(f"{model.replace('text-embedding-',''):22s} {variant:9s} {m:8s} "
                      f"{v['usable']:>3d} " + " ".join(f"{v[f'@{k}']:<4}" for k in K))
        print()
    json.dump(out, open(os.path.join(pm.HERE, "results",
             "matching_ablation.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
