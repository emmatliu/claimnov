import json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import llm
CALL = llm.claude_api if os.environ.get("SORTER_API") else llm.claude
OPUS = llm.OPUS_46_API if os.environ.get("SORTER_API") else llm.OPUS_46
from run_positive import MATCH_PROMPT, SOUND_PROMPT, parse_json, cos

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = f"{HERE}/data/anchors_full"
WORKERS = 3

slugs = sorted(d for d in os.listdir(ROOT) if os.path.isdir(f"{ROOT}/{d}") and d != "cut")
SHADOW = set(slugs[:2])

def sort_anchor(slug, model, model_tag, out_name, call=None):
    call = call or CALL
    adir = f"{ROOT}/{slug}"
    fp = f"{adir}/{out_name}"
    if os.path.exists(fp):
        return None
    ideas = json.load(open(f"{adir}/ideas.json"))
    pool = json.load(open(f"{adir}/pool_preT.json"))[:400]
    F = json.load(open(f"{adir}/F.json"))
    pool_vecs = llm.embed([f"{r['title']}. {r['abstract']}" for r in pool])
    F_vecs = llm.embed([f"{r['title']}. {r['abstract']}" for r in F])

    def sort_one(idea):
        iv = llm.embed([f"{idea['title']}. {idea['description']}"])[0]
        itxt = f"{idea['title']}: {idea['description']}"

        def best_match(records, vecs, kind):
            scored = sorted(((cos(iv, v), r) for v, r in zip(vecs, records)),
                            key=lambda x: -x[0])[:8]
            for s, r in scored:
                ans = call(model, [{"role": "user", "content": MATCH_PROMPT.format(
                    idea=itxt, ptitle=r["title"], pabstract=r["abstract"])}],
                    max_tokens=2000,
                    cache_tag=f"m{model_tag}_{slug}_{idea['idx']}_{kind}_{r['id'][:8]}")
                if ans.strip().lower().startswith("yes"):
                    return r["title"], round(s, 3)
            return None, None

        a_match, _ = best_match(pool, pool_vecs, "a")
        b_match, _ = (None, None) if a_match else best_match(F, F_vecs, "b")
        bucket = "a" if a_match else ("b" if b_match else "c")
        sound = None
        if bucket == "c":
            sound = parse_json(call(model, [{"role": "user", "content":
                SOUND_PROMPT.format(idea=itxt)}], max_tokens=1500,
                cache_tag=f"snd{model_tag}_{slug}_{idea['idx']}"))["score"]
        return {"idx": idea["idx"], "bucket": bucket, "match": a_match or b_match,
                "sound": sound}

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        srt = list(ex.map(sort_one, ideas))
    json.dump(srt, open(fp, "w"))
    from collections import Counter
    return Counter(x["bucket"] for x in srt)

if __name__ == "__main__":
  for slug in slugs:
    try:
        c = sort_anchor(slug, OPUS, "O", "sorted_full.json")
    except Exception as e:
        print(f"{slug}: FAILED this pass ({str(e)[:90]}) - continuing", flush=True)
        continue
    if c is not None:
        print(f"{slug}: {dict(c)}", flush=True)
  for slug in sorted(SHADOW):
    c = sort_anchor(slug, llm.SONNET_46, "S", "sorted_shadow_sonnet.json", call=llm.claude)
    if c is not None:
        print(f"shadow {slug}: {dict(c)}", flush=True)

  print("\n=== (b)-density readout ===")
  tot = {"a": 0, "b": 0, "c": 0}
  for slug in slugs:
    fp = f"{ROOT}/{slug}/sorted_full.json"
    if os.path.exists(fp):
        from collections import Counter
        c = Counter(x["bucket"] for x in json.load(open(fp)))
        for k in tot:
            tot[k] += c.get(k, 0)
        print(f"  {slug}: a={c.get('a',0)} b={c.get('b',0)} c={c.get('c',0)}")
  print("TOTAL:", tot, "| cost:", llm.cost())
