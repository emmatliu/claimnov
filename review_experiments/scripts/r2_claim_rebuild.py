import sys, os, json, re, ast, collections
sys.argv = [sys.argv[0], "none"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_profile_rebuild as RB
from r0_common import PILOT, OUT, cached_bedrock, SONNET_46_BEDROCK

I = json.load(open(f"{OUT}/r0_items.json"))["items"]["claimnov"]["claim"]
src = open(f"{PILOT}/make_disguises.py").read()
R1 = next(ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "R1")
pairs = [json.loads(l) for l in open("eval/data/claim_disputed/testset_clean.jsonl")]
OUTF = f"{OUT}/r2_claim_profile_prompts.jsonl"
done = {json.loads(l)["key"] for l in open(OUTF)} if os.path.exists(OUTF) else set()
fo = open(OUTF, "a"); cnt = collections.Counter()
for k in sorted(I):
    if k in done: continue
    i = int(k.split("_")[1])
    d = json.loads(re.search(r"\{.*\}", cached_bedrock(SONNET_46_BEDROCK, [{"role": "user", "content": R1.format(abstract=pairs[i]["A_abstract"])}], 3000, f"n2f_ideaify{i}"), re.S).group(0))
    recs = json.load(open(f"{PILOT}/data/pools_full/{k}.json"))["pool"]
    u = dict(set="claim", key=k, title=d["title"], description=d["description"], recs=recs, tag=f"vsq_sonnet_nf{i}")
    r = RB.rebuild(u, I[k]["candidates"])
    cnt[r["status"]] += 1
    fo.write(json.dumps({"set": "claim", "key": k, "tag": u["tag"], **r}) + "\n"); fo.flush()
print("DONE", dict(cnt))
