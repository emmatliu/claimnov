import argparse
import json
import sys

from novelty import pipeline


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help="JSONL of papers/ideas to score")
    ap.add_argument("--corpus", required=True, help="JSONL of candidate-pool papers")
    ap.add_argument("--out", default="novelty_report.json")
    ap.add_argument("--k", type=int, default=10, help="prior works judged per claim")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="max<->breadth dial for the blended score: "
                         "0 = only the strongest anticipator matters, "
                         "1 = uniform mean over the top-k (default 0.5)")
    ap.add_argument("--gate-threshold", type=float, default=None,
                    help="emit pass/fail: fail when blended novelty < threshold")
    ap.add_argument("--knockout-level", type=int, default=None, choices=[2, 3, 4, 5],
                    help="emit pass/fail: fail when any claim's best single prior "
                         "work entails at >= this level (the clone-catcher; "
                         "independent of the blended score)")
    ap.add_argument("--contrastive-veto", action="store_true",
                    help="run the pilot-calibrated contrastive gate (novelty/gate.py): "
                         "fail when a top-pool paper does what the target proposes. "
                         "Requires a Claude-4.5+ judge (config.GATE_JUDGE_MODEL).")
    args = ap.parse_args()

    targets = read_jsonl(args.target)
    corpus = read_jsonl(args.corpus)
    print(f"{len(targets)} target(s) vs {len(corpus)}-paper pool", file=sys.stderr)

    result = pipeline.score_corpus(targets, corpus, k=args.k, alpha=args.alpha)

    if args.contrastive_veto:
        from novelty import gate
        by_id = {t.get("id") or t.get("title", "?"): t for t in targets}
        for r in result["reports"]:
            t = by_id.get(r["target_id"], {})
            own = [c for c in corpus if (c.get("id") or c.get("title")) != r["target_id"]]
            v = gate.contrastive_veto(t, own)
            r["contrastive_veto"] = v

    if (args.gate_threshold is not None or args.knockout_level is not None
            or args.contrastive_veto):
        for r in result["reports"]:
            if not r.get("search_status", {}).get("resolved", True):
                r["gate"] = "unresolved"
                r["gate_reasons"] = ["no claims or no same-dimension candidates; "
                                     "verdict would not be evidence-based"]
                continue
            reasons = []
            if args.contrastive_veto and r.get("contrastive_veto", {}).get("veto"):
                ev = r["contrastive_veto"]["evidence"]
                reasons.append(f"contrastive veto: '{ev['title'][:70]}' does what "
                               "the target proposes")
            if (args.gate_threshold is not None and r["novelty"] is not None
                    and r["novelty"] < args.gate_threshold):
                reasons.append(f"novelty {r['novelty']} < {args.gate_threshold}")
            if args.knockout_level is not None:
                for c in r["claims"]:
                    lvl = c.get("max_entailment_level")
                    if lvl is not None and lvl >= args.knockout_level:
                        top = c["top_matches"][0] if c["top_matches"] else {}
                        reasons.append(
                            f"claim '{c['claim'][:60]}' entailed at level {lvl} "
                            f"by {top.get('paper_id', '?')}")
            r["gate"] = "fail" if reasons else "pass"
            if reasons:
                r["gate_reasons"] = reasons

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)

    for r in result["reports"]:
        gate = f"  [{r['gate']}]" if "gate" in r else ""
        print(f"N = {r['novelty']}  (max: {r['novelty_max']}, breadth: "
              f"{r['novelty_breadth']}, {r['n_claims']} claims)  "
              f"{r['target_title'][:60]}{gate}")
    print(f"cost: ${result['cost_usd']}  |  wrote {args.out}")


if __name__ == "__main__":
    main()
