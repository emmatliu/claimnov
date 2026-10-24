import csv, sys
from collections import Counter

def kappa(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if not pairs: return None, 0
    po = sum(x == y for x, y in pairs) / len(pairs)
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / len(pairs) ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, len(pairs)

f1, f2 = sys.argv[1], sys.argv[2]
r1 = list(csv.DictReader(open(f1))); r2 = list(csv.DictReader(open(f2)))
assert len(r1) == len(r2), "row count mismatch"
qcols = [c for c in r1[0] if c.startswith("q") or c.endswith("_yes_no") or c == "does_match_yes_no"]
qcols = sorted(set(qcols))
disagree_rows = []
for c in qcols:
    a = [row[c].strip().lower() for row in r1]
    b = [row[c].strip().lower() for row in r2]
    k, n = kappa(a, b)
    agree = sum(x == y for x, y in zip(a, b) if x and y)
    print(f"{c:34s} agreement {agree}/{n}  kappa={k:.3f}" if k is not None else f"{c}: no filled pairs")
for i, (x, y) in enumerate(zip(r1, r2)):
    if any(x[c].strip().lower() != y[c].strip().lower() for c in qcols if x[c].strip() and y[c].strip()):
        row = {k: v for k, v in x.items() if not k.startswith("q")}
        for c in qcols:
            row[f"{c}__a1"] = x[c]; row[f"{c}__a2"] = y[c]
        row["adjudicated_answer"] = ""
        disagree_rows.append(row)
if disagree_rows:
    out = f1.replace("annotator1", "ADJUDICATION").replace(".csv", "") + ".csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(disagree_rows[0].keys()))
        w.writeheader(); w.writerows(disagree_rows)
    print(f"\n{len(disagree_rows)} disagreement rows -> {out}")
else:
    print("\nno disagreements")
