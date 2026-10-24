import math
from . import llm

class Pool:
    def __init__(self, records):
        self.records = records
        texts = [f"{r['title']}. {r['abstract']}" for r in records]
        self.vecs = llm.embed(texts)

    def search(self, query, k=10):
        if not self.records: return []
        qv = llm.embed([query])[0]
        def cos(a, b):
            num = sum(x*y for x, y in zip(a, b))
            return num / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b)) + 1e-9)
        scored = sorted(((cos(qv, v), r) for v, r in zip(self.vecs, self.records)),
                        key=lambda x: -x[0])
        return [dict(r, _sim=round(s, 4)) for s, r in scored[:k]]

    def without(self, drop_ids):
        keep = [i for i, r in enumerate(self.records) if r["id"] not in drop_ids]
        p = Pool.__new__(Pool)
        p.records = [self.records[i] for i in keep]
        p.vecs = [self.vecs[i] for i in keep]
        return p
