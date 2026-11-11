"""E1.4: text loss against evidence thresholds, batch size and op budget."""

import time
from multiprocessing import Pool

from symbolic_ml.data.text import load_chars
from symbolic_ml.evaluate import summarize, write_jsonl
from symbolic_ml.models.ptree import Config, PredicateTree, prequential_batches

BASE = {"lam": 32.0, "n_min": 256, "threshold": 8.0, "batch": 256, "budget": 400_000, "suffix_only": True}
GRID = [{**BASE, "n_min": n, "suffix_only": so} for n in (32, 128, 512) for so in (True, False)] + [
    {**BASE, "n_min": 32, "budget": w, "suffix_only": so}
    for w in (25_000, 100_000, 1_600_000) for so in (True, False)
] + [{**BASE, "n_min": 32, "batch": 64, "suffix_only": so} for so in (True, False)]


def run(kw) -> dict:
    s = load_chars()
    model = PredicateTree(s.V, Config(**kw))
    t = time.time()
    bits = prequential_batches(model, s.seq)
    return {**kw, **summarize(bits), "leaves": model.leaves,
            "ops_per_update": model.ops.n / model.updates, "ops_per_token": model.ops.n / len(bits),
            "violations": int(model.budget_violations), "skipped": model.skipped, "seconds": time.time() - t}


if __name__ == "__main__":
    load_chars()
    with Pool(8) as pool:
        rows = pool.map(run, GRID)
    write_jsonl("e1_4", rows)
    print("suffix n_min thresh batch   budget    bits    tail leaves ops/token viol skipped  sec")
    for r in sorted(rows, key=lambda r: r["bits"]):
        print(f"{r['suffix_only']!s:6} {r['n_min']:5d} {r['threshold']:6.0f} {r['batch']:5d} {r['budget']:8d} "
              f"{r['bits']:7.4f} {r['tail_bits']:7.4f} {r['leaves']:6d} {r['ops_per_token']:9.0f} "
              f"{r['violations']:4d} {r['skipped']:7d} {r['seconds']:4.0f}")
