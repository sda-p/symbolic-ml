"""E1.1/E1.2: does the predicate-tree learner recover planted structure on S2-S5?"""

import sys
from multiprocessing import Pool

from symbolic_ml.data.synthetic import SOURCES
from symbolic_ml.evaluate import START, summarize, write_jsonl
from symbolic_ml.models.ptree import Config, PredicateTree, prequential_batches

N = 200_000
NAMES = ["S2-agreement", "S3-brackets", "S4-counter", "S5-xor"]


def run(job) -> dict:
    src, seed = job
    s = SOURCES[src](N, seed)
    model = PredicateTree(s.V, Config())
    bits = prequential_batches(model, s.seq)
    excess = bits - s.true_bits[START:]
    curve = [float(excess[i : i + 20_000].mean()) for i in range(0, len(excess), 20_000)]
    return {
        "source": src, "seed": seed, **summarize(bits),
        "tail_excess": summarize(excess)["tail_bits"], "curve": curve,
        "leaves": model.leaves, "classes": len(model.space.masks) - s.V,
        "ops_per_update": model.ops.n / model.updates,
        "budget_violations": int(model.budget_violations),
        "splits": model.describe(s.names, limit=12),
    }


if __name__ == "__main__":
    seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
    with Pool(8) as pool:
        rows = pool.map(run, [(src, seed) for src in NAMES for seed in seeds])
    write_jsonl("e1_1", rows)
    for r in rows:
        print(f"\n{r['source']} seed {r['seed']}: tail excess {r['tail_excess']:.4f} bits, "
              f"{r['leaves']} leaves, {r['classes']} learned classes, "
              f"{r['ops_per_update']:.0f} ops/update, {r['budget_violations']} budget violations")
        print("  excess per 20k tokens: " + " ".join(f"{c:.3f}" for c in r["curve"]))
        if r["seed"] == 0:
            for line in r["splits"]:
                print("   " + line)
