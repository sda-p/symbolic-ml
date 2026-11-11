"""E1.3: predicate tree on character-level text, with the suffix-only ablation."""

import time
from multiprocessing import Pool

from symbolic_ml.data.text import load_chars
from symbolic_ml.evaluate import summarize, write_jsonl
from symbolic_ml.models.ptree import Config, PredicateTree, prequential_batches

CONFIGS = [
    {"lam": lam, "suffix_only": so, "threshold": th}
    for lam in (8.0, 32.0)
    for so in (False, True)
    for th in (8.0,)
]


def run(kw) -> dict:
    s = load_chars()
    model = PredicateTree(s.V, Config(**kw))
    t = time.time()
    bits = prequential_batches(model, s.seq)
    kinds: dict[str, int] = {}
    for c in model.commits:
        term = c["term"]
        kind = term[0] if term[0] != "at" else ("at-offset" if term[1][0] == "off" else "at-found")
        kinds[kind] = kinds.get(kind, 0) + 1
    return {
        **kw, **summarize(bits), "leaves": model.leaves, "classes": len(model.space.masks) - s.V,
        "ops_per_update": model.ops.n / model.updates, "budget_violations": int(model.budget_violations),
        "seconds": time.time() - t, "split_kinds": kinds,
        "splits": model.describe([repr(c)[1:-1] for c in s.names], limit=25),
    }


if __name__ == "__main__":
    load_chars()
    with Pool(len(CONFIGS)) as pool:
        rows = pool.map(run, CONFIGS)
    write_jsonl("e1_3", rows)
    for r in rows:
        print(f"lam={r['lam']:4.0f} suffix_only={r['suffix_only']!s:5} bits={r['bits']:.4f} tail={r['tail_bits']:.4f} "
              f"leaves={r['leaves']} classes={r['classes']} ops/update={r['ops_per_update']:.0f} "
              f"violations={r['budget_violations']} {r['seconds']:.0f}s kinds={r['split_kinds']}")
    print("\n".join(rows[0]["splits"]))
