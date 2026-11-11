"""E0.3: are S2-S5 hard for suffix models in the intended way?"""

from multiprocessing import Pool

from symbolic_ml.data.synthetic import SOURCES
from symbolic_ml.evaluate import START, prequential, summarize, write_jsonl
from symbolic_ml.models.ctw import CTW
from symbolic_ml.models.ngram import NGram

N = 200_000
DEPTHS = [2, 4, 8, 12, 16]
CONFIGS = [
    (src, kind, d, p, seed)
    for src in SOURCES
    if src != "S1-tree"
    for kind, params in (("ctw", (1.0, 0.1)), ("ngram", (1.0,)))
    for d in DEPTHS
    for p in params
    for seed in (0, 1)
]


def run(cfg) -> dict:
    src, kind, depth, param, seed = cfg
    s = SOURCES[src](N, seed)
    model = CTW(s.V, depth, alpha=param) if kind == "ctw" else NGram(s.V, depth, lam=param)
    out = summarize(prequential(model, s.seq))
    true = summarize(s.true_bits[START:])
    return {
        "source": src, "model": kind, "depth": depth, "param": param, "seed": seed,
        "nodes": model.size, **out,
        "true_bits": true["bits"], "true_tail_bits": true["tail_bits"],
        "tail_excess": out["tail_bits"] - true["tail_bits"],
    }


if __name__ == "__main__":
    with Pool(8) as pool:
        rows = pool.map(run, CONFIGS)
    write_jsonl("e0_3", rows)
    print("source        model param depth  tail_excess (mean of 2 seeds)   entropy    nodes")
    keys = sorted({(r["source"], r["model"], r["param"], r["depth"]) for r in rows})
    for k in keys:
        rs = [r for r in rows if (r["source"], r["model"], r["param"], r["depth"]) == k]
        ex = sum(r["tail_excess"] for r in rs) / len(rs)
        print(f"{k[0]:13s} {k[1]:5s} {k[2]:5.2f} {k[3]:5d}  {ex:11.4f} {rs[0]['true_tail_bits']:28.4f} {rs[0]['nodes']:8d}")
