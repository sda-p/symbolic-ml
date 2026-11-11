"""E0.2: reference numbers for CTW and n-grams on character-level text."""

from multiprocessing import Pool

from symbolic_ml.data.text import load_chars
from symbolic_ml.evaluate import prequential, summarize, write_jsonl
from symbolic_ml.models.ctw import CTW
from symbolic_ml.models.ngram import NGram

DEPTHS = [2, 3, 4, 5, 6, 8]
CONFIGS = [("ctw", d, a) for d in DEPTHS for a in (1.0, 0.25, 0.05, 0.01)] + [
    ("ngram", d, lam) for d in DEPTHS for lam in (0.5, 2.0, 8.0, 32.0)
]


def run(cfg) -> dict:
    kind, depth, param = cfg
    s = load_chars()
    model = CTW(s.V, depth, alpha=param) if kind == "ctw" else NGram(s.V, depth, lam=param)
    out = summarize(prequential(model, s.seq))
    return {"model": kind, "depth": depth, "param": param, "nodes": model.size, **out}


if __name__ == "__main__":
    load_chars()
    with Pool(8) as pool:
        rows = pool.map(run, CONFIGS)
    write_jsonl("e0_2", rows)
    print("model depth param    bits  tail_bits    nodes")
    for r in sorted(rows, key=lambda r: (r["model"], r["bits"])):
        print(f"{r['model']:5s} {r['depth']:5d} {r['param']:5.2f} {r['bits']:7.4f} {r['tail_bits']:9.4f} {r['nodes']:9d}")
