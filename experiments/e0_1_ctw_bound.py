"""E0.1: CTW's cumulative excess loss on a tree source against the §7 bound."""

from math import comb, log2

import numpy as np

from symbolic_ml.data.synthetic import tree_source
from symbolic_ml.evaluate import START, prequential, write_jsonl
from symbolic_ml.models.ctw import CTW

N, V, H, RHO = 200_000, 8, 4, 0.5


def run(seed: int) -> dict:
    s = tree_source(N, seed, V=V, max_depth=H)
    bits = prequential(CTW(V, H, alpha=1.0, rho=RHO), s.seq)
    excess = float(bits.sum() - s.true_bits[START:].sum())
    leaves = s.meta["leaves"]
    shallow = sum(len(c) < H for c in leaves)
    prior_bits = -s.meta["internal"] * log2(1 - RHO) - shallow * log2(RHO)
    n_leaf = np.bincount(s.meta["leaf_of"][START:], minlength=len(leaves))
    param_bits = sum(log2(comb(int(n) + V - 1, V - 1)) for n in n_leaf)
    return {
        "seed": seed,
        "leaves": len(leaves),
        "excess_bits": excess,
        "bound_bits": prior_bits + param_bits,
        "ctw_bpt": float(bits.mean()),
        "true_bpt": float(s.true_bits[START:].mean()),
    }


if __name__ == "__main__":
    rows = [run(seed) for seed in range(5)]
    write_jsonl("e0_1", rows)
    for r in rows:
        ok = "ok" if r["excess_bits"] <= r["bound_bits"] else "VIOLATED"
        print(
            f"seed {r['seed']}: leaves={r['leaves']:4d} excess={r['excess_bits']:9.1f} "
            f"bound={r['bound_bits']:9.1f} {ok}  ctw={r['ctw_bpt']:.4f} true={r['true_bpt']:.4f} bits/token"
        )
