"""Interpolated n-gram: each order backs off to the next shorter context.

q_0(y) = 1/V;  q_d(y | c_d) = (count_d(y) + lam * q_{d-1}(y)) / (n_d + lam).
"""

from __future__ import annotations

from math import log2

from ..ops import Ops


class NGram:
    def __init__(self, V: int, depth: int, lam: float = 1.0, ops: Ops | None = None):
        self.V, self.depth, self.lam = V, depth, lam
        self.ops = ops or Ops()
        self.child: dict[tuple[int, int], int] = {}
        self.count: dict[tuple[int, int], int] = {}
        self.n = [0]

    def step(self, seq: list[int], t: int, update: bool = True) -> float:
        y, lam = seq[t], self.lam
        q = 1.0 / self.V
        node = 0
        for d in range(self.depth + 1):
            if d > 0:
                key = (node, seq[t - d])
                nxt = self.child.get(key, -1)
                if nxt < 0:
                    if not update:
                        break
                    nxt = len(self.n)
                    self.child[key] = nxt
                    self.n.append(0)
                node = nxt
            c = self.count.get((node, y), 0)
            q = (c + lam * q) / (self.n[node] + lam)
            if update:
                self.count[(node, y)] = c + 1
                self.n[node] += 1
        self.ops.n += self.depth + 1
        return -log2(q)

    @property
    def size(self) -> int:
        return len(self.n)
