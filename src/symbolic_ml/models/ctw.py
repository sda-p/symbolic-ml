"""Context-tree mixture over a V-ary alphabet (docs/initial-research.md §7).

Each suffix node s holds token counts and two log-quantities:
  K_s  the Dirichlet(alpha) marginal likelihood of the tokens seen at s,
  W_s  K_s at depth H, else rho*K_s + (1-rho)*prod_children W.
The predictive probability of the next token is W_root(after)/W_root(before).
Only the H+1 nodes on the current suffix path change per token.
"""

from __future__ import annotations

from math import exp, log, log1p

from ..ops import Ops

LN2 = log(2.0)


def _logaddexp(a: float, b: float) -> float:
    if a < b:
        a, b = b, a
    return a + log1p(exp(b - a))


class CTW:
    def __init__(self, V: int, depth: int, alpha: float = 1.0, rho: float = 0.5, ops: Ops | None = None):
        self.V, self.depth, self.alpha = V, depth, alpha
        self.log_rho, self.log_1mrho = log(rho), log(1 - rho)
        self.ops = ops or Ops()
        self.child: dict[tuple[int, int], int] = {}
        self.count: dict[tuple[int, int], int] = {}
        # Per node: total count, log K, log W, sum of children's log W.
        self.n = [0]
        self.logK = [0.0]
        self.logW = [0.0]
        self.logP = [0.0]

    def step(self, seq: list[int], t: int, update: bool = True) -> float:
        """Loss in bits for seq[t] given seq[:t]; requires t >= depth."""
        H, y = self.depth, seq[t]
        child = self.child
        path = [0]
        node = 0
        for d in range(1, H + 1):
            key = (node, seq[t - d])
            nxt = child.get(key, -1)
            if nxt < 0:
                if not update:
                    break
                nxt = len(self.n)
                child[key] = nxt
                self.n.append(0)
                self.logK.append(0.0)
                self.logW.append(0.0)
                self.logP.append(0.0)
            node = nxt
            path.append(node)
        self.ops.n += H + 1

        a, Va = self.alpha, self.V * self.alpha
        # Nodes missing from the path (only when not updating) are fresh:
        # they predict uniformly at every depth, so their W changes by 1/V.
        delta = -log(self.V) if len(path) <= H else 0.0
        for d in range(len(path) - 1, -1, -1):
            s = path[d]
            c = self.count.get((s, y), 0)
            newK = self.logK[s] + log((c + a) / (self.n[s] + Va))
            if d == H:
                newW = newK
                newP = 0.0
            else:
                newP = self.logP[s] + delta
                newW = _logaddexp(self.log_rho + newK, self.log_1mrho + newP)
            delta = newW - self.logW[s]
            if update:
                self.count[(s, y)] = c + 1
                self.n[s] += 1
                self.logK[s], self.logW[s], self.logP[s] = newK, newW, newP
        return -delta / LN2

    @property
    def size(self) -> int:
        return len(self.n)
