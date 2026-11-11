"""DSL v0: finite-valued features of a bounded context, evaluated columnar.

A context row is ctx[j] = the token j steps back (j = 0 is the most recent),
for j < H. Terms are plain tuples.

Position terms (value in 0..H, where H means "no such position"):
  ("off", j)          the constant offset j
  ("all",)            H itself: used as "the whole window" bound for counts
  ("first", S)        nearest offset whose token is in class S
  ("next", S, p)      nearest offset beyond position p whose token is in S
  ("shift", p, k)     p + k

Features (finite-valued, K possible values):
  ("at", p)           token at position p; K = V + 1 (V means "none")
  ("dist", p)         the position itself; K = H + 1
  ("cnt", S, p, m)    number of class-S tokens nearer than p, mod m; K = m

A class S is an index into the space's library of token sets. The library
starts with all singletons and grows as the learner commits multi-token classes.
"""

from __future__ import annotations

from math import log2

import numpy as np

from ..ops import Ops

MAX_FEATURES = 20_000


class FeatureSpace:
    def __init__(self, V: int, H: int, suffix_only: bool = False):
        self.V, self.H, self.suffix_only = V, H, suffix_only
        self.masks: list[np.ndarray] = []
        self._index: dict[bytes, int] = {}
        for v in range(V):
            m = np.zeros(V + 1, dtype=bool)
            m[v] = True
            self.add_class(m, rebuild=False)
        self.version = 0
        self.features: list[tuple[float, tuple]] = []
        self._rebuild()

    # -- classes -----------------------------------------------------------
    def add_class(self, mask: np.ndarray, rebuild: bool = True) -> int:
        key = mask.tobytes()
        if key in self._index:
            return self._index[key]
        self._index[key] = len(self.masks)
        self.masks.append(mask.copy())
        if rebuild:
            self._rebuild()
        return len(self.masks) - 1

    def class_cost(self, S: int) -> float:
        learned = len(self.masks) - self.V
        return log2(self.V) if S < self.V else log2(1 + learned)

    # -- enumeration -------------------------------------------------------
    def K(self, term: tuple) -> int:
        kind = term[0]
        return self.V + 1 if kind == "at" else self.H + 1 if kind == "dist" else term[3]

    def _rebuild(self) -> None:
        H, feats = self.H, []
        offs = [(log2(H), ("off", j)) for j in range(H)]
        if self.suffix_only:
            feats = [(c, ("at", p)) for c, p in offs]
        else:
            classes = range(len(self.masks))
            firsts = [(self.class_cost(S), ("first", S)) for S in classes]
            pos = list(offs)
            for c, p in firsts:
                if p[1] >= self.V:  # at(first(singleton)) is constant
                    pos.append((c, p))
                pos.append((c + 1, ("shift", p, -1)))
                pos.append((c + 1, ("shift", p, 1)))
                pos += [(c + self.class_cost(S), ("next", S, p)) for S in classes]
            feats = [(c, ("at", p)) for c, p in pos]
            feats += [(c + 1, ("dist", p)) for c, p in firsts]
            bounds = [(1.0, ("all",))] + firsts
            for S in classes:
                for c, p in bounds:
                    if p != ("first", S):
                        for m in (2, 3):
                            feats.append((self.class_cost(S) + c + 1, ("cnt", S, p, m)))
        feats.sort(key=lambda f: f[0])
        self.features = feats[:MAX_FEATURES]
        self.version += 1


class Batch:
    """Memoised columnar evaluation of terms over a [b, H] context matrix."""

    def __init__(self, ctx: np.ndarray, space: FeatureSpace, ops: Ops):
        self.ctx, self.space, self.ops = ctx, space, ops
        self.b, self.H = ctx.shape
        self.rows = np.arange(self.b)
        self.offsets = np.arange(self.H)
        self._memo: dict[tuple, np.ndarray] = {}

    def _in(self, S: int) -> np.ndarray:
        key = ("in", S)
        if key not in self._memo:
            self._memo[key] = self.space.masks[S][self.ctx]
            self.ops.n += self.b * self.H
        return self._memo[key]

    def _nearest(self, hit: np.ndarray) -> np.ndarray:
        self.ops.n += self.b * self.H
        return np.where(hit.any(axis=1), hit.argmax(axis=1), self.H)

    def val(self, term: tuple) -> np.ndarray:
        if term in self._memo:
            return self._memo[term]
        kind, H = term[0], self.H
        if kind == "off":
            out = np.full(self.b, term[1])
        elif kind == "all":
            out = np.full(self.b, H)
        elif kind == "first":
            out = self._nearest(self._in(term[1]))
        elif kind == "next":
            p = self.val(term[2])
            out = self._nearest(self._in(term[1]) & (self.offsets[None, :] > p[:, None]))
        elif kind == "shift":
            p = self.val(term[1])
            q = p + term[2]
            out = np.where((p == H) | (q < 0) | (q >= H), H, q)
        elif kind == "at":
            p = self.val(term[1])
            out = np.where(p == H, self.space.V, self.ctx[self.rows, np.minimum(p, H - 1)])
        elif kind == "dist":
            out = self.val(term[1])
        elif kind == "cnt":
            p = self.val(term[2])
            before = self._in(term[1]) & (self.offsets[None, :] < p[:, None])
            out = before.sum(axis=1) % term[3]
            self.ops.n += self.b * H
        else:
            raise ValueError(term)
        self.ops.n += self.b
        self._memo[term] = out
        return out


def show(term: tuple, space: FeatureSpace, names: list[str]) -> str:
    """Human-readable rendering of a term."""
    kind = term[0]

    def cls(S: int) -> str:
        members = [names[v] for v in np.flatnonzero(space.masks[S][: space.V])]
        return repr(members[0]) if len(members) == 1 else "{" + ",".join(map(repr, members)) + "}"

    if kind == "off":
        return f"-{term[1] + 1}"
    if kind == "all":
        return "window"
    if kind == "first":
        return f"last({cls(term[1])})"
    if kind == "next":
        return f"last({cls(term[1])} before {show(term[2], space, names)})"
    if kind == "shift":
        return f"{show(term[1], space, names)}{'+1' if term[2] < 0 else '-1'}"
    if kind == "at":
        return f"tok[{show(term[1], space, names)}]"
    if kind == "dist":
        return f"dist[{show(term[1], space, names)}]"
    return f"count({cls(term[1])} since {show(term[2], space, names)}) mod {term[3]}"
