"""Predicate-tree language model with bounded microbatch updates (Stage 1).

Model. A binary tree. An internal node holds a DSL feature and a set of its
values ("class"); examples whose feature value is in the class go to child 0,
the rest to child 1. Every node keeps token counts, and the prediction backs
off along the root-to-leaf path:

    q_root_parent(y) = 1/V
    q_node(y) = (count_node(y) + lam * q_parent(y)) / (n_node + lam)

Learning. Each update sees one microbatch and:
  1. predicts it with the pre-update model and adds its counts along paths;
  2. advances a bounded pool of candidates. A candidate is a (leaf, feature)
     pair with a [K, V] table of (feature value, next token) counts gathered
     since the candidate was created;
  3. scores candidates leaf by leaf in rotation, evicting poor ones and
     committing splits that clear the threshold, then refills the pool.
Steps 2 and 3 share a hard op budget per update: work that does not fit is
skipped (step 2) or resumed on the next update (step 3).

A split's score is the gain in Dirichlet marginal likelihood (concentration
lam, base measure = the leaf's current prediction) from modelling the two
sides of the split separately, in bits, minus the description cost of the
feature and class.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, log2

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import gammaln

from ..dsl.terms import Batch, FeatureSpace, show
from ..ops import Ops

LN2 = log(2.0)


@dataclass(eq=False)
class Candidate:
    leaf: int
    term: tuple
    cost: float
    table: np.ndarray
    n: int = 0
    score: float = 0.0  # best split's gain minus description cost, bits
    gain: float = 0.0
    judged_n: int = -1  # value of n when last scored
    mask: np.ndarray | None = None  # best class found when last scored


@dataclass
class Config:
    H: int = 16
    lam: float = 8.0
    batch: int = 256
    budget: int = 400_000  # ops per update allowed for candidate work
    pool: int = 8192  # memory cap on candidates; the op budget is the intended limit
    per_leaf: int = 32  # max candidates per leaf
    advance_share: float = 0.6  # fraction of the budget for updating candidate tables
    n_min: int = 256  # examples before a candidate is judged
    threshold: float = 8.0  # bits of net gain needed to commit
    rule: str = "ml"  # "ml" (marginal likelihood) or "ig" (raw information gain)
    inherit: int = 8  # features handed from a split leaf to its children
    suffix_only: bool = False
    warm_start: bool = True  # children start with the candidate table's counts


class PredicateTree:
    def __init__(self, V: int, cfg: Config | None = None, ops: Ops | None = None):
        self.V, self.cfg = V, cfg or Config()
        self.ops = ops or Ops()
        self.space = FeatureSpace(V, self.cfg.H, self.cfg.suffix_only)
        cap = 64
        self.C = np.zeros((cap, V), dtype=np.int64)
        self.n = np.zeros(cap, dtype=np.int64)
        self.parent = np.full(cap, -1)
        self.child = np.full((cap, 2), -1)
        self.mass = np.zeros(cap)  # decayed loss (bits) arriving at each node
        self.traffic = np.zeros(cap)  # decayed fraction of a batch arriving
        self.feat: dict[int, tuple] = {}
        self.cmask: dict[int, np.ndarray] = {}
        self.size = 1
        self.pool: dict[int, list[Candidate]] = {}
        self.tried: dict[int, set] = {0: set()}
        self.n_req: dict[int, int] = {0: self.cfg.n_min}  # examples a candidate at this leaf needs before judgement
        self.queue: dict[int, list[tuple[float, tuple]]] = {0: []}
        self.cursor: dict[int, tuple[int, int]] = {}
        self.updates = 0
        self.skipped = 0  # leaf-batches whose candidate tables were not advanced, for lack of budget
        self._review_pos = 0
        self._deadline = 0
        self._starved = False  # the last refill was cut short by the op budget
        self._single_bits = 0.0
        self.cand_ops_last = 0
        self.budget_violations = 0
        self.commits: list[dict] = []

    # -- prediction and count updates --------------------------------------
    def _context(self, seq: np.ndarray, t0: int, t1: int) -> np.ndarray:
        H = self.cfg.H
        return sliding_window_view(seq, H)[t0 - H : t1 - H][:, ::-1]

    def _route(self, B: Batch, y: np.ndarray):
        """Return per-example loss (bits), leaf ids, and the visited (rows, nodes) layers."""
        lam = self.cfg.lam
        cur = np.zeros(B.b, dtype=np.int64)
        q = np.full(B.b, 1.0 / self.V)
        active = B.rows
        layers = []
        while len(active):
            nodes = cur[active]
            q[active] = (self.C[nodes, y[active]] + lam * q[active]) / (self.n[nodes] + lam)
            layers.append((active, nodes))
            self.ops.n += len(active)
            inner = self.child[nodes, 0] >= 0
            active, nodes = active[inner], nodes[inner]
            for u in np.unique(nodes):
                sel = active[nodes == u]
                side = np.where(self.cmask[u][B.val(self.feat[u])[sel]], 0, 1)
                cur[sel] = self.child[u, side]
        return -np.log2(q), cur, layers

    def predictive(self, node: int) -> np.ndarray:
        """The full next-token distribution predicted at `node`."""
        path = []
        while node >= 0:
            path.append(node)
            node = self.parent[node]
        q = np.full(self.V, 1.0 / self.V)
        for u in reversed(path):
            q = (self.C[u] + self.cfg.lam * q) / (self.n[u] + self.cfg.lam)
        return q

    def step_batch(self, seq: np.ndarray, t0: int, t1: int, update: bool = True) -> np.ndarray:
        """Loss in bits for seq[t0:t1], predicted before any of it is learned."""
        B = Batch(self._context(seq, t0, t1), self.space, self.ops)
        y = seq[t0:t1]
        bits, leaf, layers = self._route(B, y)
        if not update:
            return bits
        for rows, nodes in layers:
            np.add.at(self.C, (nodes, y[rows]), 1)
            np.add.at(self.n, nodes, 1)
        self.mass[: self.size] *= 0.95
        self.traffic[: self.size] *= 0.95
        np.add.at(self.mass, leaf, bits)
        np.add.at(self.traffic, leaf, 0.05 / B.b)

        # Candidate work is hard-limited: every unit of work below checks its
        # cost against a deadline on the op counter before starting.
        start = self.ops.n
        self._deadline = start + int(self.cfg.budget * self.cfg.advance_share)
        self._advance(B, y, leaf)
        self._deadline = start + self.cfg.budget
        self._review()
        self._refill(B.b)
        self.updates += 1
        self.cand_ops_last = self.ops.n - start
        self.budget_violations += self.cand_ops_last > self.cfg.budget
        return bits

    def _affordable(self, cost: int) -> bool:
        return self.ops.n + cost <= self._deadline

    # -- candidate pool ----------------------------------------------------
    def _advance(self, B: Batch, y: np.ndarray, leaf: np.ndarray) -> None:
        """Add this batch to candidate tables, heaviest leaves first, within budget."""
        V = self.V
        for u in sorted(self.pool, key=lambda u: -self.mass[u]):
            cands = self.pool[u]
            sel = np.flatnonzero(leaf == u)
            if not len(sel):
                continue
            fresh = {c.term for c in cands if c.term not in B._memo}
            cost = len(sel) * len(cands) + sum(self._eval_cost(t, B.b) for t in fresh)
            if not self._affordable(cost):
                self.skipped += 1
                continue
            ysel = y[sel]
            for c in cands:
                K = c.table.shape[0]
                flat = B.val(c.term)[sel] * V + ysel
                c.table += np.bincount(flat, minlength=K * V).reshape(K, V).astype(np.int32)
                c.n += len(sel)
            self.ops.n += len(sel) * len(cands)

    def _log_marginal(self, counts: np.ndarray, am: np.ndarray, lg_am: np.ndarray) -> np.ndarray:
        """log Dirichlet marginal likelihood of count rows (last axis = tokens), in nats."""
        n = counts.sum(axis=-1)
        self.ops.n += counts.size
        return gammaln(am.sum()) - gammaln(n + am.sum()) + (gammaln(counts + am) - lg_am).sum(axis=-1)

    def _gain(self, side: np.ndarray, total: np.ndarray, am, lg_am, whole: float) -> np.ndarray:
        """Bits gained by modelling `side` rows and their complements separately."""
        if self.cfg.rule == "ig":
            def ent(c):
                n = c.sum(axis=-1, keepdims=True)
                p = np.where(c > 0, c / np.maximum(n, 1), 1.0)
                return -(c * np.log2(p)).sum(axis=-1)
            self.ops.n += 2 * side.size
            return ent(total) - ent(side) - ent(total - side)
        both = self._log_marginal(side, am, lg_am) + self._log_marginal(total - side, am, lg_am)
        return (both - whole) / LN2

    def _best_class(self, T: np.ndarray, total: np.ndarray, am, lg_am, whole: float):
        """Search for the set of rows of T whose split has the largest net gain.

        Start from the best single row. Then, twice, order rows by how much
        their token distribution resembles the current class against the rest
        and take the best prefix of that order. Cost is O(rows * V) per pass.
        Returns (gain_bits, class_description_bits, row_mask).
        """
        L = len(T)
        singles = self._gain(T, total, am, lg_am, whole)
        k = int(singles.argmax())
        mask = np.zeros(L, dtype=bool)
        mask[k] = True
        best = (float(singles[k]), self._single_bits, mask)
        a = am.sum()
        P = (T + am) / (T.sum(axis=1, keepdims=True) + a)
        side = T[k]
        for _ in range(2):
            ref = (side + am) / (side.sum() + a)
            rest = (total - side + am) / (total.sum() - side.sum() + a)
            order = np.argsort(-(P @ np.log(ref / rest)))
            self.ops.n += T.size
            cum = np.cumsum(T[order], axis=0)[:-1]
            gains = self._gain(cum, total, am, lg_am, whole)
            j = int(gains.argmax())
            if j == 0 or gains[j] - L <= best[0] - best[1]:
                break
            mask = np.zeros(L, dtype=bool)
            mask[order[: j + 1]] = True
            best = (float(gains[j]), float(L), mask)
            side = cum[j]
        return best

    def _review(self) -> None:
        """Score, evict and commit, one leaf at a time in rotation, within budget."""
        cfg = self.cfg
        order = sorted(self.pool)
        if not order:
            return
        start = next((i for i, u in enumerate(order) if u >= self._review_pos), 0)
        for u in order[start:] + order[:start]:
            cands, n_min = self.pool[u], self.n_req[u]
            if not any(c.n >= n_min and c.n != c.judged_n for c in cands):
                continue
            if not self._affordable(2 * self.V):
                self._review_pos = u
                return
            am = cfg.lam * self.predictive(u)
            self.ops.n += 2 * self.V
            lg_am = gammaln(am)
            keep, unreached = [], []
            seen: set[bytes] = set()
            for c in cands:
                if c.n < n_min or c.n == c.judged_n:
                    keep.append(c)
                    continue
                rows = np.flatnonzero(c.table.sum(axis=1) > 0)
                if unreached or not self._affordable(2 * c.table.size + 9 * len(rows) * self.V):
                    unreached.append(c)
                    continue
                key = c.table.tobytes()
                self.ops.n += 2 * c.table.size
                if len(rows) < 2 or key in seen:
                    continue  # constant on this leaf, or a duplicate of an earlier candidate
                seen.add(key)
                total = c.table.sum(axis=0)
                whole = float(self._log_marginal(total, am, lg_am))
                self._single_bits = log2(c.table.shape[0])
                gain, class_bits, live_mask = self._best_class(c.table[rows], total, am, lg_am, whole)
                c.score, c.gain, c.judged_n = gain - c.cost - class_bits, gain, c.n
                c.mask = np.zeros(c.table.shape[0], dtype=bool)
                c.mask[rows[live_mask]] = True
                keep.append(c)
            # Candidates the budget did not reach go first, so they are judged next time.
            keep = unreached + keep
            judged = [c for c in keep if c.judged_n == c.n and c.n >= n_min]
            winner = max(judged, key=lambda c: c.score, default=None)
            net = winner.score - log2(1 + len(self.tried[u])) if winner else -np.inf
            crowded = len(keep) >= cfg.per_leaf or self._starved
            if net <= cfg.threshold and crowded and len(judged) >= 4:
                # Slots are scarce and nothing commits here: release the weakest
                # quarter of judged candidates, ranked by score per example.
                # A negative score alone is not grounds for eviction: on few
                # examples the complexity penalty outweighs any real gain.
                judged.sort(key=lambda c: c.score / c.n)
                drop = {id(c) for c in judged[: len(judged) // 4]}
                keep = [c for c in keep if id(c) not in drop]
            if keep:
                self.pool[u] = keep
            else:
                del self.pool[u]
            if net > cfg.threshold:
                self._commit(net, winner.gain, winner, winner.mask)
            if unreached:
                self._review_pos = u
                return
        self._review_pos = 0

    def _grow(self) -> None:
        if self.size + 2 <= len(self.n):
            return
        cap = 2 * len(self.n)
        for name in ("C", "n", "parent", "child", "mass", "traffic"):
            old = getattr(self, name)
            new = np.full((cap,) + old.shape[1:], -1 if name in ("parent", "child") else 0, dtype=old.dtype)
            new[: len(old)] = old
            setattr(self, name, new)

    def _commit(self, net: float, gain: float, c: Candidate, mask: np.ndarray) -> None:
        self._grow()
        u = c.leaf
        self.feat[u], self.cmask[u] = c.term, mask
        # Features that were doing well at u are tried first at its children.
        siblings = sorted(self.pool.pop(u), key=lambda s: -s.score / max(s.n, 1))
        heirs = [(s.cost, s.term) for s in siblings[: self.cfg.inherit]]
        if c.term[0] == "at" and mask.sum() > 1 and mask.sum() < len(mask) - 1:
            self.space.add_class(mask)
        for side in (0, 1):
            v = self.size
            self.size += 1
            self.child[u, side] = v
            self.parent[v] = u
            rows = mask if side == 0 else ~mask
            if self.cfg.warm_start:
                self.C[v] = c.table[rows].sum(axis=0)
                self.n[v] = self.C[v].sum()
            share = self.n[v] / max(c.n, 1)
            self.mass[v], self.traffic[v] = self.mass[u] * share, self.traffic[u] * share
            self.tried[v], self.queue[v], self.n_req[v] = set(), list(heirs), self.cfg.n_min
        self.commits.append({
            "update": self.updates, "node": u, "term": c.term, "mask": mask,
            "gain": gain, "net": net, "n": c.n,
        })

    def _next_feature(self, u: int):
        tried = self.tried[u]
        while self.queue[u]:
            cost, term = self.queue[u].pop(0)
            if term not in tried:
                return cost, term
        version, i = self.cursor.get(u, (-1, 0))
        if version != self.space.version:
            i = 0
        feats = self.space.features
        while i < len(feats) and feats[i][1] in tried:
            i += 1
        self.cursor[u] = (self.space.version, i + 1)
        if i < len(feats):
            return feats[i]
        # Every feature has been tried at this leaf. Rejections were made on
        # limited evidence, so start over and demand twice as much next time.
        if not self.pool.get(u):
            self.tried[u] = set()
            self.n_req[u] *= 2
            self.cursor[u] = (self.space.version, 0)
        return None

    def _refill(self, b: int) -> None:
        cfg = self.cfg
        leaves = [u for u in range(self.size) if self.child[u, 0] < 0]
        active = {c.term for cs in self.pool.values() for c in cs}
        est = sum(self._eval_cost(t, b) for t in active)
        est += sum(b * self.traffic[u] / max(self.traffic[: self.size].sum(), 1e-12) * len(cs)
                   for u, cs in self.pool.items())
        count = sum(len(cs) for cs in self.pool.values())
        exhausted: set[int] = set()
        self._starved = False
        while count < cfg.pool:
            open_ = [u for u in leaves if u not in exhausted and len(self.pool.get(u, ())) < cfg.per_leaf]
            if not open_:
                break
            u = max(open_, key=lambda u: self.mass[u] / (1 + len(self.pool.get(u, ()))))
            nxt = self._next_feature(u)
            if nxt is None:
                exhausted.add(u)
                continue
            cost, term = nxt
            extra = (0 if term in active else self._eval_cost(term, b)) + b
            if est + extra > cfg.budget * cfg.advance_share:
                self.queue[u].insert(0, (cost, term))
                self._starved = True
                break
            est += extra
            active.add(term)
            self.tried[u].add(term)
            table = np.zeros((self.space.K(term), self.V), dtype=np.int32)
            self.pool.setdefault(u, []).append(Candidate(u, term, cost, table))
            count += 1

    def _eval_cost(self, term: tuple, b: int) -> int:
        """Upper bound on the ops needed to evaluate `term` on a batch from scratch."""
        scans = {"first": 2, "next": 2, "cnt": 2}
        cost, stack = 0, [term]
        while stack:
            t = stack.pop()
            cost += b * (1 + scans.get(t[0], 0) * self.cfg.H)
            stack += [x for x in t[1:] if isinstance(x, tuple)]
        return cost

    # -- reporting ---------------------------------------------------------
    @property
    def leaves(self) -> int:
        return (self.size + 1) // 2

    def describe(self, names: list[str], limit: int = 20) -> list[str]:
        out = []
        for c in self.commits[:limit]:
            K = len(c["mask"])
            vals = np.flatnonzero(c["mask"])
            if c["term"][0] == "at":
                members = [names[v] if v < self.V else "none" for v in vals]
            else:
                members = [str(v) if v < K - 1 or c["term"][0] == "cnt" else "none" for v in vals]
            if len(members) > 12:
                members = members[:12] + [f"…+{len(members) - 12}"]
            out.append(
                f"update {c['update']:5d} node {c['node']:4d}: {show(c['term'], self.space, names)} "
                f"in {{{','.join(members)}}}  gain={c['gain']:.0f} bits over {c['n']} examples"
            )
        return out


def prequential_batches(model: PredicateTree, seq: np.ndarray, start: int = 16) -> np.ndarray:
    """Per-token loss for seq[start:], one microbatch at a time."""
    b = model.cfg.batch
    return np.concatenate([model.step_batch(seq, t, min(t + b, len(seq))) for t in range(start, len(seq), b)])
