"""Synthetic token sources with known generating structure.

Every source returns a `Sample` whose `true_bits[t]` is -log2 of the true
conditional probability of `seq[t]` given the whole past. The latent state of
each generator is a deterministic function of the tokens emitted so far, so the
mean of `true_bits` is the loss of the best possible predictor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log2

import numpy as np


@dataclass
class Sample:
    seq: np.ndarray
    true_bits: np.ndarray
    V: int
    names: list[str]
    meta: dict = field(default_factory=dict)


class _Emitter:
    def __init__(self, rng: np.random.Generator, n: int):
        self.rng = rng
        self.n = n
        self.seq: list[int] = []
        self.bits: list[float] = []

    @property
    def full(self) -> bool:
        return len(self.seq) >= self.n

    def emit(self, tok: int, p: float) -> None:
        self.seq.append(tok)
        self.bits.append(-log2(p))

    def choice(self, toks: list[int], probs: list[float] | None = None) -> int:
        """Sample one of `toks`, emit it, and return its index."""
        k = len(toks)
        if probs is None:
            i = int(self.rng.integers(k))
            self.emit(toks[i], 1.0 / k)
        else:
            i = int(self.rng.choice(k, p=probs))
            self.emit(toks[i], probs[i])
        return i

    def noisy(self, toks: list[int], target: int, acc: float) -> None:
        """Emit toks[target] with probability `acc`, else another uniformly."""
        k = len(toks)
        probs = [(1 - acc) / (k - 1)] * k
        probs[target] = acc
        self.choice(toks, probs)

    def sample(self, V: int, names: list[str], **meta) -> Sample:
        return Sample(
            np.array(self.seq[: self.n], dtype=np.int64),
            np.array(self.bits[: self.n]),
            V,
            names,
            meta,
        )


def tree_source(n: int, seed: int, V: int = 8, max_depth: int = 4, p_split: float = 0.3) -> Sample:
    """S1: random variable-order Markov source.

    Contexts are tuples (x[t-1], x[t-2], ...), most recent token first.
    `meta["leaves"]` lists the leaf contexts and `meta["leaf_of"]` gives the
    leaf index used at each position (-1 where the past is too short).
    """
    rng = np.random.default_rng(seed)
    leaves: dict[tuple[int, ...], np.ndarray] = {}
    internal = 0

    def grow(ctx: tuple[int, ...]) -> None:
        nonlocal internal
        d = len(ctx)
        if d < max_depth and (d == 0 or rng.random() < p_split):
            internal += 1
            for v in range(V):
                grow(ctx + (v,))
        else:
            leaves[ctx] = rng.dirichlet(np.full(V, 0.5))

    grow(())
    leaf_index = {c: i for i, c in enumerate(leaves)}
    cdfs = {c: np.cumsum(p) for c, p in leaves.items()}
    seq = [int(v) for v in rng.integers(V, size=max_depth)]
    bits = [log2(V)] * max_depth
    leaf_of = [-1] * max_depth
    us = rng.random(n)
    for t in range(max_depth, n):
        ctx: tuple[int, ...] = ()
        while ctx not in leaves:
            ctx = ctx + (seq[t - 1 - len(ctx)],)
        y = min(int(np.searchsorted(cdfs[ctx], us[t])), V - 1)
        seq.append(y)
        bits.append(-log2(leaves[ctx][y]))
        leaf_of.append(leaf_index[ctx])
    return Sample(
        np.array(seq[:n], dtype=np.int64),
        np.array(bits[:n]),
        V,
        [str(v) for v in range(V)],
        {
            "leaves": list(leaves),
            "leaf_of": np.array(leaf_of[:n]),
            "internal": internal,
            "max_depth": max_depth,
        },
    )


def agreement(n: int, seed: int, acc: float = 0.95, p_more: float = 0.6, max_distractors: int = 6) -> Sample:
    """S2: the verb agrees with the first noun, across distractor phrases.

    Sentence: SUBJECT (PREP NOUN)* VERB FILLER END. The verb's number matches
    the subject's with probability `acc`; distractor nouns have random number.
    """
    names = ["key", "dog", "cat", "keys", "dogs", "cats", "near", "by", "is", "are", "here", "now", "."]
    sing, plur, preps, verbs, fillers, end = [0, 1, 2], [3, 4, 5], [6, 7], [8, 9], [10, 11], 12
    nouns = sing + plur
    e = _Emitter(np.random.default_rng(seed), n)
    while not e.full:
        number = e.choice(nouns) // 3
        k = 0
        while True:
            if k < max_distractors:
                more = e.rng.random() < p_more
                p_verb = 1 - p_more
            else:
                more, p_verb = False, 1.0
            if not more:
                break
            e.emit(preps[int(e.rng.integers(2))], p_more / 2)
            e.choice(nouns)
            k += 1
        probs = [1 - acc, 1 - acc]
        probs[number] = acc
        i = int(e.rng.choice(2, p=probs))
        e.emit(verbs[i], p_verb * probs[i])
        e.choice(fillers)
        e.emit(end, 1.0)
    return e.sample(len(names), names)


def brackets(n: int, seed: int, max_depth: int = 4, p_open: float = 0.45) -> Sample:
    """S3: two bracket types, bounded nesting; a closer must match its opener."""
    names = ["(", "[", ")", "]"]
    e = _Emitter(np.random.default_rng(seed), n)
    stack: list[int] = []
    while not e.full:
        d = len(stack)
        po = 1.0 if d == 0 else (0.0 if d == max_depth else p_open)
        if e.rng.random() < po:
            kind = int(e.rng.integers(2))
            e.emit(kind, po / 2)
            stack.append(kind)
        else:
            e.emit(2 + stack.pop(), 1 - po)
    return e.sample(4, names)


def counter(n: int, seed: int, m: int = 3, acc: float = 0.95, p_mark: float = 0.4) -> Sample:
    """S4: the answer is the number of markers in the segment, mod m.

    Segment: 3-10 body tokens (marker or filler), QUERY, ANSWER, END.
    """
    names = ["M", "a", "b", "c", "?", "."] + [f"r{i}" for i in range(m)]
    marker, fillers, query, end = 0, [1, 2, 3], 4, 5
    answers = list(range(6, 6 + m))
    lengths = list(range(3, 11))
    e = _Emitter(np.random.default_rng(seed), n)
    while not e.full:
        length = lengths[int(e.rng.integers(len(lengths)))]
        count = 0
        for i in range(length):
            # Stopping is only possible once 3 body tokens exist; the length is
            # uniform on 3..10, so the hazard of stopping after i tokens is
            # 1/(11-i) for i >= 3.
            p_cont = 1.0 if i < 3 else 1 - 1 / (11 - i)
            if e.rng.random() < p_mark:
                e.emit(marker, p_cont * p_mark)
                count += 1
            else:
                e.emit(fillers[int(e.rng.integers(3))], p_cont * (1 - p_mark) / 3)
        e.emit(query, 1.0 if length == 10 else 1 / (11 - length))
        e.noisy(answers, count % m, acc)
        e.emit(end, 1.0)
    return e.sample(len(names), names)


def xor(n: int, seed: int, acc: float = 0.95, max_gap: int = 5) -> Sample:
    """S5: the answer is A xor B; neither A nor B alone is informative.

    Segment: A (a0|a1), 0..max_gap fillers, B (b0|b1), ANSWER, END.
    """
    names = ["a0", "a1", "b0", "b1", "w", "x", "y", "z", "r0", "r1", "."]
    a_toks, b_toks, fillers, answers, end = [0, 1], [2, 3], [4, 5, 6, 7], [8, 9], 10
    e = _Emitter(np.random.default_rng(seed), n)
    while not e.full:
        a = e.choice(a_toks)
        gap = int(e.rng.integers(max_gap + 1))
        for i in range(gap):
            # Gap length is uniform on 0..max_gap: hazard of stopping after i
            # fillers is 1/(max_gap+1-i).
            p_cont = 1 - 1 / (max_gap + 1 - i)
            e.emit(fillers[int(e.rng.integers(4))], p_cont / 4)
        p_stop = 1 / (max_gap + 1 - gap)
        b = int(e.rng.integers(2))
        e.emit(b_toks[b], p_stop / 2)
        e.noisy(answers, a ^ b, acc)
        e.emit(end, 1.0)
    return e.sample(len(names), names)


SOURCES = {
    "S1-tree": tree_source,
    "S2-agreement": agreement,
    "S3-brackets": brackets,
    "S4-counter": counter,
    "S5-xor": xor,
}
