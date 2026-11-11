import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from symbolic_ml.dsl.terms import Batch, FeatureSpace
from symbolic_ml.models.ngram import NGram
from symbolic_ml.models.ptree import Config, PredicateTree, prequential_batches
from symbolic_ml.ops import Ops

V, H = 4, 6


def naive(term, row, space):
    """Reference semantics for one context row, written without numpy."""
    kind = term[0]
    member = lambda S, j: bool(space.masks[S][row[j]])
    if kind == "off":
        return term[1]
    if kind == "all":
        return H
    if kind == "first":
        return next((j for j in range(H) if member(term[1], j)), H)
    if kind == "next":
        p = naive(term[2], row, space)
        return next((j for j in range(H) if j > p and member(term[1], j)), H)
    if kind == "shift":
        p = naive(term[1], row, space)
        q = p + term[2]
        return H if p == H or not 0 <= q < H else q
    if kind == "at":
        p = naive(term[1], row, space)
        return V if p == H else row[p]
    if kind == "dist":
        return naive(term[1], row, space)
    p = naive(term[2], row, space)
    return sum(member(term[1], j) for j in range(min(p, H))) % term[3]


@settings(max_examples=30, deadline=None)
@given(st.lists(st.lists(st.integers(0, V - 1), min_size=H, max_size=H), min_size=1, max_size=8))
def test_columnar_evaluation_matches_reference(rows):
    space = FeatureSpace(V, H)
    space.add_class(np.array([True, False, True, False, False]))
    ctx = np.array(rows)
    batch = Batch(ctx, space, Ops())
    for _, term in space.features:
        got = batch.val(term)
        assert got.max() < space.K(term)
        assert got.tolist() == [naive(term, r, space) for r in rows], term


def test_unsplit_tree_is_a_unigram_model():
    seq = np.random.default_rng(0).integers(0, 5, size=400)
    cfg = Config(batch=1, pool=0, lam=2.0)
    tree = prequential_batches(PredicateTree(5, cfg), seq)
    ref = NGram(5, 0, lam=2.0)
    s = seq.tolist()
    assert np.allclose(tree, [ref.step(s, t) for t in range(16, len(s))])
