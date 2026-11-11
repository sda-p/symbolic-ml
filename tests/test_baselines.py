from itertools import product
from math import exp, lgamma, log

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from symbolic_ml.data.synthetic import SOURCES
from symbolic_ml.models.ctw import CTW
from symbolic_ml.models.ngram import NGram

seqs = st.integers(2, 4).flatmap(
    lambda V: st.tuples(st.just(V), st.lists(st.integers(0, V - 1), min_size=4, max_size=40))
)


def _probs(model, seq, t, V):
    return [2.0 ** -model.step(seq[:t] + [y], t, update=False) for y in range(V)]


@settings(max_examples=60, deadline=None)
@given(seqs, st.integers(0, 3), st.sampled_from([0.1, 0.5, 1.0]))
def test_predictions_normalise_and_peek_matches_update(vs, depth, alpha):
    V, seq = vs
    for model in (CTW(V, depth, alpha=alpha), NGram(V, depth, lam=alpha)):
        for t in range(depth, len(seq)):
            assert abs(sum(_probs(model, seq, t, V)) - 1) < 1e-9
            peek = model.step(seq, t, update=False)
            assert abs(peek - model.step(seq, t)) < 1e-9


def _log_dirichlet_marginal(counts, V, alpha):
    n = sum(counts)
    out = lgamma(V * alpha) - lgamma(n + V * alpha)
    return out + sum(lgamma(c + alpha) - lgamma(alpha) for c in counts)


def _mixture_brute_force(seq, V, H, alpha, rho):
    """Sum over every context tree of prior(T) * marginal likelihood under T."""

    def K(ctx):
        counts = [0] * V
        for t in range(H, len(seq)):
            if all(seq[t - 1 - i] == c for i, c in enumerate(ctx)):
                counts[seq[t]] += 1
        return exp(_log_dirichlet_marginal(counts, V, alpha))

    def trees(ctx):
        """Yield (prior, likelihood) for every subtree rooted at ctx."""
        if len(ctx) == H:
            yield 1.0, K(ctx)
            return
        yield rho, K(ctx)
        for combo in product(*[list(trees(ctx + (v,))) for v in range(V)]):
            prior, like = 1 - rho, 1.0
            for p, l in combo:
                prior *= p
                like *= l
            yield prior, like

    return sum(p * l for p, l in trees(()))


@settings(max_examples=40, deadline=None)
@given(st.lists(st.integers(0, 1), min_size=3, max_size=14), st.sampled_from([0.3, 0.5]))
def test_ctw_equals_explicit_mixture_over_trees(seq, rho):
    V, H, alpha = 2, 2, 1.0
    model = CTW(V, H, alpha=alpha, rho=rho)
    total_bits = sum(model.step(seq, t) for t in range(H, len(seq)))
    expected = -log(_mixture_brute_force(seq, V, H, alpha, rho)) / log(2)
    assert abs(total_bits - expected) < 1e-8


def test_sources_are_well_formed():
    for name, make in SOURCES.items():
        s = make(5000, seed=0)
        assert len(s.seq) == len(s.true_bits) == 5000, name
        assert s.seq.min() >= 0 and s.seq.max() < s.V, name
        assert np.all(s.true_bits >= 0), name
        assert np.array_equal(s.seq, make(5000, seed=0).seq), name
