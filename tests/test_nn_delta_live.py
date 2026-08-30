"""The live NN -> N Delta cross section agrees with the independent numpy reference.

tests/reference/nn_to_ndelta.py is a scalar-loop re-derivation kept for exactly this comparison: the
production path interpolates precomputed tables, the reference integrates directly, and they share no
code.  Agreement is limited by the discretisation on both sides, which is why the tolerance is loose
and the reference is evaluated on a finer grid than its own default.

The production total sums the two charge states allowed for a given isospin pair; the reference gives
one at a time, so pp is Delta++ (isofactor 1) plus Delta+ (isofactor 1/3).
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.slow

REF_COST, REF_M = 480, 560
TOL = 0.08


@pytest.mark.parametrize("sqrts", [2.2, 2.6, 3.0])
def test_live_matches_the_reference_integral(sqrts):
    from adonis.fsi import nn_inelastic as live
    from tests.reference import nn_to_ndelta as ref

    base = ref._base_integral(sqrts, n_cost=REF_COST, n_m=REF_M)
    expected = base * (1.0 + 1.0 / 3.0) / ref._pcm(sqrts)
    got = float(live.sigma_nn_ndelta(sqrts, ref._pcm(sqrts), True))
    assert expected > 0
    assert abs(got - expected) / expected < TOL, (
        f"sqrts={sqrts}: live {got:.4f} mb vs reference {expected:.4f} mb")


def test_isospin_factors_differ_between_like_and_unlike_pairs():
    """pp/nn reaches Delta++ and Delta+; pn reaches only the isofactor-1/3 states."""
    from adonis.fsi import nn_inelastic as live
    from tests.reference import nn_to_ndelta as ref
    pcm = ref._pcm(2.6)
    same = float(live.sigma_nn_ndelta(2.6, pcm, True))
    diff = float(live.sigma_nn_ndelta(2.6, pcm, False))
    assert same > diff > 0
    assert abs(same / diff - 2.0) < 1e-6, "the two isospin sums should stand as 4/3 to 2/3"
