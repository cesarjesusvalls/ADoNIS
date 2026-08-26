"""Gate for the GiBUU N N -> N Delta production core (adonis/fsi/mb/nn_delta.py) -- the heart
of the Propagating-Resonances mode / paper Fig 14.  Exact port of ResonanceHelper.cc; here we
gate the Delta effective width (= the pole width at the pole) and the production cross section's
physical shape (rises from the N+Delta threshold through the Delta region).
"""
import numpy as np

from tests.reference import nn_to_ndelta as nd


def test_effective_width_at_pole():
    """The running Delta -> N pi width equals the pole width at the pole mass (exact port)."""
    assert abs(float(nd.effective_width(nd.M_DELTA)) - nd.W_DELTA) < 1e-6
    # below threshold (m < mn+mpi) the width -> 0
    assert float(nd.effective_width(nd.M_N + nd.M_PI + 1e-4)) < 0.02
    # the width rises with mass above the pole
    assert nd.effective_width(1.35) > nd.effective_width(1.232) > nd.effective_width(1.15)


def test_production_xsec_physical():
    """sigma(NN->NDelta) is 0 below the 2mn+mpi threshold, rises through the Delta region, and
    is finite/positive."""
    assert nd.sigma_nn2ndelta(2.0) < 0.05                    # ~ at/below 2mn+mpi = 2.017
    s_lo = nd.sigma_nn2ndelta(2.10); s_mid = nd.sigma_nn2ndelta(2.30)
    assert 0 < s_lo < s_mid                                  # rises through the Delta region
    assert np.isfinite(s_mid) and s_mid > 10.0


def test_matrix_element_finite():
    """The Dmitriev-Sushkov matrix element is finite/positive in the physical region."""
    # representative t,u at sqrts=2.2, m_delta=1.232
    m = nd.mat_nn2ndelta(-0.3, -0.3, nd.M_DELTA ** 2)
    assert np.isfinite(m) and m > 0


def test_fig14_pp_channels():
    """Fig 14: pp->pn pi+ and pp->pp pi0 peak at the right magnitude (~20 / ~4 mb) around
    sqrts~2.3-2.5 GeV, with the 5:1 isospin ratio (Delta++ : Delta+ + branching)."""
    s = np.linspace(2.05, 3.0, 16)
    pn = np.array([nd.sigma_pp_channels(x)[0] for x in s])
    pp = np.array([nd.sigma_pp_channels(x)[1] for x in s])
    assert 15 < pn.max() < 26, pn.max()                      # pp->pn pi+ peak ~20 mb
    assert 3 < pp.max() < 6, pp.max()                        # pp->pp pi0 peak ~4 mb
    assert abs(pn.max() / pp.max() - 5.0) < 0.2              # 5:1 isospin ratio
    assert 2.2 < s[pn.argmax()] < 2.6                        # peaks in the Delta region
    assert pn[0] < 0.3 and pn[-1] < pn.max()                 # rises from threshold, falls off
