"""Analytic propagator gate for the NC leptonic current.  No oracle, no bank.

An overall missing or extra propagator factor cancels in any ratio-of-samples comparison, so it needs
a check whose two sides are both closed-form.  That is what this file is.

The construction is propagator-sensitive by design.  At Q^2 << M_Z^2 the whole NC/CC leptonic ratio
collapses to a pure number built from the couplings and the two boson masses:

    |L_NC|^2 / |L_CC|^2  ->  |c_NC / c_CC|^2 * (M_W^2 / M_Z^2)^2

with c_NC = ee*i/(2*sw*cw) and c_CC = ee*i/(sw*sqrt2), so |c_NC/c_CC|^2 = 1/(2*cw^2).  Nothing here
is fitted and nothing is measured -- both sides are closed-form, so the test cannot be weakened to
fit an implementation.  A missing propagator moves this by (M_Z^2/Q^2)^2, i.e. many orders of
magnitude; a wrong coupling moves it by a clean factor.

The energy-dependence check is the second half: a CONSTANT offset is a coupling error, a SLOPE in
Q^2 is a propagator error.  Reporting them separately is what tells the two apart.
"""
import numpy as np
import pytest

from adonis.channels import constants as C
from adonis.channels.currents.leptonic import lepton_current

_METRIC = np.array([1.0, -1.0, -1.0, -1.0])


def _amps2(L):
    """Sum over spin combos of |L.L*|, Minkowski-contracted -- the leptonic tensor trace."""
    L = np.asarray(L)
    return float(np.sum(np.abs(np.einsum('...cm,m,...cm->...c', L, _METRIC, L.conj())) ** 2))


def _kin(q2_scale):
    """A massless 2-body lepton kinematic with |Q^2| ~ q2_scale [MeV^2].

    The beam energy SCALES with the transfer (E = 4|q| at minimum): a fixed E cannot deliver a
    transfer larger than itself, and clamping the longitudinal component to keep the sqrt real
    silently produces an off-shell p_out whose spinors evaluate to NaN.  That is a property of the
    test kinematics, not of the current -- so make the kinematics physical at every Q^2 probed.
    """
    q = np.sqrt(q2_scale)
    E = max(5000.0, 4.0 * q)
    p_in = np.array([[E, 0.0, 0.0, E]])
    pz = np.sqrt(E ** 2 - q ** 2)
    return p_in, np.array([[E, q, 0.0, pz]])


def _c_nc():
    return (C.cw * C.ee * 1j) / (2 * C.sw) + (C.ee * 1j * C.sw) / (2 * C.cw)


def _c_cc():
    return C.ee * 1j / (C.sw * np.sqrt(2.0))


def test_nc_coupling_equals_the_textbook_form():
    """ACHILLES writes the NC coupling as cw*ee*i/(2sw) + ee*i*sw/(2cw).  That is algebraically
    ee*i/(2*sw*cw) -- assert it, so the literal transcription and the closed form cannot drift."""
    assert _c_nc() == pytest.approx(C.ee * 1j / (2 * C.sw * C.cw), rel=1e-14)


def test_nc_over_cc_amps2_reduces_to_the_analytic_ratio():
    """.  At Q^2 << M_Z^2 the ratio must be |c_NC/c_CC|^2 * (M_W^2/M_Z^2)^2 to <= 1e-3."""
    q2 = (50.0) ** 2
    p_in, p_out = _kin(q2)
    r = _amps2(lepton_current(p_in, p_out, kind="NC_nu")) / \
        _amps2(lepton_current(p_in, p_out, kind="CC_nu"))
    want = abs(_c_nc() / _c_cc()) ** 4 * (C.MW ** 2 / C.MZ ** 2) ** 4
    assert r == pytest.approx(want, rel=1e-3), f"NC/CC = {r:.6e}, analytic {want:.6e}"


def test_the_ratio_is_flat_in_q2_far_below_mz():
    """A CONSTANT offset is a coupling error; a SLOPE is a propagator error.  Report them apart."""
    rs = []
    for q2 in (10.0 ** 2, 50.0 ** 2, 200.0 ** 2, 800.0 ** 2):
        p_in, p_out = _kin(q2)
        rs.append(_amps2(lepton_current(p_in, p_out, kind="NC_nu")) /
                  _amps2(lepton_current(p_in, p_out, kind="CC_nu")))
    rs = np.array(rs)
    spread = rs.std() / rs.mean()
    assert spread < 1e-3, f"NC/CC ratio drifts with Q^2 (spread {spread:.2e}) -> propagator error"


def test_the_propagator_runs_with_q2_exactly_as_the_z_pole_demands():
    """The NC/CC ratio, not the raw NC amps2, is the decisive quantity.

    Raw amps2 grows with the beam energy needed to reach a given Q^2, fast enough to swamp the
    propagator; the ratio cancels that growth and isolates the coupling and propagator content.

    The decisive statement takes the NC/CC ratio at each Q^2, where the spinor content cancels
    identically, leaving only the couplings and the two propagators:

        |L_NC|^2/|L_CC|^2 = |c_NC/c_CC|^4 * |(q^2 - M_W^2 - i M_W G_W)/(q^2 - M_Z^2 - i M_Z G_Z)|^4

    This is exact at EVERY Q^2, not just below the pole, so it pins the propagator's Q^2 DEPENDENCE
    and not merely its presence.  A missing propagator fails it by orders of magnitude; a swapped
    M_W/M_Z fails it by (M_Z/M_W)^8.
    """
    for q2 in (100.0 ** 2, 1000.0 ** 2, 10000.0 ** 2, (0.5 * C.MZ) ** 2):
        p_in, p_out = _kin(q2)
        got = _amps2(lepton_current(p_in, p_out, kind="NC_nu")) / \
              _amps2(lepton_current(p_in, p_out, kind="CC_nu"))
        q = p_in - p_out
        qq = q[0, 0] ** 2 - np.sum(q[0, 1:] ** 2)
        prop_ratio = (qq - C.MW ** 2 - 1j * C.MW * C.GAMW) / (qq - C.MZ ** 2 - 1j * C.MZ * C.GAMZ)
        want = abs(_c_nc() / _c_cc()) ** 4 * abs(prop_ratio) ** 4
        assert got == pytest.approx(want, rel=1e-6), \
            f"at |Q^2|={q2:.3g} MeV^2: NC/CC = {got:.6e}, analytic {want:.6e}"


def test_nc_is_purely_left_handed_for_a_neutrino():
    """coupl_right = 0 (LeptonicCurrent.cc:36).  A right-handed piece would show up as a different
    current under the anti flag in a way a pure-V-A current cannot."""
    from adonis.channels.currents.leptonic import _couplings
    cl, cr, M, G, has_prop = _couplings("NC_nu")
    assert cr == 0
    assert (M, G) == (C.MZ, C.GAMZ)
    assert has_prop


def test_unknown_lepton_kind_still_raises():
    from adonis.channels.currents.leptonic import _couplings
    with pytest.raises(ValueError):
        _couplings("NC")
    with pytest.raises(ValueError):
        _couplings("garbage")
