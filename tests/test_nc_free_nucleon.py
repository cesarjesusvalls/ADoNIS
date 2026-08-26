"""G5(2) + G6(2) -- the ABSOLUTE free-nucleon oracle gates, in nb, against ACHILLES.

These are the gates §0 of docs/nc_implementation_plan.md calls non-negotiable, because a wrong
absolute normalisation is invisible to every shape/ratio comparison.  That is not hypothetical here:
it is the `e3319a5` failure mode (a missing photon 1/q^2 made the (e,e') scale ~1e15 too large and
the ratio gate could not see it), and during this work the RES gate caught a genuinely wrong I=1/2
isospin treatment that agreed with ACHILLES to 3.5% on the totals.

Two rules, both load-bearing:
  * NO BRIDGE CONSTANT.  Absolute nb on both sides or the gate means nothing.
  * mean and spread are reported SEPARATELY.  A flat offset is a coupling / _NORM error; a slope or
    a channel-to-channel scatter is a structural one.  A single averaged number hides the difference.

Reference: configs/achilles/run_freenucleon_nc_{res,qe}_{H,N}.yml, 20k events, E_nu = 1.5 GeV, free
nucleon, cascade off.  proc 250/251 (QE), 451/452 (RES).

Marked slow: each gate is a few 1e5-sample MC integrations.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.slow

E_NU = 1500.0

ACH_RES = {(+1, 111): 1.2133283286710014e-06,
           (+1, 211): 7.257765372213459e-07,
           (-1, 111): 1.2202023330896117e-06,
           (-1, -211): 7.414840641825235e-07}
ACH_QE = {True: 1.436153729102368e-06,
          False: 2.0569151722972532e-06}

MEAN_BAND = 0.02
SPREAD_BAND = 0.01


def _mean_spread(ratios):
    r = np.asarray(ratios)
    return float(r.mean()), float(r.std() / r.mean())


def test_g5_2_nc_res_absolute_sigma_on_a_free_nucleon():
    from adonis.channels.res_nc import sigma_free_nucleon_nc, NC_RES_CHANNELS
    ratios = []
    for ch in NC_RES_CHANNELS:
        ado, _err = sigma_free_nucleon_nc(E_NU, ch, n=40_000, seed=1)
        assert ado > 0, f"channel {ch[1]},{ch[2]} produced no cross section"
        ratios.append(ACH_RES[(ch[1], ch[2])] / ado)
    mean, spread = _mean_spread(ratios)
    assert abs(mean - 1.0) < MEAN_BAND, f"NC RES absolute scale off: mean(ACH/ADO)={mean:.4f}"
    assert spread < SPREAD_BAND, f"NC RES channel scatter {spread*100:.2f}% -> structural error"


def test_g6_2_nc_qe_absolute_sigma_on_a_free_nucleon():
    from adonis.channels.qe_nc import sigma_free_nucleon_nc_qe
    ratios = []
    for is_p in (True, False):
        ado, _err = sigma_free_nucleon_nc_qe(E_NU, is_p, n=100_000, seed=1, use_achilles_nc_coupling=True)
        assert ado > 0
        ratios.append(ACH_QE[is_p] / ado)
    mean, spread = _mean_spread(ratios)
    assert abs(mean - 1.0) < MEAN_BAND, f"NC QE absolute scale off: mean(ACH/ADO)={mean:.4f}"
    assert spread < SPREAD_BAND, f"NC QE p/n scatter {spread*100:.2f}% -> isospin structure error"


def test_nc_qe_neutron_over_proton_ratio_is_convention_free():
    """The sharpest isospin check available, and it needs NO normalisation convention: every constant
    in the chain cancels.  The proton's NC vector coupling carries (1/2 - 2 sin^2 th_W) ~ 0.037 while
    the neutron's carries -1/2, so the proton is axial-dominated and the neutron is not.  Getting the
    couplings the wrong way round gives ~1/1.43, which no overall scale can rescue."""
    from adonis.channels.qe_nc import sigma_free_nucleon_nc_qe
    p, _ = sigma_free_nucleon_nc_qe(E_NU, True, n=100_000, seed=1, use_achilles_nc_coupling=True)
    n, _ = sigma_free_nucleon_nc_qe(E_NU, False, n=100_000, seed=1, use_achilles_nc_coupling=True)
    want = ACH_QE[False] / ACH_QE[True]
    assert n / p == pytest.approx(want, rel=0.01), f"n/p = {n/p:.4f}, ACHILLES {want:.4f}"


def test_the_rotated_isovector_form_beats_the_raw_vec_one():
    """Regression guard for the D2 resolution.

    The raw-`vec` I=1/2 transcription reproduced the ACHILLES TOTALS to 3.5% -- easy to accept -- but
    broke ACHILLES's near-exact p/n mirror symmetry, giving a 6.8% channel spread against 0.5% for
    the rotated form.  This test fails if anyone reverts to it, and it fails on the SPREAD, which is
    the statistic that carries the information.
    """
    import adonis.channels.dcc.differential as D
    from adonis.channels.res_nc import sigma_free_nucleon_nc, NC_RES_CHANNELS

    def spread_now():
        rs = [ACH_RES[(c[1], c[2])] / sigma_free_nucleon_nc(E_NU, c, n=40_000, seed=1)[0]
              for c in NC_RES_CHANNELS]
        return _mean_spread(rs)[1]

    good = spread_now()
    old = D.NC_ISV_SIGN
    try:
        D.NC_ISV_SIGN = 0.0
        degraded = spread_now()
    finally:
        D.NC_ISV_SIGN = old
    assert good < degraded, (f"the isoscalar term no longer improves the channel spread "
                             f"({good*100:.2f}% vs {degraded*100:.2f}%) -- the I=1/2 rotation is wrong")
