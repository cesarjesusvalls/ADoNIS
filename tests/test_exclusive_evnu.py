"""Phase I gate: exclusive electron vs neutrino single-pion production at matched kinematics.

Both at E=2.222 GeV on 12C with the same forward lepton-angle acceptance, so the difference
between the exclusive pion observables is the current structure (EM vector vs CC vector+axial).
Validates against the ACHILLES RES oracle (data/oracle/exclusive_evnu_c12_{cos,W}.csv, columns
center, e_shape, e_err, nu_shape, nu_err):
  (1) the physical e/nu DIFFERENCE -- the neutrino is more Delta-peaked in W and less
      forward-peaked in cos(theta*) than the electron;
  (2) ADoNIS reproduces BOTH currents' distributions (chi2/ndf model-vs-oracle).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import pytest

from adonis.primary.dcc.channel import sample_final_state, weight_from_sample, assemble_event
from adonis.primary.dcc.structure import HadronStructure, CC_CHANNELS, EM_CHANNELS
from adonis.core.params import PhysicsParams
from adonis import observables as obs
from adonis.core.validation import chi2_ndf

_DIR = Path(__file__).resolve().parents[1] / "data" / "oracle"
_CSV_COS = _DIR / "exclusive_evnu_c12_cos.csv"
_CSV_W = _DIR / "exclusive_evnu_c12_W.csv"
pytestmark = pytest.mark.skipif(not _CSV_COS.exists(), reason="exclusive e/nu oracle CSVs not present")
if not (_CSV_COS.exists() and _CSV_W.exists()):          # the loadtxt below runs at import -> skip at COLLECTION
    pytest.skip("exclusive e/nu oracle CSVs absent (regenerate via the docker oracle path)", allow_module_level=True)
N = 8_000 if os.environ.get("ADONIS_CI_FAST") else 40_000
COS_BINS, COS_RNG = 16, (-1.0, 1.0)
W_BINS, W_RNG = 20, (1080.0, 1700.0)


def _model(channels, current, m_lep=0.0):
    hs = HadronStructure(channels=channels, n_theta=12, n_phi=12, spline=False)
    S = sample_final_state(jax.random.PRNGKey(5), N, hs=hs, current=current, e_nu=2222.0,
                           ep_lo=50.0, ep_hi=2200.0, theta_max_deg=17.0, theta_min_deg=14.0,
                           m_lep=m_lep)
    w, LWc = weight_from_sample(PhysicsParams(), S, use_spline=False)
    ev = assemble_event(S, w, LWc)
    return np.asarray(obs.cos_theta_star(ev)), np.asarray(obs.W(ev)), np.asarray(w)


def _nh(x, w, bins, rng):
    h, _ = np.histogram(x, bins=bins, range=rng, weights=w)
    h2, _ = np.histogram(x, bins=bins, range=rng, weights=w ** 2)
    s = h.sum(); return h / s, np.sqrt(h2) / s


# 7-col CSVs: center, {e, nue, numu} x {shape, err}  -> shape cols 1, 3, 5
oc = np.loadtxt(_CSV_COS)
ow = np.loadtxt(_CSV_W)
mcte, mWe, mwe = _model(EM_CHANNELS, "EM")
mctn, mWn, mwn = _model(CC_CHANNELS, "CC")
mctm, mWm, mwm = _model(CC_CHANNELS, "CC", m_lep=105.658)


def _fb(c, h):
    return (h[c > 0].sum() - h[c < 0].sum()) / h.sum()


def test_evnu_physical_difference():
    """The axial current makes BOTH neutrinos more Delta-peaked (lower <W>) and less
    forward-peaked (smaller cos(theta*) F/B asymmetry) than the electron -- in the ACHILLES
    oracle and the ADoNIS model.  nu_e and nu_mu agree closely (the lepton mass is a small
    kinematic shift on the same V-A structure)."""
    for col in (3, 5):                                          # nu_e, nu_mu vs e (col 1)
        assert _fb(oc[:, 0], oc[:, 1]) > _fb(oc[:, 0], oc[:, col]) + 0.02
        assert np.average(ow[:, 0], weights=ow[:, 1]) > np.average(ow[:, 0], weights=ow[:, col]) + 25
    mhe = _nh(mcte, mwe, COS_BINS, COS_RNG)[0]; mhn = _nh(mctn, mwn, COS_BINS, COS_RNG)[0]
    assert _fb(oc[:, 0], mhe) > _fb(oc[:, 0], mhn) + 0.02       # model: e more forward than nu
    assert np.average(mWe, weights=mwe) > np.average(mWn, weights=mwn) + 40
    # nu_e and nu_mu are close (lepton-mass effect small)
    assert abs(np.average(mWn, weights=mwn) - np.average(mWm, weights=mwm)) < 60


def test_evnu_model_reproduces_oracle():
    """ADoNIS reproduces all three lepton channels' exclusive distributions vs ACHILLES."""
    cases = [
        (ow, mWe, mwe, W_BINS, W_RNG, 1, 6.0),        # e: W
        (ow, mWn, mwn, W_BINS, W_RNG, 3, 6.0),        # nu_e: W
        (ow, mWm, mwm, W_BINS, W_RNG, 5, 8.0),        # nu_mu: W
        (oc, mcte, mwe, COS_BINS, COS_RNG, 1, 12.0),  # e: cos*
        (oc, mctn, mwn, COS_BINS, COS_RNG, 3, 12.0),  # nu_e: cos*
        (oc, mctm, mwm, COS_BINS, COS_RNG, 5, 12.0),  # nu_mu: cos*
    ]
    for o, x_m, w_m, bins, rng, col, tol in cases:
        ho, eo = o[:, col], o[:, col + 1]
        hm, em = _nh(x_m, w_m, bins, rng)        # include the model MC error in the chi2
        c2, ndf = chi2_ndf(hm, em, ho, eo, floor=0.05)
        assert c2 / max(ndf, 1) < tol, (c2 / max(ndf, 1), col, bins)
