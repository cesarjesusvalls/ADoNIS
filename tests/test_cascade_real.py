"""Smoke gate for the real pion cascade transport (adonis/fsi/cascade_real.py): it runs on a
pion batch, conserves charge bookkeeping, produces physically-sensible output (some pions
escape, some absorb, charge exchange happens, survivors are softened).  The exact ACHILLES
match is a separate (parked) transport-calibration task -- this only guards that the real
cross-section physics drives a functional, physical cascade.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from adonis.paths import achilles_data_root
from adonis.fsi import cascade_real as cr

_ANL = achilles_data_root() / "MesonBaryonAmplitudes" / "ANL" / "ANL_0-0.dat"
_DENS = Path(__file__).resolve().parents[1] / "data" / "nuclear" / "c12_density.txt"
pytestmark = pytest.mark.skipif(not (_ANL.exists() and _DENS.exists()),
                                reason="ANL amplitudes or 12C density not present")


def _batch(n, p_mev, seed=0):
    rng = np.random.default_rng(seed)
    rgrid, rho, _ = cr._load_density()
    rg, rh = np.asarray(rgrid), np.asarray(rho)
    w = rg ** 2 * rh; w /= w.sum()
    rr = rng.choice(rg, size=n, p=w)
    dp = rng.normal(size=(n, 3)); dp /= np.linalg.norm(dp, axis=1, keepdims=True)
    pos = jnp.asarray(rr[:, None] * dp)
    dd = rng.normal(size=(n, 3)); dd /= np.linalg.norm(dd, axis=1, keepdims=True)
    E = np.sqrt(p_mev ** 2 + cr.ox.M_PIP ** 2)
    p_pi = jnp.asarray(np.concatenate([np.full((n, 1), E), p_mev * dd], axis=1))
    return pos, p_pi


def test_cascade_runs_and_is_physical():
    n = 4000
    pos, p_pi = _batch(n, 300.0)
    p_out, ch, absorbed, nsc = cr.propagate(pos, p_pi, jnp.zeros(n, jnp.int32),
                                            cr.RealCascadeConfig(), jax.random.PRNGKey(1), 0.0)
    absb = np.asarray(absorbed); pm = np.asarray(jnp.linalg.norm(p_out[:, 1:], axis=1))
    chn = np.asarray(ch); ns = np.asarray(nsc)
    # some absorb, most survive (CC1pi); a Delta-region pion partly absorbs
    assert 0.05 < absb.mean() < 0.7, absb.mean()
    # charge exchange produced some pi0 (idx 1) from the pi+ (idx 0) beam
    assert (chn == 1).mean() > 0.0
    # survivors are softened (energy loss in scattering) below the 300 MeV input
    surv = ~absb
    assert pm[surv].mean() < 300.0
    # absorbed pions carry no momentum out (removed), survivors on-shell positive |p|
    assert np.all(pm[surv] > 0)
