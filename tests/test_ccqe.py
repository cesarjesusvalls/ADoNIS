"""Phase B1 gates: free-nucleon CCQE cross section (Llewellyn-Smith).

The analytic LS dsigma/dQ^2 with the Kelly vector FFs + dipole axial FF gives a genuinely
PHYSICAL CCQE sigma(E_nu) -- the absolute normalisation comes from G_F^2 cos^2(theta_c),
no calibration. Gates: the known plateau scale, the rise+plateau shape, and the exact
dsigma/dM_A closure. The ACHILLES QE oracle bridge lands with test_ccqe_sigma_oracle.
"""
import numpy as np
import jax
import pytest
from pathlib import Path

from adonis.primary.qe.llewellyn_smith import ccqe_sigma, ccqe_sigma_vs_enu

# the ACHILLES CCQE oracle CSV was removed in the repo cleanup (regenerable via the docker oracle path).
_ccqe_oracle_skip = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data" / "oracle" / "freenucleon_ccqe_sigma.csv").exists(),
    reason="freenucleon_ccqe_sigma.csv oracle absent (regenerate via the docker oracle path)")


def test_ccqe_plateau_scale():
    """CCQE sigma plateaus at the known free-nucleon scale ~1.0 x10^-38 cm^2 (nu_mu n->mu p)."""
    sig_nb = float(ccqe_sigma(2.0))                 # nb at 2 GeV (on the plateau)
    sigma_1e38 = sig_nb * 1e5                        # nb -> 10^-38 cm^2
    assert 0.8 < sigma_1e38 < 1.3, sigma_1e38


def test_ccqe_closure_dMA():
    """d(sigma)/dM_A: autodiff == central finite difference (analytic, exact)."""
    g = float(jax.grad(lambda MA: ccqe_sigma(1.0, MA))(1.0))
    eps = 1e-3
    fd = (float(ccqe_sigma(1.0, 1.0 + eps)) - float(ccqe_sigma(1.0, 1.0 - eps))) / (2 * eps)
    assert abs(g - fd) / abs(fd) < 1e-3, (g, fd)
    assert g > 0                                    # larger M_A -> larger sigma


def test_ccqe_rises_and_plateaus():
    """sigma(E_nu) rises from threshold to a high-energy plateau."""
    E = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    sig = ccqe_sigma_vs_enu(E)
    assert np.all(np.diff(sig[:4]) > 0), sig        # rising up to ~1 GeV
    fr = np.diff(sig) / sig[:-1]
    assert fr[-1] < fr[0], fr                        # flattening at high E


@_ccqe_oracle_skip
def test_ccqe_sigma_oracle():
    """B1 oracle: free-nucleon CCQE sigma(E_nu) vs ACHILLES QE_Spectral_Func, ABSOLUTE
    (both in nb from first principles -- no bridging constant). model/ACHILLES within ~3%."""
    from adonis.primary.qe.llewellyn_smith import ccqe_sigma_oracle
    r = ccqe_sigma_oracle()
    assert not r.skipped
    assert r.passed, r.detail
    assert r.metrics["rel_max"] < 0.06
