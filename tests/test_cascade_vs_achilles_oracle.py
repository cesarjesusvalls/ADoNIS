"""Fig 3 gate: the differentiable ADoNIS cascade reproduces the ACHILLES Virtual-Resonances
pion-carbon transparency oracle.

Both use the SAME untuned in-medium cross sections (Oset absorption + ANL-Osaka DCC scatter);
this checks the TRANSPORT.  The key, normalisation-independent physics result is the absorption
FRACTION (sigma_abs/sigma_reaction), which must agree at the Delta.  The absolute normalisation
differs by the characterised continuum-transport factor (~1.33x, smooth rho vs ACHILLES's
correlated QMC configs); we bound it loosely.  See memory cascade-absorption-fraction-029.
"""
import numpy as np
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACH = ROOT / "data" / "oracle" / "cascade_pip_c12_virt_abs.csv"
ADO = ROOT / "data" / "oracle" / "cascade_pip_c12_virt_abs_adonis.csv"
pytestmark = pytest.mark.skipif(not (ACH.exists() and ADO.exists()),
                                reason="cascade oracle CSVs not generated (need achilles:cascade)")


def _load():
    a = np.loadtxt(ACH); d = np.loadtxt(ADO)
    assert np.allclose(a[:, 0], d[:, 0])
    return a[:, 0], a[:, 1], a[:, 2], d[:, 1], d[:, 2]   # p, sr_ach, sa_ach, sr_ado, sa_ado


def test_absorption_fraction_agrees():
    """p-averaged absorption fraction: ADoNIS and ACHILLES both ~0.3 (the model prediction)."""
    p, sra, saa, srd, sad = _load()
    fa = saa.sum() / sra.sum(); fd = sad.sum() / srd.sum()
    assert 0.27 <= fa <= 0.34, fa                          # ACHILLES VirtRes ~0.31
    assert 0.27 <= fd <= 0.40, fd                          # ADoNIS ~0.35
    assert abs(fa - fd) < 0.07, (fa, fd)                   # agree to within ~0.05


def test_delta_peak_absolute_norm():
    """ABSOLUTE sigma(p) agreement in the Delta region after the oracle normalisation fix
    (radius=10 fm -> piR2=3141.6 mb; the discrete-Glauber ADoNIS cascade replaces the continuum,
    which was ~2x low for missing the transverse exp(-pi b^2/sigma) reach).  ADoNIS/ACHILLES is
    now ~0.85-1.20 absolute (within the oracle's per-bin statistics)."""
    p, sra, saa, srd, sad = _load()
    d = (p >= 200) & (p <= 360)
    assert 0.85 < np.mean(srd[d] / sra[d]) < 1.20, np.mean(srd[d] / sra[d])
    assert 0.85 < np.mean(sad[d] / saa[d]) < 1.20, np.mean(sad[d] / saa[d])


def test_both_peak_at_delta():
    """Reaction and absorption sigma are maximal in the Delta region: the Delta-window max is
    within 15% of the global max (ADoNIS abs has a noisy wing bump but the peak is at the Delta)."""
    p, sra, saa, srd, sad = _load()
    d = (p >= 220) & (p <= 360)
    for y in (sra, saa, srd, sad):
        assert y[d].max() >= 0.85 * y.max(), (p[int(np.argmax(y))], y.max(), y[d].max())
