"""Phase E1 gate: ANL-Osaka meson-baryon partial-wave cross section.

The pi+ p -> pi+ p total cross section (pure I=3/2) from the ANL partial-wave amplitudes
must reproduce the known Delta(1232) resonance: peak ~200 mb near W=1232 MeV. This
self-validates the parser + the partial-wave sum against the physical piN cross section
(the forward model IS the ANL-Osaka model; no cascade binary needed). See docs/phases/PHASE_E.md.
"""
import numpy as np
import pytest

from adonis.paths import achilles_data_root
from adonis.fsi.mb.anl_xsec import pip_p_total

_ANL = achilles_data_root() / "MesonBaryonAmplitudes" / "ANL" / "ANL_0-0.dat"
pytestmark = pytest.mark.skipif(not _ANL.exists(),
                                reason=f"ANL meson-baryon table not present ({_ANL})")


def test_pip_p_delta_peak():
    W, sig = pip_p_total()
    m = (W >= 1150) & (W <= 1350)
    ipk = int(np.argmax(sig[m]))
    Wpk, spk = W[m][ipk], sig[m][ipk]
    assert 1200 <= Wpk <= 1245, Wpk                 # Delta(1232) position
    assert 185 <= spk <= 220, spk                   # peak ~200 mb
    # well above and below the resonance the cross section is much smaller
    assert np.interp(1500, W, sig) < 30, "should fall off above the Delta"
    assert np.interp(1130, W, sig) < spk, "rises into the Delta"


def test_sigma_norm_linear():
    """The sigma-normalisation knob is exactly linear (the differentiable handle)."""
    _, s1 = pip_p_total(norm=1.0)
    _, s2 = pip_p_total(norm=2.0)
    assert np.allclose(s2, 2.0 * s1)


def test_pim_p_isospin_ratio():
    """pi- p elastic reproduces the Delta with the textbook 9:1 isospin ratio vs pi+ p
    (the I=3/2 amplitude enters with weight 1 for pi+p, 1/3 for pi-p elastic -> |1|^2/|1/3|^2=9)."""
    from adonis.fsi.mb.anl_xsec import pip_p_total, pim_p_elastic
    W, sp = pip_p_total()
    _, sm = pim_p_elastic()
    m = (W >= 1150) & (W <= 1350)
    assert 15 < sm[m].max() < 35, sm[m].max()          # pi-p elastic Delta peak ~22 mb
    assert 8.0 < sp[m].max() / sm[m].max() < 10.0       # the 9:1 ratio
