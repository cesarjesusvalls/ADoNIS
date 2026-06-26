"""Phase E1 gate: ANL-Osaka meson-baryon partial-wave cross section.

The pi+ p -> pi+ p total cross section (pure I=3/2) from the ANL partial-wave amplitudes
must reproduce the known Delta(1232) resonance: peak ~200 mb near W=1232 MeV. This
self-validates the parser + the partial-wave sum against the physical piN cross section
(the forward model IS the ANL-Osaka model; no cascade binary needed). See docs/phases/PHASE_E.md.
"""
import numpy as np
import pytest

from adonis.io import achilles_data_root
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


def test_cex_isospin_921_ratio():
    """The three piN channels at the Delta(1232) reproduce the textbook 9:2:1 isospin ratio
    pi+p : (pi-p->pi0n charge exchange) : pi-p elastic -- a pure I=3/2 prediction (weights
    1, sqrt(2)/3, 1/3 -> 1 : 2/9 : 1/9 = 9:2:1)."""
    from adonis.fsi.mb.anl_xsec import pip_p_total, pim_p_elastic, pim_p_cex
    W, sp = pip_p_total()
    _, sc = pim_p_cex()
    _, se = pim_p_elastic()
    m = (W >= 1150) & (W <= 1350)
    pk = lambda s: s[m].max()
    assert 1200 <= W[m][int(np.argmax(sc[m]))] <= 1245      # cex peaks at the Delta
    assert 8.5 < pk(sp) / pk(se) < 10.0                     # pi+p : elastic = 9
    assert 1.8 < pk(sc) / pk(se) < 2.6                      # cex   : elastic = 2 (+ I=1/2 fill-in)


def test_delta_angular_distribution():
    """pi+ p dsigma/dOmega at the Delta (W=1232) reproduces the P33 1 + 3 cos^2(theta)
    shape (forward/90deg ~ 4), with the physical S-P interference forward-backward asymmetry."""
    import numpy as np
    from adonis.fsi.mb.anl_xsec import dsigma_dOmega
    c = np.linspace(-1, 1, 41)
    d = dsigma_dOmega(1232.0, c)
    fwd, ninety = d[-1], d[len(c) // 2]
    assert 3.0 < fwd / ninety < 5.5, fwd / ninety        # ~4 for 1+3cos^2
    # symmetric-ish but forward-peaked; minimum near 90 deg
    assert d[len(c) // 2] < d[0] and d[len(c) // 2] < d[-1]


def test_eta_klambda_production():
    """piN -> etaN (N(1535)) and piN -> KLambda production channels (ANL_0-1, 0-2): the eta
    channel peaks at the N(1535) (~1535 MeV) which sits at the eta-N threshold; K Lambda turns
    on above its higher threshold."""
    import numpy as np
    from adonis.io import achilles_data_root
    import pytest
    if not (achilles_data_root() / "MesonBaryonAmplitudes" / "ANL" / "ANL_0-1.dat").exists():
        pytest.skip("ANL_0-1 (etaN) amplitudes not present")
    from adonis.fsi.mb.anl_xsec import eta_production, klambda_production
    We, se = eta_production(); Wk, sk = klambda_production()
    me = (We >= 1480) & (We <= 1800)
    assert 1510 <= We[me][se[me].argmax()] <= 1580           # N(1535) peak
    assert 1.5 < se[me].max() < 5.0                          # ~2-3 mb (eta production)
    assert np.interp(1480, We, se) < 0.5 * se[me].max()      # rises from the eta-N threshold
    # K Lambda turns on at a higher W than eta
    assert np.interp(1500, Wk, sk) < 0.1                     # below K-Lambda threshold
