"""Phase B2 gate: PWIA quasi-elastic inclusive (e,e') response on 12C.

The spectral-function fold gives a QE peak near the relativistic quasi-free position
(omega ~ sqrt(q^2+M^2)-M + E_b) with a Fermi-motion width (~ k_F q/M). The QE component
of Fig 1. See docs/phases/PHASE_B.md.
"""
import numpy as np
from adonis.nuclear.qe_inclusive import qe_dsigma_domega


def test_qe_peak_and_fermi_width():
    w = np.linspace(20, 400, 80)
    d = qe_dsigma_domega(2222.0, 15.541, w)
    assert np.all(d >= 0)
    wpk = w[int(np.argmax(d))]
    assert 180 <= wpk <= 280, wpk                 # relativistic QE peak region (+ binding/skew)
    half = d.max() / 2
    above = w[d > half]
    fwhm = above[-1] - above[0]
    assert 100 <= fwhm <= 220, fwhm               # Fermi-motion width ~ k_F q/M


import pytest
from adonis.paths import achilles_data_root
_PKE40 = achilles_data_root() / "Spectral_Functions" / "pke40p_tot.data"


@pytest.mark.skipif(not _PKE40.exists(), reason="40Ar spectral function not present")
def test_ar40_qe_broader_than_c12():
    """B3: 40Ar QE peak is broader than 12C (higher Fermi momentum, k_F 250 vs 225)."""
    w = np.linspace(20, 400, 80)
    dC = qe_dsigma_domega(2222.0, 15.541, w, sf="pke12p_tot.data", n_p=6, n_n=6)
    dAr = qe_dsigma_domega(2222.0, 15.541, w, sf="pke40p_tot.data", n_p=18, n_n=22)
    def fwhm(d):
        h = d.max() / 2; a = w[d > h]; return a[-1] - a[0]
    assert fwhm(dAr) > fwhm(dC), (fwhm(dAr), fwhm(dC))


from pathlib import Path as _P
_EE_CSV = _P(__file__).resolve().parents[1] / "data" / "oracle" / "inclusive_ee_12C_res.csv"


_QE_CSV = _P(__file__).resolve().parents[1] / "data" / "oracle" / "inclusive_ee_12C_qe.csv"


@pytest.mark.skipif(not _EE_CSV.exists(), reason="ACHILLES RES (e,e') oracle CSV not present")
def test_onepi_bump_matches_achilles_res():
    """B2 oracle (chi2): the model 1pi/Delta dsigma/domega vs the ACHILLES RES_Spectral_Func
    inclusive (e,e') generator (E=2.222 GeV, theta~15.5 deg).  After the removal-energy
    correction the bump centroids agree to ~1 MeV and the shape chi2/ndf ~15 (residuals only
    at the rising edge + the high-omega resonance tail the Delta-only EM fold omits)."""
    from adonis.nuclear.inclusive_1pi import onepi_dsigma_domega
    from adonis.core.validation import chi2_ndf
    d = np.loadtxt(_EE_CSV)
    ow, osh, oerr = d[:, 0], d[:, 1], d[:, 2]
    m = onepi_dsigma_domega(2222.0, 15.541, ow, "pke12p_tot.data", 12)
    chi2, ndf = chi2_ndf(m, np.zeros_like(m), osh, oerr, floor=0.05)

    def centroid(wv, yv, lo=380.0, hi=680.0):
        k = (wv >= lo) & (wv <= hi)
        return float(np.sum(wv[k] * yv[k]) / np.sum(yv[k]))

    assert abs(centroid(ow, m) - centroid(ow, osh)) < 20.0       # bump region agrees (binding fix)
    assert chi2 / max(ndf, 1) < 20.0, chi2 / max(ndf, 1)         # shape chi2/ndf gate


@pytest.mark.skipif(not _QE_CSV.exists(), reason="ACHILLES QE (e,e') oracle CSV not present")
def test_qe_peak_vs_achilles():
    """B2 oracle (chi2): the model QE dsigma/domega vs the ACHILLES QE_Spectral_Func generator
    (same spectral function, pke12p).  With the proper longitudinal/transverse Rosenbluth
    separation (v_L R_L + v_T R_T) the peak sits at 208 vs the ACHILLES 192 MeV and chi2/ndf
    ~9 (was ~28 with the isotropic G_E^2+tau G_M^2 combination).  The residual ~16 MeV is the
    deeper off-shell single-nucleon cross-section (de Forest cc1/cc2) treatment."""
    from adonis.core.validation import chi2_ndf
    d = np.loadtxt(_QE_CSV)
    ow, osh, oerr = d[:, 0], d[:, 1], d[:, 2]
    m = qe_dsigma_domega(2222.0, 15.541, ow, "pke12p_tot.data", 6, 6)
    chi2, ndf = chi2_ndf(m, np.zeros_like(m), osh, oerr, floor=0.05)
    assert abs(ow[int(np.argmax(m))] - ow[int(np.argmax(osh))]) < 30.0   # peak offset bounded
    assert chi2 / max(ndf, 1) < 15.0, chi2 / max(ndf, 1)                 # v_L/v_T Rosenbluth fix


def test_inclusive_two_peak_structure():
    """Full inclusive (e,e') has the QE peak AND a 1pi/Delta bump above it (Fig 1 structure):
    the Delta bump sits near omega ~ (m_Delta^2 - M^2 + Q^2)/(2M), above the QE peak."""
    from adonis.nuclear.inclusive_1pi import onepi_dsigma_domega
    w = np.linspace(20, 500, 60)
    qe = qe_dsigma_domega(2222., 15.541, w, "pke12p_tot.data", 6, 6)
    pi = onepi_dsigma_domega(2222., 15.541, w, "pke12p_tot.data", 12)
    assert np.all(pi >= 0)
    w_qe, w_pi = w[int(np.argmax(qe))], w[int(np.argmax(pi))]
    assert w_pi > w_qe + 100, (w_qe, w_pi)              # Delta bump well above the QE peak
    assert 380 <= w_pi <= 520, w_pi                     # near the expected Delta position
