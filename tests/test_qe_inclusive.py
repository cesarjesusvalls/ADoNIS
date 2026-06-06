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
