"""Phase F gate: Oset pion absorption self-energy (transcription + differentiability).

Faithful transcription of ACHILLES OsetCrossSections.cc: the absorption self-energy peaks
in the Delta region, the 3N piece is clamped >=0 at low T_pi, and Im Sigma_abs is exactly
differentiable in the C_A2/C_A3/C_Q knobs (the paper's tunable FSI absorption parameters).
"""
import numpy as np
import jax

from tests.reference.oset_selfenergy import (absorption_self_energy, self_energy_abs_NNN, self_energy_qe,
                                 C_A2, C_Q)


def test_absorption_peaks_in_delta_region():
    T = np.array([20, 50, 85, 120, 165, 200, 250, 300.])
    s = np.array([float(absorption_self_energy(t)) for t in T])
    assert np.all(s > 0)
    assert 150 <= T[int(np.argmax(s))] <= 260            # absorption peaks near the Delta


def test_abs_NNN_clamped_low_energy():
    """3N absorption is 0 below ~50-80 MeV (the C_A3 quadratic goes negative there)."""
    assert float(self_energy_abs_NNN(30.0)) == 0.0
    assert float(self_energy_abs_NNN(200.0)) > 0.0


def test_oset_differentiable_in_knobs():
    """Im Sigma_abs is exactly differentiable in the absorption-strength knob (closure)."""
    def total(scale):
        return absorption_self_energy(120.0, c_a2=tuple(scale * c for c in C_A2))
    g = float(jax.grad(total)(1.0))
    fd = (float(total(1.001)) - float(total(0.999))) / 0.002
    assert abs(g - fd) / abs(fd) < 1e-3, (g, fd)
    # QE channel differentiable in C_Q too
    def qe(scale):
        return self_energy_qe(120.0, c_q=tuple(scale * c for c in C_Q))
    assert np.isfinite(float(jax.grad(qe)(1.0)))
