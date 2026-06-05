"""Phase A3 gates: free-nucleon single-pion σ(E_ν).

Closure (differentiability of σ wrt M_A) + forward physics sanity (channel ratios
match the DCC isospin structure; σ rises from threshold to a high-energy plateau).
The absolute-normalisation / ACHILLES-oracle / ANL-BNL gates land with A3.4–A3.5
(see docs/phases/PHASE_A3.md).
"""
import os
from pathlib import Path

import numpy as np
import jax

from adonis.params import PhysicsParams, GenConfig
from adonis.nuclear.free import FreeNucleon
from adonis.primary.dcc.sigma_enu import (sigma_vs_enu, sigma_vs_enu_nb, sigma_channels_at,
                                          dsigma_dMA_closure, freenucleon_sigma_oracle,
                                          CM2_1E38_PER_NB, em_sigma_channels_at,
                                          em_dsigma_dpw_closure)

N = 8_000 if os.environ.get("ADONIS_CI_FAST") else 40_000
N_ORACLE = 100_000 if os.environ.get("ADONIS_CI_FAST") else 150_000
_CSV = Path(__file__).resolve().parent.parent / "data" / "oracle"


def test_sigma_closure_dMA():
    """d(total σ)/dM_A: autodiff == central finite difference (kind-1 exact gradient)."""
    r = dsigma_dMA_closure(key=jax.random.PRNGKey(0), n=N)
    assert not r.skipped
    assert r.passed, r.detail
    assert r.metrics["rel_err"] < 1e-3


def test_free_nucleon_channel_ratios():
    """Free-nucleon channel split reproduces the DCC isospin fractions (STATUS.md:
    p→pπ⁺ ~0.66, n→pπ⁰ ~0.19, n→nπ⁺ ~0.15 in the Δ region)."""
    _, sc = sigma_channels_at(PhysicsParams(), jax.random.PRNGKey(1), e_nu=1000.0, n=N)
    sc = np.asarray(sc)
    f = sc / sc.sum()
    assert f[2] > f[0] > f[1], f               # p→pπ⁺ largest, n→nπ⁺ smallest
    assert 0.60 < f[2] < 0.75, f               # Δ⁺⁺-like dominance
    assert 0.12 < f[0] < 0.26, f               # n→pπ⁰
    assert f.sum() == np.float64(f.sum())      # finite


def test_sigma_rises_and_plateaus():
    """σ(E_ν) increases monotonically over the rise and is near-flat at high E_ν."""
    E = [500., 700., 1000., 1500., 2000.]
    tot, _ = sigma_vs_enu(PhysicsParams(), jax.random.PRNGKey(2), E, n=N)
    assert np.all(np.diff(tot) > 0), tot       # monotonic rise across this range
    assert tot[0] > 0
    # high-energy flattening: last step's fractional increase < the first step's
    fr = np.diff(tot) / tot[:-1]
    assert fr[-1] < fr[0], fr


def test_freenucleon_sigma_oracle():
    """A3.4 oracle: σ(E_ν) for all 3 CC channels vs ACHILLES on stationary nucleons
    (1H + 1N), bridged by a single universal constant. Tight residual + c-spread."""
    r = freenucleon_sigma_oracle(key=jax.random.PRNGKey(11), n=N_ORACLE)
    assert not r.skipped
    assert r.passed, r.detail
    assert r.metrics["rel_max"] < 0.06
    assert r.metrics["c_spread_std"] < 0.03


M_MU = 105.658


def test_freenucleon_sigma_oracle_muon():
    """A3.3 oracle: the SAME vs ACHILLES with the muon mass (ν_μ → μ⁻). Validates that
    the muon-mass kinematics (|p'| in both the lepton momentum and the leptonic
    phase-space factor) are faithful."""
    r = freenucleon_sigma_oracle(csv=_CSV / "freenucleon_numu_sigma.csv",
                                 key=jax.random.PRNGKey(11), n=N_ORACLE, m_lep=M_MU)
    assert not r.skipped
    assert r.passed, r.detail


def test_physical_absolute_scale():
    """A3.5 (units): with the calibrated model→nb constant, the free-proton CC1π⁺ cross
    section sits at the known physical scale ~0.6–0.9 ×10⁻³⁸ cm² at the high-energy
    plateau (matching the paper's Fig-2 axis)."""
    _, perch = sigma_vs_enu_nb(PhysicsParams(), jax.random.PRNGKey(11), [2000.], n=N_ORACLE)
    sigma_ppi_cm2 = float(perch[0, 2]) * CM2_1E38_PER_NB     # p->p pi+ in 10^-38 cm^2
    assert 0.5 < sigma_ppi_cm2 < 1.0, sigma_ppi_cm2


def test_muon_electron_constant_consistency():
    """The model→ACHILLES bridging constant is the SAME for ν_e (massless) and ν_μ
    (massive) — they share the CC coupling, so the muon mass must be purely kinematic.
    Agreement of c_μ and c_e is a strong check of the muon-mass implementation."""
    n = max(N_ORACLE, 80_000)
    r_e = freenucleon_sigma_oracle(csv=_CSV / "freenucleon_nue_sigma.csv",
                                   key=jax.random.PRNGKey(11), n=n, m_lep=0.0)
    r_mu = freenucleon_sigma_oracle(csv=_CSV / "freenucleon_numu_sigma.csv",
                                    key=jax.random.PRNGKey(11), n=n, m_lep=M_MU)
    ratio = r_mu.metrics["c"] / r_e.metrics["c"]
    assert abs(ratio - 1.0) < 0.02, f"c_mu/c_e = {ratio:.4f}"


# ---------------------------------------------------------------------------- #
#  Phase A1 — EM single-pion (electron probe)
# ---------------------------------------------------------------------------- #
def test_em_closure_dpw():
    """EM differentiability: d(total EM σ)/d(vector-FF pw_norm), autodiff == FD.
    (EM has no axial; the pw_norm wave-normalisation is the differentiable handle.)"""
    r = em_dsigma_dpw_closure(key=jax.random.PRNGKey(0), n=N)
    assert not r.skipped
    assert r.passed, r.detail
    assert r.metrics["rel_err"] < 1e-3


def test_em_proton_channel_ratio():
    """EM proton channel ratio σ(e p->e p π0)/σ(e p->e n π+) ≈ 0.91, matching ACHILLES
    (325.3/355.9 = 0.914) within the [10,90] acceptance — validates the EM isospin +
    1/Q^4 weight (the crude structure-function proxy got this wrong; the real weight nails
    it)."""
    _, sc = em_sigma_channels_at(PhysicsParams(), jax.random.PRNGKey(7), 1500.0, N,
                                 theta_min_deg=10.0, theta_max_deg=90.0)
    sc = np.asarray(sc)
    ratio = sc[0] / sc[1]                              # p->p pi0 / p->n pi+
    assert 0.85 < ratio < 0.98, (ratio, sc)


def test_em_sigma_oracle():
    """A1 oracle: EM σ(E_e) in the [10,90] acceptance vs ACHILLES (electron on 1H+1N),
    proton channels + neutron total, single-constant bridge. (The neutron π0/π- split is
    excluded -- a known open EM-isospin refinement; the neutron total is right to <2%.)"""
    from adonis.primary.dcc.sigma_enu import em_sigma_oracle
    r = em_sigma_oracle(key=jax.random.PRNGKey(7), n=N_ORACLE)
    assert not r.skipped
    assert r.passed, r.detail
    assert r.metrics["rel_max"] < 0.06
