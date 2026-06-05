"""Free-nucleon charged-current quasi-elastic (CCQE) cross section, Llewellyn-Smith
(Phase B1).

The 1-nucleon CC current with the Kelly vector form factors (F1, F2) and a dipole/
z-expansion axial form factor (F_A, with the pseudoscalar F_P from PCAC) gives the
standard Llewellyn-Smith dsigma/dQ^2 for nu_mu n -> mu- p (and nubar p -> mu+ n):

    dsigma/dQ^2 = (M^2 G_F^2 cos^2(theta_c)) / (8 pi E_nu^2)
                  [ A(Q^2) -/+ B(Q^2)(s-u)/M^2 + C(Q^2)(s-u)^2/M^4 ]

with (s-u) = 4 M E_nu - Q^2 - m_l^2 and tau = Q^2/(4 M^2):

    A = (m_l^2 + Q^2)/M^2 [ (1+tau)F_A^2 - (1-tau)F_1^2 + tau(1-tau)F_2^2 + 4 tau F_1 F_2
        - (m_l^2/4M^2)((F_1+F_2)^2 + (F_A + 2 F_P)^2 - (Q^2/M^2 + 4)F_P^2) ]
    B = (Q^2/M^2) F_A (F_1 + F_2)
    C = (1/4)(F_A^2 + F_1^2 + tau F_2^2)

Everything is analytic and differentiable in M_A (through F_A, and F_P = 2M^2 F_A/(Q^2+m_pi^2)).
This is the QE analog of the A3 free-nucleon 1pi sigma(E_nu); the same single-constant
bridge validates it against ACHILLES QE_Spectral_Func on a stationary nucleon.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from adonis.primary.dcc.form_factors import kelly_dirac_pauli, axial_dipole, M_N_GEV, M_PI_GEV

# weak constants
GF = 1.1663787e-5            # [GeV^-2]
COSTHC = 0.97373             # cos(theta_Cabibbo)
HBARC2_GEV2_TO_NB = 0.3893793721e6   # (hbar c)^2 = 0.389 mb GeV^2 -> nb: x1e6
M = M_N_GEV
M_MU_GEV = 0.105658


def _vector_isovector(Q2_GeV2):
    """CC isovector vector form factors F1_V = F1p - F1n, F2_V = F2p - F2n (Kelly)."""
    f1p, f1n, f2p, f2n = kelly_dirac_pauli(Q2_GeV2)
    return f1p - f1n, f2p - f2n


def ls_dsigma_dQ2(Q2_GeV2, E_nu_GeV, MA=1.0, m_l=M_MU_GEV, anti=False):
    """Llewellyn-Smith dsigma/dQ^2 [nb/GeV^2] at (Q^2, E_nu). Differentiable in MA."""
    tau = Q2_GeV2 / (4.0 * M ** 2)
    F1, F2 = _vector_isovector(Q2_GeV2)
    FA = axial_dipole(Q2_GeV2, MA)                  # = -g_A/(1+Q^2/MA^2)^2  (negative)
    FP = 2.0 * M ** 2 * FA / (Q2_GeV2 + M_PI_GEV ** 2)
    ml2 = m_l ** 2
    A = ((ml2 + Q2_GeV2) / M ** 2) * (
        (1 + tau) * FA ** 2 - (1 - tau) * F1 ** 2 + tau * (1 - tau) * F2 ** 2 + 4 * tau * F1 * F2
        - (ml2 / (4 * M ** 2)) * ((F1 + F2) ** 2 + (FA + 2 * FP) ** 2
                                  - (Q2_GeV2 / M ** 2 + 4) * FP ** 2))
    B = (Q2_GeV2 / M ** 2) * FA * (F1 + F2)
    C = 0.25 * (FA ** 2 + F1 ** 2 + tau * F2 ** 2)
    s_u = 4 * M * E_nu_GeV - Q2_GeV2 - ml2
    sign = +1.0 if anti else -1.0                   # -/+ : nu uses -, nubar +
    pref = (M ** 2 * GF ** 2 * COSTHC ** 2) / (8 * jnp.pi * E_nu_GeV ** 2)
    dsig = pref * (A + sign * B * s_u / M ** 2 + C * s_u ** 2 / M ** 4)
    return dsig * HBARC2_GEV2_TO_NB                 # GeV^-2 -> nb


def _Q2_limits(E_nu_GeV, m_l=M_MU_GEV):
    """Kinematic Q^2 range for nu n -> l p at beam energy E_nu (nucleon at rest)."""
    s = M ** 2 + 2 * M * E_nu_GeV
    # outgoing lepton energy/momentum in CM, then Q^2 = 2 E_nu (E_l - p_l cos) - m_l^2 bounds
    El_cm = (s - M ** 2 + m_l ** 2) / (2 * jnp.sqrt(s))
    pl_cm = jnp.sqrt(jnp.clip(El_cm ** 2 - m_l ** 2, 0.0, None))
    Enu_cm = (s - M ** 2) / (2 * jnp.sqrt(s))
    # Q^2 = -m_l^2 + 2 Enu_cm (El_cm -/+ pl_cm)
    Q2_min = -m_l ** 2 + 2 * Enu_cm * (El_cm - pl_cm)
    Q2_max = -m_l ** 2 + 2 * Enu_cm * (El_cm + pl_cm)
    return jnp.clip(Q2_min, 0.0, None), Q2_max


def ccqe_sigma(E_nu_GeV, MA=1.0, m_l=M_MU_GEV, anti=False, nq=400):
    """Total CCQE cross section sigma(E_nu) [nb] by integrating dsigma/dQ^2 over the
    allowed Q^2 range. Differentiable in MA (the Q^2 grid is detached)."""
    q2lo, q2hi = _Q2_limits(E_nu_GeV, m_l)
    x = (jnp.arange(nq) + 0.5) / nq
    Q2 = jax.lax.stop_gradient(q2lo + (q2hi - q2lo) * x)
    dQ2 = jax.lax.stop_gradient((q2hi - q2lo) / nq)
    vals = jax.vmap(lambda q: ls_dsigma_dQ2(q, E_nu_GeV, MA, m_l, anti))(Q2)
    return jnp.sum(vals) * dQ2


def ccqe_sigma_vs_enu(energies_GeV, MA=1.0, m_l=M_MU_GEV, anti=False):
    """sigma(E_nu) [nb] over a list of beam energies."""
    return np.array([float(ccqe_sigma(float(e), MA, m_l, anti)) for e in energies_GeV])


def ccqe_sigma_oracle(csv=None, MA=1.0, rel_max=0.06):
    """B1 oracle: free-nucleon CCQE sigma(E_nu) vs ACHILLES QE_Spectral_Func (nu_mu on a
    stationary neutron). Both sides are ABSOLUTE nb (the LS prefactor is G_F^2 cos^2 theta_c,
    no fit), so this checks the absolute agreement directly -- |model/ACH - 1| < rel_max at
    every energy (no bridging constant). CSV cols: E_nu[MeV], sigma_nb."""
    from pathlib import Path
    from adonis.core.validation import TestResult
    if csv is None:
        csv = Path(__file__).resolve().parents[3] / "data" / "oracle" / "freenucleon_ccqe_sigma.csv"
    ref = np.loadtxt(csv)
    E_MeV, ach = ref[:, 0], ref[:, 1]
    mod = ccqe_sigma_vs_enu(E_MeV / 1000.0, MA)
    rel = np.abs(mod - ach) / ach
    passed = bool(rel.max() < rel_max)
    detail = "  ".join(f"{e/1000:.1f}GeV {mod[i]/ach[i]:.3f}" for i, e in enumerate(E_MeV))
    return TestResult(
        "CCQE.sigma.oracle", "oracle", passed, False,
        f"CCQE sigma(E) model/ACHILLES (absolute nb): max rel {rel.max():.3f} "
        f"mean {rel.mean():.3f}  [{detail}] (tol {rel_max})",
        {"rel_max": float(rel.max()), "rel_mean": float(rel.mean()), "n": int(len(E_MeV))})
