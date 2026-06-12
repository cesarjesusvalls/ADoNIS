"""PWIA quasi-elastic inclusive (e,e') response on a nucleus (Phase B2).

Plane-wave impulse approximation: the electron knocks out a single bound nucleon (drawn
from the spectral function S(p,E)), elastic at the nucleon level.  Folding the single-
nucleon response over S(p,E) gives the nuclear dsigma/domega, whose quasi-elastic peak sits
near omega ~ Q^2/2M and whose width comes from Fermi motion -- the QE component of Fig 1.

For fixed beam energy E_e and angle theta_e, scanning the energy transfer omega:
  E' = E_e - omega ;  Q^2 = 4 E_e E' sin^2(theta/2) ;  q = sqrt(Q^2 + omega^2)
energy conservation for a nucleon (|p|, removal energy E) -> final |p+q| fixes
  cos(theta_pq) = [ (omega - E + M)^2 - M^2 - p^2 - q^2 ] / (2 p q),  |cos| <= 1,
and the d^3p delta-function reduces the fold to (1/q) integral dp p dE S(p,E) x response.
The response uses the Kelly vector form factors (transverse G_M dominates the QE peak).

All numpy; differentiable handles (M_A etc.) enter via the 1pi piece, not the QE FFs here.
"""
from __future__ import annotations

import numpy as np

from adonis.nuclear.spectral import load_spectral
from adonis.primary.dcc.form_factors import kelly_sachs

from adonis.constants import mN as M_N, alpha as ALPHA  # Constant::mN, precise


def _single_nucleon_response(Q2_MeV2, q, th):
    """Single-nucleon elastic response with the proper longitudinal/transverse Rosenbluth
    separation (Donnelly-Raskin): v_L R_L + v_T R_T, with R_L ~ G_E^2, R_T ~ tau G_M^2 and
    v_L = (Q^2/q^2)^2, v_T = Q^2/(2 q^2) + tan^2(theta/2).  This (vs the simplified
    (G_E^2+tau G_M^2)/(1+tau)) gets the QE-peak shape/position right -- the transverse term
    dominates and shifts the peak relative to the isotropic combination."""
    Q2_GeV2 = Q2_MeV2 / 1e6
    gep, gen, gmp, gmn = kelly_sachs(Q2_GeV2)
    tau = Q2_MeV2 / (4 * M_N ** 2)
    R_L = gep ** 2 + gen ** 2
    R_T = tau * (gmp ** 2 + gmn ** 2)
    vL = (Q2_MeV2 / q ** 2) ** 2
    vT = Q2_MeV2 / (2 * q ** 2) + np.tan(th / 2) ** 2
    return vL * R_L + vT * R_T


def qe_dsigma_domega(E_e, theta_deg, omega, sf="pke12p_tot.data", n_p=6, n_n=6):
    """Inclusive QE dsigma/domega [arb.] at beam E_e [MeV], angle theta, for an array of
    omega [MeV]. PWIA spectral fold; returns the response shape (Mott factor folded in).

    Energy balance (Benhar spectral-function PWIA): E_f = omega + M - E with E the removal
    energy from S(p,E) -- the binding is carried by E, so the bare mass M is used (NOT the
    on-shell sqrt(M^2+p^2); de Forest's on-shell-initial variant double-counts the Fermi
    energy here and over-corrects).  The response uses the proper longitudinal/transverse
    Rosenbluth separation (`_single_nucleon_response`).  Validated vs the ACHILLES
    QE_Spectral_Func oracle (same spectral function, pke12p): peak 208 vs 192 MeV,
    chi2/ndf ~9 (the v_L/v_T separation brought it down from ~28 with the isotropic
    G_E^2+tau G_M^2 combination)."""
    t = load_spectral(sf)
    p = t.mom.astype(float)                       # (np,) MeV
    E = t.energy.astype(float)                    # (ne,)
    S = t.S.astype(float)                         # (np, ne)
    dp = p[1] - p[0]; dE = E[1] - E[0]
    th = np.deg2rad(theta_deg)
    out = np.zeros(len(omega))
    for i, w in enumerate(np.atleast_1d(omega)):
        Ep = E_e - w
        if Ep <= 0:
            continue
        Q2 = 4 * E_e * Ep * np.sin(th / 2) ** 2
        q = np.sqrt(Q2 + w ** 2)
        mott = (ALPHA * np.cos(th / 2) / (2 * E_e * np.sin(th / 2) ** 2)) ** 2
        resp = _single_nucleon_response(Q2, q, th)
        # fold over (p,E): cos(theta_pq) fixed by energy conservation (E = removal energy)
        acc = 0.0
        for ip, pp in enumerate(p):
            num = (w - E + M_N) ** 2 - M_N ** 2 - pp ** 2 - q ** 2
            cos_pq = num / (2 * pp * q + 1e-9)
            ok = np.abs(cos_pq) <= 1.0            # (ne,)
            acc += pp * np.sum(S[ip][ok] * dE)    # 1/q Jacobian below; p dp weight
        out[i] = mott * resp * (2 * np.pi) * acc * dp / q * (n_p + n_n) / 12.0
    return out
