"""1-pion-production contribution to the inclusive (e,e') cross section (Phase B2).

The Delta/resonance bump in dsigma/domega, above the quasi-elastic peak.  At the lepton
momentum transfer (omega, q), the electron makes a piN final state off a bound nucleon
(spectral function S(p,E)); the hadronic response is the EM transverse/longitudinal
structure functions W_T, W_L (the angle-integrated DCC hadron tensor, EM_CHANNELS) at the
invariant mass W = sqrt((q + p_struck)^2).  Folding over S(p,E) Fermi-broadens the bump,
which sits at omega ~ (m_Delta^2 - M^2 + Q^2)/(2M) -- above the QE peak.  Adding this to
the QE response (qe_inclusive) gives the full dsigma/domega ~ paper Fig 1.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from adonis.nuclear.spectral import load_spectral
from adonis.primary.dcc.structure import HadronStructure, EM_CHANNELS
from adonis.params import PhysicsParams

M_N = 938.919
ALPHA = 1.0 / 137.036
_HS = {}


def _em_hs():
    if "hs" not in _HS:
        _HS["hs"] = HadronStructure(channels=EM_CHANNELS, n_theta=10, n_phi=10, spline=False)
    return _HS["hs"]


def onepi_dsigma_domega(E_e, theta_deg, omega, sf="pke12p_tot.data", n_nuc=12, knobs=None):
    """1pi inclusive dsigma/domega [arb.] at beam E_e, angle theta, array omega [MeV]."""
    knobs = PhysicsParams() if knobs is None else knobs
    t = load_spectral(sf)
    p = t.mom.astype(float); n_p = np.clip(t.n_p, 0, None)
    dp = p[1] - p[0]
    pdist = p ** 2 * n_p                              # momentum distribution (un-normalised)
    pdist = pdist / (np.sum(pdist) * dp + 1e-30)
    # mean nucleon removal energy E_rm from the spectral function: the struck nucleon is
    # BOUND, so its initial energy is M_N - E_rm, not M_N.  Omitting this puts the Delta
    # bump ~60 MeV too low in omega (it takes more omega to reach W=1232 off a bound
    # nucleon).  The QE response already folds E(p,E); here we use the S-weighted mean.
    E_grid = t.energy.astype(float); S2 = t.S.astype(float)             # (np, ne)
    Pp = (p ** 2)[:, None] * S2
    E_rm = float(np.sum(E_grid[None, :] * Pp) / (np.sum(Pp) + 1e-30))
    th = np.deg2rad(theta_deg)
    omega = np.atleast_1d(omega).astype(float)

    # build the (omega) -> (W, Q2) and structure-function lookup, folding over |p| (cos uniform)
    out = np.zeros(len(omega))
    cosg = np.linspace(-1, 1, 9)                      # struck-nucleon angle vs q (coarse fold)
    for i, w in enumerate(omega):
        Ep = E_e - w
        if Ep <= 0:
            continue
        Q2 = 4 * E_e * Ep * np.sin(th / 2) ** 2
        q = np.sqrt(Q2 + w ** 2)
        mott = (ALPHA * np.cos(th / 2) / (2 * E_e * np.sin(th / 2) ** 2)) ** 2
        vT = Q2 / (2 * q ** 2) + np.tan(th / 2) ** 2
        vL = (Q2 / q ** 2) ** 2
        # W for each (|p|, cos) of the struck nucleon (off-shell E ~ M); W^2=(q+p_struck)^2
        Ws, weights = [], []
        for ip, pp in enumerate(p):
            for cg in cosg:
                tot0 = w + M_N - E_rm                  # struck nucleon is bound: E = M_N - E_rm
                W2 = tot0 ** 2 - (q ** 2 + pp ** 2 + 2 * q * pp * cg)
                if W2 > (M_N + 130) ** 2:              # above pion threshold
                    Ws.append(np.sqrt(W2)); weights.append(pdist[ip])
        if not Ws:
            continue
        Ws = jnp.asarray(Ws); Q2a = jnp.full_like(Ws, Q2)
        WT, WL = _em_hs().structures_at(Ws, jnp.clip(Q2a, 1.0, None), knobs)
        resp = np.asarray(vT * WT + vL * WL)
        out[i] = mott * float(np.sum(np.asarray(weights) * resp)) * dp / len(cosg) * n_nuc / 12.0
    return out
