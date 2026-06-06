"""GiBUU N N -> N Delta production cross section -- the core of the ACHILLES Propagating-
Resonances mode and the paper's Fig 14 (pp -> pn pi+, pp -> pp pi0 via N N -> N Delta -> N N pi).

Faithful port of `src/Achilles/ResonanceHelper.cc` (the Dmitriev-Sushkov / GiBUU one-pion-
exchange matrix element `MatNN2NDelta`, the Delta effective width `GetEffectiveWidth`, and the
`DSigmaDM` differential) -- all in GeV.  sigma(NN->NDelta)(sqrts) = integral over the Delta
invariant mass (Breit-Wigner spectral function) and cos(theta) of the differential.

This is the production cross section for one N Delta charge channel (the matrix element is
charge-independent at the Delta pole); the full Fig-14 pp->pn pi+ / pp pi0 breakdown applies
the NN->NDelta isospin Clebsch + the Delta -> N pi branching on top (NucleonNucleon.cc).
"""
from __future__ import annotations

import numpy as np

# masses [GeV]
M_N = (0.93827208816 + 0.93956542054) / 2.0
M_PI = 0.13957018
M_DELTA = 1.232
W_DELTA = 0.117             # Delta pole width [GeV]
HBARC2 = 0.3893793721       # GeV^2 mb  (Constant::HBARC2 = hbarc^2 in GeV^2 mb)


def pcm2(s, m1sq, m2sq):
    """CM momentum^2 from Mandelstam s and the two squared masses (Kallen)."""
    return (s ** 2 + m1sq ** 2 + m2sq ** 2 - 2 * s * m1sq - 2 * s * m2sq - 2 * m1sq * m2sq) / (4 * s)


def blatt_weisskopf_l1(x):
    return x / np.sqrt(1 + x * x)


def effective_width(mdelta):
    """Delta -> N pi running width [GeV] (Blatt-Weisskopf L=1), GetEffectiveWidth."""
    hbarc = 0.1973269804   # GeV fm
    k = np.sqrt(np.clip(pcm2(mdelta ** 2, M_N ** 2, M_PI ** 2), 0.0, None))
    k0 = np.sqrt(pcm2(M_DELTA ** 2, M_N ** 2, M_PI ** 2))
    rho = k / mdelta * blatt_weisskopf_l1(k / hbarc) ** 2
    rho0 = k0 / M_DELTA * blatt_weisskopf_l1(k0 / hbarc) ** 2
    return W_DELTA * rho / rho0


def mat_nn2ndelta(t, u, sdelta):
    """Dmitriev-Sushkov OPE matrix element |M|^2 for N N -> N Delta (ResonanceHelper.cc)."""
    fps, fp, lambda2, kappa2 = 2.202, 1.008, 0.63 ** 2, 0.2 ** 2
    mpi2 = M_PI ** 2
    mn, mn2 = M_N, M_N ** 2
    gp = fp * 2 * mn / M_PI
    mdelta2 = M_DELTA ** 2
    zt = (pcm2(mdelta2, mn2, t) + kappa2) / (pcm2(sdelta, mn2, t) + kappa2)
    zu = (pcm2(mdelta2, mn2, u) + kappa2) / (pcm2(sdelta, mn2, u) + kappa2)
    md = np.sqrt(sdelta)
    m1 = t * (t - (md - mn) ** 2) * ((md + mn) ** 2 - t) ** 2 / (3 * sdelta)
    m2 = u * (u - (md - mn) ** 2) * ((md + mn) ** 2 - u) ** 2 / (3 * sdelta)
    m12 = (1.0 / (2 * sdelta)
           * ((t * u + (sdelta - mn2) * (t + u) - sdelta ** 2 + mn2 ** 2)
              * (t * u + mn * (md + mn) * (sdelta - mn2))
              - 1.0 / 3.0 * (t * u - (md + mn) ** 2 * (t + u) + (md + mn) ** 4)
              * (t * u - mn * (md - mn) * (sdelta - mn2))))
    ft = (lambda2 - mpi2) / (lambda2 - t)
    fu = (lambda2 - mpi2) / (lambda2 - u)
    pt = 1 / (t - mpi2)
    pu = 1 / (u - mpi2)
    return (fps * gp / M_PI) ** 2 * (ft ** 4 * zt * pt * pt * m1 + fu ** 4 * zu * pu * pu * m2
                                     + (ft * fu) ** 2 * np.sqrt(zt * zu) * pt * pu * m12)


def _dsigma_dcost_dm(cost, sqrts, mdelta):
    """Integrand d^2 sigma /(dcos dm_delta) [mb/GeV] (ResonanceHelper TestDeltaDSigmaDOmegaDM)."""
    mn = M_N
    if sqrts < 2 * mn + M_PI or sqrts < mdelta + mn:
        return 0.0
    pin2 = sqrts ** 2 / 4 - mn ** 2
    pout2 = max((sqrts ** 2 - (mdelta + mn) ** 2) * (sqrts ** 2 - (mdelta - mn) ** 2)
                / (4 * sqrts ** 2), 0.0)
    e2 = np.sqrt(mn ** 2 + pin2)
    e3 = np.sqrt(mn ** 2 + pout2)
    e4 = np.sqrt(mdelta ** 2 + pout2)
    prop = 0.0
    if mdelta > mn + M_PI:
        width = effective_width(mdelta)
        prop = (1 / np.pi * mdelta * width
                / ((mdelta ** 2 - M_DELTA ** 2) ** 2 + (mdelta * width) ** 2))
    t = mdelta ** 2 + mn ** 2 - 2 * e4 * e2 + 2 * cost * np.sqrt(pin2 * pout2)
    u = 2 * mn ** 2 - 2 * e3 * e2 - 2 * cost * np.sqrt(pin2 * pout2)
    mat = 1.0 / (32 * np.pi * sqrts ** 2) * mat_nn2ndelta(t, u, mdelta ** 2) * HBARC2
    return mat * np.sqrt(pout2) * 2 * mdelta * prop


def sigma_nn2ndelta(sqrts, n_cost=40, n_m=60):
    """sigma(N N -> N Delta) [mb] at CM energy sqrts [GeV] for ONE charge channel
    (matrix element charge-independent at the pole; isospin applied by the caller).
    2-D integral over cos(theta) in [-1,1] and m_delta in [mn+mpi, sqrts-mn]."""
    sqrts = float(sqrts)
    m_lo, m_hi = M_N + M_PI, sqrts - M_N
    if m_hi <= m_lo:
        return 0.0
    cg = np.linspace(-1, 1, n_cost); mg = np.linspace(m_lo, m_hi, n_m)
    dC = cg[1] - cg[0]; dM = mg[1] - mg[0]
    tot = 0.0
    for m in mg:
        row = np.array([_dsigma_dcost_dm(c, sqrts, m) for c in cg])
        tot += np.sum(row) * dC * dM
    return float(2 * np.pi * tot)
