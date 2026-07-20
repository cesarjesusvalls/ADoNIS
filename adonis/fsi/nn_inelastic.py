"""NN -> N Delta inelastic channel: faithful port of ACHILLES NucleonNucleon (GiBUU /
Dmitriev-Sushkov) -- src/Achilles/CascadeInteractions/NucleonNucleon.cc::SigmaNN2NDelta{,Interp}
+ ResonanceHelper.cc::{MatNN2NDelta, DSigmaDM, GetEffectiveWidth} + Distributions.hh::
BlattWeisskopf.  This is the channel that degrades fast nucleons and CREATES pions in the
ACHILLES cascade (ResonanceMode: Decay -> Delta -> N pi immediately); it was missing from the
ADoNIS nucleon cascade (elastic-only), the offender behind the CC1pi carbon excess
(docs/logbook/cc1pi_t2k.md #6).

Conventions mirrored exactly:
* ParticleInfo masses = Particles.yml (MASS_PDG_*); mn_avg = (938.27+939.57)/2 MeV.
* MatNN2NDelta constants: fps=2.202, fp=1.008, lambda2=0.63^2, kappa2=0.2^2, mpi=0.14 GeV
  (hardcoded in ACHILLES).
* DSigmaDM: mpi_avg = (2 m_pi+ + m_pi0)/3 (yml); Breit-Wigner with the L=1 Blatt-Weisskopf
  effective width; sigma in mb via HBARC2.
* SigmaNN2NDelta integrates dm over [m_neutron + m_pi+, sqrts - m_neutron] ("always the
  heavier particles") and the table is built for delta++ at pcm = 1 GeV, then scaled by
  isofactor (1 for delta++/-, 1/3 for delta+/0) / pcm[GeV] (SigmaNN2NDeltaInterp).
* Charge states (AllowedResonanceStates): pp -> {delta++ n, delta+ p}; pn -> {delta+ n,
  delta0 p}; nn -> {delta0 n, delta- p}.  ExpSup = 0 in the T2K card (no density suppression).
All tables cached as NUMPY; jnp.asarray per call (tracer-safe)."""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from adonis.constants import MASS_PDG_PROTON, MASS_PDG_NEUTRON, MASS_PDG_PIP, MASS_PDG_PI0, HBARC2

_GEV = 1000.0
MN_AVG = (MASS_PDG_PROTON + MASS_PDG_NEUTRON) / 2 / _GEV       # ParticleInfo avg [GeV]
MPI_AVG = (2 * MASS_PDG_PIP + MASS_PDG_PI0) / 3 / _GEV
MN_HEAVY = MASS_PDG_NEUTRON / _GEV                              # "heavier" choices (integration limits)
MPI_HEAVY = MASS_PDG_PIP / _GEV
# Particles.yml delta masses/widths [GeV]
DELTA_MASS = {"pp": 1230.55 / _GEV, "p": 1234.90 / _GEV, "0": 1231.30 / _GEV, "m": 1230.55 / _GEV}
DELTA_WIDTH = {"pp": 112.2 / _GEV, "p": 131.1 / _GEV, "0": 112.5 / _GEV, "m": 112.2 / _GEV}
HBARC_GEVFM = 197.3269804 / 1000.0                             # GeV.fm (BlattWeisskopf x = k/HBARC); full precision (was 0.19732, 3.5e-5 coarse) [audit 2026-07-20]
HBARC2_GEV2_MB = HBARC2 / 1e6                                   # mb GeV^2

_CACHE = {}


def _pcm2(s, s1, s2):
    return (s + s1 - s2) ** 2 / (4 * s) - s1


def _blatt_weisskopf1(x):
    return x / np.sqrt(1 + x * x)


def _eff_width(mass, delta="pp"):
    """GetEffectiveWidth (L=1) for Delta -> N pi at running mass [GeV]."""
    m0, w0 = DELTA_MASS[delta], DELTA_WIDTH[delta]
    k = np.sqrt(np.clip(_pcm2(mass ** 2, MN_AVG ** 2, MPI_AVG ** 2), 1e-12, None))
    k0 = np.sqrt(np.clip(_pcm2(m0 ** 2, MN_AVG ** 2, MPI_AVG ** 2), 1e-12, None))
    rho = k / mass * _blatt_weisskopf1(k / HBARC_GEVFM) ** 2
    rho0 = k0 / m0 * _blatt_weisskopf1(k0 / HBARC_GEVFM) ** 2
    return w0 * rho / rho0


def _mat_nn2ndelta(t, u, sdelta, delta="pp"):
    """MatNN2NDelta (GiBUU Dmitriev-Sushkov), all args GeV^2."""
    fps, fp, lambda2, kappa2, mpi = 2.202, 1.008, 0.63 ** 2, 0.2 ** 2, 0.14
    mpi2 = mpi * mpi
    mn, mn2 = MN_AVG, MN_AVG ** 2
    gp = fp * 2 * mn / mpi
    mdelta2 = DELTA_MASS[delta] ** 2
    zt = (_pcm2(mdelta2, mn2, t) + kappa2) / (_pcm2(sdelta, mn2, t) + kappa2)
    zu = (_pcm2(mdelta2, mn2, u) + kappa2) / (_pcm2(sdelta, mn2, u) + kappa2)
    md = np.sqrt(sdelta)
    m1 = t * (t - (md - mn) ** 2) * ((md + mn) ** 2 - t) ** 2 / (3 * sdelta)
    m2 = u * (u - (md - mn) ** 2) * ((md + mn) ** 2 - u) ** 2 / (3 * sdelta)
    m12 = (1.0 / (2 * sdelta)) * (
        (t * u + (sdelta - mn2) * (t + u) - sdelta ** 2 + mn2 ** 2)
        * (t * u + mn * (md + mn) * (sdelta - mn2))
        - (1.0 / 3.0) * (t * u - (md + mn) ** 2 * (t + u) + (md + mn) ** 4)
        * (t * u - mn * (md - mn) * (sdelta - mn2)))
    ft = (lambda2 - mpi2) / (lambda2 - t)
    fu = (lambda2 - mpi2) / (lambda2 - u)
    pt = 1 / (t - mpi2)
    pu = 1 / (u - mpi2)
    return (fps * gp / mpi) ** 2 * (ft ** 4 * zt * pt * pt * m1 + fu ** 4 * zu * pu * pu * m2
                                    + (ft * fu) ** 2 * np.sqrt(np.clip(zt * zu, 0, None)) * pt * pu * m12)


def dsigma_dm(sqrts, mdelta, delta="pp", ncos=200):
    """DSigmaDM(iresonance=False): dsigma/dm [mb/GeV] for NN -> N Delta(mdelta) at sqrts [GeV]."""
    sqrts = np.atleast_1d(np.asarray(sqrts, float))
    mdelta = np.atleast_1d(np.asarray(mdelta, float))
    out = np.zeros(np.broadcast_shapes(sqrts.shape, mdelta.shape))
    sqrts, mdelta = np.broadcast_arrays(sqrts, mdelta)
    for i in np.ndindex(out.shape):
        rs, md = sqrts[i], mdelta[i]
        if rs < 2 * MN_AVG + MPI_AVG or rs < md + MN_AVG:
            continue
        pin2 = rs * rs / 4 - MN_AVG ** 2
        pout2 = max((rs * rs - (md + MN_AVG) ** 2) * (rs * rs - (md - MN_AVG) ** 2) / (4 * rs * rs), 0.0)
        e2 = np.sqrt(MN_AVG ** 2 + pin2)
        e3 = np.sqrt(MN_AVG ** 2 + pout2)
        e4 = np.sqrt(md * md + pout2)
        prop = 0.0
        if md > MN_AVG + MPI_AVG:
            m0 = DELTA_MASS[delta]
            width = _eff_width(md, delta)
            prop = (1 / np.pi) * md * width / ((md * md - m0 * m0) ** 2 + (md * width) ** 2)
        cost = np.linspace(-1, 1, ncos)
        t = md * md + MN_AVG ** 2 - 2 * e4 * e2 + 2 * cost * np.sqrt(pin2 * pout2)
        u = 2 * MN_AVG ** 2 - 2 * e3 * e2 - 2 * cost * np.sqrt(pin2 * pout2)
        mat = (1.0 / (32 * np.pi * rs * rs)) * _mat_nn2ndelta(t, u, md * md, delta) * HBARC2_GEV2_MB
        integrand = mat * np.sqrt(pout2) * 2 * md * prop
        out[i] = np.trapezoid(integrand, cost)
    return out


def _build_tables(n_s=240, n_m=160, s_lo=None, s_hi=4.0):
    """sigma_ref(sqrts) for delta++ at pcm = 1 GeV (SigmaNN2NDelta table), and the
    mass inverse-CDF table for Delta-mass sampling."""
    if "sig" in _CACHE:
        return
    s_lo = s_lo or (2 * MN_HEAVY + MPI_HEAVY + 1e-4)
    # NON-UNIFORM sqrts grid: DENSE near threshold (the sigma turn-on is steep -> a uniform ~8 MeV grid
    # gave +36% from linear-interp overshoot, cf. scripts/test_sigma_nn_ndelta.py) then coarse above the
    # cascade-relevant region.  The per-point mass/cos integral is ALREADY exact vs ACHILLES (validated
    # direct), so n_m/ncos are unchanged -- only the table density near threshold is the fix.
    s_knee = min(2.30, s_hi)
    S = np.unique(np.concatenate([np.linspace(s_lo, s_knee, 4 * n_s),
                                  np.linspace(s_knee, s_hi, n_s)]))
    sig = np.zeros(len(S))                                       # len(S) != n_s (non-uniform grid)
    icdf = np.zeros((len(S), 65))
    ugrid = np.linspace(0, 1, 65)
    for i, rs in enumerate(S):
        m_lo = MN_HEAVY + MPI_HEAVY
        m_hi = rs - MN_HEAVY
        if m_hi <= m_lo:
            icdf[i] = m_lo
            continue
        M = np.linspace(m_lo, m_hi, n_m)
        d = dsigma_dm(rs, M)
        sig[i] = np.trapezoid(d, M)                              # mb.GeV / (pcm) applied later
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(M))])
        cdf = cdf / cdf[-1] if cdf[-1] > 0 else np.linspace(0, 1, n_m)
        icdf[i] = np.interp(ugrid, cdf, M)
    _CACHE["S"] = S
    _CACHE["sig"] = sig                                          # integrate dm at pcm=1GeV scale
    _CACHE["icdf"] = icdf
    _CACHE["u"] = ugrid


def sigma_nn_ndelta(sqrts_gev, pcm_gev, same_iso):
    """TOTAL NN -> N Delta cross section [mb] summed over the two allowed charge states
    (isofactors 1 + 1/3 = 4/3 for pp/nn; 1/3 + 1/3 = 2/3 for pn), at sqrts [GeV] with the
    1/pcm flux scaling (SigmaNN2NDeltaInterp).  jnp-friendly (interp on the numpy table)."""
    _build_tables()
    base = jnp.interp(jnp.asarray(sqrts_gev), jnp.asarray(_CACHE["S"]), jnp.asarray(_CACHE["sig"]),
                      left=0.0, right=0.0)
    iso_sum = jnp.where(jnp.asarray(same_iso), 4.0 / 3.0, 2.0 / 3.0)
    return base * iso_sum / jnp.clip(jnp.asarray(pcm_gev), 1e-6, None)


def sample_delta_mass(sqrts_gev, u):
    """Delta mass [GeV] ~ dsigma/dm at sqrts, via the inverse-CDF table (u ~ U[0,1))."""
    _build_tables()
    S = jnp.asarray(_CACHE["S"]); icdf = jnp.asarray(_CACHE["icdf"]); ug = jnp.asarray(_CACHE["u"])
    si = jnp.clip(jnp.searchsorted(S, jnp.asarray(sqrts_gev)) - 1, 0, len(_CACHE["S"]) - 2)
    fs = (jnp.asarray(sqrts_gev) - S[si]) / (S[si + 1] - S[si])
    ui = jnp.clip((jnp.asarray(u) * (len(_CACHE["u"]) - 1)).astype(jnp.int32), 0, len(_CACHE["u"]) - 2)
    fu = jnp.asarray(u) * (len(_CACHE["u"]) - 1) - ui
    def pick(row):
        a = icdf[row, ui] * (1 - fu) + icdf[row, ui + 1] * fu
        return a
    return pick(si) * (1 - fs) + pick(si + 1) * fs
