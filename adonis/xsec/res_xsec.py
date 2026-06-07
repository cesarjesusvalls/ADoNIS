"""Absolute ACHILLES CC RES (single-pion) cross section by flat-MC, using the ported mappers
(Beam/QESpectral/ThreeBody) + the exclusive DCC matrix element (dcc_current).  sigma_RES = sum
over the 3 CC channels {n->n pi+, n->p pi0, p->p pi+} of E_u[ amps2 * flux * initwgt * spinavg *
event.Weight() ].  Same OneBodySpectral struck-nucleon mapper as QE; the mu+N+pi final state is
sampled with two isotropic 2-body splits (same integral as ACHILLES's t-channel sampler, higher
variance).  10 random dims: [0:4] struck nucleon, [4] beam, [5] s23, [6:8] split-A angles, [8:10]
split-B angles.  nu_mu + 12C, T2K flux.  Target: ACHILLES sigma_RES ~ 1.698e-5 nb.

STATUS: the matrix element (dcc_current, validated) and the ThreeBody Jacobian (magnitude
consistent with ACHILLES psw) are correct, but the ISOTROPIC ThreeBody sampling is severely
variance-limited for the peaked RES integrand (ACHILLES uses t-channel + Vegas; the instrumented
dump shows integrand*weight mean/median ~ 4000x).  At low N the flat-MC sigma_RES overshoots from
rare high-weight events.  TODO for the absolute sigma_RES: port the t-channel TChannelMomenta
importance sampling (FinalStateMapper.cc, source read) OR vectorise dcc_current for high-N.  The
RES DIFFERENTIAL shapes (for Figs 7/8 TKI) are unbiased per-event regardless of variance.
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from adonis.xsec import constants as C
from adonis.xsec.flux import T2KFlux
from adonis.xsec.spectral import SpectralFunction
from adonis.xsec.backend import flux_factor, MASS_PDG_NEUTRON, MASS_PDG_PROTON
from adonis.xsec.dcc_current import exclusive_amps2

_MN = C.mN
M_MU = 105.7
_TWO_PI = 2 * np.pi
N_NUC = 6
M_PIP = 139.57018; M_PI0 = 134.9764
M_P = 938.27; M_N = 939.57
SPIN_AVG = 0.5

# (init pid, itiz, final-N mass, pion pid, pion mass, struck-nucleon mass for flux/spectral)
CHANNELS = [
    (2112, -1, M_N, 211, M_PIP, MASS_PDG_NEUTRON),     # n -> n pi+
    (2112, -1, M_P, 111, M_PI0, MASS_PDG_NEUTRON),     # n -> p pi0
    (2212, +1, M_P, 211, M_PIP, MASS_PDG_PROTON),      # p -> p pi+
]


def _sqlam(s, s1, s2):
    a = (s - s1 - s2) ** 2 - 4 * s1 * s2
    return np.sqrt(a) / s if a > 0 else 0.0


def _boost_to_lab(p4cm, P):
    rsq = np.sqrt(max(P[0] ** 2 - P[1:] @ P[1:], 1e-9))
    E = (P[0] * p4cm[0] + P[1:] @ p4cm[1:]) / rsq
    c1 = (p4cm[0] + E) / (rsq + P[0])
    return np.array([E, *(p4cm[1:] + c1 * P[1:])])


def _final_state(u, Enu, m_pi, m_Nf):
    """Return (k_nu, p_struck, k_mu, p_N, p_pi, J_had, J_3body, mom, energy) or None."""
    k_nu = np.array([Enu, 0.0, 0.0, Enu])
    Smin = (M_MU + m_Nf + m_pi) ** 2
    rad = max(Enu ** 2 + 2 * Enu * _MN + _MN ** 2 - Smin, 0.0)
    pmin = max(Enu - np.sqrt(rad), 0.0); pmax = min(Enu + np.sqrt(rad), 800.0)
    dp = pmax - pmin
    if dp <= 0:
        return None
    mom = dp * u[0] + pmin
    cosTm = (2 * Enu * _MN + _MN ** 2 - mom ** 2 - Smin) / (2 * Enu * max(mom, 1e-9))
    cosTm = min(max(cosTm, -1), 1)
    cosT = (cosTm + 1) * u[1] - 1; sinT = np.sqrt(max(1 - cosT ** 2, 0))
    phi = _TWO_PI * u[2]
    pvec = np.array([mom * sinT * np.cos(phi), mom * sinT * np.sin(phi), mom * cosT])
    det = Enu ** 2 + mom ** 2 + 2 * pvec[2] * Enu + Smin
    emax = min(min(_MN + Enu - np.sqrt(max(det, 0)), _MN - mom), 400.0)
    if emax <= 0:
        return None
    energy = emax * u[3] - 1e-8
    p_struck = np.array([_MN - energy, *pvec])
    J_had = mom ** 2 * dp * (cosTm + 1) * _TWO_PI * emax

    P = k_nu + p_struck
    s = P[0] ** 2 - P[1:] @ P[1:]
    if s <= Smin:
        return None
    sqrts = np.sqrt(s)
    s23max = (sqrts - m_pi) ** 2; s23min = max((M_MU + m_Nf) ** 2, 1e-8)
    if s23max <= s23min:
        return None
    s23 = s23min + (s23max - s23min) * u[5]
    rs23 = np.sqrt(s23)
    # split A: total -> muN(mass rs23) + pi, isotropic in total CM
    EmuN = (s + s23 - m_pi ** 2) / (2 * sqrts); pA = sqrts * _sqlam(s, s23, m_pi ** 2) / 2
    ctA = 2 * u[6] - 1; stA = np.sqrt(max(1 - ctA ** 2, 0)); phA = _TWO_PI * u[7]
    dA = np.array([stA * np.cos(phA), stA * np.sin(phA), ctA])
    muN_cm = np.array([EmuN, *(pA * dA)]); pi_cm = np.array([np.sqrt(m_pi ** 2 + pA ** 2), *(-pA * dA)])
    p_muN = _boost_to_lab(muN_cm, P); p_pi = _boost_to_lab(pi_cm, P)
    I2W_A = 2.0 / np.pi / _sqlam(s, s23, m_pi ** 2)
    # split B: muN -> mu + N, isotropic in muN rest frame
    Emu = (s23 + M_MU ** 2 - m_Nf ** 2) / (2 * rs23); pB = rs23 * _sqlam(s23, M_MU ** 2, m_Nf ** 2) / 2
    ctB = 2 * u[8] - 1; stB = np.sqrt(max(1 - ctB ** 2, 0)); phB = _TWO_PI * u[9]
    dB = np.array([stB * np.cos(phB), stB * np.sin(phB), ctB])
    mu_cm = np.array([Emu, *(pB * dB)]); N_cm = np.array([np.sqrt(m_Nf ** 2 + pB ** 2), *(-pB * dB)])
    k_mu = _boost_to_lab(mu_cm, p_muN); p_N = _boost_to_lab(N_cm, p_muN)
    I2W_B = 2.0 / np.pi / _sqlam(s23, M_MU ** 2, m_Nf ** 2)
    if I2W_A <= 0 or I2W_B <= 0:
        return None
    density = (2 * np.pi) ** 5 * I2W_A * I2W_B / (s23max - s23min)
    J_3body = 1.0 / density
    return k_nu, p_struck, k_mu, p_N, p_pi, J_had, J_3body, mom, energy


def generate(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    flux = T2KFlux()
    minE = flux.seed_min_GeV(); maxE = flux.max_energy; dE_beam = maxE - minE
    sf = SpectralFunction("data/Spectral_Functions/pke12n_tot.data")
    out = {}
    sig_tot = 0.0
    for (ipid, itiz, mNf, ppid, mpi, mstr) in CHANNELS:
        w = np.zeros(n)
        for i in range(n):
            u = rng.random(10)
            E_GeV = u[4] * dE_beam + minE; Enu = E_GeV * 1000.0
            fs = _final_state(u, Enu, mpi, mNf)
            if fs is None:
                continue
            k_nu, p_str, k_mu, p_N, p_pi, J_had, J_3b, mom, energy = fs
            iw = N_NUC * sf.batch(np.array([mom]), np.array([energy]))[0]
            if iw <= 0:
                continue
            a2 = exclusive_amps2(k_nu, k_mu, p_str, p_N, p_pi, itiz, ppid)
            if not np.isfinite(a2) or a2 <= 0:
                continue
            fl = float(flux_factor(np.asarray(k_nu)[None], np.asarray(p_str)[None], had_mass=mstr)[0])
            J_beam = (dE_beam * flux.f(E_GeV)) / flux.flux_integral
            w[i] = a2 * fl * iw * SPIN_AVG * J_beam * J_had * J_3b
        sc = w[np.isfinite(w)].mean()
        out[(ipid, ppid)] = sc; sig_tot += sc
    out["sigma"] = sig_tot
    return out


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    r = generate(n)
    print("RES channel sigmas (nb):")
    for k, v in r.items():
        if k != "sigma":
            print(f"  init {k[0]} -> pi {k[1]}: {v:.4e}")
    print(f"sigma_RES = {r['sigma']:.4e} nb   ACHILLES target ~1.698e-5 nb  ratio={r['sigma']/1.698e-5:.3f}")
