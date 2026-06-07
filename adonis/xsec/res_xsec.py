"""Absolute ACHILLES CC RES (single-pion) cross section by flat-MC, VECTORISED.

sigma_RES = sum over the 3 CC channels {n->n pi+, n->p pi0, p->p pi+} of E_u[ amps2 * flux *
initwgt * spinavg * event.Weight() ], using the ported mappers (Beam/QESpectral/ThreeBody) + the
batched exclusive DCC matrix element (dcc_current.exclusive_amps2_batch).  The mu+N+pi final state
is sampled with two isotropic 2-body splits (same integral as ACHILLES's t-channel sampler, higher
variance -> needs high N, now affordable via the vectorised current).  10 random dims: [0:4] struck
nucleon, [4] beam, [5] s23, [6:8] split-A angles, [8:10] split-B angles.  nu_mu + 12C, T2K flux.
ACHILLES target sigma_RES ~ 1.698e-5 nb.
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from adonis.xsec import constants as C
from adonis.xsec.flux import T2KFlux
from adonis.xsec.spectral import SpectralFunction
from adonis.xsec.backend import flux_factor, MASS_PDG_NEUTRON, MASS_PDG_PROTON
from adonis.xsec.dcc_current import exclusive_amps2_batch

_MN = C.mN
M_MU = 105.7
_TWO_PI = 2 * np.pi
N_NUC = 6
M_PIP = 139.57018; M_PI0 = 134.9764
M_P = 938.27; M_N = 939.57
SPIN_AVG = 0.5

CHANNELS = [
    (2112, -1, M_N, 211, M_PIP, MASS_PDG_NEUTRON),     # n -> n pi+
    (2112, -1, M_P, 111, M_PI0, MASS_PDG_NEUTRON),     # n -> p pi0
    (2212, +1, M_P, 211, M_PIP, MASS_PDG_PROTON),      # p -> p pi+
]


def _sqlam(s, s1, s2):
    a = (s - s1 - s2) ** 2 - 4 * s1 * s2
    return np.where(a > 0, np.sqrt(np.clip(a, 0, None)) / s, 0.0)


def _boost_to_lab(p4cm, P):
    """Boost CM 4-vec (N,4) to lab where parent P (N,4) (Boost lflag=0)."""
    rsq = np.sqrt(np.clip(P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, axis=1), 1e-9, None))
    E = (P[:, 0] * p4cm[:, 0] + np.sum(P[:, 1:] * p4cm[:, 1:], axis=1)) / rsq
    c1 = (p4cm[:, 0] + E) / (rsq + P[:, 0])
    return np.concatenate([E[:, None], p4cm[:, 1:] + c1[:, None] * P[:, 1:]], axis=1)


def _sample_channel(n, rng, flux, minE, maxE, m_pi, m_Nf):
    """Vectorised sampling of n RES events for one channel.  Returns dict of arrays + validity."""
    u = rng.random((n, 10))
    dE_beam = maxE - minE
    E_GeV = u[:, 4] * dE_beam + minE; Enu = E_GeV * 1000.0
    k_nu = np.stack([Enu, np.zeros(n), np.zeros(n), Enu], axis=1)
    J_beam = (dE_beam * flux.f(E_GeV)) / flux.flux_integral
    Smin = (M_MU + m_Nf + m_pi) ** 2
    rad = np.clip(Enu ** 2 + 2 * Enu * _MN + _MN ** 2 - Smin, 0, None)
    pmin = np.clip(Enu - np.sqrt(rad), 0, None); pmax = np.clip(Enu + np.sqrt(rad), None, 800.0)
    dp = pmax - pmin
    mom = dp * u[:, 0] + pmin
    cosTm = np.clip((2 * Enu * _MN + _MN ** 2 - mom ** 2 - Smin) / (2 * Enu * np.clip(mom, 1e-9, None)), -1, 1)
    cosT = (cosTm + 1) * u[:, 1] - 1; sinT = np.sqrt(np.clip(1 - cosT ** 2, 0, None))
    phi = _TWO_PI * u[:, 2]
    pvec = np.stack([mom * sinT * np.cos(phi), mom * sinT * np.sin(phi), mom * cosT], axis=1)
    det = Enu ** 2 + mom ** 2 + 2 * pvec[:, 2] * Enu + Smin
    emax = np.minimum(np.minimum(_MN + Enu - np.sqrt(np.clip(det, 0, None)), _MN - mom), 400.0)
    energy = emax * u[:, 3] - 1e-8
    p_struck = np.concatenate([(_MN - energy)[:, None], pvec], axis=1)
    J_had = mom ** 2 * dp * (cosTm + 1) * _TWO_PI * emax
    P = k_nu + p_struck
    s = P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, axis=1)
    sqrts = np.sqrt(np.clip(s, 1e-9, None))
    s23max = (sqrts - m_pi) ** 2; s23min = max((M_MU + m_Nf) ** 2, 1e-8)
    s23 = s23min + (s23max - s23min) * u[:, 5]; rs23 = np.sqrt(np.clip(s23, 1e-9, None))
    # split A: total -> muN + pi
    EmuN = (s + s23 - m_pi ** 2) / (2 * sqrts); pA = sqrts * _sqlam(s, s23, m_pi ** 2) / 2
    ctA = 2 * u[:, 6] - 1; stA = np.sqrt(np.clip(1 - ctA ** 2, 0, None)); phA = _TWO_PI * u[:, 7]
    dA = np.stack([stA * np.cos(phA), stA * np.sin(phA), ctA], axis=1)
    muN_cm = np.concatenate([EmuN[:, None], pA[:, None] * dA], axis=1)
    pi_cm = np.concatenate([np.sqrt(m_pi ** 2 + pA ** 2)[:, None], -pA[:, None] * dA], axis=1)
    p_muN = _boost_to_lab(muN_cm, P); p_pi = _boost_to_lab(pi_cm, P)
    I2W_A = 2.0 / np.pi / np.clip(_sqlam(s, s23, m_pi ** 2), 1e-12, None)
    # split B: muN -> mu + N
    Emu = (s23 + M_MU ** 2 - m_Nf ** 2) / (2 * rs23); pB = rs23 * _sqlam(s23, M_MU ** 2, m_Nf ** 2) / 2
    ctB = 2 * u[:, 8] - 1; stB = np.sqrt(np.clip(1 - ctB ** 2, 0, None)); phB = _TWO_PI * u[:, 9]
    dB = np.stack([stB * np.cos(phB), stB * np.sin(phB), ctB], axis=1)
    mu_cm = np.concatenate([Emu[:, None], pB[:, None] * dB], axis=1)
    N_cm = np.concatenate([np.sqrt(m_Nf ** 2 + pB ** 2)[:, None], -pB[:, None] * dB], axis=1)
    k_mu = _boost_to_lab(mu_cm, p_muN); p_N = _boost_to_lab(N_cm, p_muN)
    I2W_B = 2.0 / np.pi / np.clip(_sqlam(s23, M_MU ** 2, m_Nf ** 2), 1e-12, None)
    density = (2 * np.pi) ** 5 * I2W_A * I2W_B / (s23max - s23min)
    J_3body = np.where(density > 0, 1.0 / np.clip(density, 1e-300, None), 0.0)
    valid = (dp > 0) & (emax > 0) & (s > Smin) & (s23max > s23min) & (_sqlam(s, s23, m_pi ** 2) > 0) & (_sqlam(s23, M_MU ** 2, m_Nf ** 2) > 0)
    return dict(k_nu=k_nu, p_struck=p_struck, k_mu=k_mu, p_N=p_N, p_pi=p_pi,
                J=J_beam * J_had * J_3body, mom=mom, energy=energy, Enu=Enu, E_GeV=E_GeV, valid=valid)


def generate(n=20000, seed=0, return_events=False):
    rng = np.random.default_rng(seed)
    flux = T2KFlux(); minE = flux.seed_min_GeV(); maxE = flux.max_energy
    sf = SpectralFunction("data/Spectral_Functions/pke12n_tot.data")
    out = {}; sig = 0.0
    ev = {k: [] for k in ("k_nu", "k_mu", "p_struck", "p_N", "p_pi", "w", "ppid", "Npid")}
    for (ipid, itiz, mNf, ppid, mpi, mstr) in CHANNELS:
        Npid = 2212 if mNf == M_P else 2112
        s = _sample_channel(n, rng, flux, minE, maxE, mpi, mNf)
        v = s["valid"]
        iw = N_NUC * sf.batch(s["mom"], s["energy"])
        a2 = np.zeros(n)
        idx = np.where(v & (iw > 0) & (s["J"] > 0))[0]
        if len(idx):
            a2[idx] = exclusive_amps2_batch(s["k_nu"][idx], s["k_mu"][idx], s["p_struck"][idx],
                                            s["p_N"][idx], s["p_pi"][idx], itiz, ppid)
        fl = np.asarray(flux_factor(s["k_nu"], s["p_struck"], had_mass=mstr))
        w = np.where(v, a2 * fl * iw * SPIN_AVG * s["J"], 0.0)
        w = np.where(np.isfinite(w) & (a2 > 0), w, 0.0)
        sc = w.mean(); out[(ipid, ppid)] = sc; sig += sc
        if return_events:
            keep = w > 0
            ev["k_nu"].append(s["k_nu"][keep]); ev["k_mu"].append(s["k_mu"][keep])
            ev["p_struck"].append(s["p_struck"][keep]); ev["p_N"].append(s["p_N"][keep])
            ev["p_pi"].append(s["p_pi"][keep])
            # weight per event so that sum(w_event) over the sampled set = sigma_channel
            ev["w"].append(w[keep] / n)
            ev["ppid"].append(np.full(keep.sum(), ppid)); ev["Npid"].append(np.full(keep.sum(), Npid))
    out["sigma"] = sig
    if return_events:
        out["events"] = {k: np.concatenate(ev[k]) if ev[k] else np.empty((0,)) for k in ev}
    return out


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    nseed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    import numpy as _np
    sigs = []
    for sd in range(nseed):
        r = generate(n, seed=sd); sigs.append(r["sigma"])
        if sd == 0:
            for k, v in r.items():
                if k != "sigma":
                    print(f"  init {k[0]} -> pi {k[1]}: {v:.4e} nb")
    sigs = _np.array(sigs)
    print(f"sigma_RES = {sigs.mean():.4e} nb  (N={n} x{nseed} seeds, sem {sigs.std()/max(_np.sqrt(nseed),1):.2e})"
          f"  ACHILLES ~1.698e-5  ratio={sigs.mean()/1.698e-5:.3f}")
