"""sigma_RES the FLAT way: explicit QESpectralMapper struck-nucleon Jacobian + initwgt = N*S(p,E),
NO importance sampling.  If flat gives ~1.15e-5 (ACHILLES p->ppi+) while importance gives 0.85e-5,
the importance iw=N replacement is the bug.  If flat ALSO gives 0.85e-5, the bias is elsewhere.
Dominant channel p->p pi+ only.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from adonis.xsec import constants as C
from adonis.xsec.flux import T2KFlux
from adonis.xsec.spectral import SpectralFunction
from adonis.xsec.backend import flux_factor, MASS_PDG_PROTON
from adonis.xsec.dcc_current import exclusive_amps2_batch

MN = C.mN; M_MU = 105.7; M_P = 938.27; M_PIP = 139.57018
N_NUC = 6; SPIN = 0.5; TWO_PI = 2 * np.pi
_SFp = SpectralFunction("data/Spectral_Functions/pke12p_tot.data")   # proton channel -> proton SF


def _sqlam(s, a, b):
    v = (s - a - b) ** 2 - 4 * a * b
    return np.where(v > 0, np.sqrt(np.clip(v, 0, None)) / s, 0.0)


def _boost_to_lab(p4cm, P):
    rsq = np.sqrt(np.clip(P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, axis=1), 1e-9, None))
    E = (P[:, 0] * p4cm[:, 0] + np.sum(P[:, 1:] * p4cm[:, 1:], axis=1)) / rsq
    c1 = (p4cm[:, 0] + E) / (rsq + P[:, 0])
    return np.concatenate([E[:, None], p4cm[:, 1:] + c1[:, None] * P[:, 1:]], axis=1)


def sigma_flat(n, seed, m_Nf=M_P, mpi=M_PIP, itiz=+1, ppid=211):
    rng = np.random.default_rng(seed); u = rng.random((n, 10))
    flux = T2KFlux(); maxE = flux.max_energy
    Smin = (M_MU + m_Nf + mpi) ** 2
    minE = max((Smin - m_Nf ** 2) / (2 * m_Nf) / 1000.0, flux.min_energy)
    E_GeV = u[:, 4] * (maxE - minE) + minE; Enu = E_GeV * 1000.0
    k_nu = np.stack([Enu, np.zeros(n), np.zeros(n), Enu], axis=1)
    J_beam = ((maxE - minE) * flux.f(E_GeV)) / flux.flux_integral
    # FLAT struck nucleon (QESpectralMapper), Smin = 3-body threshold
    radical = np.clip(Enu ** 2 + 2 * Enu * MN + MN ** 2 - Smin, 0, None)
    pmin = np.clip(Enu - np.sqrt(radical), 0, None); pmax = np.clip(Enu + np.sqrt(radical), None, 800.0)
    dp = pmax - pmin; mom = dp * u[:, 0] + pmin
    cosTm = np.clip((2 * Enu * MN + MN ** 2 - mom ** 2 - Smin) / (2 * Enu * np.clip(mom, 1e-9, None)), -1, 1)
    cosT = (cosTm + 1) * u[:, 1] - 1; sinT = np.sqrt(np.clip(1 - cosT ** 2, 0, None)); phi = TWO_PI * u[:, 2]
    pvec = np.stack([mom * sinT * np.cos(phi), mom * sinT * np.sin(phi), mom * cosT], axis=1)
    det = Enu ** 2 + mom ** 2 + 2 * pvec[:, 2] * Enu + Smin
    emax = MN + Enu - np.sqrt(np.clip(det, 0, None)); emax = np.minimum(np.minimum(emax, MN - mom), 400.0)
    energy = emax * u[:, 3] - 1e-8
    p_struck = np.concatenate([(MN - energy)[:, None], pvec], axis=1)
    J_had = mom ** 2 * dp * (cosTm + 1) * TWO_PI * emax       # QESpectralMapper forward Jacobian
    initwgt = N_NUC * _SFp.batch(mom, energy)                  # N * S(p, removal)
    # 3-body (isotropic, same as res_xsec)
    P = k_nu + p_struck
    s = P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, axis=1); sqrts = np.sqrt(np.clip(s, 1e-9, None))
    s23max = (sqrts - mpi) ** 2; s23min = (M_MU + m_Nf) ** 2
    s23 = s23min + (s23max - s23min) * u[:, 5]; rs23 = np.sqrt(np.clip(s23, 1e-9, None))
    pA = sqrts * _sqlam(s, s23, mpi ** 2) / 2
    ctA = 2 * u[:, 6] - 1; stA = np.sqrt(np.clip(1 - ctA ** 2, 0, None)); phA = TWO_PI * u[:, 7]
    dA = np.stack([stA * np.cos(phA), stA * np.sin(phA), ctA], axis=1)
    EmuN = (s + s23 - mpi ** 2) / (2 * sqrts)
    muN_cm = np.concatenate([EmuN[:, None], pA[:, None] * dA], axis=1)
    pi_cm = np.concatenate([np.sqrt(mpi ** 2 + pA ** 2)[:, None], -pA[:, None] * dA], axis=1)
    p_muN = _boost_to_lab(muN_cm, P); p_pi = _boost_to_lab(pi_cm, P)
    I2W_A = 2.0 / np.pi / np.clip(_sqlam(s, s23, mpi ** 2), 1e-12, None)
    Emu = (s23 + M_MU ** 2 - m_Nf ** 2) / (2 * rs23); pB = rs23 * _sqlam(s23, M_MU ** 2, m_Nf ** 2) / 2
    ctB = 2 * u[:, 8] - 1; stB = np.sqrt(np.clip(1 - ctB ** 2, 0, None)); phB = TWO_PI * u[:, 9]
    dB = np.stack([stB * np.cos(phB), stB * np.sin(phB), ctB], axis=1)
    mu_cm = np.concatenate([Emu[:, None], pB[:, None] * dB], axis=1)
    N_cm = np.concatenate([np.sqrt(m_Nf ** 2 + pB ** 2)[:, None], -pB[:, None] * dB], axis=1)
    k_mu = _boost_to_lab(mu_cm, p_muN); p_N = _boost_to_lab(N_cm, p_muN)
    I2W_B = 2.0 / np.pi / np.clip(_sqlam(s23, M_MU ** 2, m_Nf ** 2), 1e-12, None)
    density = (2 * np.pi) ** 5 * I2W_A * I2W_B / (s23max - s23min)
    J3 = np.where(density > 0, 1.0 / density, 0.0)
    valid = (dp > 0) & (emax > 0) & (s > Smin) & (s23max > s23min) & (_sqlam(s, s23, mpi ** 2) > 0) & (_sqlam(s23, M_MU ** 2, m_Nf ** 2) > 0)
    a2 = np.zeros(n); idx = np.where(valid & (J3 > 0))[0]
    if len(idx):
        a2[idx] = exclusive_amps2_batch(k_nu[idx], k_mu[idx], p_struck[idx], p_N[idx], p_pi[idx], itiz, ppid)
    fl = np.asarray(flux_factor(k_nu, p_struck, had_mass=MASS_PDG_PROTON))
    w = np.where(valid & (a2 > 0), a2 * fl * initwgt * SPIN * J_beam * J_had * J3, 0.0)
    w = np.where(np.isfinite(w), w, 0.0)
    return w.mean()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    sigs = np.array([sigma_flat(n, seed=s) for s in range(4)])
    print(f"p->p pi+ FLAT (explicit J_had + N*S, no importance): sigma = {sigs.mean():.4e}  sem {sigs.std()/2:.2e}")
    print(f"  vs ACHILLES 1.151e-5 (ratio {sigs.mean()/1.151e-5:.3f}); importance ~8.5e-6 (0.74)")
