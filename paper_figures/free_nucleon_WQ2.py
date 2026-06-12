"""FREE-NUCLEON bisection (res_investigation_log.md Iter 3; doc retired to git history): dsigma/dW(omega) & dsigma/dQ^2
for a struck nucleon AT REST (on-shell) -- de Forest shift ~0, no Fermi/spectral smearing -- so
the comparison isolates the ELEMENTARY DCC current (RES) / QE current Q^2 dependence.

RES: nu p -> mu- p pi+ (free proton = ACHILLES hydrogen, Coherent mapper).  Uses res_xsec's OWN
     exclusive_amps2_batch (the same amplitude as the nuclear figure).
QE : nu n -> mu- p     (free neutron).  Uses qe_xsec's me_cross_section.
ACHILLES ref: nofsi_res_H.hepmc (1H) / nofsi_qe_N.hepmc (1N).

Usage: python paper_figures/free_nucleon_WQ2.py res 1000000
       python paper_figures/free_nucleon_WQ2.py qe  1000000
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from adonis.xsec.flux import T2KFlux
from adonis.xsec.backend import flux_factor, MASS_PDG_PROTON, MASS_PDG_NEUTRON, me_cross_section
from adonis.xsec.dcc_current import exclusive_amps2_batch
from adonis.xsec import res_xsec as R
from adonis.xsec import qe_xsec as Q
from paper_figures.diagnostic_WQ2 import parse_hepmc, panel

ROOT = Path(__file__).resolve().parents[1]
ACH = ROOT.parent / "Achilles/_resrun_out"
M_MU = 105.7; M_PIP = 139.57018; _TWO_PI = 2 * np.pi


def _free_sample_common(n, rng, flux, m_rest):
    """Beam (T2K) + struck nucleon AT REST (m_rest,0,0,0).  Returns Enu,k_nu,p_struck,J_beam."""
    u = rng.random((n, 8))
    # beam seed: for the free nucleon process Masses()[1] = outgoing-nucleon^2 ~ m_rest^2
    Smin_seed = (M_MU + m_rest + (M_PIP if m_rest == MASS_PDG_PROTON else 0.0))  # placeholder, set per mode
    return u


def adonis_free_res(n, seed):
    """nu p(rest) -> mu p pi+ ; struck proton at rest, initwgt=1, J_had=1."""
    rng = np.random.default_rng(seed); flux = T2KFlux(); maxE = flux.max_energy
    mP, mpi, mmu = MASS_PDG_PROTON, M_PIP, M_MU
    Smin = (mmu + mP + mpi) ** 2
    minE = max((Smin - mP ** 2) / (2 * mP) / 1000.0, flux.min_energy); dE = maxE - minE
    u = rng.random((n, 8))
    E_GeV = u[:, 0] * dE + minE; Enu = E_GeV * 1000.0
    k_nu = np.stack([Enu, np.zeros(n), np.zeros(n), Enu], axis=1)
    J_beam = (dE * flux.f(E_GeV)) / flux.flux_integral
    p_struck = np.tile([mP, 0.0, 0.0, 0.0], (n, 1))
    P = k_nu + p_struck
    s = P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, axis=1); sqrts = np.sqrt(np.clip(s, 1e-9, None))
    s23max = (sqrts - mpi) ** 2; s23min = (mmu + mP) ** 2
    s23 = s23min + (s23max - s23min) * u[:, 1]; rs23 = np.sqrt(np.clip(s23, 1e-9, None))
    EmuN = (s + s23 - mpi ** 2) / (2 * sqrts); pA = sqrts * R._sqlam(s, s23, mpi ** 2) / 2
    ctA = 2 * u[:, 2] - 1; stA = np.sqrt(np.clip(1 - ctA ** 2, 0, None)); phA = _TWO_PI * u[:, 3]
    dA = np.stack([stA * np.cos(phA), stA * np.sin(phA), ctA], axis=1)
    muN_cm = np.concatenate([EmuN[:, None], pA[:, None] * dA], axis=1)
    pi_cm = np.concatenate([np.sqrt(mpi ** 2 + pA ** 2)[:, None], -pA[:, None] * dA], axis=1)
    p_muN = R._boost_to_lab(muN_cm, P); p_pi = R._boost_to_lab(pi_cm, P)
    I2W_A = 2.0 / np.pi / np.clip(R._sqlam(s, s23, mpi ** 2), 1e-12, None)
    Emu = (s23 + mmu ** 2 - mP ** 2) / (2 * rs23); pB = rs23 * R._sqlam(s23, mmu ** 2, mP ** 2) / 2
    ctB = 2 * u[:, 4] - 1; stB = np.sqrt(np.clip(1 - ctB ** 2, 0, None)); phB = _TWO_PI * u[:, 5]
    dB = np.stack([stB * np.cos(phB), stB * np.sin(phB), ctB], axis=1)
    mu_cm = np.concatenate([Emu[:, None], pB[:, None] * dB], axis=1)
    N_cm = np.concatenate([np.sqrt(mP ** 2 + pB ** 2)[:, None], -pB[:, None] * dB], axis=1)
    k_mu = R._boost_to_lab(mu_cm, p_muN); p_N = R._boost_to_lab(N_cm, p_muN)
    I2W_B = 2.0 / np.pi / np.clip(R._sqlam(s23, mmu ** 2, mP ** 2), 1e-12, None)
    density = (2 * np.pi) ** 5 * I2W_A * I2W_B / (s23max - s23min)
    J3 = np.where(density > 0, 1.0 / np.clip(density, 1e-300, None), 0.0)
    valid = (s > Smin) & (s23max > s23min) & (R._sqlam(s, s23, mpi ** 2) > 0) & (R._sqlam(s23, mmu ** 2, mP ** 2) > 0) & (J3 > 0)
    a2 = np.zeros(n); idx = np.where(valid)[0]
    if len(idx):
        a2[idx] = exclusive_amps2_batch(k_nu[idx], k_mu[idx], p_struck[idx], p_N[idx], p_pi[idx], +1, 211)
    fl = np.asarray(flux_factor(k_nu, p_struck, had_mass=mP))
    w = np.where(valid, a2 * fl * 0.5 * J_beam * J3, 0.0)
    w = np.where(np.isfinite(w) & (a2 > 0), w, 0.0)
    q = k_nu - k_mu; q2 = (np.sum(q[:, 1:] ** 2, axis=1) - q[:, 0] ** 2) / 1e6
    W = np.sqrt(np.clip((p_N[:, 0] + p_pi[:, 0]) ** 2 - np.sum((p_N[:, 1:] + p_pi[:, 1:]) ** 2, axis=1), 0, None))
    keep = w > 0
    return W[keep], q2[keep], w[keep] / n


def adonis_free_qe(n, seed):
    """nu n(rest) -> mu p ; struck neutron at rest, initwgt=1, J_had=1."""
    rng = np.random.default_rng(seed); flux = T2KFlux(); maxE = flux.max_energy
    mN, mP, mmu = MASS_PDG_NEUTRON, MASS_PDG_PROTON, M_MU
    Smin = (mmu + mP) ** 2
    minE = max((Smin - mP ** 2) / (2 * mP) / 1000.0, flux.min_energy); dE = maxE - minE
    u = rng.random((n, 3))
    E_GeV = u[:, 0] * dE + minE; Enu = E_GeV * 1000.0
    k_nu = np.stack([Enu, np.zeros(n), np.zeros(n), Enu], axis=1)
    J_beam = (dE * flux.f(E_GeV)) / flux.flux_integral
    p_struck = np.tile([mN, 0.0, 0.0, 0.0], (n, 1))
    P = k_nu + p_struck
    s = P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, axis=1); sqrts = np.sqrt(np.clip(s, 1e-9, None))
    s2, s3 = mmu ** 2, mP ** 2
    E1 = sqrts / 2 * (1 + s2 / s - s3 / s); E2 = sqrts / 2 * (1 + s3 / s - s2 / s)
    lam = np.sqrt(np.clip((s - s2 - s3) ** 2 - 4 * s2 * s3, 0, None)); pcm = lam / (2 * sqrts)
    cts = 2 * u[:, 1] - 1; sts = np.sqrt(np.clip(1 - cts ** 2, 0, None)); php = _TWO_PI * u[:, 2]
    dirn = np.stack([sts * np.cos(php), sts * np.sin(php), cts], axis=1)
    beta = P[:, 1:] / P[:, 0:1]
    k_mu = Q._boost(np.concatenate([E1[:, None], pcm[:, None] * dirn], axis=1), beta)
    p_out = Q._boost(np.concatenate([E2[:, None], -pcm[:, None] * dirn], axis=1), beta)
    J2 = 2.0 * _TWO_PI * pcm / (sqrts * 16 * np.pi ** 2)
    valid = (s > Smin) & (lam > 0)
    d = me_cross_section(jnp.asarray(k_nu), jnp.asarray(k_mu), jnp.asarray(p_struck),
                         jnp.asarray(p_out), spin_avg=0.5, had_mass=mN)
    me = np.asarray(d["me_xsec"])
    w = np.where(valid, me * J_beam * J2, 0.0)
    w = np.where(np.isfinite(w), w, 0.0)
    q = k_nu - k_mu; q2 = (np.sum(q[:, 1:] ** 2, axis=1) - q[:, 0] ** 2) / 1e6
    omega = k_nu[:, 0] - k_mu[:, 0]
    keep = w > 0
    return omega[keep], q2[keep], w[keep] / n


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "res"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000000
    hepmc = ACH / ("nofsi_res_H.hepmc" if mode == "res" else "nofsi_qe_N.hepmc")
    print(f"[free-{mode}] ACHILLES ref {hepmc.name}")
    alx, aq2, aw, sig_ach = parse_hepmc(hepmc, mode)
    print(f"  {len(alx)} ACHILLES events, sigma_total={sig_ach:.4e} nb")
    nchunks = 10; ch = n // nchunks
    dlx_l, dq2_l, dw_l, sig_chunks = [], [], [], []
    import time; t0 = time.time()
    for i in range(nchunks):
        dlx, dq2, dw = (adonis_free_res(ch, i) if mode == "res" else adonis_free_qe(ch, i))
        sig_chunks.append(dw.sum()); dlx_l.append(dlx); dq2_l.append(dq2); dw_l.append(dw / nchunks)
        print(f"  [ADoNIS free-{mode}] {100*(i+1)/nchunks:5.1f}%  running σ={np.mean(sig_chunks):.4e}  {time.time()-t0:.0f}s", flush=True)
    dlx = np.concatenate(dlx_l); dq2 = np.concatenate(dq2_l); dw = np.concatenate(dw_l)
    sig_ado = dw.sum()
    print(f"  ADoNIS sigma={sig_ado:.4e}  total ADoNIS/ACH={sig_ado/sig_ach:.3f}")
    if mode == "res":
        lbins = np.linspace(1080, 1800, 16); llab = "W=M(Nπ) [MeV]"
    else:
        lbins = np.linspace(0, 1500, 20); llab = "ω=E_ν−E_μ [MeV]"
    qbins = np.linspace(0, 2.0, 18)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), height_ratios=[3, 1], sharex="col")
    fig.suptitle(f"FREE-NUCLEON {mode.upper()} (struck at rest): ACHILLES vs ADoNIS — "
                 f"total {sig_ado/sig_ach:.3f}  (N_ACH={len(alx)}, N_ADO={len(dlx)})")
    panel(axes[0, 0], axes[1, 0], alx, aw, dlx, dw, lbins, llab)
    panel(axes[0, 1], axes[1, 1], aq2, aw, dq2, dw, qbins, "Q² [GeV²]")
    fig.tight_layout()
    out = ROOT / "paper_figures" / (f"free_{mode}_WQ2.png")
    fig.savefig(out, dpi=110); print("  wrote", out)
    for cut in (0.2, 0.4):
        la = aw[aq2 < cut].sum(); ld = dw[dq2 < cut].sum()
        ha = aw[aq2 >= cut].sum(); hd = dw[dq2 >= cut].sum()
        print(f"  Q2<{cut}: ADO/ACH={ld/la:.3f}   Q2>={cut}: ADO/ACH={hd/ha:.3f}")


if __name__ == "__main__":
    main()
