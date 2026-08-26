"""Free-proton (hydrogen) contribution to T2K CC0pi-Np, the H half of the CH target.
nu_mu on a FREE proton at rest, T2K flux, RES (the only CC channel on a free p: p -> p pi+).
Runs the produced pion through the discrete cascade, then applies the CC0pi-Np selection.
Reports the H absolute RES sigma and how many H events pass CC0pi.
"""
import os, sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from adonis.flux.spectrum import SpectrumFlux, M_MU, M_P
from adonis.channels.res import free_nucleon_weights, _pi_kin_mass, M_PIP
from adonis.channels.currents.matrix_element import MASS_PDG_PROTON
from adonis.fsi.cascade import DiscreteCascadeConfig
from adonis.fsi.pool_fsi import run_fsi

MU_LO = 250.0; COSMU = -0.6; P_LO, P_HI = 450.0, 1000.0; COSP = 0.4
_CFG = lambda **k: DiscreteCascadeConfig(step=0.04, max_steps=600, engine="pool", **k)


def generate_H(n, seed=0, flux=None):
    """Flux-averaged nu_mu p -> mu- p pi+ on a FREE proton at rest (thin caller over the shared
    res.free_nucleon_weights primitive: multiply the per-event weight by the flux-sampling J_beam, sum).

    `flux` is a spectrum table relative to the ACHILLES data root; None takes SpectrumFlux's default.
    """
    rng = np.random.default_rng(seed)
    flux = SpectrumFlux(flux)
    m_pi = _pi_kin_mass(M_PIP); m_Nf = M_P
    Smin = (M_MU + m_Nf + m_pi) ** 2
    minE = max((Smin - m_Nf ** 2) / (2 * m_Nf) / 1000.0, flux.min_energy)
    maxE = flux.max_energy; dE = maxE - minE
    u = rng.random((n, 6))
    E_GeV = u[:, 0] * dE + minE; Enu = E_GeV * 1000.0
    J_beam = (dE * flux.f(E_GeV)) / flux.flux_integral
    k_nu = np.stack([Enu, np.zeros(n), np.zeros(n), Enu], axis=1)
    w0, kin = free_nucleon_weights(k_nu, +1, M_P, 211, M_PIP, MASS_PDG_PROTON, u[:, 1:6])
    w = w0 * J_beam
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0) / n
    return k_nu, kin["k_lep"], kin["p_N"], kin["p_pi"], w


def _cc0pi_obs(knu, mu, lead, w):
    pmu = np.linalg.norm(mu[:, 1:], axis=1); cmu = mu[:, 3] / np.clip(pmu, 1e-9, None)
    pl = np.linalg.norm(lead[:, 1:], axis=1); cl = lead[:, 3] / np.clip(pl, 1e-9, None)
    sel = (w > 0) & (pmu > MU_LO) & (cmu > COSMU) & (pl > P_LO) & (pl < P_HI) & (cl > COSP)
    lt = mu[:, 1:3]; pt = lead[:, 1:3]; dv = lt + pt
    dpt = np.linalg.norm(dv, axis=1)
    c = -np.sum(lt * dv, axis=1) / (np.linalg.norm(lt, axis=1) * dpt + 1e-9)
    dat = np.arccos(np.clip(c, -1, 1))
    q = knu - mu; Q2 = ((q[:, 1:] ** 2).sum(1) - q[:, 0] ** 2) / 1e6
    return dpt[sel], dat[sel], Q2[sel], w[sel]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400000
    knu, kmu, pN, pPi, w = generate_H(n, seed=0)
    print(f"H (free p) nu_mu RES p->p pi+ :  sigma = {w.sum():.4e} nb  ({int((w>0).sum())} events w>0)")
    m = len(w)
    out = run_fsi(jnp.asarray(pPi), jnp.asarray(pN), jnp.full(m, 211, jnp.int32),
                  jnp.full(m, 2212, jnp.int32), jnp.full(m, 2212, jnp.int32),
                  _CFG(seed=1), jax.random.PRNGKey(7), channel="res")
    absorbed = np.asarray(out["pterm"]["pid"] == 0)
    lead = np.asarray(out["lead_prot"]); mu = kmu
    has_p = np.linalg.norm(lead[:, 1:], axis=1) > 1
    wcc = w * (absorbed & has_p)
    dpt, dat, Q2, ww = _cc0pi_obs(knu, mu, lead, wcc)
    print(f"  H CC0pi-Np events: {len(ww)}   sigma_CC0pi(H) = {ww.sum():.4e} nb")
    os.makedirs("output/adonis", exist_ok=True)
    np.savez("output/adonis/t2k_C_cc0pi_H.npz", dpt=dpt, dalphat=dat, Q2=Q2, w=ww)
    print("wrote output/adonis/t2k_C_cc0pi_H.npz")
