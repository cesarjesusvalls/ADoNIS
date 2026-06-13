"""Free-proton (hydrogen) contribution to T2K CC0pi-Np, the H half of the CH target.
nu_mu on a FREE proton at rest, T2K flux, RES (the only CC channel on a free p: p -> p pi+).
Runs the produced pion through the discrete cascade, then applies the CC0pi-Np selection.
Reports the H absolute RES sigma and how many H events pass CC0pi.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import adonis.xsec.dcc_current as dcc
dcc.BATCH_INTERP = "spline"
from adonis.xsec.flux import T2KFlux, M_MU, M_P
from adonis.xsec.res_xsec import _sample_3body, _sample_3body_dispatch, _pi_kin_mass, M_PIP, SPIN_AVG
from adonis.xsec.dcc_current import exclusive_amps2_batch
from adonis.xsec.backend import flux_factor, MASS_PDG_PROTON
from adonis.core.event import EventRecord
from adonis.fsi.cascade_discrete import DiscreteCascadeFSI, DiscreteNucleonFSI, DiscreteCascadeConfig

MU_LO = 250.0; COSMU = -0.6; P_LO, P_HI = 450.0, 1000.0; COSP = 0.4
_CFG = lambda **k: DiscreteCascadeConfig(cylinder=True, step=0.04, max_steps=325, **k)


def generate_H(n, seed=0):
    rng = np.random.default_rng(seed)
    flux = T2KFlux()
    m_pi = _pi_kin_mass(M_PIP); m_Nf = M_P            # outgoing N = proton; kinematic pion mass
    Smin = (M_MU + m_Nf + m_pi) ** 2
    minE = max((Smin - m_Nf ** 2) / (2 * m_Nf) / 1000.0, flux.min_energy)
    maxE = flux.max_energy; dE = maxE - minE
    u = rng.random((n, 6))
    E_GeV = u[:, 0] * dE + minE; Enu = E_GeV * 1000.0
    J_beam = (dE * flux.f(E_GeV)) / flux.flux_integral
    k_nu = np.stack([Enu, np.zeros(n), np.zeros(n), Enu], axis=1)
    p_struck = np.tile([M_P, 0, 0, 0], (n, 1)).astype(float)    # FREE proton at rest
    tb = _sample_3body_dispatch(k_nu, p_struck, m_pi, m_Nf, u[:, 1:6])   # honors SAMPLER_3BODY
    k_mu, p_N, p_pi, J3, valid = tb["k_mu"], tb["p_N"], tb["p_pi"], tb["J_3body"], tb["valid3"]
    a2 = np.zeros(n); ch = 1000
    idx = np.where(valid & (J3 > 0))[0]
    for i in range(0, len(idx), ch):
        sl = idx[i:i + ch]
        a2[sl] = np.asarray(exclusive_amps2_batch(k_nu[sl], k_mu[sl], p_struck[sl], p_N[sl], p_pi[sl], +1, 211))
    fl = np.asarray(flux_factor(k_nu, p_struck, had_mass=MASS_PDG_PROTON))
    w = np.where(valid, a2 * fl * SPIN_AVG * J3 * J_beam, 0.0)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0) / n           # absolute nb per event (mean)
    return k_nu, k_mu, p_N, p_pi, w


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
    # pion cascade (NOTE: DiscreteCascadeFSI propagates through a 12C config -- the H proton is
    # treated as embedded in carbon; a strictly free H would not absorb.  See report.)
    ev = EventRecord(k=jnp.asarray(knu), kp=jnp.asarray(kmu), p_struck=jnp.zeros((m, 4)),
                     p_pi=jnp.asarray(pPi), p_N=jnp.asarray(pN), w=jnp.asarray(w),
                     channel=jnp.zeros(m, jnp.int32), pid_pi=jnp.full((m,), 211, jnp.int32),
                     pid_N=jnp.full((m,), 2212, jnp.int32), pid_Ni=jnp.full((m,), 2212, jnp.int32),
                     W=jnp.zeros(m), Q2_adj=jnp.zeros(m))
    pion = DiscreteCascadeFSI(_CFG(seed=1))
    ev = pion.apply(None, ev, key=jax.random.PRNGKey(7))
    absorbed = np.asarray(pion.last_absorbed); abs_p = np.asarray(pion.last_abs_proton)
    ev = DiscreteNucleonFSI(_CFG(seed=2)).apply(None, ev, key=jax.random.PRNGKey(13))
    mu = np.asarray(ev.kp); pNf = np.asarray(ev.p_N)
    mom_prim = np.linalg.norm(pNf[:, 1:], axis=1) * (np.asarray(ev.pid_N) == 2212)
    mom_abs = np.linalg.norm(abs_p[:, 1:], axis=1)
    lead = np.where((mom_abs > mom_prim)[:, None], abs_p, pNf)
    has_p = (mom_abs > 1) | (np.asarray(ev.pid_N) == 2212)
    wcc = w * (absorbed & has_p)
    dpt, dat, Q2, ww = _cc0pi_obs(knu, mu, lead, wcc)
    print(f"  H CC0pi-Np events: {len(ww)}   sigma_CC0pi(H) = {ww.sum():.4e} nb")
    np.savez("data/oracle/t2k_cc0pi_tki_adonis_H.npz", dpt=dpt, dalphat=dat, Q2=Q2, w=ww)
    print("wrote data/oracle/t2k_cc0pi_tki_adonis_H.npz")
