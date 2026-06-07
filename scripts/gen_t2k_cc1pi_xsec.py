"""T2K CC1pi+ STV (delta_pTT, p_N, delta_alphaT) from the BIT-EXACT ACHILLES RES primary
(adonis/xsec.res_xsec) + the discrete-Glauber cascade -- first-principles replacement for the
DCC-bridge surrogate.  Signal: the produced pi+ SURVIVES FSI (pid 211) and the leading nucleon is
a proton.  Tight T2K CC1pi+ phase space (NUISANCE PRD 103 112009): mu/pi/p momentum windows +
theta<70deg.  Each event carries its absolute RES nb weight.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from adonis.xsec import res_xsec
from adonis.core.event import EventRecord
from adonis.fsi.cascade_discrete import DiscreteCascadeFSI, DiscreteNucleonFSI, DiscreteCascadeConfig
from adonis import observables as obs

ROOT = Path(__file__).resolve().parents[1]
COS70 = np.cos(70.0 * np.pi / 180.0)
MU_LO, MU_HI = 250.0, 7000.0; PI_LO, PI_HI = 150.0, 1200.0; P_LO, P_HI = 450.0, 1200.0
_CFG = DiscreteCascadeConfig


def _acc(p4, lo, hi):
    m = np.linalg.norm(p4[:, 1:], axis=1)
    return (m > lo) & (m < hi) & (p4[:, 3] / np.clip(m, 1e-9, None) > COS70)


def generate(n_res=400000, seed=0, nseed=1):
    dptt, pn, dat, w = [], [], [], []
    for sd in range(nseed):
        r = res_xsec.generate(n_res, seed=sd + seed, return_events=True); e = r["events"]
        m = len(e["w"])
        if m == 0:
            continue
        ev = EventRecord(k=jnp.asarray(e["k_nu"]), kp=jnp.asarray(e["k_mu"]),
                         p_struck=jnp.asarray(e["p_struck"]), p_pi=jnp.asarray(e["p_pi"]),
                         p_N=jnp.asarray(e["p_N"]), w=jnp.asarray(e["w"]),
                         channel=jnp.zeros(m, jnp.int32), pid_pi=jnp.asarray(e["ppid"], jnp.int32),
                         pid_N=jnp.asarray(e["Npid"], jnp.int32), pid_Ni=jnp.full((m,), 2112),
                         W=jnp.zeros(m), Q2_adj=jnp.zeros(m))
        ev = DiscreteCascadeFSI(_CFG(seed=1)).apply(None, ev, key=jax.random.PRNGKey(sd + 11))
        ev = DiscreteNucleonFSI(_CFG(seed=2)).apply(None, ev, key=jax.random.PRNGKey(sd + 13))
        pid_pi = np.asarray(ev.pid_pi); pid_N = np.asarray(ev.pid_N); wv = np.asarray(ev.w) / n_res
        mu = np.asarray(ev.kp); ppi = np.asarray(ev.p_pi); pN = np.asarray(ev.p_N)
        sel = ((pid_pi == 211) & (pid_N == 2212) & (wv > 0) & _acc(mu, MU_LO, MU_HI)
               & _acc(ppi, PI_LO, PI_HI) & _acc(pN, P_LO, P_HI))
        dptt.append(np.asarray(obs.delta_pTT(ev))[sel]); pn.append(np.asarray(obs.p_N_tki(ev))[sel])
        dat.append(np.asarray(obs.delta_alphaT(ev))[sel]); w.append(wv[sel])
    cat = lambda a: np.concatenate(a) if a else np.array([])
    return dict(dptt=cat(dptt), pn=cat(pn), dalphat=cat(dat), w=cat(w))


if __name__ == "__main__":
    n_res = int(sys.argv[1]) if len(sys.argv) > 1 else 400000
    nseed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    d = generate(n_res, nseed=nseed)
    print(f"CC1pi+: {len(d['w'])} events  "
          f"rms(dpTT)={np.sqrt(np.average(d['dptt']**2, weights=d['w'])):.1f}  "
          f"<p_N>={np.average(d['pn'], weights=d['w']):.1f}" if len(d['w']) else "no events")
    np.savez(ROOT / "data" / "oracle" / "t2k_cc1pi_tki_adonis_xsec.npz", **d)
    print("wrote data/oracle/t2k_cc1pi_tki_adonis_xsec.npz")
