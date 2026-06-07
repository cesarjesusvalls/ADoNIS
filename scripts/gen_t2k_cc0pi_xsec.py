"""T2K CC0pi-Np STV (delta_pT, delta_alphaT) from the BIT-EXACT ACHILLES primary cross sections
(adonis/xsec) + the discrete-Glauber cascade -- the first-principles replacement for the LS/DCC-
bridge surrogate.  CC0pi = CCQE (xsec.qe_xsec) + RES-with-pion-absorbed (xsec.res_xsec, pion FSI),
each carrying its OWN absolute nb weight (sigma_QE 4.31e-5, sigma_RES 1.46e-5), so the QE:RES mix
is first-principles -- no fitted fraction.  Leading proton = highest-momentum proton among the
primary recoil (after NN FSI) and the pion-absorption protons.  CC0pi-Np cuts: p_mu>250,
cos_mu>-0.6, leading proton 450-1000 MeV/c, cos_p>0.4, no surviving meson.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from adonis.xsec import qe_xsec, res_xsec
from adonis.core.event import EventRecord
from adonis.fsi.cascade_discrete import DiscreteCascadeFSI, DiscreteNucleonFSI, DiscreteCascadeConfig
from adonis import observables as obs

ROOT = Path(__file__).resolve().parents[1]
MU_LO = 250.0; COSMU = -0.6; P_LO, P_HI = 450.0, 1000.0; COSP = 0.4
_CFG = lambda **k: DiscreteCascadeConfig(cylinder=True, **k)


def _cc0pi(mu, lead, w):
    pmu = np.linalg.norm(mu[:, 1:], axis=1); cmu = mu[:, 3] / np.clip(pmu, 1e-9, None)
    pl = np.linalg.norm(lead[:, 1:], axis=1); cl = lead[:, 3] / np.clip(pl, 1e-9, None)
    sel = (w > 0) & (pmu > MU_LO) & (cmu > COSMU) & (pl > P_LO) & (pl < P_HI) & (cl > COSP)
    lt = mu[:, 1:3]; pt = lead[:, 1:3]; dv = lt + pt
    dpt = np.linalg.norm(dv, axis=1)
    c = -np.sum(lt * dv, axis=1) / (np.linalg.norm(lt, axis=1) * dpt + 1e-9)
    dat = np.arccos(np.clip(c, -1, 1))
    return dpt[sel], dat[sel], w[sel]


def qe_sample(n, seed):
    r = qe_xsec.generate(n, seed=seed)
    w = np.asarray(r["w"]) / n                                   # per-event absolute nb
    k_mu = np.asarray(r["k_mu"]); p_out = np.asarray(r["p_out"]); p_str = np.asarray(r["p_struck"])
    z = jnp.zeros((len(w), 4))
    ev = EventRecord(k=jnp.asarray(r["k_nu"]), kp=jnp.asarray(k_mu), p_struck=jnp.asarray(p_str),
                     p_pi=z, p_N=jnp.asarray(p_out), w=jnp.asarray(w),
                     channel=jnp.zeros(len(w), jnp.int32), pid_pi=jnp.zeros(len(w), jnp.int32),
                     pid_N=jnp.full((len(w),), 2212), pid_Ni=jnp.full((len(w),), 2112),
                     W=jnp.zeros(len(w)), Q2_adj=jnp.zeros(len(w)))
    ev = DiscreteNucleonFSI(_CFG(seed=2)).apply(None, ev, key=jax.random.PRNGKey(seed + 7))
    mu = np.asarray(ev.kp); lead = np.asarray(ev.p_N)
    return _cc0pi(mu, lead, w)


def res_absorbed_sample(n, seed):
    r = res_xsec.generate(n, seed=seed, return_events=True); e = r["events"]
    m = len(e["w"])
    if m == 0:
        return np.array([]), np.array([]), np.array([])
    ev = EventRecord(k=jnp.asarray(e["k_nu"]), kp=jnp.asarray(e["k_mu"]), p_struck=jnp.asarray(e["p_struck"]),
                     p_pi=jnp.asarray(e["p_pi"]), p_N=jnp.asarray(e["p_N"]), w=jnp.asarray(e["w"]),
                     channel=jnp.zeros(m, jnp.int32),
                     pid_pi=jnp.asarray(e["ppid"], jnp.int32), pid_N=jnp.asarray(e["Npid"], jnp.int32),
                     pid_Ni=jnp.full((m,), 2112), W=jnp.zeros(m), Q2_adj=jnp.zeros(m))
    pion = DiscreteCascadeFSI(_CFG(seed=1))
    ev = pion.apply(None, ev, key=jax.random.PRNGKey(seed + 11))
    absorbed = np.asarray(pion.last_absorbed); abs_p = np.asarray(pion.last_abs_proton)
    ev = DiscreteNucleonFSI(_CFG(seed=2)).apply(None, ev, key=jax.random.PRNGKey(seed + 13))
    mu = np.asarray(ev.kp); pN = np.asarray(ev.p_N); pidN = np.asarray(ev.pid_N)
    prim_is_p = pidN == 2212
    mom_prim = np.linalg.norm(pN[:, 1:], axis=1) * prim_is_p
    mom_abs = np.linalg.norm(abs_p[:, 1:], axis=1)
    lead = np.where((mom_abs > mom_prim)[:, None], abs_p, pN)
    has_p = (mom_abs > 1) | prim_is_p
    w = np.asarray(e["w"]) * (absorbed & has_p)                  # CC0pi only if pion absorbed
    return _cc0pi(mu, lead, w)


def generate(n_qe=400000, n_res=120000, seed=0):
    qd = qe_sample(n_qe, seed)
    rd = res_absorbed_sample(n_res, seed)
    dpt = np.concatenate([qd[0], rd[0]]); dat = np.concatenate([qd[1], rd[1]])
    w = np.concatenate([qd[2], rd[2]])
    return dict(dpt=dpt, dalphat=dat, w=w, sig_qe=qd[2].sum(), sig_res=rd[2].sum())


if __name__ == "__main__":
    n_qe = int(sys.argv[1]) if len(sys.argv) > 1 else 400000
    n_res = int(sys.argv[2]) if len(sys.argv) > 2 else 120000
    nseed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    acc = {"dpt": [], "dalphat": [], "w": []}; sq = sr = 0.0
    for sd in range(nseed):
        d = generate(n_qe, n_res, seed=sd)
        for k in acc:
            acc[k].append(d[k])
        sq += d["sig_qe"]; sr += d["sig_res"]
        print(f"  seed {sd}: {len(d['w'])} CC0pi events")
    dpt = np.concatenate(acc["dpt"]); dat = np.concatenate(acc["dalphat"]); w = np.concatenate(acc["w"])
    f = sr / (sq + sr) if (sq + sr) > 0 else 0
    print(f"CC0pi-Np TOTAL: {len(w)} events  RES-absorbed fraction = {100*f:.1f}%")
    np.savez(ROOT / "data" / "oracle" / "t2k_cc0pi_tki_adonis_xsec.npz", dpt=dpt, dalphat=dat, w=w)
    print("wrote data/oracle/t2k_cc0pi_tki_adonis_xsec.npz")
