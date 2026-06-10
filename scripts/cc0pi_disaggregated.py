"""Disaggregated ADoNIS CC0pi-Np prediction (absolute nb, first-principles) by
{QE,RES} x {C,H} x {no-FSI, FSI}, for dsigma/d{W,Q2,delta_alphaT,delta_pT}.

W = vertex invariant mass sqrt((q+p_struck)^2) -- QE sits ~M_N, RES at the Delta.
CC0pi-Np cuts: p_mu>250, cos_mu>-0.6, leading proton 450-1000 MeV/c cos_p>0.4, no surviving meson.
Physically-fixed cells (the code confirms): QE in H = 0 (no nu_mu CCQE on a free proton);
RES no-FSI = 0 CC0pi (pion survives the no-meson cut).
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

import adonis.xsec.dcc_current as dcc
dcc.BATCH_INTERP = "spline"
from adonis.xsec import qe_xsec, res_xsec
from adonis.core.event import EventRecord
from adonis.fsi.cascade_discrete import DiscreteCascadeFSI, DiscreteNucleonFSI, DiscreteCascadeConfig
from scripts.h_cc0pi import generate_H

MU_LO, COSMU, P_LO, P_HI, COSP = 250.0, -0.6, 450.0, 1000.0, 0.4
_CFG = lambda **k: DiscreteCascadeConfig(cylinder=True, step=0.04, max_steps=325, **k)
NQE, NRES, NH, NSEED = 200000, 200000, 2000, 4   # H contributes 0 CC0pi (free p can't absorb) -> token NH


def _obs(knu, kmu, pstr, lead, w):
    """CC0pi-Np selection + (W_vertex, Q2, delta_alphaT, delta_pT) for the passing events."""
    pmu = np.linalg.norm(kmu[:, 1:], axis=1); cmu = kmu[:, 3] / np.clip(pmu, 1e-9, None)
    pl = np.linalg.norm(lead[:, 1:], axis=1); cl = lead[:, 3] / np.clip(pl, 1e-9, None)
    sel = (w > 0) & (pmu > MU_LO) & (cmu > COSMU) & (pl > P_LO) & (pl < P_HI) & (cl > COSP)
    q = knu - kmu
    tot = q + pstr
    W = np.sqrt(np.clip(tot[:, 0] ** 2 - np.sum(tot[:, 1:] ** 2, axis=1), 0, None))
    Q2 = ((q[:, 1:] ** 2).sum(1) - q[:, 0] ** 2) / 1e6
    lt = kmu[:, 1:3]; pt = lead[:, 1:3]; dv = lt + pt
    dpt = np.linalg.norm(dv, axis=1)
    c = -np.sum(lt * dv, axis=1) / (np.linalg.norm(lt, axis=1) * dpt + 1e-9)
    dat = np.arccos(np.clip(c, -1, 1))
    return dict(W=W[sel], Q2=Q2[sel], dalphat=dat[sel], dpt=dpt[sel], w=w[sel])


def _ev(knu, kmu, pstr, ppi, pN, w, pid_pi):
    m = len(w)
    return EventRecord(k=jnp.asarray(knu), kp=jnp.asarray(kmu), p_struck=jnp.asarray(pstr),
                       p_pi=jnp.asarray(ppi), p_N=jnp.asarray(pN), w=jnp.asarray(w),
                       channel=jnp.zeros(m, jnp.int32), pid_pi=jnp.asarray(pid_pi, jnp.int32),
                       pid_N=jnp.full((m,), 2212, jnp.int32), pid_Ni=jnp.full((m,), 2112, jnp.int32),
                       W=jnp.zeros(m), Q2_adj=jnp.zeros(m))


def qe_C(n, seed, fsi):
    r = qe_xsec.sample_importance(n, seed=seed)
    knu = np.asarray(r["k_nu"]); kmu = np.asarray(r["k_mu"]); pstr = np.asarray(r["p_struck"])
    pout = np.asarray(r["p_out"]); w = np.asarray(r["w"]) / n
    if fsi:
        ev = _ev(knu, kmu, pstr, np.zeros_like(knu), pout, w, np.zeros(len(w)))
        ev = DiscreteNucleonFSI(_CFG(seed=2)).apply(None, ev, key=jax.random.PRNGKey(seed + 7))
        lead = np.asarray(ev.p_N)
    else:
        lead = pout                                              # bare recoil proton
    return _obs(knu, kmu, pstr, lead, w)


def res_C(n, seed, fsi):
    e = res_xsec.generate(n, seed=seed, return_events=True)["events"]
    knu = np.asarray(e["k_nu"]); kmu = np.asarray(e["k_mu"]); pstr = np.asarray(e["p_struck"])
    ppi = np.asarray(e["p_pi"]); pN = np.asarray(e["p_N"]); w = np.asarray(e["w"]); ppid = np.asarray(e["ppid"])
    if not fsi:
        return _obs(knu, kmu, pstr, pN, w * 0.0)                 # pion survives -> no CC0pi
    ev = _ev(knu, kmu, pstr, ppi, pN, w, ppid)
    pion = DiscreteCascadeFSI(_CFG(seed=1)); ev = pion.apply(None, ev, key=jax.random.PRNGKey(seed + 11))
    absorbed = np.asarray(pion.last_absorbed); abs_p = np.asarray(pion.last_abs_proton)
    ev = DiscreteNucleonFSI(_CFG(seed=2)).apply(None, ev, key=jax.random.PRNGKey(seed + 13))
    pNf = np.asarray(ev.p_N); mom_p = np.linalg.norm(pNf[:, 1:], axis=1) * (np.asarray(ev.pid_N) == 2212)
    mom_a = np.linalg.norm(abs_p[:, 1:], axis=1)
    lead = np.where((mom_a > mom_p)[:, None], abs_p, pNf)
    has_p = (mom_a > 1) | (np.asarray(ev.pid_N) == 2212)
    return _obs(knu, kmu, pstr, lead, w * (absorbed & has_p))


def res_H(n, seed, fsi):
    # Free proton (hydrogen): the produced pi+ has NO second nucleon to be absorbed onto
    # (pi N N -> N N needs two nucleons), so it always survives the no-meson cut -> RES-H is
    # never CC0pi, with OR without FSI.  ACHILLES confirms abs_frac(H) = 0 exactly.  The earlier
    # 12C-config cascade applied to H was an artifact (faked ~0.27e-6 nb of CC0pi).
    knu, kmu, pN, pPi, w = generate_H(n, seed=seed)
    return _obs(knu, kmu, np.tile([938.27, 0, 0, 0], (len(w), 1)), pN, w * 0.0)


def accumulate(fn, n, fsi, perseed_div):
    """Run NSEED seeds, concatenate; divide weights so they sum to the MEAN nb sigma."""
    acc = {k: [] for k in ("W", "Q2", "dalphat", "dpt", "w")}
    for sd in range(NSEED):
        d = fn(n, sd, fsi)
        for k in acc: acc[k].append(d[k])
        print(f"    {fn.__name__} fsi={fsi} seed {sd+1}/{NSEED}: +{len(d['w'])}", flush=True)
    out = {k: np.concatenate(v) if v[0].size or len(v) else np.array([]) for k, v in acc.items()}
    out["w"] = out["w"] / perseed_div
    return out


if __name__ == "__main__":
    CONTRIB = {}
    for fsi in (False, True):
        print(f"=== FSI={fsi} ===", flush=True)
        CONTRIB[("QE-C", fsi)] = accumulate(qe_C, NQE, fsi, NSEED)
        CONTRIB[("RES-C", fsi)] = accumulate(res_C, NRES, fsi, NSEED)
        CONTRIB[("RES-H", fsi)] = accumulate(res_H, NH, fsi, NSEED)
    np.savez("data/oracle/cc0pi_disaggregated.npz",
             **{f"{c}_{f}_{k}": CONTRIB[(c, f)][k] for (c, f) in CONTRIB for k in CONTRIB[(c, f)]})
    for (c, f), d in CONTRIB.items():
        print(f"  {c:6s} FSI={str(f):5s}: sigma_CC0pi = {d['w'].sum():.4e} nb  ({len(d['w'])} ev)")

    OBS = [("W", np.linspace(850, 1700, 26), "vertex W [MeV]"),
           ("Q2", np.linspace(0, 1.4, 22), r"$Q^2$ [GeV$^2$]"),
           ("dalphat", np.linspace(0, np.pi, 22), r"$\delta\alpha_T$ [rad]"),
           ("dpt", np.linspace(0, 800, 22), r"$\delta p_T$ [MeV]")]
    COL = {"QE-C": "C0", "RES-C": "C3", "RES-H": "C2"}
    fig, axes = plt.subplots(len(OBS), 2, figsize=(13, 15))
    for r, (key, bins, xl) in enumerate(OBS):
        bw = np.diff(bins); ctr = 0.5 * (bins[1:] + bins[:-1])
        for col, fsi in enumerate((False, True)):
            ax = axes[r, col]; tot = np.zeros(len(ctr))
            for c in ("QE-C", "RES-C", "RES-H"):
                d = CONTRIB[(c, fsi)]
                if d["w"].sum() <= 0:
                    continue
                h, _ = np.histogram(d[key], bins=bins, weights=d["w"]); h = h / bw
                ax.step(bins, np.append(h, h[-1]), where="post", color=COL[c], lw=1.6,
                        label=f"{c} ({d['w'].sum():.2e})")
                tot += h
            ax.step(bins, np.append(tot, tot[-1]), where="post", color="k", lw=2.0, ls="--", label="sum")
            ax.set_xlabel(xl); ax.set_ylabel(r"d$\sigma$/dx [nb/unit]")
            ax.set_title(f"{'with FSI' if fsi else 'no FSI'} — {xl}")
            ax.legend(fontsize=7); ax.set_ylim(bottom=0)
    fig.suptitle("ADoNIS CC0$\\pi$-Np disaggregated (absolute, first-principles): QE/RES x C/H x FSI   "
                 "[QE-H=0; RES no-FSI=0]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig("paper_figures/cc0pi_disaggregated.png", dpi=120); print("wrote paper_figures/cc0pi_disaggregated.png")
