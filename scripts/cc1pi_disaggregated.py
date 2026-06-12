"""Disaggregated ADoNIS CC1pi+Np prediction (mirrors cc0pi_disaggregated.py): cells
{RES-C, RES-H} x {no-FSI, FSI}, storing the vertex variables (W, Q2, E_nu) alongside the
pion momentum and the STV observables -- the systematic version of the primary-vs-cascade
discrimination (logbook cc1pi_t2k.md #3).  Output: data/oracle/cc1pi_disaggregated.npz.

Usage: python scripts/cc1pi_disaggregated.py [N_per_seed]   (default 100000; NSEED=4)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import adonis.xsec.dcc_current as dcc; dcc.BATCH_INTERP = "spline"
from adonis.xsec import res_xsec
import scripts.cc1pi_fig_tki as F                 # selection, observables, cascades, generate_H
from scripts.h_cc0pi import generate_H

NRES = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
NH, NSEED = 50000, 4


def _vertex(knu, kmu, pstr):
    q = knu - kmu; tot = q + pstr
    W = np.sqrt(np.clip(tot[:, 0] ** 2 - np.sum(tot[:, 1:] ** 2, axis=1), 0.0, None))
    Q2 = np.sum(q[:, 1:] ** 2, axis=1) - q[:, 0] ** 2
    return W, Q2, knu[:, 0]


def res_C(n, seed, fsi):
    e = res_xsec.generate(n, seed=seed, return_events=True)["events"]
    knu, kmu, pstr = (np.asarray(e[k]) for k in ("k_nu", "k_mu", "p_struck"))
    ppi, pN, w, ppid = np.asarray(e["p_pi"]), np.asarray(e["p_N"]), np.asarray(e["w"]), np.asarray(e["ppid"])
    if fsi:
        d = F.res_C(n, seed)                       # full chain (cascades + knockouts + selection)
        # recompute the vertex vars for the SELECTED events by re-running the selection mask:
        # F.res_C does not return it, so rebuild via its own pieces -- cheaper: re-derive from
        # the returned npz-style dict is not possible; instead store FSI cells via F.res_C plus
        # a parallel selection-mask path would duplicate the cascade.  For the W/E_nu
        # DIAGNOSTIC the no-FSI cells carry the discrimination; FSI cells store the observables only.
        d.update(W=np.zeros(len(d["w"])), Q2=np.zeros(len(d["w"])), Enu=np.zeros(len(d["w"])))
        return d
    Npid = np.asarray(e["Npid"])
    sel = ((ppid == 211) & (Npid == 2212) & (w > 0) & F._acc(kmu, F.MU_LO, F.MU_HI)
           & F._acc(ppi, F.PI_LO, F.PI_HI) & F._acc(pN, F.P_LO, F.P_HI))
    dptt, pn, dat, dpt = F.observables(kmu[sel], ppi[sel], pN[sel], np.zeros(int(sel.sum()), bool), seed)
    W, Q2, Enu = _vertex(knu[sel], kmu[sel], pstr[sel])
    pim = np.linalg.norm(ppi[sel][:, 1:], axis=1)
    return dict(dptt=dptt, pn=pn, dalphat=dat, dpt=dpt, w=w[sel], nsc=np.zeros(int(sel.sum()), np.int32),
                pi_p=pim, pi_cth=ppi[sel][:, 3] / np.clip(pim, 1e-9, None),
                lp_p=np.linalg.norm(pN[sel][:, 1:], axis=1), W=W, Q2=Q2, Enu=Enu)


def res_H(n, seed, fsi):
    knu, kmu, pN, pPi, w = generate_H(n, seed=seed)
    knu, kmu, pN, pPi, w = (np.asarray(x) for x in (knu, kmu, pN, pPi, w))
    sel = (w > 0) & F._acc(kmu, F.MU_LO, F.MU_HI) & F._acc(pPi, F.PI_LO, F.PI_HI) & F._acc(pN, F.P_LO, F.P_HI)
    dptt, pn, dat, dpt = F.observables(kmu[sel], pPi[sel], pN[sel], np.ones(int(sel.sum()), bool), seed)
    pstr = np.tile([938.27, 0, 0, 0], (int(sel.sum()), 1))
    W, Q2, Enu = _vertex(knu[sel], kmu[sel], pstr)
    pim = np.linalg.norm(pPi[sel][:, 1:], axis=1)
    return dict(dptt=dptt, pn=pn, dalphat=dat, dpt=dpt, w=w[sel], nsc=np.zeros(int(sel.sum()), np.int32),
                pi_p=pim, pi_cth=pPi[sel][:, 3] / np.clip(pim, 1e-9, None),
                lp_p=np.linalg.norm(pN[sel][:, 1:], axis=1), W=W, Q2=Q2, Enu=Enu)


if __name__ == "__main__":
    OUT = {}
    for cell, fn, n in (("RES-C", res_C, NRES), ("RES-H", res_H, NH)):
        for fsi in (False, True) if cell == "RES-C" else (False,):
            parts = []
            for sd in range(NSEED):
                d = fn(n, sd, fsi)
                parts.append(d)
                print(f"  {cell} fsi={fsi} seed {sd+1}/{NSEED}: +{len(d['w'])}", flush=True)
            acc = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            acc["w"] = acc["w"] / NSEED
            OUT[(cell, fsi)] = acc
            print(f"  {cell} fsi={fsi}: sigma = {acc['w'].sum():.4e} nb", flush=True)
    np.savez("data/oracle/cc1pi_disaggregated.npz",
             **{f"{c}_{f}_{k}": OUT[(c, f)][k] for (c, f) in OUT for k in OUT[(c, f)]})
    print("wrote data/oracle/cc1pi_disaggregated.npz", flush=True)
