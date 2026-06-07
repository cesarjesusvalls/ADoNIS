"""ADoNIS QE PRIMARY (no-FSI) T2K CC0pi-Np STV, current bit-exact code.  Controlled twin of the
ACHILLES `run_nofsi_qe.yml` run (QE_Spectral_Func, T2K flux, 12C, Cascade:Run:False).  Same
CC0pi-Np cuts as scripts/extract_t2k_cc0pi_tki.py so the two are compared apples-to-apples.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from adonis.xsec import qe_xsec

ROOT = Path(__file__).resolve().parents[1]
MU_LO, COSMU = 250.0, -0.6
P_LO, P_HI, COSP = 450.0, 1000.0, 0.4


def cc0pi(mu, prot, w):
    pmu = np.linalg.norm(mu[:, 1:], axis=1); cmu = mu[:, 3] / np.clip(pmu, 1e-9, None)
    pl = np.linalg.norm(prot[:, 1:], axis=1); cl = prot[:, 3] / np.clip(pl, 1e-9, None)
    sel = (w > 0) & (pmu > MU_LO) & (cmu > COSMU) & (pl > P_LO) & (pl < P_HI) & (cl > COSP)
    lt = mu[:, 1:3]; pt = prot[:, 1:3]; dv = lt + pt
    dpt = np.linalg.norm(dv, axis=1)
    c = -np.sum(lt * dv, axis=1) / (np.linalg.norm(lt, axis=1) * dpt + 1e-9)
    dat = np.arccos(np.clip(c, -1, 1))
    return dpt[sel], dat[sel], w[sel]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500000
    nseed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    dpt, dat, w = [], [], []
    for sd in range(nseed):
        r = qe_xsec.sample_importance(n, seed=sd)
        wv = np.asarray(r["w"]) / n
        a, b, c = cc0pi(np.asarray(r["k_mu"]), np.asarray(r["p_out"]), wv)
        dpt.append(a); dat.append(b); w.append(c)
    dpt = np.concatenate(dpt); dat = np.concatenate(dat); w = np.concatenate(w)
    neff = w.sum() ** 2 / np.sum(w ** 2)
    print(f"ADoNIS no-FSI QE CC0pi-Np: {len(w)} events  Neff={neff:.0f}  <dpT>={np.average(dpt,weights=w):.1f}")
    np.savez(ROOT / "data" / "oracle" / "nofsi_qe_adonis.npz", dpt=dpt, dalphat=dat, w=w)
    print("wrote data/oracle/nofsi_qe_adonis.npz")
