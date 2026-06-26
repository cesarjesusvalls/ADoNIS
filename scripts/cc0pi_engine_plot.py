"""Render the T2K CC0pi-Np TKI figure (dsigma/d delta_pT, d delta_alphaT, dQ2) from the saved CC0pi
engine RICH bank (data/oracle/t2k_cc0pi.npz, gen_cc_engine_rich.py qe).  ADoNIS = the
FAITHFUL BFS cascade engine on 12C QE (+ free-proton H, no FSI) -- apples-to-apples QE-vs-QE against
ACHILLES proc==200 (QE on CH).  No engine rerun: any cut/binning is pure re-binning of the bank.

Signal (T2K CC0pi-Np): exactly-0 pion (veto any surviving NN-created pion), p_mu>250, cos_mu>-0.6,
leading proton 450<p_p<1000 MeV/c with cos_p>0.4.  delta_pT/alphaT from muon + leading proton.

Usage: python -u scripts/cc0pi_engine_plot.py [bank.npz]
Output: paper_figures/cc0pi_tki_ENGINE_compare.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BANK = sys.argv[1] if len(sys.argv) > 1 else "data/oracle/t2k_cc0pi.npz"
_PIONS = (211, 111, -211)
# T2K CC0pi-Np cuts
MU_LO, COSMU_LO = 250.0, -0.6
P_LO, P_HI, COSP_LO = 450.0, 1000.0, 0.4


def engine_signal(bank):
    b = dict(np.load(bank))
    n = len(b["w"]); ar = np.arange(n)
    mu = b["mu"]
    pmu = np.linalg.norm(mu[:, 1:], axis=1); cmu = mu[:, 3] / np.clip(pmu, 1e-9, None)
    # leading in-window proton among the stored top-6 (window 450-1000, cos_p>0.4)
    prot = b["prot"]
    pm = np.linalg.norm(prot[:, :, 1:], axis=2); cth = prot[:, :, 3] / np.clip(pm, 1e-9, None)
    inwin = (pm > P_LO) & (pm < P_HI) & (cth > COSP_LO)
    mm = np.where(inwin, pm, -1.0); j = np.argmax(mm, axis=1); bm = mm[ar, j]
    lead = prot[ar, j]; has_p = bm > 0
    no_pion = ~np.isin(b["cr_pid"], _PIONS)               # CC0pi: veto any surviving created pion
    sel = (b["w"] > 0) & has_p & no_pion & (pmu > MU_LO) & (cmu > COSMU_LO)
    mu, lead = mu[sel], lead[sel]
    # TKI from muon + leading proton (no pion in CC0pi):  delta_pT = |pT_mu + pT_p|
    lt = mu[:, 1:3]; ht = lead[:, 1:3]; dpt_vec = lt + ht
    dpt = np.linalg.norm(dpt_vec, axis=1)
    c = -np.sum(lt * dpt_vec, axis=1) / (np.linalg.norm(lt, axis=1) * dpt + 1e-12)
    dat = np.arccos(np.clip(c, -1.0, 1.0))
    q = (b["nu"] - b["mu"])[sel]
    Q2 = (np.sum(q[:, 1:] ** 2, axis=1) - q[:, 0] ** 2) / 1e6
    return dict(dpt=dpt, dalphat=dat, Q2=Q2, w=b["w"][sel])


VARS = [("dpt", np.linspace(0, 800, 21), r"$\delta p_T$ [MeV]"),
        ("dalphat", np.linspace(0, np.pi, 21), r"$\delta\alpha_T$ [rad]"),
        ("Q2", np.linspace(0, 1.4, 21), r"$Q^2$ [GeV$^2$]")]


def main():
    A = engine_signal(BANK)
    H = dict(np.load("data/oracle/t2k_cc0pi_tki_adonis_H.npz"))      # free-proton H, no FSI (1 sample)
    # ADoNIS QE on CH = 12C engine (already absolute) + free-H
    ado = {k: np.concatenate([A[k], H[k]]) for k in ("dpt", "dalphat", "Q2", "w")}
    ach = dict(np.load("data/oracle/t2k_cc0pi_tki_achilles.npz"))
    qe = ach["proc"] == 200                                          # QE on CH (apples-to-apples)
    ascale = float(ach["weight_to_nb"])                              # GenCrossSection/sum_w from header
    ach_v = {k: ach[k][qe] for k in ("dpt", "dalphat", "Q2")}; ach_w = ach["w"][qe] * ascale
    sA, sH = ado["w"].sum(), ach_w.sum()
    print(f"ADoNIS engine-12C({A['w'].sum():.4e}) + free-H({H['w'].sum():.4e}) = {sA:.4e} nb")
    print(f"ACHILLES QE(proc200) {sH:.4e} nb   ACH/ADO {sH/sA:.4f}")
    print(f"{'var':8s} chi2/ndf  ACH/ADO")
    fig, ax = plt.subplots(2, len(VARS), figsize=(6.2 * len(VARS), 7.6), height_ratios=[3, 1],
                           squeeze=False, sharex="col")
    for c, (k, edg, xl) in enumerate(VARS):
        bw = np.diff(edg); ctr = 0.5 * (edg[1:] + edg[:-1])
        da, _ = np.histogram(ach_v[k], edg, weights=ach_w); ea = np.sqrt(np.histogram(ach_v[k], edg, weights=ach_w**2)[0]) / bw; da /= bw
        dd, _ = np.histogram(ado[k], edg, weights=ado["w"]); ed = np.sqrt(np.histogram(ado[k], edg, weights=ado["w"]**2)[0]) / bw; dd /= bw
        a0, a1 = ax[0, c], ax[1, c]
        a0.fill_between(edg, np.append(da-ea, (da-ea)[-1]), np.append(da+ea, (da+ea)[-1]), step="post", color="0.55", alpha=0.55, lw=0, label="ACH stat")
        a0.step(edg, np.append(da, da[-1]), where="post", color="0.3", lw=1.3, label="ACHILLES QE")
        a0.errorbar(ctr, dd, yerr=ed, fmt="s", color="C0", ms=3, capsize=2, lw=0.9, label="ENGINE (12C+H)")
        a0.set_title(xl, fontsize=9); a0.set_ylim(bottom=0)
        if c == 0: a0.legend(fontsize=8)
        m = (da > 0) & (dd > 0); chi2 = float(np.sum((da[m]-dd[m])**2 / (ea[m]**2+ed[m]**2))); ndf = int(m.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            r = da/dd; re = r*np.sqrt((ed/dd)**2+(ea/da)**2)
        a1.axhspan(0.9, 1.1, color="green", alpha=0.12); a1.axhline(1.0, ls="--", color="green", lw=0.7)
        a1.errorbar(ctr[m], r[m], yerr=re[m], fmt="o", color="C3", ms=3, capsize=2, lw=0.8)
        a1.set_ylim(0.6, 1.4); a1.set_xlabel(xl, fontsize=9); a1.text(0.04, 0.83, f"{chi2/max(ndf,1):.2f}", transform=a1.transAxes, fontsize=9)
        print(f"{k:8s}  {chi2/max(ndf,1):6.2f}   {np.sum(da*bw)/max(np.sum(dd*bw),1e-30):.3f}")
    fig.suptitle(f"T2K CC0$\\pi$-Np  FAITHFUL ENGINE (12C QE + H) vs ACHILLES QE (proc200)  "
                 f"ACH/ADO {sH/sA:.3f}  N={len(A['w'])}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); out = "paper_figures/cc0pi_tki_ENGINE_compare.png"
    fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    main()
