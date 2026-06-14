"""Render the CC1pi+Np diff-xsec + ACH/ADO ratio figure FROM the saved engine RICH bank
(data/oracle/t2k_cc1pi_engine_rich.npz, produced by gen_cc_engine_rich.py res).  No engine rerun:
the bank holds per-event everything needed to apply ANY signal definition as pure re-binning.

Signal (mirrors cascade_engine_figure.py exactly):
  pi+ = primary pion (pid_pi/pi_post) OR NN-created pion (cr_pid/cr_p4); require exactly one pi+ and
        no other meson over {primary, created} (111/-211/-1 = other meson).
  proton = leading in-window proton among the stored top-M proton terminals (prot/prot_origin/prot_gen).
  acceptance windows from cc1pi_signal.DEFAULT; observables from cc1pi_fig_tki.observables (single source).

Usage: python -u scripts/cc1pi_engine_plot.py [bank.npz]
Output: paper_figures/cc1pi_ratios_ENGINE_carbon.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import scripts.cc1pi_fig_tki as F            # observables() (single source)
import scripts.cc1pi_signal as S

BANK = sys.argv[1] if len(sys.argv) > 1 else "data/oracle/t2k_cc1pi_engine_rich.npz"
COS70 = np.cos(np.deg2rad(70.0))
MU_LO, MU_HI = S.DEFAULT["mu_win"]; PI_LO, PI_HI = S.DEFAULT["pi_win"]; P_LO, P_HI = S.DEFAULT["p_win"]


def _acc(p4, lo, hi):
    m = np.linalg.norm(p4[:, 1:], axis=1)
    return (m > lo) & (m < hi) & (p4[:, 3] / np.clip(m, 1e-9, None) > COS70)


def qe_created_signal(bank="data/oracle/t2k_cc0pi_engine_rich.npz"):
    """CC1pi contribution from the QE bank: a QE event whose CASCADE created a surviving pi+ (NN
    inelastic).  Symmetric to the RES-pion-absorbed term in CC0pi -- a full-CC ACHILLES run (the
    reference) contains these, so the RES-only engine must add them for an apples-to-apples CC1pi."""
    b = dict(np.load(bank))
    n = len(b["w"]); ar = np.arange(n)
    kmu, cr_p4 = b["mu"], b["cr_p4"]
    cp = b["cr_pid"]                                      # primary pion is absent for QE (pid_pi==0)
    cr_pip = (cp == 211)
    n_pip = cr_pip.astype(int)
    n_other = np.isin(cp, [111, -211, -1]).astype(int)
    pi_f = cr_p4
    signal_meson = (n_pip == 1) & (n_other == 0)
    prot = b["prot"]
    pm = np.linalg.norm(prot[:, :, 1:], axis=2); cth = prot[:, :, 3] / np.clip(pm, 1e-9, None)
    inwin = (pm > P_LO) & (pm < P_HI) & (cth > COS70)
    mm = np.where(inwin, pm, -1.0); j = np.argmax(mm, axis=1); bm = mm[ar, j]
    best = prot[ar, j]; has_p = bm > 0
    sel = signal_meson & has_p & (b["w"] > 0) & _acc(kmu, MU_LO, MU_HI) & _acc(pi_f, PI_LO, PI_HI)
    dptt, pn, dat, _ = F.observables(kmu[sel], pi_f[sel], best[sel], np.zeros(int(sel.sum()), bool), 0)
    qv = (b["nu"] - kmu)[sel]; totv = qv + b["struck"][sel]
    Wv = np.sqrt(np.clip(totv[:, 0] ** 2 - np.sum(totv[:, 1:] ** 2, axis=1), 0, None))
    Q2v = (np.sum(qv[:, 1:] ** 2, axis=1) - qv[:, 0] ** 2) / 1e6
    pim = np.linalg.norm(pi_f[sel][:, 1:], axis=1); lpm = np.linalg.norm(best[sel][:, 1:], axis=1)
    return dict(pn=pn, dptt=dptt, dalphat=dat, W=Wv, Q2=Q2v, pi_p=pim, lp_p=lpm, w=b["w"][sel])


def engine_signal(bank):
    b = dict(np.load(bank))
    n = len(b["w"]); ar = np.arange(n)
    kmu, pi_post, cr_p4 = b["mu"], b["pi_post"], b["cr_p4"]
    pp, cp = b["pid_pi"], b["cr_pid"]                    # cr_pid already 0 when the created pion is dead
    prim_pip = (pp == 211); cr_pip = (cp == 211)
    n_pip = prim_pip.astype(int) + cr_pip.astype(int)
    n_other = np.isin(pp, [111, -211, -1]).astype(int) + np.isin(cp, [111, -211, -1]).astype(int)
    pi_f = np.where(prim_pip[:, None], pi_post, cr_p4)   # the surviving pi+ (primary or created)
    signal_meson = (n_pip == 1) & (n_other == 0)
    # leading in-window proton among the stored top-M proton terminals
    prot = b["prot"]                                     # (n, M, 4); zeros for empty slots
    pm = np.linalg.norm(prot[:, :, 1:], axis=2); cth = prot[:, :, 3] / np.clip(pm, 1e-9, None)
    inwin = (pm > P_LO) & (pm < P_HI) & (cth > COS70)
    mm = np.where(inwin, pm, -1.0); j = np.argmax(mm, axis=1); bm = mm[ar, j]
    best = prot[ar, j]; has_p = bm > 0
    sel = signal_meson & has_p & (b["w"] > 0) & _acc(kmu, MU_LO, MU_HI) & _acc(pi_f, PI_LO, PI_HI)
    dptt, pn, dat, _ = F.observables(kmu[sel], pi_f[sel], best[sel], np.zeros(int(sel.sum()), bool), 0)
    qv = (b["nu"] - kmu)[sel]; totv = qv + b["struck"][sel]
    Wv = np.sqrt(np.clip(totv[:, 0] ** 2 - np.sum(totv[:, 1:] ** 2, axis=1), 0, None))
    Q2v = (np.sum(qv[:, 1:] ** 2, axis=1) - qv[:, 0] ** 2) / 1e6
    pim = np.linalg.norm(pi_f[sel][:, 1:], axis=1); lpm = np.linalg.norm(best[sel][:, 1:], axis=1)
    return dict(pn=pn, dptt=dptt, dalphat=dat, W=Wv, Q2=Q2v, pi_p=pim, lp_p=lpm, w=b["w"][sel])


VARS = [("pn", np.array([0, 120, 240, 600, 1500.]), r"$p_N$"),
        ("dptt", np.array([-700, -300, -100, 100, 300, 700.]), r"$\delta p_{TT}$"),
        ("dalphat", np.linspace(0, np.pi, 7), r"$\delta\alpha_T$"),
        ("W", np.linspace(1080, 1700, 13), r"$W$"),
        ("Q2", np.linspace(0, 1.5, 13), r"$Q^2$"),
        ("pi_p", np.linspace(150, 1200, 13), r"$p_\pi$"),
        ("lp_p", np.linspace(450, 1200, 13), r"lead $p_p$")]


def main():
    A_res = engine_signal(BANK)
    A_qe = qe_created_signal()                                      # QE -> cascade-created pi+ (CC1pi too)
    A = {k: np.concatenate([A_res[k], A_qe[k]]) for k in A_res}
    ach = dict(np.load("data/oracle/t2k_cc1pi_rich_ach_FSI.npz"))
    H = S.ach_select(ach, dict(S.DEFAULT)); H["w"] = H["w"] * float(ach["weight_to_nb"])
    print(f"bank {BANK}  signal {len(A['w'])} ev  (RES {len(A_res['w'])} + QE-created-pi {len(A_qe['w'])}, "
          f"sigma RES {A_res['w'].sum():.3e} + QE {A_qe['w'].sum():.3e})")
    print(f"selected sigma: ENGINE {A['w'].sum():.4e}  ACH {H['w'].sum():.4e}  "
          f"ACH/ADO {H['w'].sum()/max(A['w'].sum(),1e-30):.3f}")
    print(f"{'var':8s} chi2/ndf  ACH/ADO")
    fig, ax = plt.subplots(2, len(VARS), figsize=(3.4 * len(VARS), 6.2), height_ratios=[3, 1],
                           squeeze=False, sharex="col")
    for c, (k, edg, xl) in enumerate(VARS):
        bw = np.diff(edg); ctr = 0.5 * (edg[1:] + edg[:-1])
        da, _ = np.histogram(H[k], edg, weights=H["w"]); ea = np.sqrt(np.histogram(H[k], edg, weights=H["w"]**2)[0]) / bw; da /= bw
        dd, _ = np.histogram(A[k], edg, weights=A["w"]); ed = np.sqrt(np.histogram(A[k], edg, weights=A["w"]**2)[0]) / bw; dd /= bw
        a0, a1 = ax[0, c], ax[1, c]
        a0.fill_between(edg, np.append(da-ea, (da-ea)[-1]), np.append(da+ea, (da+ea)[-1]), step="post", color="0.55", alpha=0.55, lw=0, label="ACH stat")
        a0.step(edg, np.append(da, da[-1]), where="post", color="0.3", lw=1.3, label="ACHILLES")
        a0.errorbar(ctr, dd, yerr=ed, fmt="s", color="C0", ms=3, capsize=2, lw=0.9, label="ENGINE")
        a0.set_title(xl, fontsize=9); a0.set_ylim(bottom=0)
        if c == 0: a0.legend(fontsize=7)
        m = (da > 0) & (dd > 0); chi2 = float(np.sum((da[m]-dd[m])**2 / (ea[m]**2+ed[m]**2))); ndf = int(m.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            r = da/dd; re = r*np.sqrt((ed/dd)**2+(ea/da)**2)
        a1.axhspan(0.9, 1.1, color="green", alpha=0.12); a1.axhline(1.0, ls="--", color="green", lw=0.7)
        a1.errorbar(ctr[m], r[m], yerr=re[m], fmt="o", color="C3", ms=3, capsize=2, lw=0.8)
        a1.set_ylim(0.5, 1.6); a1.set_xlabel(xl, fontsize=8); a1.text(0.04, 0.83, f"{chi2/max(ndf,1):.1f}", transform=a1.transAxes, fontsize=9)
        print(f"{k:8s}  {chi2/max(ndf,1):6.2f}   {np.sum(da*bw)/max(np.sum(dd*bw),1e-30):.3f}")
    fig.suptitle(f"CC1$\\pi^+$Np  FAITHFUL ENGINE (rich bank) vs ACHILLES (carbon)  "
                 f"ACH/ADO {H['w'].sum()/max(A['w'].sum(),1e-30):.3f}  N={len(A['w'])}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); out = "paper_figures/cc1pi_ratios_ENGINE_carbon.png"
    fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    main()
