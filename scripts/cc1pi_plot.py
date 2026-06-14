"""Render all CC1pi+Np diff-xsec + ACH/ADO ratios from the RICH banks under a chosen signal
definition -- pure re-binning, NO regeneration.  Change the sigdef (CLI flags) to redraw.

Banks: data/oracle/t2k_cc1pi_rich_adonis.npz  (ADoNIS, pre+post)
       data/oracle/t2k_cc1pi_rich_ach_FSI.npz / _nofsi.npz  (ACHILLES, post / pre-FSI)
ADoNIS weights are absolute nb already; ACHILLES weights x weight_to_nb (hepmc header).

Usage examples:
  python scripts/cc1pi_plot.py                              # default FSI signal, carbon
  python scripts/cc1pi_plot.py --nofsi                      # no-FSI signal
  python scripts/cc1pi_plot.py --count-recoil-neutron       # ADoNIS counts the recoil neutron (the OLD bug)
  python scripts/cc1pi_plot.py --target CH
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import scripts.cc1pi_signal as S

VARS = [("pn", np.array([0, 120, 240, 600, 1500.]), r"$p_N$ [MeV/c]"),
        ("dptt", np.array([-700, -300, -100, 100, 300, 700.]), r"$\delta p_{TT}$"),
        ("dalphat", np.linspace(0, np.pi, 7), r"$\delta\alpha_T$ [rad]"),
        ("W", np.linspace(1080, 1700, 13), r"$W$ [MeV]"),
        ("Q2", np.linspace(0, 1.5, 13), r"$Q^2$ [GeV$^2$]"),
        ("pi_p", np.linspace(150, 1200, 13), r"$p_\pi$ [MeV/c]"),
        ("lp_p", np.linspace(450, 1200, 13), r"lead $p_p$ [MeV/c]")]


def sigdef_from_argv():
    sd = dict(S.DEFAULT)
    if "--nofsi" in sys.argv: sd["fsi"] = False
    if "--count-recoil-neutron" in sys.argv: sd["count_recoil_neutron"] = True
    if "--native-only" in sys.argv: sd["proton_source"] = "native"
    if "--no-proton" in sys.argv: sd["require_proton"] = False
    if "--one-proton" in sys.argv: sd["proton_count"] = "eq1"
    if "--no-kine" in sys.argv:                              # require >=1 proton, but NO kinematic cuts on anyone
        sd["mu_win"] = (0., 1e12); sd["pi_win"] = (0., 1e12); sd["p_win"] = (0., 1e12); sd["cth"] = -2.0
    if "--target" in sys.argv: sd["target"] = sys.argv[sys.argv.index("--target") + 1]
    return sd


def main():
    sd = sigdef_from_argv()
    # without a proton cut there is no leading proton -> only nucleon-independent observables are defined
    vars_ = VARS if sd.get("require_proton", True) else [v for v in VARS if v[0] in ("W", "Q2", "pi_p")]
    ado = dict(np.load("data/oracle/t2k_cc1pi_rich_adonis.npz"))
    achf = "data/oracle/t2k_cc1pi_rich_ach_%s.npz" % ("FSI" if sd["fsi"] else "nofsi")
    ach = dict(np.load(achf))
    aw2nb = float(ach["weight_to_nb"])
    A = S.ado_select(ado, sd)
    H = S.ach_select(ach, sd); H["w"] = H["w"] * aw2nb
    print(f"sigdef = {sd}")
    print(f"selected sigma: ADoNIS {A['w'].sum():.4e}  ACHILLES {H['w'].sum():.4e}  ACH/ADO {H['w'].sum()/max(A['w'].sum(),1e-30):.3f}\n")
    print(f"{'variable':10s} chi2/ndf  integral ACH/ADO")
    fig, axes = plt.subplots(2, len(vars_), figsize=(3.4 * len(vars_), 6.2), height_ratios=[3, 1],
                             squeeze=False, sharex="col")
    for c, (key, edges, xlab) in enumerate(vars_):
        bw = np.diff(edges); ctr = 0.5 * (edges[1:] + edges[:-1])
        da, _ = np.histogram(H[key], edges, weights=H["w"]); ea2, _ = np.histogram(H[key], edges, weights=H["w"] ** 2)
        dd, _ = np.histogram(A[key], edges, weights=A["w"]); ed2, _ = np.histogram(A[key], edges, weights=A["w"] ** 2)
        da, ea, dd, ed = da / bw, np.sqrt(ea2) / bw, dd / bw, np.sqrt(ed2) / bw
        ax, axr = axes[0, c], axes[1, c]
        ax.fill_between(edges, np.append(da - ea, (da - ea)[-1]), np.append(da + ea, (da + ea)[-1]),
                        step="post", color="0.55", alpha=0.55, lw=0, label="ACHILLES stat. unc.")
        ax.step(edges, np.append(da, da[-1]), where="post", color="0.3", lw=1.3, label="ACHILLES")
        ax.errorbar(ctr, dd, yerr=ed, fmt="s", color="C0", ms=3, capsize=2, lw=0.9, label="ADoNIS")
        ax.set_title(xlab, fontsize=9); ax.set_ylim(bottom=0)
        if c == 0: ax.legend(fontsize=7)
        m = (da > 0) & (dd > 0)
        chi2 = float(np.sum((da[m] - dd[m]) ** 2 / (ea[m] ** 2 + ed[m] ** 2))); ndf = int(m.sum())
        iA, iD = np.sum(da * bw), np.sum(dd * bw)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = da / dd; re = r * np.sqrt((ed / dd) ** 2 + (ea / da) ** 2)
        axr.axhspan(0.9, 1.1, color="green", alpha=0.12); axr.axhline(1.0, ls="--", color="green", lw=0.7)
        axr.errorbar(ctr[m], r[m], yerr=re[m], fmt="o", color="C3", ms=3, capsize=2, lw=0.8)
        axr.set_ylim(0.5, 1.6); axr.set_xlabel(xlab, fontsize=8)
        axr.text(0.04, 0.83, f"{chi2/max(ndf,1):.1f}", transform=axr.transAxes, fontsize=9)
        print(f"{key:10s}  {chi2/max(ndf,1):6.2f}   {iA/max(iD,1e-30):.3f}")
    tag = ("noFSI" if not sd["fsi"] else "FSI") + ("_recoilN" if sd["count_recoil_neutron"] else "") + ("_noProtonCut" if not sd.get("require_proton", True) else "") + ("_noKine" if "--no-kine" in sys.argv else "") + ("_1prot" if sd.get("proton_count") == "eq1" else "") + f"_{sd['target']}"
    fig.suptitle(f"CC1$\\pi^+$Np  ADoNIS vs ACHILLES  [{tag}]   ACH/ADO integral {H['w'].sum()/max(A['w'].sum(),1e-30):.3f}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = f"paper_figures/cc1pi_ratios_{tag}.png"; fig.savefig(out, dpi=120); print("\nwrote", out)


if __name__ == "__main__":
    main()
