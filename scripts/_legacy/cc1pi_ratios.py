"""Disaggregated ACH/ADO ratio check for T2K CC1pi+Np with the DIFFERENTIABLE factorized chain.
Per-variable (pN, dpTT, daT, vertex W, Q2, pi_p, leading-proton p) ACH/ADO ratios + chi2/ndf,
CONSISTENTLY WEIGHTED on both sides (ACH: w x weight_to_nb from the hepmc header; ADO: per-event
nb weights).  Establishes where the factorized prediction actually stands vs ACHILLES.

Usage: python scripts/cc1pi_ratios.py [N_per_seed=120000] [NSEED=4]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import scripts.cc1pi_fig_tki as F            # factorized res_C / res_H, selection, observables
from analysis.utils.hepmc import weight_to_nb_of

NRES = int(sys.argv[1]) if len(sys.argv) > 1 else 120000
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 4
NH = 40000
KEYS = ("pn", "dptt", "dalphat", "W", "Q2", "pi_p", "lp_p")

# (var, ACH-npz key, edges, x-label, optional rad->deg)
VARS = [
    ("pn",      np.array([0, 120, 240, 600, 1500.0]),      r"$p_N$ [MeV/c]"),
    ("dptt",    np.array([-700, -300, -100, 100, 300, 700.0]), r"$\delta p_{TT}$ [MeV/c]"),
    ("dalphat", np.linspace(0, np.pi, 7),                   r"$\delta\alpha_T$ [rad]"),
    ("W",       np.linspace(1080, 1700, 13),                r"vertex $W$ [MeV]"),
    ("Q2",      np.linspace(0, 1.5, 13),                    r"$Q^2$ [GeV$^2$]"),
    ("pi_p",    np.linspace(150, 1200, 13),                 r"$p_\pi$ [MeV/c]"),
    ("lp_p",    np.linspace(450, 1200, 13),                 r"lead $p_p$ [MeV/c]"),
]


def accumulate():
    cells = {}
    for cell, fn, n in (("RES-C", F.res_C, NRES), ("RES-H", F.res_H, NH)):
        parts = [fn(n, sd) for sd in range(NSEED)]
        for sd in range(NSEED):
            print(f"  {cell} seed {sd+1}/{NSEED}: +{len(parts[sd]['w'])}", flush=True)
        d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        d["w"] = d["w"] / NSEED
        cells[cell] = d
    ado = {k: np.concatenate([cells[c][k] for c in cells]) for k in KEYS + ("w",)}
    return ado, cells


def main():
    ado, cells = accumulate()
    np.savez("data/oracle/t2k_cc1pi_ratios_adonis.npz", **{k: ado[k] for k in KEYS + ("w",)})
    print(f"sigma_CC1pi(tight) factorized: RES-C {cells['RES-C']['w'].sum():.4e}  "
          f"RES-H {cells['RES-H']['w'].sum():.4e} nb  (ADoNIS arrays cached)", flush=True)
    ach = np.load("data/oracle/t2k_cc1pi_tki_achilles.npz")
    ach_w = np.asarray(ach["w"]) * weight_to_nb_of(ach)          # consistent absolute weighting

    fig, axes = plt.subplots(2, len(VARS), figsize=(3.6 * len(VARS), 6.4),
                             height_ratios=[3, 1])
    print("  [W is a vertex DIAGNOSTIC: ACH uses the status-2 struck nucleon, res_C the de-Forest"
          "\n   off-shell p_struck -> the two struck-nucleon conventions differ; its chi2 is not a clean test]")
    print(f"\n{'variable':12s}  chi2/ndf   integral ACH/ADO")
    for c, (key, edges, xlab) in enumerate(VARS):
        bw = np.diff(edges); ctr = 0.5 * (edges[1:] + edges[:-1])
        av = np.asarray(ach[key]); dv = ado[key]
        if key == "Q2":
            av = av / 1e6                                       # ACH Q2 is MeV^2; res_C is GeV^2

        def hist(v, w):
            h, _ = np.histogram(v, bins=edges, weights=w)
            e2, _ = np.histogram(v, bins=edges, weights=w ** 2)
            return h / bw, np.sqrt(e2) / bw
        da, ea = hist(av, ach_w); dd, ed = hist(dv, ado["w"])
        ax, axr = axes[0, c], axes[1, c]
        ax.step(edges, np.append(da, da[-1]), where="post", color="0.35", lw=1.4, label="ACHILLES")
        ax.step(edges, np.append(dd, dd[-1]), where="post", color="C0", lw=1.4, label="ADoNIS (fact.)")
        ax.set_title(xlab, fontsize=9); ax.set_ylim(bottom=0)
        if c == 0:
            ax.legend(fontsize=7); ax.set_ylabel(r"d$\sigma$/dx [nb/unit]")
        with np.errstate(divide="ignore", invalid="ignore"):
            r = da / dd; re = r * np.sqrt((ed / dd) ** 2 + (ea / da) ** 2)
        m = (da > 0) & (dd > 0) & np.isfinite(re) & (re > 0)
        chi2 = float(np.sum((da[m] - dd[m]) ** 2 / (ea[m] ** 2 + ed[m] ** 2))); ndf = int(m.sum())
        axr.axhspan(0.9, 1.1, color="green", alpha=0.12); axr.axhline(1.0, ls="--", color="green", lw=0.8)
        axr.errorbar(ctr[m], r[m], yerr=re[m], fmt="o", color="C3", ms=3, capsize=2, lw=1.0)
        axr.set_ylim(0.5, 1.6); axr.set_xlabel(xlab, fontsize=8)
        if c == 0:
            axr.set_ylabel("ACH/ADO")
        axr.text(0.04, 0.83, f"{chi2/max(ndf,1):.1f}", transform=axr.transAxes, fontsize=9)
        iA = float(np.sum(da * bw)); iD = float(np.sum(dd * bw))
        print(f"{key:12s}  {chi2/max(ndf,1):6.2f}    {iA/iD:.3f}   (ACH {iA:.3e}  ADO {iD:.3e})")
    fig.suptitle("T2K CC1$\\pi^+$Np disaggregated ACH/ADO ratios (factorized differentiable chain)",
                 fontsize=12)
    fig.tight_layout()
    out = "paper_figures/cc1pi_ratios.png"; fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    main()
