"""SAME 7-variable sampler-comparison figure but H-ONLY (free proton): ADoNIS res_H under
isotropic vs t-channel (fixed struck-nucleon axis) vs ACHILLES H-tagged events (is_h==True from
t2k_cc1pi_4way_ach_fsi.npz, signal selection).  H has no cascade / no spectral function -> isolates
the 3-body sampler in the clean on-shell free-proton regime.

Usage: python scripts/cc1pi_sampler_fig_H.py [NH=160000] [NSEED=4]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import adonis.xsec.dcc_current as DC
import adonis.xsec.res_xsec as R
import scripts.cc1pi_fig_tki as F
from adonis.data.oracle.normalization import weight_to_nb_of

NH = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 160000
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 4
KEYS = ("pn", "dptt", "dalphat", "W", "Q2", "pi_p", "lp_p")
VARS = [("pn", np.array([0, 120, 240, 600, 1500.0]), r"$p_N$ [MeV/c]"),
        ("dptt", np.array([-700, -300, -100, 100, 300, 700.0]), r"$\delta p_{TT}$"),
        ("dalphat", np.linspace(0, np.pi, 7), r"$\delta\alpha_T$ [rad]"),
        ("W", np.linspace(1080, 1700, 13), r"vertex $W$ [MeV]"),
        ("Q2", np.linspace(0, 1.5, 13), r"$Q^2$ [GeV$^2$]"),
        ("pi_p", np.linspace(150, 1200, 13), r"$p_\pi$ [MeV/c]"),
        ("lp_p", np.linspace(450, 1200, 13), r"lead $p_p$ [MeV/c]")]


def accH(samp):
    DC.BATCH_INTERP = "spline"; R.SAMPLER_3BODY = samp
    parts = [F.res_H(NH, sd) for sd in range(NSEED)]
    d = {k: np.concatenate([p[k] for p in parts]) for k in KEYS + ("w",)}
    d["w"] = d["w"] / NSEED
    print(f"  res_H {samp}: {len(d['w'])} ev  sigma_H={d['w'].sum():.4e}", flush=True)
    return d


def ach_H():
    """ACHILLES H-only, signal selection, from the 4-way topology npz (is_h==True)."""
    a = np.load("data/oracle/t2k_cc1pi_4way_ach_fsi.npz")
    sel = np.asarray(a["is_h"]) & np.asarray(a["has_acc"]) & np.asarray(a["mu_acc"]) & np.asarray(a["pi_acc"])
    w = np.asarray(a["w"])[sel] * weight_to_nb_of(a)
    out = {"pn": np.asarray(a["pn_acc"])[sel], "dptt": np.asarray(a["dptt_acc"])[sel],
           "dalphat": np.asarray(a["dat_acc"])[sel], "W": np.asarray(a["W"])[sel],
           "Q2": np.asarray(a["Q2"])[sel] / 1e6, "pi_p": np.asarray(a["pi_p"])[sel],
           "lp_p": np.asarray(a["lp_p_acc"])[sel], "w": w}
    print(f"  ACHILLES H: {int(sel.sum())} ev  sigma_H={w.sum():.4e}", flush=True)
    return out


def main():
    cache = "data/oracle/t2k_cc1pi_sampler_H_adonis.npz"
    if os.path.exists(cache) and "--recompute" not in sys.argv:
        d = np.load(cache)
        iso = {k: d["iso_" + k] for k in KEYS + ("w",)}; tc = {k: d["tc_" + k] for k in KEYS + ("w",)}
        print("loaded cached ADoNIS H arrays", flush=True)
    else:
        iso = accH("isotropic"); tc = accH("tchannel")
        np.savez(cache, **{"iso_" + k: iso[k] for k in KEYS + ("w",)},
                 **{"tc_" + k: tc[k] for k in KEYS + ("w",)})
    ach = ach_H()
    fig, axes = plt.subplots(2, len(VARS), figsize=(3.4 * len(VARS), 6.6), height_ratios=[3, 1.2])
    print(f"\n{'var':10s} {'chi2 iso':>9} {'chi2 tchan':>10}  {'ACH/ADO iso':>11} {'tchan':>7}")
    for c, (key, edges, xlab) in enumerate(VARS):
        bw = np.diff(edges); ctr = 0.5 * (edges[1:] + edges[:-1])

        def H(d, w):
            h, _ = np.histogram(np.asarray(d), bins=edges, weights=np.asarray(w))
            e2, _ = np.histogram(np.asarray(d), bins=edges, weights=np.asarray(w) ** 2)
            return h / bw, np.sqrt(e2) / bw
        da, ea = H(ach[key], ach["w"]); di, ei = H(iso[key], iso["w"]); dt, et = H(tc[key], tc["w"])
        ax, axr = axes[0, c], axes[1, c]
        ax.fill_between(edges, np.append(da - ea, (da - ea)[-1]), np.append(da + ea, (da + ea)[-1]),
                        step="post", color="0.5", alpha=0.25, lw=0)
        ax.step(edges, np.append(da, da[-1]), where="post", color="0.35", lw=1.4, label="ACHILLES H")
        ax.errorbar(ctr, di, yerr=ei, fmt="o", color="C1", ms=3, capsize=2, lw=0.9, label="ADO iso")
        ax.errorbar(ctr, dt, yerr=et, fmt="s", color="C0", ms=3, capsize=2, lw=0.9, label="ADO t-chan")
        ax.set_title(xlab, fontsize=9); ax.set_ylim(bottom=0)
        if c == 0:
            ax.legend(fontsize=7); ax.set_ylabel(r"d$\sigma$/dx [nb/unit]")

        def ratio(d, e):
            with np.errstate(divide="ignore", invalid="ignore"):
                r = da / d; re = r * np.sqrt((e / d) ** 2 + (ea / da) ** 2)
            m = (da > 0) & (d > 0) & np.isfinite(re)
            c2 = float(np.sum((da[m] - d[m]) ** 2 / (ea[m] ** 2 + e[m] ** 2))) / max(int(m.sum()), 1)
            return r, re, m, c2
        ri, rei, mi, c2i = ratio(di, ei); rt, ret, mt, c2t = ratio(dt, et)
        axr.axhspan(0.9, 1.1, color="green", alpha=0.12); axr.axhline(1.0, ls="--", color="green", lw=0.7)
        axr.errorbar(ctr[mi], ri[mi], yerr=rei[mi], fmt="o", color="C1", ms=3, capsize=2, lw=0.9)
        axr.errorbar(ctr[mt], rt[mt], yerr=ret[mt], fmt="s", color="C0", ms=3, capsize=2, lw=0.9)
        axr.set_ylim(0.5, 1.6); axr.set_xlabel(xlab, fontsize=8)
        if c == 0:
            axr.set_ylabel("ACH/ADO")
        axr.text(0.04, 0.86, f"iso {c2i:.1f}", transform=axr.transAxes, fontsize=7, color="C1")
        axr.text(0.04, 0.70, f"tch {c2t:.1f}", transform=axr.transAxes, fontsize=7, color="C0")
        print(f"{key:10s} {c2i:9.2f} {c2t:10.2f}  {float(np.sum(da*bw))/max(float(np.sum(di*bw)),1e-30):11.3f} "
              f"{float(np.sum(da*bw))/max(float(np.sum(dt*bw)),1e-30):7.3f}")
    fig.suptitle("T2K CC1$\\pi^+$Np H-ONLY: ACHILLES vs ADoNIS isotropic vs t-channel (fixed axis, spline)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = "paper_figures/cc1pi_sampler_H.png"; fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    main()
