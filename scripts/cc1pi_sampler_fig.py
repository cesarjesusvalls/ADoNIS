"""Figure: cc1pi_ratios 7-variable diagnostic, ACHILLES vs ADoNIS-isotropic vs ADoNIS-tchannel
(both spline amps2).  Top row overlay (dsigma/dx), bottom row ACH/ADO ratio for both samplers.
Isotropic arrays come from the cached t2k_cc1pi_ratios_adonis.npz; t-channel is generated here
(SAMPLER_3BODY=tchannel) and cached to t2k_cc1pi_ratios_adonis_tchan.npz.

Usage: python scripts/cc1pi_sampler_fig.py [NRES=120000] [NSEED=4] [--recompute]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import adonis.xsec.dcc_current as DC
import adonis.xsec.res_xsec as R
import scripts.cc1pi_fig_tki as F
from adonis.data.oracle.normalization import weight_to_nb_of

NRES = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 120000
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 4
NH = 40000
KEYS = ("pn", "dptt", "dalphat", "W", "Q2", "pi_p", "lp_p")
VARS = [("pn", np.array([0, 120, 240, 600, 1500.0]), r"$p_N$ [MeV/c]"),
        ("dptt", np.array([-700, -300, -100, 100, 300, 700.0]), r"$\delta p_{TT}$"),
        ("dalphat", np.linspace(0, np.pi, 7), r"$\delta\alpha_T$ [rad]"),
        ("W", np.linspace(1080, 1700, 13), r"vertex $W$ [MeV]"),
        ("Q2", np.linspace(0, 1.5, 13), r"$Q^2$ [GeV$^2$]"),
        ("pi_p", np.linspace(150, 1200, 13), r"$p_\pi$ [MeV/c]"),
        ("lp_p", np.linspace(450, 1200, 13), r"lead $p_p$ [MeV/c]")]


def accumulate_tchan():
    DC.BATCH_INTERP = "spline"; R.SAMPLER_3BODY = "tchannel"
    cells = {}
    for cell, fn, n in (("RES-C", F.res_C, NRES), ("RES-H", F.res_H, NH)):
        parts = [fn(n, sd) for sd in range(NSEED)]
        print(f"  tchan {cell}: {sum(len(p['w']) for p in parts)} ev", flush=True)
        d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        d["w"] = d["w"] / NSEED
        cells[cell] = d
    return {k: np.concatenate([cells[c][k] for c in cells]) for k in KEYS + ("w",)}


def main():
    iso = np.load("data/oracle/t2k_cc1pi_ratios_adonis.npz")
    tc_path = "data/oracle/t2k_cc1pi_ratios_adonis_tchan.npz"
    if os.path.exists(tc_path) and "--recompute" not in sys.argv:
        tc = np.load(tc_path)
    else:
        tcd = accumulate_tchan(); np.savez(tc_path, **{k: tcd[k] for k in KEYS + ("w",)}); tc = tcd
    ach = np.load("data/oracle/t2k_cc1pi_tki_achilles.npz")
    aw = np.asarray(ach["w"]) * weight_to_nb_of(ach)

    fig, axes = plt.subplots(2, len(VARS), figsize=(3.4 * len(VARS), 6.6), height_ratios=[3, 1.2])
    print(f"\n{'var':10s} {'chi2 iso':>9} {'chi2 tchan':>10}")
    for c, (key, edges, xlab) in enumerate(VARS):
        bw = np.diff(edges); ctr = 0.5 * (edges[1:] + edges[:-1])
        av = np.asarray(ach[key]) / (1e6 if key == "Q2" else 1.0)

        def H(v, w):
            h, _ = np.histogram(v, bins=edges, weights=w); e2, _ = np.histogram(v, bins=edges, weights=w ** 2)
            return h / bw, np.sqrt(e2) / bw
        da, ea = H(av, aw); di, ei = H(np.asarray(iso[key]), np.asarray(iso["w"]))
        dt, et = H(np.asarray(tc[key]), np.asarray(tc["w"]))
        ax, axr = axes[0, c], axes[1, c]
        ax.step(edges, np.append(da, da[-1]), where="post", color="0.35", lw=1.5, label="ACHILLES")
        ax.step(edges, np.append(di, di[-1]), where="post", color="C1", lw=1.3, label="ADO isotropic")
        ax.step(edges, np.append(dt, dt[-1]), where="post", color="C0", lw=1.3, label="ADO t-channel")
        ax.set_title(xlab, fontsize=9); ax.set_ylim(bottom=0)
        if c == 0:
            ax.legend(fontsize=7); ax.set_ylabel(r"d$\sigma$/dx [nb/unit]")

        def chi2(d, e):
            m = (da > 0) & (d > 0); return float(np.sum((da[m] - d[m]) ** 2 / (ea[m] ** 2 + e[m] ** 2))) / max(int(m.sum()), 1)
        c2i, c2t = chi2(di, ei), chi2(dt, et)
        axr.axhspan(0.9, 1.1, color="green", alpha=0.12); axr.axhline(1.0, ls="--", color="green", lw=0.7)
        mi = (da > 0) & (di > 0); mt = (da > 0) & (dt > 0)
        axr.plot(ctr[mi], (da / di)[mi], "o-", color="C1", ms=2.5, lw=0.8)
        axr.plot(ctr[mt], (da / dt)[mt], "s-", color="C0", ms=2.5, lw=0.8)
        axr.set_ylim(0.5, 1.6); axr.set_xlabel(xlab, fontsize=8)
        if c == 0:
            axr.set_ylabel("ACH/ADO")
        axr.text(0.04, 0.86, f"iso {c2i:.1f}", transform=axr.transAxes, fontsize=7, color="C1")
        axr.text(0.04, 0.70, f"tch {c2t:.1f}", transform=axr.transAxes, fontsize=7, color="C0")
        print(f"{key:10s} {c2i:9.2f} {c2t:10.2f}")
    fig.suptitle("T2K CC1$\\pi^+$Np: ACHILLES vs ADoNIS isotropic vs t-channel 3-body sampler (spline)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = "paper_figures/cc1pi_sampler.png"; fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    main()
