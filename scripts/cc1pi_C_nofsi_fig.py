"""CARBON, NO-FSI test: does the high-W tail survive without the cascade?
ADoNIS res_C(fsi=False) -- carbon primary RES (spectral fn / Fermi / de-Forest off-shell struck
nucleon, NO cascade), t-channel sampler, spline amps2 -- vs ACHILLES no-FSI carbon (is_h==False
from t2k_cc1pi_4way_ach_nofsi.npz, signal selection).  Same 7-variable / T2K style.

If the W/pi_p tail is present here -> NUCLEAR primary (spectral/de-Forest).  If clean here but present
with FSI (cc1pi_sampler CH) -> the cascade (FSI) is the driver.

Usage: python scripts/cc1pi_C_nofsi_fig.py [NRES=120000] [NSEED=4] [--recompute]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import adonis.xsec.dcc_current as DC
import adonis.xsec.res_xsec as R
import scripts.cc1pi_fig_tki as F          # pins spline at import
from adonis.data.oracle.normalization import weight_to_nb_of

NRES = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 120000
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 4
KEYS = ("pn", "dptt", "dalphat", "W", "Q2", "pi_p", "lp_p")
VARS = [("pn", np.array([0, 120, 240, 600, 1500.0]), r"$p_N$ [MeV/c]"),
        ("dptt", np.array([-700, -300, -100, 100, 300, 700.0]), r"$\delta p_{TT}$"),
        ("dalphat", np.linspace(0, np.pi, 7), r"$\delta\alpha_T$ [rad]"),
        ("W", np.linspace(1080, 1700, 13), r"vertex $W$ [MeV]"),
        ("Q2", np.linspace(0, 1.5, 13), r"$Q^2$ [GeV$^2$]"),
        ("pi_p", np.linspace(150, 1200, 13), r"$p_\pi$ [MeV/c]"),
        ("lp_p", np.linspace(450, 1200, 13), r"lead $p_p$ [MeV/c]")]


def acc_ado():
    DC.BATCH_INTERP = "spline"; R.SAMPLER_3BODY = "tchannel"
    parts = [F.res_C(NRES, sd, fsi=False) for sd in range(NSEED)]
    d = {k: np.concatenate([p[k] for p in parts]) for k in KEYS + ("w",)}
    d["w"] = d["w"] / NSEED
    print(f"  ADoNIS res_C(no-FSI, tchan): {len(d['w'])} ev  sigma_C={d['w'].sum():.4e}", flush=True)
    return d


def ach_C_nofsi():
    a = np.load("data/oracle/t2k_cc1pi_4way_ach_nofsi.npz")
    sel = (~np.asarray(a["is_h"])) & np.asarray(a["has_acc"]) & np.asarray(a["mu_acc"]) & np.asarray(a["pi_acc"])
    w = np.asarray(a["w"])[sel] * weight_to_nb_of(a)
    out = {"pn": np.asarray(a["pn_acc"])[sel], "dptt": np.asarray(a["dptt_acc"])[sel],
           "dalphat": np.asarray(a["dat_acc"])[sel], "W": np.asarray(a["W"])[sel],
           "Q2": np.asarray(a["Q2"])[sel] / 1e6, "pi_p": np.asarray(a["pi_p"])[sel],
           "lp_p": np.asarray(a["lp_p_acc"])[sel], "w": w}
    print(f"  ACHILLES no-FSI carbon: {int(sel.sum())} ev  sigma_C={w.sum():.4e}", flush=True)
    return out


def main():
    cache = "data/oracle/t2k_cc1pi_C_nofsi_adonis.npz"
    if os.path.exists(cache) and "--recompute" not in sys.argv:
        d = np.load(cache); ado = {k: d[k] for k in KEYS + ("w",)}; print("loaded cached ADoNIS", flush=True)
    else:
        ado = acc_ado(); np.savez(cache, **{k: ado[k] for k in KEYS + ("w",)})
    ach = ach_C_nofsi()
    fig, axes = plt.subplots(2, len(VARS), figsize=(3.4 * len(VARS), 6.6), height_ratios=[3, 1.2])
    print(f"\n{'var':10s} {'chi2 tchan':>10}  {'ACH/ADO':>8}")
    for c, (key, edges, xlab) in enumerate(VARS):
        bw = np.diff(edges); ctr = 0.5 * (edges[1:] + edges[:-1])

        def H(d, w):
            h, _ = np.histogram(np.asarray(d), bins=edges, weights=np.asarray(w))
            e2, _ = np.histogram(np.asarray(d), bins=edges, weights=np.asarray(w) ** 2)
            return h / bw, np.sqrt(e2) / bw
        da, ea = H(ach[key], ach["w"]); dd, ed = H(ado[key], ado["w"])
        ax, axr = axes[0, c], axes[1, c]
        ax.fill_between(edges, np.append(da - ea, (da - ea)[-1]), np.append(da + ea, (da + ea)[-1]),
                        step="post", color="0.45", alpha=0.35, lw=0, label="ACHILLES stat. unc.")
        ax.step(edges, np.append(da, da[-1]), where="post", color="0.35", lw=1.4, label="ACHILLES (no-FSI C)")
        ax.errorbar(ctr, dd, yerr=ed, fmt="s", color="C0", ms=4, capsize=2, lw=1.0,
                    label="ADoNIS C no-FSI (t-chan)", zorder=4)
        ax.set_title(xlab, fontsize=9); ax.set_ylim(bottom=0)
        if c == 0:
            ax.legend(fontsize=6); ax.set_ylabel(r"d$\sigma$/dx [nb/unit]")
        with np.errstate(divide="ignore", invalid="ignore"):
            r = da / dd; re = r * np.sqrt((ed / dd) ** 2 + (ea / da) ** 2)
        m = (da > 0) & (dd > 0) & np.isfinite(re)
        c2 = float(np.sum((da[m] - dd[m]) ** 2 / (ea[m] ** 2 + ed[m] ** 2))) / max(int(m.sum()), 1)
        axr.axhspan(0.9, 1.1, color="green", alpha=0.12); axr.axhline(1.0, ls="--", color="green", lw=0.7)
        axr.errorbar(ctr[m], r[m], yerr=re[m], fmt="s", color="C0", ms=3, capsize=2, lw=0.9)
        axr.set_ylim(0.5, 1.6); axr.set_xlabel(xlab, fontsize=8)
        if c == 0:
            axr.set_ylabel("ACH/ADO")
        axr.text(0.04, 0.84, f"{c2:.1f}", transform=axr.transAxes, fontsize=8, color="C0")
        print(f"{key:10s} {c2:10.2f}  {float(np.sum(da*bw))/max(float(np.sum(dd*bw)),1e-30):8.3f}")
    fig.suptitle("T2K CC1$\\pi^+$Np CARBON NO-FSI: ACHILLES vs ADoNIS RES primary (t-channel, spline)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = "paper_figures/cc1pi_C_nofsi.png"; fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    main()
