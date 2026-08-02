"""Section 4 figure: the multisample closure on the 16 Gate-I dials.

Reads ONLY the persisted run npz (analysis.paper.physfit.multisample -> output/altgen/
sec4_closure_multisample.npz); no bank, instant, re-styles without re-fitting.  The story: inject a known
truth on the 16 dials sections 2/3 say are constrainable, build fake data by EXACT nonlinear reweight of
every S2/S3 sample (T2K + MINERvA + (e,e') + beams), fit blind from nominal, and show the dials come back.

  (a) recovery  -- injected displacement vs blind BFP, in prior-sigma units, rows grouped by physics block
                   (same block language + tabs as the §2/§3 heatmaps).
  (b) pulls     -- (BFP - truth)/sigma_post, M0 (LM) and M2 (Huber), with the +/-2 sigma calibration band.
  (c-e) overlays-- one sample per family (T2K dpt, MINERvA dpt, pi+->C reaction): closure data vs nominal
                   vs fitted, so the recovery is visible as an actual binned prediction across sample types.

Usage:  python -m analysis.paper.sec4_closure.make
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style

C_FIT, C_NOM = "#1f4b9c", "0.55"                        # fit BFP = deep IBM blue; nominal = grey
OVERLAYS = [("dpt", "T2K CC0$\\pi$  $\\delta p_T$"),
            ("mnv_dpt", "MINERvA CC0$\\pi$  $\\delta p_T$"),
            ("pip_react", "$\\pi^+$–C  $\\sigma_{\\rm reac}$")]


def main(label="sec4_closure_multisample"):
    style.use()
    z = np.load(style.ALTGEN / f"{label}.npz", allow_pickle=True)
    from analysis.paper.physical_fit import PRIOR, theta_nominal
    from adonis.reweight.reweight_model import nominal_knobs
    pnames = [str(x) for x in z["pnames"]]
    subset = [int(k) for k in z["subset"]]
    truth = np.asarray(z["truth"])
    nom = np.asarray(theta_nominal(nominal_knobs()))                   # canonical nominal for every dial
    priorv = np.asarray(z["prior"]) if "prior" in z.files else np.asarray(PRIOR)   # prior sigma per dial
    row0 = np.asarray(z["row0"]); dskeys = [str(x) for x in z["dskeys"]]
    data = np.asarray(z["data"]); sigma = np.asarray(z["sigma"]); m_nom = np.asarray(z["model_nom"])

    # rows ordered by physics block (like §2/§3); prior sigma to put every dial on one scale
    order = sorted(range(len(subset)), key=lambda c: (style.knob_group(pnames[subset[c]]), subset[c]))
    ksub = [subset[c] for c in order]
    gid = [style.knob_group(pnames[k]) for k in ksub]
    labels = [style.plab(pnames[k]) for k in ksub]
    prior = np.asarray(priorv)[ksub]                                   # prior sigma of each shown dial

    mkey = "fit" if "fit_th" in z.files else "M0"                      # single-fit npz (fallback: old M0)
    th_fit = np.asarray(z[f"{mkey}_th"]); s_fit = np.sqrt(np.abs(np.diag(np.asarray(z[f"{mkey}_V"]))))
    sig = {subset[c]: s_fit[c] for c in range(len(subset))}            # post-sigma per dial (subset order)

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        fig = plt.figure(figsize=(13.5, 8.4))
        gs = fig.add_gridspec(4, 2, width_ratios=[1.35, 1.0], height_ratios=[1.5, 1.0, 1.0, 1.0],
                              hspace=0.6, wspace=0.28, left=0.13, right=0.975, top=0.91, bottom=0.08)
        yy = np.arange(len(ksub))

        # (a) recovery in prior-sigma units: injected truth vs blind chi2 BFP ----------------------- #
        axr = fig.add_subplot(gs[:, 0])
        for i, k in enumerate(ksub):
            p = max(prior[i], 1e-9)
            axr.plot((truth[k] - nom[k]) / p, yy[i], marker="*", ms=13, color="k", zorder=5, ls="none")
            axr.errorbar((th_fit[k] - nom[k]) / p, yy[i], xerr=sig[k] / p, fmt="o", color=C_FIT,
                         ms=6, capsize=3, lw=1.4, zorder=4)
        axr.axvline(0, color="0.8", lw=0.8, zorder=0)
        axr.set_yticks(yy); axr.set_yticklabels(labels, fontsize=9)
        axr.set_ylim(len(ksub) - 0.5, -0.5)
        axr.set_xlabel(r"recovered value $-$ nominal  (prior $\sigma$ units)", fontsize=9.5)
        style.knob_group_tabs(axr, gid, tabx=-0.30, tabw=0.028, labx=-0.37, fontsize=8)
        for sp in ("top", "right"):
            axr.spines[sp].set_visible(False)
        axr.plot([], [], "*", color="k", ms=11, label="injected truth")
        axr.plot([], [], "o", color=C_FIT, ms=6, label=r"$\chi^2$ fit (BFP $\pm\sigma$)")
        axr.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
        axr.set_title("(a)  blind recovery of the 16 dials", fontsize=10, loc="left")

        # (b) pulls (BFP - truth)/sigma_post, with the +/-1,+/-2 sigma calibration band ------------- #
        axp = fig.add_subplot(gs[0, 1])
        axp.axvspan(-2, 2, color="0.93", zorder=0)
        axp.axvspan(-1, 1, color="0.86", zorder=0)
        axp.axvline(0, color="k", lw=0.9)
        pulls = [(th_fit[k] - truth[k]) / max(sig[k], 1e-12) for k in ksub]
        axp.plot(pulls, yy, "o", color=C_FIT, ms=5, zorder=3)
        axp.set_yticks(yy); axp.set_yticklabels([]); axp.set_ylim(len(ksub) - 0.5, -0.5)
        axp.set_xlim(-4, 4); axp.set_xlabel(r"pull  $(\hat\theta-\theta^\star)/\sigma$", fontsize=9)
        axp.set_title("(b)  pulls (band $\\pm1,\\pm2\\sigma$)", fontsize=10, loc="left")
        for sp in ("top", "right"):
            axp.spines[sp].set_visible(False)

        # (c-e) sample overlays: closure data vs nominal vs fitted ---------------------------------- #
        for row, (key, ttl) in enumerate([o for o in OVERLAYS if o[0] in dskeys], start=1):
            j = dskeys.index(key); sl = slice(int(row0[j]), int(row0[j + 1]))
            edges = np.asarray(z[f"{key}_edges"]); x = 0.5 * (edges[1:] + edges[:-1]); xe = 0.5 * np.diff(edges)
            ax = fig.add_subplot(gs[row, 1])
            ax.errorbar(x, data[sl], xerr=xe, yerr=sigma[sl], fmt="o", color="k", ms=2.6, lw=0.7,
                        capsize=0, zorder=3, label="closure data")
            xs = np.repeat(edges, 2)[1:-1]
            ax.plot(xs, np.repeat(m_nom[sl], 2), color=C_NOM, ls=":", lw=1.4, label="nominal")
            ax.plot(xs, np.repeat(np.asarray(z[f"{mkey}_model"])[sl], 2), color=C_FIT, ls="-", lw=1.5, label="fit BFP")
            ax.set_ylim(bottom=0); ax.set_title(f"({'cde'[row-1]})  {ttl}", fontsize=9, loc="left")
            ax.tick_params(labelsize=7.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if row == 1:
                ax.legend(fontsize=7, loc="upper right", framealpha=0.9)

        c2 = f"$\\chi^2_{{\\rm data}}$/ndf = {float(z[f'{mkey}_chi2data']):.0f}/{row0[-1]}"
        fig.suptitle(f"Multisample closure — 16 dials, {len(dskeys)} observables (T2K+MINERvA+$(e,e')$+beams)"
                     f"    ·    {c2}", fontsize=11, x=0.13, ha="left")
        style.save(fig, label)


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*(pos[:1] or []))
