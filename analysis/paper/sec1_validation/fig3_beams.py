"""Paper Fig 3: pi+ nucleus ABSORPTION + REACTION sigma(p), ADoNIS vs ACHILLES, on 12C and 40Ar.

A sigma-vs-scan (efficiency) figure, NOT a per-event histogram: sigma_X(p_bin) = piR^2 * N_X(bin)/N_tried(bin)
with binomial errors, computed on both sides and drawn through the shared chi2_ratio_panel in CURVE mode.
Reuses the validated beam sigma(p) computation (beams.make_figs.adonis_sigma on the paper_banks_p4 beam
banks + beams.achilles_beam.sigma_of_p on the ACHILLES CrossSection-mode oracle).

Paper layout: rows = nucleus (C top, Ar bottom), cols = absorption (left) / reaction (right).

  python -m analysis.paper.sec1_validation.fig3_beams [--nbins 15]
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ADONIS_BEAM_PATTERN", "output/paper_banks_p4/beam_{beam}_{target}/merged")
from analysis.paper.beams.make_figs import adonis_sigma        # noqa: E402
from analysis.paper.beams import achilles_beam as AB           # noqa: E402
from adonis.workflow.plotting import chi2_ratio_panel          # noqa: E402
from analysis.paper import style                               # noqa: E402

NUC_TEX = {"C": r"$^{12}$C", "Ar": r"$^{40}$Ar"}


def main(nbins=30, out="fig03_pi_nucleus_sigma"):
    style.use()
    fig, ax = plt.subplots(4, 2, figsize=(8.4, 8.2), sharex="col",
                           gridspec_kw={"height_ratios": [3, 1, 3, 1], "hspace": 0.0, "wspace": 0.24})
    for ni, nuc in enumerate(("C", "Ar")):
        try:
            edges, sr, ss, er, es = adonis_sigma("pip", nbins, nuc)
            hr, hs, her, hes, _nr, _ns, _nt = AB.sigma_of_p("pip", edges, nuc)
        except Exception as e:
            for r in range(2):
                ax[2 * ni + r, 0].text(0.5, 0.5, f"{nuc}: {e}", ha="center", fontsize=7,
                                       transform=ax[2 * ni + r, 0].transAxes)
            continue
        cen = 0.5 * (edges[:-1] + edges[1:])
        for oi, (A, EA, H, EH, lab) in enumerate(((ss, es, hs, hes, "absorption"),
                                                  (sr, er, hr, her, "reaction"))):
            a0, a1 = ax[2 * ni, oi], ax[2 * ni + 1, oi]
            res = chi2_ratio_panel(a0, a1, None, {"x": cen, "y": H, "yerr": EH},
                                   {"x": cen, "y": A, "yerr": EA},
                                   label=rf"$\pi^+$ {NUC_TEX[nuc]}: $\sigma$ ({lab})",
                                   xlabel=r"beam $|p|$ [MeV/c]", ratio_ylim=(0.75, 1.25),
                                   ado_label="ADoNIS", ref_label="ACHILLES")
            a0.set_ylabel(r"$\sigma$ [mb]"); a1.set_ylabel("ratio")
            print(f"  {nuc} {lab:10s} chi2/ndf {res['chi2']/max(res['ndf'],1):6.2f} (ndf={res['ndf']})", flush=True)
            if ni == 0 and oi == 0:
                a0.legend(fontsize=7)
    fig.suptitle(r"ADoNIS vs ACHILLES --- $\pi^+$ absorption + reaction on $^{12}$C / $^{40}$Ar "
                 r"(pure transport)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.subplots_adjust(hspace=0.0)
    style.save(fig, out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbins", type=int, default=30)
    main(nbins=ap.parse_args().nbins)
