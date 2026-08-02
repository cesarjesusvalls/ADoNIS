"""Assemble the Sec 4 & Sec 5 draft figures into ONE PDF (one figure per page, with captions) for review.

Pulls the rendered PNGs from output/paper/ and lays them out with a title + caption per page.  Regenerate
the figures first (see docs/sec4_sec5_figures.md) if any are stale.

    python -m analysis.paper.figure_draft            # -> output/paper/sec4_sec5_figure_draft.pdf
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path("output/paper")

# (png, section-tag, title, caption)
FIGS = [
    ("__H__", "Section 4", "Fitting & statistical interpretability", ""),
    ("sec4_fig41_recovery.png", "Fig 4.1", "Recovery + 1-D uncertainty (MLE)",
     "Blind fit recovers every injected-truth (star) unbiased; per dial the Gaussian sigma (grey) vs the "
     "profile Delta-chi2=1 interval (blue). They coincide for well-measured dials; the weak cross-section "
     "dials show wide, mildly asymmetric profile intervals."),
    ("sec4_ebwall.png", "Fig 4.1b", "E_b at the physical wall",
     "E_b injected at 0.15 MeV near its E_b>=0 wall. The profile likelihood is one-sided (flat below 0, the "
     "model clamps E_b<0); the Gaussian sigma is symmetric and leaks to -0.10 MeV, unphysical. The clean "
     "boundary illustration: symmetric errors mislead, the likelihood-ratio interval respects the physics."),
    ("sec4_fig42_dists.png", "Fig 4.2", "Pre/post-fit distributions across the probes",
     "Nominal (pre-fit) vs best-fit (post-fit) vs closure data for T2K + MINERvA CC0pi dpT (nu), (e,e') QE "
     "omega (e-beam), and pi+ -> C reaction sigma (hadron beam). One fit across all sample types."),
    ("sec4_fig43a_coverage_mle.png", "Fig 4.3", "Coverage / calibration (bulk)",
     "80-toy MLE ensemble: pooled pull +0.01 +/- 1.07 (KS p=0.50) ~ N(0,1); per-dial all centered/unit-width; "
     "chi2_data ~ chi2(354). The Gaussian intervals cover correctly away from boundaries. (At a boundary, "
     "see Fig 4.1b: the Gaussian interval goes unphysical and one must use the profile.)"),
    ("sec4_fig44_corner.png", "Fig 4.4", "Parameter corner (correlations)",
     "16-dial corner: exact 68/95% contours (blue) vs Gaussian ellipses (orange). Most pairs uncorrelated; "
     "clear degeneracies M_A_res x {kF, S_Delta, N_SRC} and s_NN_pn x {f_NN_cex, s_NN_inel} -- the "
     "correlations 1-D intervals cannot show."),
    ("__H__", "Section 5", "Computational performance & novel methodologies", ""),
    ("sec5_fig_scaling.png", "Fig 5.1", "Differentiability advantage (scaling)",
     "Measured on the multisample model. (a) 120-pair non-Gaussian corner: Taylor 10 s vs exact grid 1.6 hr "
     "vs true profile 45 hr. (b) fit Jacobian: autodiff 935 ms vs finite-diff 4.0 s (and exact vs noisy). "
     "(c) derivative-tensor sweeps ~ n^m/m! (16 -> 3876) vs order m."),
    ("sec4_closure_r16_noprior_corner_validate.png", "Fig 5.2", "Non-Gaussian contours from autodiff",
     "Exact grid (black) vs the 2nd-order-model Taylor corner built from autodiff derivatives J + d2m "
     "(orange), vs the Gaussian ellipse (blue), for four pairs. The Taylor tracks the exact contour through "
     "1-2 sigma at a fraction of the cost; both peel away from the Gaussian."),
    ("sec5_fig_hybrid.png", "Fig 5.4", "Hybrid: grid-bad x Taylor-good",
     "(E_b, X) contour: exact (black) vs pure Taylor (orange, a smooth ellipse that misses the E_b wall) vs "
     "hybrid (blue: grid E_b exactly, Taylor the good dial). The hybrid recovers the exact walled contour; "
     "for the degenerate M_A_res some residual Taylor error remains (would tighten at 3rd order)."),
    ("sec5_fig_table.png", "Fig 5.5", "Differentiability advantage per task",
     "Per statistical task (gradient/Fisher, Hessian, error propagation, non-Gaussian corner, m-th "
     "derivative, hybrid): the differentiable-engine cost vs what a finite-difference generator pays."),
    ("__H__", "Appendix", "Supporting / diagnostic figures", ""),
    ("sec4_prior_pull.png", "A1", "MLE vs MAP (prior pull)",
     "The fit is unbiased: MLE (no prior) lands exactly on truth; MAP (Gate-I prior) is pulled toward "
     "nominal, the pull tracking each dial's shrinkage."),
    ("sec4_closure_random16_asym.png", "A2", "Asymmetric (MINOS) errors",
     "Profile minimum + asymmetric Delta-chi2=1 interval vs the GN symmetric sigma vs injected truth, per dial."),
    ("sec4_traj_s10_convergence.png", "A3", "Fit convergence",
     "Distance-to-truth per dial vs LM/Gauss-Newton iteration (Newton-decrement quadratic convergence)."),
    ("sec4_closure_random16_profile.png", "A4", "1-D profile vs GN parabola",
     "Per-dial exact profiled Delta-chi2 vs the Gauss-Newton parabola -- the 1-D non-Gaussianity (tails)."),
]


def main():
    out = OUT / "sec4_sec5_figure_draft.pdf"
    with PdfPages(out) as pdf:
        for png, tag, title, cap in FIGS:
            if png == "__H__":                                    # section divider page
                fig = plt.figure(figsize=(11, 8.5)); fig.text(0.5, 0.55, tag, ha="center", fontsize=34,
                                                              fontweight="bold")
                fig.text(0.5, 0.45, title, ha="center", fontsize=18, color="0.35")
                pdf.savefig(fig); plt.close(fig); continue
            p = OUT / png
            fig = plt.figure(figsize=(11, 8.5))
            fig.text(0.06, 0.955, f"{tag}", fontsize=13, fontweight="bold")
            fig.text(0.14, 0.955, title, fontsize=13)
            if p.exists():
                ax = fig.add_axes([0.05, 0.20, 0.90, 0.72]); ax.axis("off")
                ax.imshow(mpimg.imread(p))
            else:
                fig.text(0.5, 0.55, f"[missing: {png}]", ha="center", color="red")
            fig.text(0.06, 0.13, cap, fontsize=9.5, wrap=True, va="top",
                     bbox=dict(boxstyle="round", fc="0.96", ec="0.8"))
            pdf.savefig(fig); plt.close(fig)
    print(f"[out] {out}  ({len([f for f in FIGS if f[0] != '__H__'])} figures)")


if __name__ == "__main__":
    sys.exit(main())
