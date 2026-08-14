"""FIGURE J -- prior versus post-fit uncertainty, for every parameter in the fit.

The standard summary of what a fit actually learned.  Each parameter's post-fit uncertainty is drawn as a
box against its prior, so the question "which of these did the data constrain?" has a visual answer:

  box at 1.0        the prior, by construction
  box below 1.0     the data shrank it -- this parameter was measured
  box at 1.0 still  the prior is doing all the work; the data says nothing about it

The templates are shown separately and in ABSOLUTE units, because they have no prior at all.  That is the
whole design: an unfolded spectrum must not be pulled toward the generator.  It is also the reason the
flux barely moves in figure H -- the templates can absorb a distortion for free, while every flux bin
that tries to do the same pays a penalty.

Usage:  python -m analysis.paper.sec5_unfold.fig_constraint [label] [study]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style

COL = {"template": "#1f4b9c", "flux": "#e08214", "xsec": "#1a9e57", "detector": "#7b5aa6"}

# Cross-section knobs are ordered and coloured by PHYSICS BLOCK, using the same grouping and the same
# muted clay/sage/violet/rose palette as the section-2 and section-3 knob figures -- the reader has
# already learned what those colours mean by the time this figure appears.
GROUP_COL = style.KNOB_GROUP_COLOR
GROUP_NAME = {g: n.replace("\n", " ") for g, n in style.KNOB_GROUP_NAME.items()}


def _boxes(ax, lo, hi, color=None, width=0.62, alpha=0.75, lw=0.8, x=None):
    """Uncertainty as a filled box per parameter, rather than a bar with caps."""
    lo, hi = np.atleast_1d(lo), np.atleast_1d(hi)
    x = np.arange(len(lo)) if x is None else np.atleast_1d(x)
    for xi, l, h in zip(x, lo, hi):
        ax.add_patch(plt.Rectangle((xi - width / 2, l), width, max(h - l, 1e-12),
                                   facecolor=color, edgecolor=color, alpha=alpha, lw=lw, zorder=3))


def main(label="sec5", study="asimov"):
    style.use()
    z = np.load(style.ALTGEN / f"{label}_unfold.npz", allow_pickle=True)
    blocks = np.array([str(b) for b in z["param_block"]])
    names = np.array([str(n) for n in z["param_name"]])
    prior = np.asarray(z["param_prior"], float)
    post = np.sqrt(np.abs(np.diag(np.asarray(z[f"{study}_cov"]))))
    val = np.concatenate([z[f"{study}_c"], z[f"{study}_f"], z[f"{study}_th"][z["dial_idx"]],
                          z[f"{study}_det"]])
    edges = np.asarray(z["flux_edges"], float)

    fig = plt.figure(figsize=(12.2, 3.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 1.05, 1.25, 0.95], wspace=0.34)

    # ---- (a) templates: no prior at all, so quote them in absolute units --------------------------- #
    ax = fig.add_subplot(gs[0, 0])
    m = blocks == "template"
    x = np.arange(int(m.sum()))
    _boxes(ax, val[m] - post[m], val[m] + post[m], COL["template"])
    ax.plot(x, val[m], "_", color="k", ms=9, mew=1.2, zorder=5)
    ax.axhline(1.0, color="k", lw=0.8, ls=":", zorder=1)
    ax.set_xticks(x); ax.set_xlabel("truth cell"); ax.set_ylabel(r"$c_j\pm\sigma$")
    ax.set_ylim(0, 2.0); ax.set_xlim(-0.7, len(x) - 0.3)
    ax.set_title("(a)  templates — no prior", fontsize=9.5, loc="left")

    # ---- (b),(c) parameters with a prior: post-fit as a fraction of it ----------------------------- #
    for k, (blk, ttl, rot, lab) in enumerate(
            [("flux", r"(b)  flux — $E_\nu$ intervals", 45, None),
             ("xsec", "(c)  cross section", 90, None)]):
        ax = fig.add_subplot(gs[0, 1 + k])
        m = blocks == blk
        r = post[m] / prior[m]
        nm = names[m]
        if blk == "xsec":                      # sort by physics block, as sections 2 and 3 do
            o = sorted(range(len(nm)), key=lambda k: (style.knob_group(nm[k]), k))
            r, nm, m = r[o], nm[o], m          # m stays a mask; nm/r are now block-ordered
        x = np.arange(len(r))
        if blk == "xsec":
            for xi, ri, n in zip(x, r, nm):
                _boxes(ax, [0.0], [ri], GROUP_COL[style.knob_group(n)], width=0.74, alpha=0.85, x=[xi])
            seen = []
            for n in nm:
                g = style.knob_group(n)
                if g not in seen:
                    seen.append(g)
            ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=GROUP_COL[g], alpha=0.85, ec="none",
                                             label=GROUP_NAME[g]) for g in seen],
                      frameon=False, fontsize=6.5, loc="lower left", handlelength=1.0)
        else:
            _boxes(ax, np.zeros_like(r), r, COL[blk], width=0.74, alpha=0.8)
        ax.axhline(1.0, color="k", lw=0.9, zorder=4)
        ax.set_ylim(0, 1.15); ax.set_xlim(-0.7, len(x) - 0.3)
        ax.set_ylabel(r"post-fit $\sigma$ / prior $\sigma$" if k == 0 else "")
        ax.set_xticks(x)
        if blk == "flux":
            lo = edges[:-1]
            ax.set_xticklabels([f"{int(e)}" if np.isfinite(e) else "" for e in lo],
                               rotation=rot, fontsize=6.5, ha="right")
            ax.set_xlabel(r"bin low edge [MeV]", fontsize=8)
        else:
            ax.set_xticklabels([style.plab(n) for n in nm], rotation=rot, fontsize=7.5)
        ax.set_title(ttl, fontsize=9.5, loc="left")

    # ---- (d) 60 detector dials: a distribution, not 60 tick labels --------------------------------- #
    ax = fig.add_subplot(gs[0, 3])
    m = blocks == "detector"
    r = post[m] / prior[m]
    ax.hist(r, bins=14, color=COL["detector"], alpha=0.85)
    ax.axvline(np.median(r), color="k", lw=1.0, ls="--", label=f"median {np.median(r):.2f}")
    ax.axvline(1.0, color="k", lw=0.9)
    ax.set_xlabel(r"post-fit $\sigma$ / prior $\sigma$", fontsize=8)
    ax.set_ylabel("detector dials")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_title(f"(d)  detector — {int(m.sum())} dials", fontsize=9.5, loc="left")

    style.save(fig, f"{label}_figJ")


if __name__ == "__main__":
    main(*(sys.argv[1:3] or []))
