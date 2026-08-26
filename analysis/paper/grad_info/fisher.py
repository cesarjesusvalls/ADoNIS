"""Paper section 3 -- Fisher information per subset: what is worth fitting, and why not.

Gate I asks, for a given subset of bins: with a prior on every knob, does the DATA (not the prior)
determine it?  Asimov posterior V = (J^T C^-1 J + Pi^-1)^-1; a knob is FIT when
    shrinkage = sigma_post / prior < 0.5     (the data at least halves the prior width).

sigma_post is MARGINALIZED (a diagonal element of V), so a knob fails either because the data cannot
SEE it or because another knob can MIMIC it.  Those are physically opposite and want opposite fixes, so
we report both axes:
    raw  = 1/sqrt(F_kk) / prior   -- other knobs held FIXED: pure sensitivity, degeneracy-blind
    marg = sqrt(V_kk)  / prior    -- other knobs free: what the fit actually delivers
  raw < 0.5, marg < 0.5  -> MEASURABLE
  raw < 0.5, marg > 0.5  -> DEGENERATE (seen clearly, cannot be disentangled -- a better observable can fix it)
  raw > 0.5              -> INVISIBLE  (the sample carries no information at this precision -- nothing can)

Everything is a row slice of ONE persisted Jacobian, so every subset is exact and costs no bank pass.
Fisher is additive (F = sum_s F_s), so stacking samples is stacking rows -- which is why the same
machinery answers both "which observable class measures this knob" and "which SAMPLE does".

WHICH slices get plotted is not hardcoded here: `configs/paper/sec2_subsets.yaml` declares the column
axes as named groups of dataset keys (literals, globs, or references to an earlier group), and
`subsets.py` resolves them against whatever `dskeys` the input npz actually carries.  Changing the
sample composition or the binning is a config edit; a sample that is absent drops out with a warning
and a sample that no column claims is reported as `uncovered`.

Usage:
    python -m analysis.paper.grad_info.make [npz_label]     # default: the config's `npz:`
    ADONIS_SEC2_CONFIG=... ADONIS_SEC2_NPZ=... python -m analysis.paper.grad_info.make
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from analysis.paper import style
from adonis.stats import fisher as FE
from analysis.paper.grad_info import subsets as SS

plab = style.plab


def gate1(J, sigma, prior, rows):
    """(marginalized shrinkage, raw shrinkage, Fisher) for the given bin rows -- via the shared engine."""
    F, _V, _sig_post, marg, reach = FE.fisher_shrinkage(J, sigma, prior, rows)
    with np.errstate(divide="ignore"):
        raw = np.where(reach > 0, 1.0 / (reach * prior), np.inf)
    return marg, raw, F



def fig_shrinkage_grouped(M, pnames, labels, names, figname, fit_cut):
    """Single marginalized panel, rows grouped by physics: dark = tighter constraint, orange outline = FIT.
    The last column ('all' = every sample combined) is set apart.  The key goes in the caption, not on axes."""
    order = sorted(range(len(pnames)), key=lambda k: (style.knob_group(pnames[k]), k))
    M = M[order]
    plabels = [plab(pnames[k]) for k in order]
    gid = [style.knob_group(pnames[k]) for k in order]
    nk, ng = M.shape
    cm = style.CMAP_CONSTRAINT

    def _tc(v):
        r, g, b, _ = cm(min(max(v, 0.0), 1.0))
        return "w" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "0.15"

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        SCALE, CELL_W, MARGIN_W, CELL_H, MARGIN_H = 0.8, 0.736, 2.9, 0.34, 2.35
        fig, ax = plt.subplots(figsize=(SCALE * (CELL_W * ng + MARGIN_W),
                                        SCALE * (CELL_H * nk + MARGIN_H)))
        im = ax.imshow(np.clip(M, 0, 1), aspect="auto", cmap=cm, vmin=0, vmax=1)
        for k in range(nk):
            for c in range(ng):
                v = M[k, c]; fit = v < fit_cut
                if fit:
                    ax.add_patch(Rectangle((c - .5, k - .5), 1, 1, fill=False, ec=style.FIT_EC, lw=1.9, zorder=3))
                ax.text(c, k, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.6 if fit else 7, fontweight="bold" if fit else "normal",
                        color=_tc(v), alpha=1.0 if fit else 0.6, zorder=4)
        ax.set_xticks(np.arange(-.5, ng, 1), minor=True)
        ax.set_yticks(np.arange(-.5, nk, 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.1); ax.tick_params(which="minor", length=0)
        style.knob_group_tabs(ax, gid)
        if names and names[-1] == "all":
            ax.axvline(ng - 1.5, color="0.25", lw=1.6, zorder=6)
        ax.set_yticks(range(nk)); ax.set_yticklabels(plabels, fontsize=8.5)
        ax.set_xticks(range(ng)); ax.set_xticklabels(labels, fontsize=8.5)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        cax = make_axes_locatable(ax).append_axes("bottom", size="1.4%", pad=0.62)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0.0, 0.5, 1.0])
        cb.set_label("posterior / prior width   $\\sigma_{\\rm post}/\\sigma_{\\rm prior}$", fontsize=8.5)
        cb.ax.tick_params(labelsize=7.5)
        return style.save(fig, figname)



def main(label=None):
    style.use()
    cfg = SS.load_config()
    default_label = label or os.environ.get("ADONIS_SEC2_NPZ") or cfg.get("npz", "multisample_carbon")
    fit_cut = float(cfg.get("fit_cut", 0.5))

    _cache = {}
    def load(lbl):
        """Load + schema-check one npz, cached.  An axis may point at its own npz via `npz:` (must share
        the default's knob basis); currently every axis uses the default multi-sample npz."""
        if lbl not in _cache:
            src = style.ALTGEN / f"{lbl}.npz"
            if not src.exists():
                raise SystemExit(f"[sec3] no Jacobian at {src}\n"
                                 f"       available: {sorted(p.stem for p in style.ALTGEN.glob('*.npz'))}\n"
                                 f"       pass a label:  python -m analysis.paper.grad_info.make <label>")
            _cache[lbl] = (str(src),) + SS.check_schema(np.load(src, allow_pickle=True), str(src))
        return _cache[lbl]

    dsrc, dJ, _ds, prior, pnames, ddk, _dr = load(default_label)
    print(f"[sec3] default npz {default_label}: {dJ.shape[0]} bins x {dJ.shape[1]} knobs, {len(ddk)} datasets")
    print(f"       datasets: {ddk}")

    tables = {}
    for aname, axis in cfg["axes"].items():
        axis = dict(axis, _name=aname)
        alabel = axis.get("npz", default_label)
        src, J, sigma, _pr, pn, dskeys, row0 = load(alabel)
        if pn != pnames:
            raise SystemExit(f"[sec3] axis '{aname}' npz {alabel}: knob basis differs from {default_label}")
        groups = SS.resolve_axis(axis, dskeys)
        if not groups:
            print(f"  [sec3] axis '{aname}': no columns survived — skipped")
            continue
        M = np.zeros((len(pnames), len(groups)))
        R = np.zeros_like(M)
        for c, (_n, _l, keys) in enumerate(groups):
            M[:, c], R[:, c], _ = gate1(J, sigma, prior, SS.rows_for(keys, dskeys, row0))

        print(f"\n==== GATE I · axis '{aname}' ({alabel}) -- shrinkage, FIT < {fit_cut} ====")
        names = [g[0] for g in groups]
        print(f"{'knob':>20} " + " ".join(f"{s:>10}" for s in names))
        for k in np.argsort(M[:, -1]):
            print(f"{pnames[k]:>20} " + " ".join(f"{M[k, c]:10.2f}" for c in range(len(groups))))
        for c, s in enumerate(names):
            print(f"  {s:>10}: {int((M[:, c] < fit_cut).sum()):2d}/{len(pnames)} FIT  "
                  f"[{', '.join(pnames[k] for k in np.where(M[:, c] < fit_cut)[0])}]")

        figname = axis.get("figure", f"sec2_shrinkage_{aname}")
        fig_shrinkage_grouped(M, pnames, [g[1] for g in groups], names, figname, fit_cut)
        tables[aname] = (M, R, names)

    if not tables:
        raise SystemExit("[sec2] no axis resolved against this npz — check configs/paper/sec2_subsets.yaml")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
