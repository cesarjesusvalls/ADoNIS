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

WHICH slices get plotted is not hardcoded here: `configs/paper/sec3_subsets.yaml` declares the column
axes as named groups of dataset keys (literals, globs, or references to an earlier group), and
`subsets.py` resolves them against whatever `dskeys` the input npz actually carries.  Changing the
sample composition or the binning is a config edit; a sample that is absent drops out with a warning
and a sample that no column claims is reported as `uncovered`.

Usage:
    python -m analysis.paper.sec3_fisher.make [npz_label]     # default: the config's `npz:`
    ADONIS_SEC3_CONFIG=... ADONIS_SEC3_NPZ=... python -m analysis.paper.sec3_fisher.make
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from analysis.paper import style
from analysis.paper import fisher_engine as FE
from analysis.paper.sec3_fisher import subsets as SS

C_MEAS, C_DEG, C_INV = "#2ca02c", "#ff7f0e", "#7f7f7f"

plab = style.plab      # knob name -> LaTeX symbol; single-sourced in style.py (identical to sec2's labels)


def gate1(J, sigma, prior, rows):
    """(marginalized shrinkage, raw shrinkage, Fisher) for the given bin rows -- via the shared engine."""
    F, _V, _sig_post, marg, reach = FE.gate1(J, sigma, prior, rows)     # reach = sqrt(diag F)
    with np.errstate(divide="ignore"):
        raw = np.where(reach > 0, 1.0 / (reach * prior), np.inf)         # raw shrinkage = 1/sqrt(F_kk)/prior
    return marg, raw, F


# ---- figures ---------------------------------------------------------------------------------- #

def fig_shrinkage(M, R, pnames, labels, title, figname, fit_cut, marg_only=False):
    """knob x column shrinkage, marginalized next to raw: green-without-red == DEGENERATE.
    marg_only: draw ONLY the marginalized panel (what the fit delivers) -- no raw/degeneracy panel."""
    n = len(labels)
    panels = [(M, "MARGINALIZED (other knobs free)\nwhat the fit delivers", "#d62728")]
    if not marg_only:
        panels.append((R, "RAW (other knobs fixed)\nwhat the data can see", C_MEAS))
    ncol = len(panels)
    fig, axes = plt.subplots(1, ncol, figsize=(1.55 * n * ncol / 2 + 3.4, 8.2), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (Z, ttl, box) in zip(axes, panels):
        im = ax.imshow(np.clip(Z, 0, 1.2), aspect="auto", cmap="Blues", vmin=0, vmax=1.2)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(ttl, fontsize=8.5)
        for k in range(Z.shape[0]):
            for c in range(n):
                v = Z[k, c]                                   # UNCLIPPED: >1.2 and inf must read ">1"
                txt = "$>$1" if (not np.isfinite(v)) or v > 1.2 else f"{v:.2f}"
                ax.text(c, k, txt, ha="center", va="center", fontsize=6,
                        color="w" if min(v, 1.2) > 0.7 else "k")
                if v < fit_cut:
                    ax.add_patch(Rectangle((c - .5, k - .5), 1, 1, fill=False, ec=box, lw=1.4))
    axes[0].set_yticks(range(len(pnames)))
    axes[0].set_yticklabels([plab(p) for p in pnames], fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="shrinkage  $\\sigma_{post}/\\sigma_{prior}$")
    if marg_only:
        fig.suptitle(f"{title}.   red box = FIT: the joint fit measures it "
                     f"($\\sigma_{{post}}/\\sigma_{{prior}} < {fit_cut:g}$)", fontsize=9)
    else:
        fig.suptitle(f"{title}.   red box = FIT (the fit measures it) · "
                     "green box = the data sees it with other knobs held fixed\n"
                     "green without red  $\\Rightarrow$  DEGENERATE (sensitivity exists, another knob spends it)",
                     fontsize=9)
    return style.save(fig, figname)


def _place_labels(ax, xs, ys, texts, colors, fontsize=6.5, marker_px=7.0):
    """Annotate points with labels that overlap neither each other NOR another point's marker.

    Matters here because the interesting knobs cluster tightly in the MEASURABLE corner.  Avoiding only
    other labels is not enough: a label parked next to a *different* marker reads as that marker's name
    (it did — s_NN_elastic[1] pointed straight at sabs).  So every marker is an obstacle from the start,
    and any label pushed off the first ring gets a leader line back to its own point.

    Label boxes are MEASURED with the renderer, not estimated from the character count: the labels are
    LaTeX symbols ($s^{\\rm inel}_{NN,pp}$ is 22 characters and renders about as wide as four), so a
    character-count estimate reserves several times the real width and flings labels off their points.
    """
    fig = ax.figure
    fig.canvas.draw()                                     # transData must be final before we use it
    rend = fig.canvas.get_renderer()
    px = fig.dpi / 72.0                                   # points -> pixels
    pts = ax.transData.transform(np.column_stack([xs, ys]))

    def _size(t):
        a = ax.text(0, 0, t, fontsize=fontsize, transform=None)
        bb = a.get_window_extent(renderer=rend)
        a.remove()
        return bb.width, bb.height

    # obstacles: [x0, y0, x1, y1] in pixels — every marker, up front
    blocked = [np.array([x - marker_px, y - marker_px, x + marker_px, y + marker_px]) for x, y in pts]

    ang = np.deg2rad([0, 45, -45, 90, -90, 135, -135, 180])
    order = np.argsort(-pts[:, 1])                        # top-down: crowded low corner resolves last
    for i in order:
        x, y = pts[i]
        w, h = _size(texts[i])
        w, h = w + 2 * px, h + 2 * px                      # a hair of breathing room around each label
        for r in (marker_px + 0.62 * h, 2.0 * h, 3.1 * h, 4.4 * h):
            best = None
            for a in ang:
                cx, cy = x + (r + w / 2) * np.cos(a), y + (r + h / 2) * np.sin(a)
                box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                if not any(box[0] < p[2] and p[0] < box[2] and box[1] < p[3] and p[1] < box[3]
                           for p in blocked):
                    best = (cx, cy, box, r)
                    break
            if best:
                break
        cx, cy, box, r = best or (x + w / 2 + marker_px, y, np.array([x, y - h / 2, x + w, y + h / 2]),
                                  marker_px)
        blocked.append(box)
        far = r > 1.4 * h
        ax.annotate(texts[i], (xs[i], ys[i]), fontsize=fontsize, color=colors[i], zorder=4,
                    ha="center", va="center", xytext=((cx - x) / px, (cy - y) / px),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=colors[i], alpha=0.8,
                                    shrinkA=0, shrinkB=3) if far else None)


def fig_failure_modes(marg, raw, pnames, colname, figname, fit_cut):
    """The two-axis plane: can the data SEE it (x) vs can the fit DELIVER it (y)."""
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    xlo, xhi = 5e-3, 4.0
    rp = np.clip(raw, xlo * 1.05, xhi * 0.92)             # inf / huge raw pile up at the right edge
    mp = np.clip(marg, 0, 1.05)
    ax.set_xscale("log")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(-0.02, 1.12)

    # shaded regions, so the classification reads without hunting for the caption
    ax.add_patch(Rectangle((xlo, -0.02), fit_cut - xlo, fit_cut + 0.02, color=C_MEAS, alpha=0.05, lw=0))
    ax.add_patch(Rectangle((xlo, fit_cut), fit_cut - xlo, 1.14 - fit_cut, color=C_DEG, alpha=0.05, lw=0))
    ax.add_patch(Rectangle((fit_cut, -0.02), xhi - fit_cut, 1.14, color=C_INV, alpha=0.07, lw=0))
    ax.axvline(fit_cut, color="k", ls=":", lw=.8)
    ax.axhline(fit_cut, color="k", ls=":", lw=.8)

    kind = np.where(marg < fit_cut, 0, np.where(raw < fit_cut, 1, 2))
    cols = [(C_MEAS, C_DEG, C_INV)[k] for k in kind]
    clipped = ~np.isfinite(raw) | (raw > xhi * 0.92)
    ax.scatter(rp[~clipped], mp[~clipped], s=34, c=[c for c, q in zip(cols, clipped) if not q], zorder=3)
    ax.scatter(rp[clipped], mp[clipped], s=44, marker=">", zorder=3,
               c=[c for c, q in zip(cols, clipped) if q])

    # The region key lives OUTSIDE the axes: a knob can sit anywhere in the plane, so any in-plot
    # caption box eventually lands on a point (it did: it swallowed Eb_shift).  Shading + legend
    # carries the same information and leaves the label placer the whole area to work with.
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=c, alpha=0.25, label=t) for c, t in (
        (C_MEAS, "MEASURABLE — seen, and no other knob can spend it"),
        (C_DEG, "DEGENERATE — seen, but mimicked (a better observable can recover it)"),
        (C_INV, "INVISIBLE — no information at this precision (nothing can)"))],
        loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, fontsize=8, handlelength=1.2,
        handleheight=0.9, borderpad=0.3, labelspacing=0.35)

    _place_labels(ax, rp, mp, [plab(p) for p in pnames], cols, fontsize=8.5)
    ax.set_xlabel("raw shrinkage  (other knobs FIXED)  —  can the data see it?")
    ax.set_ylabel("marginalized shrinkage  —  can the fit deliver it?")
    ax.set_title(f"Two ways to fail Gate I   ({colname})", fontsize=10)
    return style.save(fig, figname)


def fig_degeneracy(F, prior, pnames, colname, figname, kmodes=8):
    """Prior-scaled Fisher eigen-spectrum + the composition of the best-measured modes."""
    Fs = F * prior[:, None] * prior[None, :]
    evals, evecs = np.linalg.eigh(Fs)
    order = np.argsort(evals)[::-1]
    K = min(kmodes, len(evals))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), gridspec_kw={"width_ratios": [1, 2.4]})
    axes[0].semilogy(range(1, len(evals) + 1), np.maximum(evals[order], 1e-12), "o-", ms=4,
                     color=style.C_ADONIS)
    axes[0].axhline(1.0, color="k", ls=":", lw=.8)
    axes[0].text(len(evals) * .55, 1.3, "$\\lambda<1$: prior-dominated", fontsize=7)
    axes[0].set_xlabel("eigenmode"); axes[0].set_ylabel("$\\lambda$  (prior-scaled Fisher)")
    axes[0].set_title("degeneracy spectrum", fontsize=9)
    im = axes[1].imshow(evecs[:, order[:K]], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_xticks(range(K))
    axes[1].set_xticklabels([f"{evals[order[i]]:.3g}" for i in range(K)], fontsize=7, rotation=45)
    axes[1].set_xlabel("$\\lambda$ of the mode")
    axes[1].set_yticks(range(len(pnames)))
    axes[1].set_yticklabels([plab(p) for p in pnames], fontsize=9)
    axes[1].set_title(f"eigenvector composition, best-measured modes  ({colname})", fontsize=9)
    fig.colorbar(im, ax=axes[1], fraction=0.03, pad=0.02, label="knob weight in the mode")
    return style.save(fig, figname)


# ---- driver ----------------------------------------------------------------------------------- #

def main(label=None):
    style.use()
    cfg = SS.load_config()
    default_label = label or os.environ.get("ADONIS_SEC3_NPZ") or cfg.get("npz", "multisample_carbon")
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
                                 f"       pass a label:  python -m analysis.paper.sec3_fisher.make <label>")
            _cache[lbl] = (str(src),) + SS.check_schema(np.load(src, allow_pickle=True), str(src))
        return _cache[lbl]

    # the default npz fixes the shared knob basis (pnames/prior) every axis must agree on
    dsrc, dJ, _ds, prior, pnames, ddk, _dr = load(default_label)
    print(f"[sec3] default npz {default_label}: {dJ.shape[0]} bins x {dJ.shape[1]} knobs, {len(ddk)} datasets")
    print(f"       datasets: {ddk}")

    tables, resolved, arrays = {}, {}, {}
    for aname, axis in cfg["axes"].items():
        axis = dict(axis, _name=aname)
        alabel = axis.get("npz", default_label)                     # per-axis input; else the file default
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

        fig_shrinkage(M, R, pnames, [g[1] for g in groups], axis.get("title", aname),
                      axis.get("figure", f"sec3_shrinkage_{aname}"), fit_cut,
                      marg_only=bool(axis.get("marginalized_only", False)))
        tables[aname] = (M, R, names)
        resolved[aname] = groups
        arrays[aname] = (J, sigma, dskeys, row0)

    if not tables:
        raise SystemExit("[sec3] no axis resolved against this npz — check configs/paper/sec3_subsets.yaml")

    # ---- figs 2 and 3 are computed on ONE designated column (config `reference:`) ---------------- #
    ref = cfg.get("reference") or {}
    aname = ref.get("axis") if ref.get("axis") in resolved else list(resolved)[-1]
    groups, (M, R, names) = resolved[aname], tables[aname]
    J, sigma, dskeys, row0 = arrays[aname]                          # the reference axis's own npz
    g = ref.get("group", -1)
    c = names.index(g) if isinstance(g, str) and g in names else (int(g) if isinstance(g, int) else -1)
    colname = f"{aname}/{names[c]}"
    print(f"\n[sec3] failure-mode + degeneracy reference column: {colname} "
          f"({len(groups[c][2])} datasets, {int((M[:, c] < fit_cut).sum())}/{len(pnames)} FIT)")

    fig_failure_modes(M[:, c], R[:, c], pnames, colname, "sec3_failure_modes", fit_cut)
    _, _, F = gate1(J, sigma, prior, SS.rows_for(groups[c][2], dskeys, row0))
    fig_degeneracy(F, prior, pnames, colname, "sec3_degeneracy")

    out = style.ALTGEN / f"sec3_shrinkage_{default_label}.npz"
    np.savez(out, pnames=pnames, source=str(dsrc), fit_cut=fit_cut, reference=colname,
             **{f"{a}_marg": t[0] for a, t in tables.items()},
             **{f"{a}_raw": t[1] for a, t in tables.items()},
             **{f"{a}_cols": np.asarray(t[2]) for a, t in tables.items()})
    print(f"[out] {out}")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
