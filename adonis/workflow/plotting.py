"""The shared chi2 + ACH/ADO ratio panel used by every ADoNIS-vs-ACHILLES comparison figure.

`chi2_ratio_panel` draws: weighted histogram with sqrt(sum w^2)/binwidth errors, ACHILLES stat band +
step, ADoNIS errorbar, ratio panel with error propagation, per-bin chi2 over bins where both are
positive, and the per-variable ACH/ADO integral.  `make_figure` builds the 2xN grid and loops.
"""
from __future__ import annotations
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from adonis.constants import COS70


def hist_with_errors(values, weights, edges):
    """Weighted dsigma/dx [per bin width] and its sqrt(sum w^2)/binwidth Gaussian error."""
    bw = np.diff(edges)
    h, _ = np.histogram(values, edges, weights=weights)
    h2, _ = np.histogram(values, edges, weights=weights ** 2)
    return h / bw, np.sqrt(h2) / bw


def darker(c, f):
    """Blend a colour toward BLACK by fraction f (f=0 -> unchanged)."""
    if not f:
        return c
    r, g, b = mcolors.to_rgb(c)
    return (r * (1.0 - f), g * (1.0 - f), b * (1.0 - f))


def lighter(c, f):
    """Blend a colour toward WHITE by fraction f (f=0 -> unchanged).  NB this lowers contrast against
    a white page, so the pair is symmetric about the validated base rather than resting on it."""
    if not f:
        return c
    r, g, b = mcolors.to_rgb(c)
    return (r + (1.0 - r) * f, g + (1.0 - g) * f, b + (1.0 - b) * f)


def _headroom(a0, frac):
    """Grow the top of the y range so in-axes labels/legends do not sit on the curves."""
    if frac:
        a0.set_ylim(a0.get_ylim()[0], a0.get_ylim()[1] * (1.0 + frac))


def _curve_panel(a0, a1, ref, ado, *, label, ratio_band, ratio_ylim, ado_label, ref_label, xlabel=None,
                 curve_color="0.1", ratio_color="navy", ref_darken=0.0, headroom=0.0, ado_lighten=0.0,
                 show_ratio=True):
    """Precomputed-curve/points variant: ref/ado are {x, y, yerr} (NOT per-event {values, w}).
    For sigma-vs-scan (E_nu, W) curves and per-bin efficiency points -- same line+band + ratio style as
    the histogram panel, but the y is handed in already reduced (integrated sigma / per-bin efficiency)."""
    x = np.asarray(ref["x"]); ry = np.asarray(ref["y"]); rye = np.asarray(ref.get("yerr", np.zeros_like(ry)))
    ay = np.asarray(ado["y"]); aye = np.asarray(ado.get("yerr", np.zeros_like(ay)))
    edges = ref.get("edges")
    edges = None if edges is None else np.asarray(edges)

    def _draw(a, y, ye, col, lw, ls=None, label=None):
        if edges is None:
            a.fill_between(x, y - ye, y + ye, color=col, alpha=0.22, lw=0)
            a.plot(x, y, color=col, lw=lw, ls=ls, label=label)
        else:
            a.fill_between(edges, np.append(y - ye, (y - ye)[-1]), np.append(y + ye, (y + ye)[-1]),
                           step="post", color=col, alpha=0.22, lw=0)
            a.step(edges, np.append(y, y[-1]), where="post", color=col, lw=lw, ls=ls, label=label)
    for y, ye, ls, lab, lw, col in [(ay, aye, "-", ado_label, 1.4, lighter(curve_color, ado_lighten)),
                                    (ry, rye, "--", ref_label, 1.3, darker(curve_color, ref_darken))]:
        _draw(a0, y, ye, col, lw, ls, lab)
    a0.set_title(label, fontsize=9); a0.set_ylim(bottom=0); _headroom(a0, headroom)
    m = (ry > 0) & (ay > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(m, ry / ay, np.nan)
        re = np.where(m, r * np.sqrt((aye / ay) ** 2 + (rye / ry) ** 2), np.nan)
    if show_ratio:
        a1.axhline(1.0, ls="-", color="0.6", lw=0.8)
        for off in (0.1, 0.2):
            a1.axhline(1.0 - off, ls="--", color="0.7", lw=0.6); a1.axhline(1.0 + off, ls="--", color="0.7", lw=0.6)
        _draw(a1, r, re, ratio_color, 1.2)
        a1.set_ylim(*ratio_ylim); a1.set_xlabel(xlabel or label, fontsize=8)
    else:
        a0.set_xlabel(xlabel or label, fontsize=8)
    a0.set_xlim(*(( x[0], x[-1]) if edges is None else (edges[0], edges[-1])))
    chi2 = float(np.nansum((ry[m] - ay[m]) ** 2 / (rye[m] ** 2 + aye[m] ** 2 + 1e-30))); ndf = int(m.sum())
    return dict(chi2=chi2, ndf=ndf, ach_ado=float(np.nansum(ry[m]) / max(np.nansum(ay[m]), 1e-30)), n_shadow=0)


def chi2_ratio_panel(a0, a1, edges, ref, ado, *, label, ratio_band=(0.9, 1.1),
                     ratio_ylim=(0.5, 1.6), ado_label="ENGINE", ref_label="ACHILLES", data=None,
                     logy=False, shadow_frac=None, ref_parts=None, ado_parts=None, xlabel=None,
                     total_color="0.1", part_colors=None, ratio_color="navy",
                     ref_darken=0.0, headroom=0.0, ado_lighten=0.0, ratio_xmax=None,
                     ratio_yticks=None, show_ratio=True):
    """Render one observable into top axis a0 (dsigma/dx) + bottom a1 (ACH/ADO ratio).
    ref/ado: dict with the observable key -> values, plus 'w'.  Returns dict(chi2, ndf, ach_ado).
    total_color / part_colors / ratio_color are palette hooks (unset -> default colors); part_colors
    maps a component label -> colour.
    shadow_frac (opt-in, e.g. 0.01): bins contributing < this fraction of the panel's total cross section
    (max of the ACH and ADO integrals, so a real discrepancy is never hidden) are shadowed (greyed) and
    excluded from chi2/ndf -- removes stat-inflated low-sigma tail bins.  None -> off (all bins count).
    Curve/points mode: if ref/ado carry precomputed {x, y, yerr} (not {values, w}), delegate to
    _curve_panel (sigma-vs-scan curves, per-bin efficiency) -- same style, y handed in already reduced."""
    if "y" in ref:
        return _curve_panel(a0, a1, ref, ado, label=label, ratio_band=ratio_band, ratio_ylim=ratio_ylim,
                            ado_label=ado_label, ref_label=ref_label, xlabel=xlabel,
                            curve_color=total_color, ratio_color=ratio_color,
                            ref_darken=ref_darken, headroom=headroom, ado_lighten=ado_lighten,
                            show_ratio=show_ratio)
    bw = np.diff(edges); ctr = 0.5 * (edges[1:] + edges[:-1])
    da, ea = hist_with_errors(ref["values"], ref["w"], edges)
    dd, ed = hist_with_errors(ado["values"], ado["w"], edges)
    def _components(parts, ls, prefix, shade):
        _pc = {"QE": "tab:blue", "RES": "tab:green"} if part_colors is None else part_colors
        for lbl, part in parts.items():
            dp, dpe = hist_with_errors(part["values"], part["w"], edges)
            _base = _pc.get(lbl, "0.6")
            col = darker(_base, shade) if shade >= 0 else lighter(_base, -shade)
            a0.fill_between(edges, np.append(dp - dpe, (dp - dpe)[-1]), np.append(dp + dpe, (dp + dpe)[-1]),
                            step="post", color=col, alpha=0.15, lw=0)
            a0.step(edges, np.append(dp, dp[-1]), where="post", color=col, lw=1.0, ls=ls,
                    label=f"{prefix} {lbl}")
    m = (da > 0) & (dd > 0)
    shadow = np.zeros(len(bw), bool)
    if shadow_frac:
        ca = da * bw / max(float(np.sum(da * bw)), 1e-30)
        cd = dd * bw / max(float(np.sum(dd * bw)), 1e-30)
        shadow = m & (np.maximum(ca, cd) < float(shadow_frac))
    keep = m & ~shadow
    ddm = np.where(keep, dd, np.nan); edm = np.where(keep, ed, np.nan)
    _ado_col = lighter(total_color, ado_lighten)
    a0.fill_between(edges, np.append(ddm - edm, (ddm - edm)[-1]), np.append(ddm + edm, (ddm + edm)[-1]),
                    step="post", color=_ado_col, alpha=0.22, lw=0)
    a0.step(edges, np.append(ddm, ddm[-1]), where="post", color=_ado_col, lw=1.4, ls="-", label=ado_label)
    if ado_parts:
        _components(ado_parts, "-", ado_label, -ado_lighten)
    _ref_col = darker(total_color, ref_darken)
    a0.fill_between(edges, np.append(da - ea, (da - ea)[-1]), np.append(da + ea, (da + ea)[-1]),
                    step="post", color=_ref_col, alpha=0.22, lw=0)
    a0.step(edges, np.append(da, da[-1]), where="post", color=_ref_col, lw=1.3, ls="--", label=ref_label)
    if ref_parts:
        _components(ref_parts, "--", ref_label, ref_darken)
    if shadow.any():
        for i in np.where(shadow)[0]:
            a0.axvspan(edges[i], edges[i + 1], color="0.88", zorder=0)
            if show_ratio:
                a1.axvspan(edges[i], edges[i + 1], color="0.88", zorder=0)
    if data is not None:
        a0.errorbar(data["ctr"], data["val"], yerr=data["err"], fmt="o", color="k", ms=3,
                    capsize=2, lw=0.9, label="data")
    a0.set_title(label, fontsize=9)
    if logy:
        a0.set_yscale("log")
        pos = np.concatenate([da[da > 0], dd[dd > 0]])
        if pos.size:
            a0.set_ylim(0.5 * pos.min(), 2.0 * pos.max())
    else:
        a0.set_ylim(bottom=0)
    _headroom(a0, headroom)
    rkeep = keep if ratio_xmax is None else (keep & (ctr < float(ratio_xmax)))
    chi2 = float(np.sum((da[rkeep] - dd[rkeep]) ** 2 / (ea[rkeep] ** 2 + ed[rkeep] ** 2)))
    ndf = int(rkeep.sum())
    if show_ratio:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = da / dd; re = r * np.sqrt((ed / dd) ** 2 + (ea / da) ** 2)
        a1.axhline(1.0, ls="-", color="0.6", lw=0.8)
        for off in (0.1, 0.2):
            a1.axhline(1.0 - off, ls="--", color="0.7", lw=0.6)
            a1.axhline(1.0 + off, ls="--", color="0.7", lw=0.6)
        rm = np.where(rkeep, r, np.nan); rem = np.where(rkeep, re, np.nan)
        a1.fill_between(edges, np.append(rm - rem, (rm - rem)[-1]), np.append(rm + rem, (rm + rem)[-1]),
                        step="post", color=ratio_color, alpha=0.25, lw=0)
        a1.step(edges, np.append(rm, rm[-1]), where="post", color=ratio_color, lw=1.2)
        lo, hi = ratio_ylim
        if ratio_yticks is not None:
            a1.set_yticks(ratio_yticks)
            tlo, thi = min(ratio_yticks), max(ratio_yticks)
            step = (thi - tlo) / max(len(ratio_yticks) - 1, 1)
            lo, hi = min(lo, tlo - 0.6 * step), max(hi, thi + 0.6 * step)
        a1.set_ylim(lo, hi); a1.set_xlabel(xlabel or label, fontsize=8)
    else:
        a0.set_xlabel(xlabel or label, fontsize=8)

    a0.set_xlim(edges[0], edges[-1])
    ach_ado = float(np.sum(da * bw) / max(np.sum(dd * bw), 1e-30))
    return dict(chi2=chi2, ndf=ndf, ach_ado=ach_ado, n_shadow=int(shadow.sum()))


def _ylabel(label, name_var=False):
    """y-axis label for a top panel.  name_var=True turns the spec label into the differential
    variable: "$p_{\\pi^0}$ [MeV/c]" -> "$d\\sigma/dp_{\\pi^0}$ [nb]".  The unit suffix is dropped
    because the y unit is nb, not the x unit."""
    if not name_var or not label:
        return r"$d\sigma/dx$ [nb]"
    v = label.split("[")[0].strip().strip("$").strip()
    return rf"$d\sigma/d{v}$ [nb]" if v else r"$d\sigma/dx$ [nb]"


def make_figure(specs, ref_sel, ado_sel, *, title="", ratio_band=(0.9, 1.1), ratio_ylim=(0.5, 1.6),
                ado_label="ENGINE", ref_label="ACHILLES", data=None, panel_w=3.4,
                panel_kw=None, legend_fn=None, fig_h=6.4, title_kw=None, label_as_xlabel=False,
                rect_top=0.96, min_w=7.5, max_cols=None, ylabel_var=False,
                wspace=None, annotations=None, ylabel=None, ratio_xmax=None, show_ratio=True):
    """specs: list of (key, edges, label).  ref_sel/ado_sel: full selection dicts (key->array + 'w').
    Returns (fig, results{key: {chi2,ndf,ach_ado}}, sigma{'ref','ado','ach_ado'}).
    Style hooks (all optional): panel_kw forwards palette/headroom args to chi2_ratio_panel; legend_fn
    (ax, has_parts) replaces the default legend; fig_h/title_kw size the canvas and title;
    label_as_xlabel puts the spec label on the x axis instead of on top of the panel (avoiding a
    duplicate of the title)."""
    nv = len(specs)
    panel_kw = dict(panel_kw or {})
    _rx_default = panel_kw.pop("ratio_xmax", None)
    ncol = nv if not max_cols else min(max_cols, nv)
    nrow = 1 if not max_cols else int(np.ceil(nv / ncol))
    total_w = max(panel_w * ncol, min_w)
    nsub = 2 if show_ratio else 1
    hr = [3, 1] if show_ratio else [1]
    row_h = fig_h if show_ratio else fig_h * 0.75
    fig = plt.figure(figsize=(total_w, row_h * nrow))
    outer = fig.add_gridspec(nrow, 1, hspace=0.30 if nrow > 1 else 0.0)
    ax = np.empty((2, nv), dtype=object)
    for r in range(nrow):
        k = min(ncol, nv - r * ncol)
        if k == ncol:
            inner = outer[r].subgridspec(nsub, ncol, height_ratios=hr, hspace=0.0, wspace=wspace)
            gcols = list(range(ncol))
        else:
            pad = (ncol - k) / 2.0
            inner = outer[r].subgridspec(nsub, k + 2, width_ratios=[pad] + [1.0] * k + [pad],
                                         height_ratios=hr, hspace=0.0, wspace=wspace)
            gcols = list(range(1, k + 1))
        for cc in range(k):
            i = r * ncol + cc
            gc = gcols[cc]
            top = fig.add_subplot(inner[0, gc])
            if show_ratio:
                rat = fig.add_subplot(inner[1, gc], sharex=top)
                top.tick_params(labelbottom=False)
                ax[0, i], ax[1, i] = top, rat
            else:
                ax[0, i], ax[1, i] = top, None
    results = {}
    for c, (key, edges, label) in enumerate(specs):
        if isinstance(ref_sel[key], dict) and "y" in ref_sel[key]:
            ref, ado = ref_sel[key], ado_sel[key]
        else:
            ref = {"values": np.asarray(ref_sel[key]), "w": np.asarray(ref_sel["w"])}
            ado = {"values": np.asarray(ado_sel[key]), "w": np.asarray(ado_sel["w"])}
        d = None if data is None else data.get(key)

        def _parts(sel):
            chan = sel.get("chan")
            if chan is None or len(np.unique(chan)) < 2:
                return None
            out = {}
            for cv, lbl in ((0, "QE"), (1, "RES")):
                mk = np.asarray(chan) == cv
                if mk.any():
                    out[lbl] = {"values": np.asarray(sel[key])[mk], "w": np.asarray(sel["w"])[mk]}
            return out
        rp, ap = _parts(ref_sel), _parts(ado_sel)
        rx = (ratio_xmax.get(key) if isinstance(ratio_xmax, dict) else ratio_xmax)
        if rx is None:
            rx = _rx_default
        res = chi2_ratio_panel(ax[0, c], ax[1, c], np.asarray(edges), ref, ado,
                               label="" if label_as_xlabel else label,
                               xlabel=label if label_as_xlabel else None,
                               ratio_band=ratio_band, ratio_ylim=ratio_ylim,
                               ado_label=ado_label, ref_label=ref_label, data=d,
                               ref_parts=rp, ado_parts=ap, ratio_xmax=rx, show_ratio=show_ratio, **panel_kw)
        if c == 0:
            if legend_fn is None:
                ax[0, c].legend(fontsize=7)
            else:
                legend_fn(ax[0, c], ap is not None)
            ax[0, c].set_ylabel(ylabel if ylabel is not None else _ylabel(label, name_var=ylabel_var))
            if show_ratio:
                ax[1, c].set_ylabel("ratio")
        results[key] = res
    if annotations:
        for c, (key, _e, _l) in enumerate(specs):
            if key in annotations:
                ax[0, c].text(0.97, 0.95, annotations[key], transform=ax[0, c].transAxes,
                              fontsize=7, va="top", ha="right")
    sr = float(np.sum(ref_sel["w"])) if "w" in ref_sel else 0.0
    sa = float(np.sum(ado_sel["w"])) if "w" in ado_sel else 0.0
    sig = {"ref": sr, "ado": sa, "ach_ado": sr / max(sa, 1e-30)}
    if title:
        fig.suptitle(title, **(title_kw or {"fontsize": 12, "wrap": True}))
    fig.tight_layout(rect=[0, 0, 1, rect_top])
    if nrow == 1 and show_ratio:
        fig.subplots_adjust(hspace=0.0)
    return fig, results, sig
