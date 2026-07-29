"""The ONE chi2 + ACH/ADO ratio panel, replacing the ~25 copy-pasted plotting loops.

`chi2_ratio_panel` is verbatim the loop body shared by cc1pi_engine_plot.py and
cc0pi_engine_combined.py: weighted histogram with sqrt(sum w^2)/binwidth errors, ACHILLES stat band +
step, ADoNIS errorbar, ratio panel with error propagation, per-bin chi2 over bins where both are
positive, and the per-variable ACH/ADO integral.  `make_figure` builds the 2xN grid and loops.
"""
from __future__ import annotations
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from adonis.constants import COS70   # single source (adonis.constants)


def hist_with_errors(values, weights, edges):
    """Weighted dsigma/dx [per bin width] and its sqrt(sum w^2)/binwidth Gaussian error."""
    bw = np.diff(edges)
    h, _ = np.histogram(values, edges, weights=weights)
    h2, _ = np.histogram(values, edges, weights=weights ** 2)
    return h / bw, np.sqrt(h2) / bw


def chi2_ratio_panel(a0, a1, edges, ref, ado, *, label, ratio_band=(0.9, 1.1),
                     ratio_ylim=(0.5, 1.6), ado_label="ENGINE", ref_label="ACHILLES", data=None,
                     logy=False, shadow_frac=None, ref_parts=None):
    """Render one observable into top axis a0 (dsigma/dx) + bottom a1 (ACH/ADO ratio).
    ref/ado: dict with the observable key -> values, plus 'w'.  Returns dict(chi2, ndf, ach_ado).
    shadow_frac (opt-in, e.g. 0.01): bins contributing < this fraction of the panel's TOTAL cross section
    (max of the ACH and ADO integrals, so a real discrepancy is never hidden) are SHADOWED (greyed) and
    EXCLUDED from chi2/ndf -- removes stat-inflated low-sigma tail bins.  None -> off (all bins count)."""
    bw = np.diff(edges); ctr = 0.5 * (edges[1:] + edges[:-1])
    da, ea = hist_with_errors(ref["values"], ref["w"], edges)
    dd, ed = hist_with_errors(ado["values"], ado["w"], edges)
    a0.fill_between(edges, np.append(da - ea, (da - ea)[-1]), np.append(da + ea, (da + ea)[-1]),
                    step="post", color="0.55", alpha=0.55, lw=0, label="ACH stat")
    a0.step(edges, np.append(da, da[-1]), where="post", color="0.3", lw=1.3, label=ref_label)
    if ref_parts:                               # reference QE/RES breakdown (thin dashed component steps)
        _pc = {"QE": "tab:blue", "RES": "tab:green"}
        for lbl, part in ref_parts.items():
            dp, _ = hist_with_errors(part["values"], part["w"], edges)
            a0.step(edges, np.append(dp, dp[-1]), where="post", color=_pc.get(lbl, "0.6"),
                    lw=1.0, ls="--", label=f"{ref_label} {lbl}")
    m = (da > 0) & (dd > 0)                     # bins with content on both sides
    shadow = np.zeros(len(bw), bool)            # low-contribution bins: shown greyed, excluded from chi2
    if shadow_frac:
        ca = da * bw / max(float(np.sum(da * bw)), 1e-30)
        cd = dd * bw / max(float(np.sum(dd * bw)), 1e-30)
        shadow = m & (np.maximum(ca, cd) < float(shadow_frac))
    keep = m & ~shadow
    # ADoNIS: step line + shaded error band (SAME visual language as the ACHILLES total/QE/RES);
    # nan-mask non-kept bins so the line/band break cleanly at empties/shadowed bins.
    ddm = np.where(keep, dd, np.nan); edm = np.where(keep, ed, np.nan)
    a0.fill_between(edges, np.append(ddm - edm, (ddm - edm)[-1]), np.append(ddm + edm, (ddm + edm)[-1]),
                    step="post", color="darkorange", alpha=0.30, lw=0)
    a0.step(edges, np.append(ddm, ddm[-1]), where="post", color="darkorange", lw=1.4, label=ado_label)
    if shadow.any():                            # shadowed bins: grey vertical spans in both panels (no markers)
        for i in np.where(shadow)[0]:
            a0.axvspan(edges[i], edges[i + 1], color="0.88", zorder=0)
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
    chi2 = float(np.sum((da[keep] - dd[keep]) ** 2 / (ea[keep] ** 2 + ed[keep] ** 2))); ndf = int(keep.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        r = da / dd; re = r * np.sqrt((ed / dd) ** 2 + (ea / da) ** 2)
    a1.axhline(1.0, ls="-", color="0.6", lw=0.8)                 # central reference
    for off in (0.1, 0.2):                                       # +/-10% and +/-20% shift guides (thin grey lines)
        a1.axhline(1.0 - off, ls="--", color="0.7", lw=0.6)
        a1.axhline(1.0 + off, ls="--", color="0.7", lw=0.6)
    rm = np.where(keep, r, np.nan); rem = np.where(keep, re, np.nan)     # ratio: navy step + shaded band
    a1.fill_between(edges, np.append(rm - rem, (rm - rem)[-1]), np.append(rm + rem, (rm + rem)[-1]),
                    step="post", color="navy", alpha=0.25, lw=0)
    a1.step(edges, np.append(rm, rm[-1]), where="post", color="navy", lw=1.2)
    a1.set_ylim(*ratio_ylim); a1.set_xlabel(label, fontsize=8)
    a0.set_xlim(edges[0], edges[-1])           # clamp to bin edges (sharex -> a1 too): no "floating" margin
    ach_ado = float(np.sum(da * bw) / max(np.sum(dd * bw), 1e-30))
    return dict(chi2=chi2, ndf=ndf, ach_ado=ach_ado, n_shadow=int(shadow.sum()))


def make_figure(specs, ref_sel, ado_sel, *, title="", ratio_band=(0.9, 1.1), ratio_ylim=(0.5, 1.6),
                ado_label="ENGINE", ref_label="ACHILLES", data=None, panel_w=3.4):
    """specs: list of (key, edges, label).  ref_sel/ado_sel: full selection dicts (key->array + 'w').
    Returns (fig, results{key: {chi2,ndf,ach_ado}}, sigma{'ref','ado','ach_ado'})."""
    nv = len(specs)
    total_w = max(panel_w * nv, 7.5)          # floor so single-panel figs are not narrow/clipped
    fig, ax = plt.subplots(2, nv, figsize=(total_w, 6.4),
                           gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0},
                           squeeze=False, sharex="col")
    results = {}
    for c, (key, edges, label) in enumerate(specs):
        ref = {"values": np.asarray(ref_sel[key]), "w": np.asarray(ref_sel["w"])}
        ado = {"values": np.asarray(ado_sel[key]), "w": np.asarray(ado_sel["w"])}
        d = None if data is None else data.get(key)
        rp = None                                    # reference QE/RES breakdown (from the 'chan' column)
        chan = ref_sel.get("chan")
        if chan is not None and len(np.unique(chan)) > 1:
            rp = {}
            for cv, lbl in ((0, "QE"), (1, "RES")):
                mk = np.asarray(chan) == cv
                if mk.any():
                    rp[lbl] = {"values": np.asarray(ref_sel[key])[mk], "w": np.asarray(ref_sel["w"])[mk]}
        res = chi2_ratio_panel(ax[0, c], ax[1, c], np.asarray(edges), ref, ado, label=label,
                               ratio_band=ratio_band, ratio_ylim=ratio_ylim,
                               ado_label=ado_label, ref_label=ref_label, data=d, ref_parts=rp)
        if c == 0:
            ax[0, c].legend(fontsize=7); ax[0, c].set_ylabel(r"$d\sigma/dx$ [nb]")
            ax[1, c].set_ylabel("ratio")
        results[key] = res
    sr, sa = float(np.sum(ref_sel["w"])), float(np.sum(ado_sel["w"]))
    sig = {"ref": sr, "ado": sa, "ach_ado": sr / max(sa, 1e-30)}
    if title:
        fig.suptitle(title, fontsize=12, wrap=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(hspace=0.0)          # top + ratio panels share one x-axis (no gap)
    return fig, results, sig
