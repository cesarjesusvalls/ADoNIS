"""The ONE chi2 + ACH/ADO ratio panel, replacing the ~25 copy-pasted plotting loops.

`chi2_ratio_panel` is verbatim the loop body shared by cc1pi_engine_plot.py and
cc0pi_engine_combined.py: weighted histogram with sqrt(sum w^2)/binwidth errors, ACHILLES stat band +
step, ADoNIS errorbar, ratio panel with error propagation, per-bin chi2 over bins where both are
positive, and the per-variable ACH/ADO integral.  `make_figure` builds the 2xN grid and loops.
"""
from __future__ import annotations
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

COS70 = float(np.cos(np.deg2rad(70.0)))


def hist_with_errors(values, weights, edges):
    """Weighted dsigma/dx [per bin width] and its sqrt(sum w^2)/binwidth Gaussian error."""
    bw = np.diff(edges)
    h, _ = np.histogram(values, edges, weights=weights)
    h2, _ = np.histogram(values, edges, weights=weights ** 2)
    return h / bw, np.sqrt(h2) / bw


def chi2_ratio_panel(a0, a1, edges, ref, ado, *, label, ratio_band=(0.9, 1.1),
                     ratio_ylim=(0.5, 1.6), ado_label="ENGINE", ref_label="ACHILLES", data=None,
                     logy=False):
    """Render one observable into top axis a0 (dsigma/dx) + bottom a1 (ACH/ADO ratio).
    ref/ado: dict with the observable key -> values, plus 'w'.  Returns dict(chi2, ndf, ach_ado)."""
    bw = np.diff(edges); ctr = 0.5 * (edges[1:] + edges[:-1])
    da, ea = hist_with_errors(ref["values"], ref["w"], edges)
    dd, ed = hist_with_errors(ado["values"], ado["w"], edges)
    a0.fill_between(edges, np.append(da - ea, (da - ea)[-1]), np.append(da + ea, (da + ea)[-1]),
                    step="post", color="0.55", alpha=0.55, lw=0, label="ACH stat")
    a0.step(edges, np.append(da, da[-1]), where="post", color="0.3", lw=1.3, label=ref_label)
    a0.errorbar(ctr, dd, yerr=ed, fmt="s", color="C0", ms=3, capsize=2, lw=0.9, label=ado_label)
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
    m = (da > 0) & (dd > 0)
    chi2 = float(np.sum((da[m] - dd[m]) ** 2 / (ea[m] ** 2 + ed[m] ** 2))); ndf = int(m.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        r = da / dd; re = r * np.sqrt((ed / dd) ** 2 + (ea / da) ** 2)
    a1.axhspan(ratio_band[0], ratio_band[1], color="green", alpha=0.12)
    a1.axhline(1.0, ls="--", color="green", lw=0.7)
    a1.errorbar(ctr[m], r[m], yerr=re[m], fmt="o", color="C3", ms=3, capsize=2, lw=0.8)
    a1.set_ylim(*ratio_ylim); a1.set_xlabel(label, fontsize=8)
    a0.set_xlim(edges[0], edges[-1])           # clamp to bin edges (sharex -> a1 too): no "floating" margin
    a1.text(0.04, 0.83, f"{chi2/max(ndf,1):.1f}", transform=a1.transAxes, fontsize=9)
    ach_ado = float(np.sum(da * bw) / max(np.sum(dd * bw), 1e-30))
    return dict(chi2=chi2, ndf=ndf, ach_ado=ach_ado)


def make_figure(specs, ref_sel, ado_sel, *, title="", ratio_band=(0.9, 1.1), ratio_ylim=(0.5, 1.6),
                ado_label="ENGINE", ref_label="ACHILLES", data=None, panel_w=3.4):
    """specs: list of (key, edges, label).  ref_sel/ado_sel: full selection dicts (key->array + 'w').
    Returns (fig, results{key: {chi2,ndf,ach_ado}}, sigma{'ref','ado','ach_ado'})."""
    nv = len(specs)
    total_w = max(panel_w * nv, 7.5)          # floor so single-panel figs are not narrow/clipped
    fig, ax = plt.subplots(2, nv, figsize=(total_w, 6.4), height_ratios=[3, 1],
                           squeeze=False, sharex="col")
    results = {}
    for c, (key, edges, label) in enumerate(specs):
        ref = {"values": np.asarray(ref_sel[key]), "w": np.asarray(ref_sel["w"])}
        ado = {"values": np.asarray(ado_sel[key]), "w": np.asarray(ado_sel["w"])}
        d = None if data is None else data.get(key)
        res = chi2_ratio_panel(ax[0, c], ax[1, c], np.asarray(edges), ref, ado, label=label,
                               ratio_band=ratio_band, ratio_ylim=ratio_ylim,
                               ado_label=ado_label, ref_label=ref_label, data=d)
        if c == 0:
            ax[0, c].legend(fontsize=7)
        results[key] = res
    sr, sa = float(np.sum(ref_sel["w"])), float(np.sum(ado_sel["w"]))
    sig = {"ref": sr, "ado": sa, "ach_ado": sr / max(sa, 1e-30)}
    if title:
        fig.suptitle(title + f"  ACH/ADO {sig['ach_ado']:.3f}", fontsize=12, wrap=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, results, sig
