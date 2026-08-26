"""Shared publication style + paths for the paper figures. Style only -- no physics, no data."""
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.legend_handler import HandlerBase

OUTDIR = Path(os.environ.get("ADONIS_PAPER_OUT", "output/paper"))
ALTGEN = Path(os.environ.get("ADONIS_ALTGEN", "output/altgen"))

PANEL_W = 3.5
PANEL_H = 3.0
STD_COLS = 2

C_ADONIS = "#1f77b4"
C_ACHILLES = "#d62728"
C_DATA = "k"

C_TOTAL = "#dc267f"
C_QE = "#648fff"
C_RES = "#fe6100"
PARTS = {"QE": C_QE, "RES": C_RES}
C_RATIO = "k"

GEN_GREY = "0.35"
ADO_LIGHTEN = 0.25
REF_DARKEN = 0.25

PLATEX = {
    "M_A_qe": r"$M_A^{\rm QE}$", "M_A_res": r"$M_A^{\rm RES}$",
    "axial_strength": r"$S_A^{\rm QE}$", "vector_strength": r"$S_V^{\rm QE}$",
    "mu_p": r"$\mu_p$", "mu_n": r"$\mu_n$", "gep": r"$G_E^p$", "gen": r"$G_E^n$",
    "res_axial_strength": r"$C_5^A$", "pion_pole": r"$F_{\rm pp}$", "delta_strength": r"$S_\Delta$",
    "sabs": r"$s_{\rm abs}^{\pi}$", "s_piN_elastic": r"$s^{\rm el}_{\pi N}$",
    "s_piN_cex": r"$s^{\rm cex}_{\pi N}$", "s_conv": r"$s_{\rm conv}$",
    "s_NN_elastic[0]": r"$s^{\rm el}_{NN,pp}$", "s_NN_elastic[1]": r"$s^{\rm el}_{NN,pn}$",
    "s_NN_elastic[2]": r"$s^{\rm el}_{NN,nn}$",
    "s_NN_inelastic[0]": r"$s^{\rm inel}_{NN,pp}$", "s_NN_inelastic[1]": r"$s^{\rm inel}_{NN,pn}$",
    "s_NN_inelastic[2]": r"$s^{\rm inel}_{NN,nn}$", "f_NN_cex": r"$f^{\rm cex}_{NN}$",
    "kF_sf": r"$k_F$", "Eb_shift": r"$\Delta E_b$", "sf_norm": r"$N_{\rm SF}$",
    "src_tail": r"$N_{\rm SRC}$", "qe_norm": r"$N_{\rm QE}$", "res_norm": r"$N_{\rm RES}$",
}


def plab(p):
    """LaTeX label for a knob name (falls back to the raw name for anything not in PLATEX)."""
    return PLATEX.get(p, p)


class DashSlashSolid(HandlerBase):
    """Legend handle drawn as dashed segment / '/' / solid segment: linestyle carries the generator,
    colour carries the physics."""
    def __init__(self, color=None, lighten=ADO_LIGHTEN, darken=REF_DARKEN, **kw):
        super().__init__(**kw)
        self._c = color or GEN_GREY
        self._lighten, self._darken = lighten, darken

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        y = height / 2.0 - ydescent
        x0 = -xdescent
        seg = 0.40 * width
        arts = [Line2D([x0, x0 + seg], [y, y], color=darker(self._c, self._darken), lw=1.3,
                       dashes=(2.2, 1.4)),
                Line2D([x0 + width - seg, x0 + width], [y, y],
                       color=lighter(self._c, self._lighten), lw=1.4, ls="-"),
                Text(x0 + 0.5 * width, y, "/", color="k", fontsize=fontsize * 1.25,
                     ha="center", va="center")]
        for a in arts:
            a.set_transform(trans)
        return arts


def darker(c, f=REF_DARKEN):
    """Blend a colour toward black by fraction f.  Mirrors adonis.workflow.plotting.darker."""
    r, g, b = mcolors.to_rgb(c)
    return (r * (1.0 - f), g * (1.0 - f), b * (1.0 - f))


def lighter(c, f=ADO_LIGHTEN):
    """Blend a colour toward white by fraction f.  Mirrors adonis.workflow.plotting.lighter."""
    r, g, b = mcolors.to_rgb(c)
    return (r + (1.0 - r) * f, g + (1.0 - g) * f, b + (1.0 - b) * f)


def panel_kw(**over):
    """The paper palette as chi2_ratio_panel kwargs (for make_figure's panel_kw hook)."""
    kw = dict(total_color=C_TOTAL, part_colors=PARTS, ratio_color=C_RATIO,
              ado_lighten=ADO_LIGHTEN, ref_darken=REF_DARKEN, headroom=0.30,
              ratio_yticks=[0.9, 1.0, 1.1])
    kw.update(over)
    return kw


def panel_legend(ax, has_parts, loc="upper right", bbox_to_anchor=None):
    """Legend for a make_figure panel: Total/QE/RES when the selection carries a QE/RES breakdown,
    nothing at all when it is a single series (the caption states dashed=ACHILLES / solid=ADoNIS)."""
    if not has_parts:
        return
    handles, labels, hmap = swatches(
        [("Total", C_TOTAL), ("QE", C_QE), ("RES", C_RES)])
    if bbox_to_anchor is None:
        bbox_to_anchor = (0.84, 1.0) if loc == "upper right" else None
    ax.legend(handles, labels, handler_map=hmap, loc=loc, bbox_to_anchor=bbox_to_anchor,
              fontsize=7, ncol=1, handlelength=3.4, labelspacing=0.35, borderpad=0.2)


def paper_run_analysis_kw():
    """The run_analysis(**kw) styling the paper's config-driven figures use.  Pure-config figures pass
    this to run_analysis; the standalone drivers self-style via style.use() + the style.* palette instead."""
    return dict(panel_w=PANEL_W, fig_h=PANEL_H, min_w=PANEL_W, max_cols=STD_COLS,
                panel_kw=panel_kw(ratio_yticks=[0.8, 1.0, 1.2]),
                legend_fn=panel_legend, label_as_xlabel=True,
                rect_top=1.0)


def swatches(entries, lighten=ADO_LIGHTEN, darken=REF_DARKEN):
    """(handles, labels, handler_map) where every entry is a '-- / -' pair in its own colour.
    entries: [(label, colour_or_None), ...]; None -> the neutral grey generator key."""
    handles, labels, hmap = [], [], {}
    for lab, col in entries:
        h = Line2D([], [])
        hmap[h] = DashSlashSolid(color=col, lighten=lighten, darken=darken)
        handles.append(h); labels.append(lab)
    return handles, labels, hmap


RC = {
    "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "legend.frameon": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "axes.linewidth": 0.8, "lines.linewidth": 1.3,
    "mathtext.default": "regular",
}


def use():
    mpl.rcParams.update(RC)


def _degraded():
    """Degradations accepted in this process (--allow-partial).  Imported lazily so `style` stays usable
    without the fit layer."""
    try:
        from adonis.fit.merge import DEGRADED
        return list(DEGRADED)
    except Exception:
        return []


def save(fig, name):
    """Write <name>.png + <name>.pdf into OUTDIR and return the png path.

    A figure built from a knowingly incomplete shard set is marked: a banner on the image, and the
    reason in the file's metadata.
    """
    OUTDIR.mkdir(parents=True, exist_ok=True)
    png = OUTDIR / f"{name}.png"
    bad = _degraded()
    if bad:
        txt = "; ".join(f"{w}: {r}" for w, r in bad)
        fig.text(0.5, -0.012, "PARTIAL DATA -- " + (txt[:150] + "..." if len(txt) > 150 else txt),
                 ha="center", va="top", color="#c00", fontsize=6.5, zorder=1e6,
                 bbox=dict(facecolor="white", edgecolor="#c00", lw=0.8, pad=3.0))
    meta = {"Title": name, "Software": "ADoNIS", "Comment": "; ".join(r for _w, r in bad) or "complete"}
    fig.savefig(png, metadata=meta)
    fig.savefig(OUTDIR / f"{name}.pdf",
                metadata={"Title": name, "Creator": "ADoNIS", "Subject": meta["Comment"]})
    plt.close(fig)
    print(f"[out] {png}" + ("   *** PARTIAL ***" if bad else ""))
    return png


CMAP_CONSTRAINT = mcolors.LinearSegmentedColormap.from_list(
    "constraint_blue", ["#07204d", "#1f4b9c", "#4f7fe0", "#8fb2f5", "#c9dcfb", "#f6f9ff"])
CMAP_GRAD_DIV = mcolors.LinearSegmentedColormap.from_list(
    "grad_div", ["#0a2a5c", "#648fff", "#f4f4f6", "#fe8a3c", "#7a2f00"])
FIT_EC = "#fe6100"
KNOB_GROUP_NAME = {0: "amplitude", 1: "pion\nFSI", 2: "nucleon\nFSI", 3: "nuclear"}
KNOB_GROUP_COLOR = {0: "#b07d56", 1: "#5f8a6f", 2: "#7b6f9e", 3: "#a86f82"}


def knob_group(p):
    """Physics block for a knob name: 0 amplitude (hard vertex), 1 pion FSI, 2 nucleon FSI, 3 nuclear."""
    if p.startswith("s_piN") or p in ("sabs", "s_conv"):
        return 1
    if p.startswith("s_NN") or p.startswith("f_NN"):
        return 2
    if p in ("kF_sf", "Eb_shift", "sf_norm", "src_tail", "qe_norm", "res_norm"):
        return 3
    return 0


def knob_group_tabs(ax, gid, tabx=-0.15, tabw=0.022, labx=-0.185, lw=1.6, fontsize=8):
    """Draw physics-block separators + a colour-coded tab bracketing each block to its rotated label.
    `gid` is the per-row group id (rows already ordered by block).  x in axes-fraction, y in data."""
    from matplotlib.patches import Rectangle
    nk = len(gid)
    yt = ax.get_yaxis_transform()
    bounds = [i for i in range(1, nk) if gid[i] != gid[i - 1]]
    for b in bounds:
        ax.axhline(b - .5, color="0.22", lw=lw, zorder=6)
    seg = [-.5] + [b - .5 for b in bounds] + [nk - .5]
    for a, b in zip(seg[:-1], seg[1:]):
        g = gid[int((a + b) / 2 + .5)]
        ax.add_patch(Rectangle((tabx, a), tabw, b - a, transform=yt, clip_on=False,
                               facecolor=KNOB_GROUP_COLOR[g], edgecolor="none", zorder=5))
        ax.text(labx, (a + b) / 2, KNOB_GROUP_NAME[g], ha="center", va="center", rotation=90,
                fontsize=fontsize, color="0.2", fontweight="bold", transform=yt)
