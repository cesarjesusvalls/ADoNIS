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
ALTGEN = Path(os.environ.get("ADONIS_ALTGEN", "output/altgen"))   # persisted physfit npz

# ADoNIS / ACHILLES / data colours, used identically in every section.
C_ADONIS = "#1f77b4"
C_ACHILLES = "#d62728"
C_DATA = "k"

# Channel decomposition (total / QE / RES): IBM colourblind-safe palette, indices 2/3/5.
# IBM index 1 (#ffb000 amber) is deliberately NOT used -- lightness 0.81 and only 1.78:1 contrast
# on white, i.e. invisible as a hairline in print.  This triple passes every check of the dataviz
# validator on a light surface over ALL pairs (lightness band, chroma floor, CVD separation,
# normal-vision separation).  #fe6100/#648fff sit just under 3:1 contrast, which the
# always-present legend covers.  Total gets #dc267f -- the only one clearing 3:1 -- because it
# is the headline curve.  Known weak pair: total<->RES is dE 5.2 under tritanopia (~1e-4 of
# readers); position (total always above its components) and linestyle disambiguate.
C_TOTAL = "#dc267f"
C_QE = "#648fff"
C_RES = "#fe6100"
PARTS = {"QE": C_QE, "RES": C_RES}
C_RATIO = "k"

GEN_GREY = "0.35"          # the generator key is an annotation, not a series -> neutral grey
# ADoNIS and ACHILLES agree to within the line width almost everywhere, so linestyle ALONE cannot be
# read: each series is a light/dark PAIR of the same hue (ADoNIS light+solid, ACHILLES dark+dashed,
# drawn over it).  The pair straddles the validated base symmetrically (+-25%): the light member
# therefore sits BELOW the base's contrast, which the thick solid stroke + the legend have to carry.
ADO_LIGHTEN = 0.25
REF_DARKEN = 0.25

# Compact, physics-intuitive LaTeX for each physical_fit SPEC knob (name -> symbol).  Single-sourced HERE
# so the section-2 gradient figures and the section-3 Fisher figures name the same knob identically (they
# plot the same Jacobian).  QE/RES form-factor strengths S_{A,V}; Sachs FFs; C5A (res axial); pion-pole;
# Delta P33 strength; FSI scale factors s_* (pi absorption / piN elastic+cex / conversion / NN
# elastic+inelastic per pp,pn,nn; NN charge-exchange fraction); spectral function k_F / removal-energy
# shift / SF & SRC norms; QE & RES channel norms.
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
    """Legend handle drawn as  -- / --  : dashed segment, a literal '/', solid segment.
    ONE entry replaces the 'ACHILLES x / ADoNIS x' product: linestyle carries the generator,
    colour carries the physics.  HandlerTuple cannot do this (it tiles artists, no text)."""
    def __init__(self, color=None, lighten=ADO_LIGHTEN, darken=REF_DARKEN, **kw):
        super().__init__(**kw)
        self._c = color or GEN_GREY               # None -> neutral grey key
        self._lighten, self._darken = lighten, darken

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        y = height / 2.0 - ydescent
        x0 = -xdescent
        seg = 0.40 * width                       # each line segment (long: little dead space)
        # mirrors the plot: DARK dashed = ACHILLES, LIGHT solid = ADoNIS
        arts = [Line2D([x0, x0 + seg], [y, y], color=darker(self._c, self._darken), lw=1.3,
                       dashes=(2.2, 1.4)),
                Line2D([x0 + width - seg, x0 + width], [y, y],
                       color=lighter(self._c, self._lighten), lw=1.4, ls="-"),
                # the slash is punctuation between the two shades, not part of either -> black
                Text(x0 + 0.5 * width, y, "/", color="k", fontsize=fontsize * 1.25,
                     ha="center", va="center")]
        for a in arts:
            a.set_transform(trans)
        return arts


def darker(c, f=REF_DARKEN):
    """Blend a colour toward black by fraction f.  Mirrors adonis.workflow.plotting.darker so the
    legend swatch and the curves cannot drift apart."""
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
    if bbox_to_anchor is None:          # inset from the right edge only when hugging it, so the
        bbox_to_anchor = (0.84, 1.0) if loc == "upper right" else None   # in-axes tag stays clear
    ax.legend(handles, labels, handler_map=hmap, loc=loc, bbox_to_anchor=bbox_to_anchor,
              fontsize=7, ncol=1, handlelength=3.4, labelspacing=0.35, borderpad=0.2)


def paper_run_analysis_kw():
    """The EXACT run_analysis(**kw) styling the paper's config-driven figures use -- SINGLE SOURCE (was
    inlined at the make.py call site).  Pure-config figures pass this to run_analysis; the standalone
    drivers already self-style via style.use() + the style.* palette, so they need nothing from here."""
    return dict(panel_w=2.4, fig_h=3.0, min_w=3.2,
                panel_kw=panel_kw(ratio_yticks=[0.8, 1.0, 1.2]),
                legend_fn=panel_legend, label_as_xlabel=True,
                title_kw={"fontsize": 9, "y": 0.955, "va": "top"}, rect_top=1.0)


def swatches(entries, lighten=ADO_LIGHTEN, darken=REF_DARKEN):
    """(handles, labels, handler_map) where EVERY entry is a '-- / -' pair in its own colour, so each
    series advertises both of its shades instead of a single stroke the plot never actually draws.
    entries: [(label, colour_or_None), ...]; None -> the neutral grey generator key."""
    handles, labels, hmap = [], [], {}
    for lab, col in entries:
        h = Line2D([], [])                        # one sentinel per entry: handler_map keys on identity
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


def save(fig, name):
    """Write <name>.png + <name>.pdf into OUTDIR and return the png path."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    png = OUTDIR / f"{name}.png"
    fig.savefig(png)
    fig.savefig(OUTDIR / f"{name}.pdf")
    plt.close(fig)
    print(f"[out] {png}")
    return png
