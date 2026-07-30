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
