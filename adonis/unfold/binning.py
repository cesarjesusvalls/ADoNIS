"""Rectangular 2-D binnings in (delta-p_T, delta-alpha_T), one for truth and a finer one for reco.

Unfolding needs MORE reco bins than truth bins -- the reco distribution is the constraint and the truth
distribution is the unknown, so an over-determined system is what makes the inverse stable.  Both live
here so the two can never drift apart, and so the flattening convention (row-major, delta-p_T slowest) is
stated exactly once: a truth index that means something different in the response builder than in the
figure is the kind of bug that produces a plausible, wrong unfolded spectrum.

The truth grid is 10 cells and the reco grid is 30, as agreed.  Within those counts the factorisation is
chosen so the delta-p_T axis nests exactly (each truth bin splits into two reco bins), which keeps the
response matrix strongly diagonal in the variable that carries most of the structure.  delta-alpha_T does
not nest (2 -> 3); it does not need to, because the projections quoted in the paper are taken in TRUTH
space, where nesting is trivial.
"""
from __future__ import annotations

import numpy as np

PI = float(np.pi)

# EQUAL-OCCUPANCY edges, from the weighted quantiles of the nominal sample (truth edges from the true
# distribution, reco edges from the smeared one).  The first attempt used the NUISANCE release edges
# coarsened by eye, and left a reco bin holding 0.32 events out of 11 500 -- a bin where sqrt(N) is not an
# uncertainty and the Gaussian chi2 term is meaningless.  Quantiles fix that by construction.
#
# The last bin of each delta-p_T axis is OPEN.  Smearing pushes the reco tail out to ~4.9 GeV, and an
# event outside the grid is not binned at all: a closed axis would silently drop ~1% of the reco sample
# from the fit, and any true signal above the top edge would be reclassified as background.
TRUE_DPT = [0.0, 85.0, 125.0, 175.0, 270.0, np.inf]
RECO_DPT = [0.0, 80.0, 120.0, 150.0, 185.0, 220.0, 260.0, 310.0, 380.0, 520.0, np.inf]

# delta-alpha_T [rad], likewise equal-occupancy.  pi is a hard kinematic bound, so these axes are closed.
TRUE_DAT = [0.0, 1.75, PI]
RECO_DAT = [0.0, 1.34, 2.38, PI]


class Grid2D:
    """A rectangular grid over (dpt, dat) with a flat index.  Out-of-range events are index -1, never
    silently clipped into the edge bin -- an event at dpt = 3 GeV is not a 1 GeV event."""

    def __init__(self, dpt_edges, dat_edges, name=""):
        self.dpt = np.asarray(dpt_edges, dtype=float)
        self.dat = np.asarray(dat_edges, dtype=float)
        self.name = name
        self.ndpt = len(self.dpt) - 1
        self.ndat = len(self.dat) - 1
        self.n = self.ndpt * self.ndat

    def index(self, dpt, dat):
        """Flat bin per event, row-major with dpt slowest: k = i_dpt * ndat + i_dat.  -1 = outside."""
        i = np.digitize(np.asarray(dpt, dtype=float), self.dpt) - 1
        j = np.digitize(np.asarray(dat, dtype=float), self.dat) - 1
        ok = (i >= 0) & (i < self.ndpt) & (j >= 0) & (j < self.ndat)
        return np.where(ok, i * self.ndat + j, -1)

    def unflatten(self, v):
        """Flat (n,) -> (ndpt, ndat), for projecting onto either axis."""
        return np.asarray(v).reshape(self.ndpt, self.ndat)

    def project(self, v, axis):
        """Collapse a flat vector onto 'dpt' or 'dat' by summing over the other axis."""
        M = self.unflatten(v)
        return M.sum(axis=1) if axis == "dpt" else M.sum(axis=0)

    def widths(self, axis):
        return np.diff(self.dpt) if axis == "dpt" else np.diff(self.dat)

    def __repr__(self):
        return f"Grid2D({self.name}: {self.ndpt} dpt x {self.ndat} dat = {self.n} cells)"


def truth_grid():
    return Grid2D(TRUE_DPT, TRUE_DAT, "truth")


def reco_grid():
    return Grid2D(RECO_DPT, RECO_DAT, "reco")
