"""Rectangular 2-D binnings in (delta-p_T, delta-alpha_T), one for truth and a finer one for reco.

Unfolding needs MORE reco bins than truth bins -- the reco distribution is the constraint and the truth
distribution is the unknown, so an over-determined system is what makes the inverse stable.  Both live
here so the two can never drift apart, and so the flattening convention (row-major, delta-p_T slowest) is
stated exactly once: a truth index that means something different in the response builder than in the
figure is the kind of bug that produces a plausible, wrong unfolded spectrum.

The truth grid is 9 cells by default (see TRUTH_EDGES) and the reco grid is 60.  Six reco bins per truth bin leaves the system
comfortably over-determined once the 28 physics knobs are floating too: 60 reco bins against 9 unpriored
templates, rather than 30, which data alone could not close.  (The other three blocks -- flux,
cross-section knobs and detector -- each carry a prior, so each contributes a constraint alongside its
parameter and is to first order self-financing; only the templates spend the data.)

The factorisation was measured, not guessed.  Of the ways to split 60 cells, 10 x 6 is the only one that
keeps at least 10 SIGNAL events in every reco bin (min 11.5; 20 x 3 falls to 2.0 and 12 x 5 to 8.1) while
also giving the best-conditioned response.  6 x 10 has more occupancy margin but conditions worse and
leaves only 6 delta-p_T bins against the truth delta-p_T bins -- too weak a ratio in the variable carrying
most of the structure.  delta-p_T nests exactly (each truth bin splits into two reco bins); delta-alpha_T
does not (2 -> 6 is a clean 3x, but its edges are quantiles rather than subdivisions), and it does not
need to, because the projections quoted in the paper are taken in TRUTH space.
"""
from __future__ import annotations

import os

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
RECO_DPT = [0.0, 80.0, 120.0, 150.0, 185.0, 220.0, 260.0, 310.0, 380.0, 520.0, np.inf]
# delta-alpha_T [rad], likewise equal-occupancy.  pi is a hard kinematic bound, so these axes are closed.
RECO_DAT = [0.0, 0.68, 1.34, 1.91, 2.38, 2.78, PI]

# TWO TRUTH GRIDS, both equal-occupancy by the same rule, selected by TRUTH_GRID below.  Kept side by
# side rather than one overwriting the other: they answer different questions and the comparison between
# them IS a result -- see the resolution note above.
#
#   "3x3"  9 cells.  Every delta-p_T bin is wider than the 106-152 MeV resolution.  cond(A) = 41,
#          c_err 0.23.  This is the grid the detector supports.
#   "5x6"  30 cells.  Occupancy is ample (min 134 true-signal events per cell), so this is NOT a
#          statistics limit, but the delta-p_T widths are 42.8-95.6 MeV -- every one of them NARROWER
#          than the resolution.  These edges land within a few MeV of the old 5x2 grid, which returned
#          c_err 0.98, a x27 inflation over the perfect-detector limit, from conditioning alone.
TRUTH_EDGES = {
    "3x3": ([0.0, 150.0, 350.0, np.inf], [0.0, 1.2, 2.2, PI]),
    "5x6": ([0.0, 84.4, 127.2, 175.7, 271.3, np.inf],
            [0.0, 0.627, 1.218, 1.745, 2.216, 2.669, PI]),
}
TRUTH_GRID = os.environ.get("ADONIS_TRUTH_GRID", "3x3")
TRUE_DPT, TRUE_DAT = TRUTH_EDGES[TRUTH_GRID]


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


# WHICH OBSERVABLE PAIR THE SECTION UNFOLDS.  "stv" is the transverse-kinematic pair the section was
# built on; "lep" is muon momentum and angle on the published T2K binning.  The switch exists because
# the two behave completely differently under the same detector: delta-p_T is a VECTOR SUM of muon and
# proton momenta, so it inherits both resolutions and ends up coarser than either, while p_mu and
# cos theta_mu are measured directly.  Measured on this bank at the section working point, every STV
# truth bin was narrower than its resolution and 54 of 58 lepton bins are wider than theirs.
UNFOLD_OBS = os.environ.get("ADONIS_UNFOLD_OBS", "stv")


def truth_grid():
    if UNFOLD_OBS == "lep":
        return t2k_truth_grid()
    return Grid2D(TRUE_DPT, TRUE_DAT, "truth")


def reco_grid():
    if UNFOLD_OBS == "lep":
        return t2k_reco_grid(int(os.environ.get("ADONIS_RECO_SPLIT", "2")))
    return Grid2D(RECO_DPT, RECO_DAT, "reco")


# ======================================================================================================
# T2K CC0pi DOUBLE-DIFFERENTIAL BINNING (arXiv:2002.09323), for the lepton-kinematics unfolding.
#
# WHY A STAIRCASE AND NOT A RECTANGLE.  p_mu and cos theta_mu are strongly correlated -- fast muons are
# forward -- so a rectangular grid built from 1-D marginal quantiles empties its corners: measured on
# this bank, a 5 x 6 rectangle put 0.0 events in the high-p_mu / backward cell.  A template with no
# events is identically zero, its c_j is unconstrained, and the fit is singular.  The published analysis
# solves this by binning p_mu CONDITIONALLY on the cos theta_mu slice, and these are its edges.
#
# p_mu is in GeV/c (the unit of the measurement); cos theta_mu is dimensionless.  58 truth bins.
#
# RESOLUTION, against the section's detector (sigma_p = 10%, sigma_theta = 5 deg): 54 of the 58 bins have
# width > resolution.  The binding constraint is now the ANGLE, not the momentum -- the cos slices
# spanning 0.80-0.94 come out at ratio 1.01-1.18 -- which is the opposite of the STV grid, where every
# delta-p_T bin was below 1.0.  One p_mu bin (3.0-3.25 GeV inside cos 0.94-0.98) is genuinely
# unresolved at ratio 0.80; it sits in the far tail and holds almost nothing.
# EVERY SLICE'S TOP BIN IS AN OVERFLOW TO 30 GeV/c, not a closed edge.  Checked against the published
# table for cos theta_mu in [0.94, 0.98], whose last bin is 3 -> 30 rather than 3 -> 3.25: its value,
# 4.698e-41, sits ~100x below its neighbours, which is what dividing by a width of 27 instead of ~1
# produces.  Closing those axes at the second-to-last edge pushed every high-momentum signal event
# outside the truth grid, where build() correctly but silently reclassifies it as BACKGROUND -- an
# event with no template to scale it.  The measurement loses no rate; the fit did.
T2K_CC0PI_SLICES = [
    ((-1.00, 0.20), [0.0, 30.0]),                                               # backward catch-all
    (( 0.20, 0.60), [0.0, 0.3, 0.4, 0.5, 0.6, 30.0]),
    (( 0.60, 0.70), [0.0, 0.3, 0.4, 0.5, 0.6, 0.8, 30.0]),
    (( 0.70, 0.80), [0.0, 0.3, 0.4, 0.5, 0.6, 0.8, 30.0]),
    (( 0.80, 0.85), [0.0, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 30.0]),
    (( 0.85, 0.90), [0.0, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 30.0]),
    (( 0.90, 0.94), [0.0, 0.4, 0.5, 0.6, 0.8, 1.25, 2.0, 30.0]),
    (( 0.94, 0.98), [0.0, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 30.0]),
    (( 0.98, 1.00), [0.0, 0.5, 0.7, 0.9, 1.25, 2.0, 3.0, 5.0, 30.0]),
]


def _subdivide(slices, k=2, open_top=True):
    """Split every p_mu bin into k, for the RECO grid.

    Unfolding needs MORE reco bins than truth bins, and the templates carry no prior, so 58 truth
    against 60 reco (the old rectangular reco grid) leaves almost no margin.  k=2 gives 116.

    The top p_mu edge becomes infinite: smearing pushes the momentum tail past any closed edge, and an
    event outside the grid is not binned at all, so a closed axis silently drops reco events from the
    fit.  Truth keeps the published finite edges -- true signal outside the truth grid has no template
    to scale it and is background by construction.
    """
    out = []
    for (c0, c1), e in slices:
        fine = []
        for a, b in zip(e[:-1], e[1:]):
            fine.extend(a + (b - a) * np.arange(k) / k)
        fine.append(e[-1])
        if open_top:
            fine[-1] = np.inf
        out.append(((c0, c1), fine))
    return out


class StaircaseGrid:
    """A cos-theta-sliced grid whose p_mu edges differ per slice.  Flat index runs slice-major.

    Same contract as Grid2D where it matters -- .n, .index(), out-of-range is -1 and never clipped --
    but projections cannot be a reshape, because the rows have different lengths and different edges.
    Use projector() for that.
    """

    def __init__(self, slices, name=""):
        self.slices = [((float(c0), float(c1)), np.asarray(e, float)) for (c0, c1), e in slices]
        self.name = name
        self.cos_edges = np.array([s[0][0] for s in self.slices] + [self.slices[-1][0][1]])
        self.counts = [len(e) - 1 for _, e in self.slices]
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)])
        self.n = int(self.offsets[-1])
        self.nslice = len(self.slices)

    def index(self, pmu, cos):
        """Flat bin per event; -1 outside the grid (either axis)."""
        p = np.asarray(pmu, float); c = np.asarray(cos, float)
        out = np.full(p.shape, -1, int)
        si = np.digitize(c, self.cos_edges) - 1
        for k, ((_c0, _c1), e) in enumerate(self.slices):
            m = si == k
            if not np.any(m):
                continue
            j = np.digitize(p[m], e) - 1
            ok = (j >= 0) & (j < len(e) - 1)
            idx = np.full(int(m.sum()), -1, int)
            idx[ok] = self.offsets[k] + j[ok]
            out[m] = idx
        return out

    def cell_bounds(self):
        """(p_lo, p_hi, c_lo, c_hi) per flat bin."""
        pl, ph, cl, ch = [], [], [], []
        for (c0, c1), e in self.slices:
            pl.extend(e[:-1]); ph.extend(e[1:]); cl.extend([c0] * (len(e) - 1)); ch.extend([c1] * (len(e) - 1))
        return np.array(pl), np.array(ph), np.array(cl), np.array(ch)

    def projector(self, axis, target_edges, weights):
        """Linear operator P mapping the flat truth vector onto `target_edges` along `axis`.

        Row i, column j is weights[j] times the fraction of cell j's extent along `axis` that falls in
        target bin i -- a flat-within-cell assumption, which is the only assumption available without
        re-running the generator on the finer binning, and is stated because it is not free.

        Being a LINEAR operator is the point: the projected value is P c and its covariance is exactly
        P C P^T, so the correlations between cells folded into one projected bin are carried properly
        instead of being approximated by adding errors in quadrature.
        """
        pl, ph, cl, ch = self.cell_bounds()
        lo, hi = (pl, ph) if axis == "pmu" else (cl, ch)
        te = np.asarray(target_edges, float)
        P = np.zeros((len(te) - 1, self.n))
        for i in range(len(te) - 1):
            a, b = te[i], te[i + 1]
            ov = np.clip(np.minimum(ph if axis == "pmu" else ch, b)
                         - np.maximum(pl if axis == "pmu" else cl, a), 0.0, None)
            wdt = np.where(np.isfinite(hi - lo), hi - lo, np.nan)
            frac = np.divide(ov, wdt, out=np.zeros_like(ov), where=np.isfinite(wdt) & (wdt > 0))
            P[i] = frac * np.asarray(weights, float)
        return P


def t2k_truth_grid():
    return StaircaseGrid(T2K_CC0PI_SLICES, "t2k-truth")


def t2k_reco_grid(k=2):
    return StaircaseGrid(_subdivide(T2K_CC0PI_SLICES, k=k), f"t2k-reco-x{k}")
