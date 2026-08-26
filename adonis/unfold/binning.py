"""Rectangular 2-D binnings in (delta-p_T, delta-alpha_T): one truth grid, one finer reco grid.

Unfolding needs more reco bins than truth bins: the reco distribution is the constraint, the truth
distribution is the unknown, so the system must be over-determined for the inverse to be stable. Both
grids live here so they cannot drift apart, and the flattening convention (row-major, delta-p_T
slowest) is defined exactly once.

The truth grid is 9 cells by default (TRUTH_EDGES), the reco grid 60: 60 reco bins against 9 unpriored
templates stays over-determined even once the 28 physics knobs float (the other three blocks -- flux,
cross-section knobs, detector -- each carry a prior and so are self-financing).

Of the ways to split 60 reco cells, 10 x 6 keeps at least 10 signal events in every bin (min 11.5;
20 x 3 falls to 2.0, 12 x 5 to 8.1) while best-conditioning the response; 6 x 10 conditions worse and
leaves too few delta-p_T bins against the truth axis. delta-p_T nests exactly into reco; delta-alpha_T
does not and does not need to, since the projections quoted downstream are taken in truth space.
"""
from __future__ import annotations

import numpy as np

PI = float(np.pi)

RECO_DPT = [0.0, 80.0, 120.0, 150.0, 185.0, 220.0, 260.0, 310.0, 380.0, 520.0, np.inf]
RECO_DAT = [0.0, 0.68, 1.34, 1.91, 2.38, 2.78, PI]

TRUTH_EDGES = {
    "3x3": ([0.0, 150.0, 350.0, np.inf], [0.0, 1.2, 2.2, PI]),
    "5x6": ([0.0, 84.4, 127.2, 175.7, 271.3, np.inf],
            [0.0, 0.627, 1.218, 1.745, 2.216, 2.669, PI]),
}
DEFAULT_TRUTH = "3x3"
TRUE_DPT, TRUE_DAT = TRUTH_EDGES[DEFAULT_TRUTH]


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




def truth_grid(obs="stv", variant=DEFAULT_TRUTH):
    """Truth binning: "stv" -> rectangular Grid2D from TRUTH_EDGES[variant]; "lep" -> a published staircase."""
    if obs == "lep":
        from adonis.measurements.t2k_cc0pi_binning import t2k_truth_grid
        return t2k_truth_grid()
    return Grid2D(*TRUTH_EDGES[variant], name="truth")


def reco_grid(obs="stv", split=2):
    """Reco binning.  `split` subdivides each staircase p_mu bin (lep only); unused for "stv"."""
    if obs == "lep":
        from adonis.measurements.t2k_cc0pi_binning import t2k_reco_grid
        return t2k_reco_grid(int(split))
    return Grid2D(RECO_DPT, RECO_DAT, "reco")


def _subdivide(slices, k=2, open_top=True):
    """Split every p_mu bin into k, for the reco grid: a 58 truth / 58 reco split would leave almost no
    margin since the templates carry no prior, so k=2 gives 116 reco bins.

    The top p_mu edge becomes infinite: smearing pushes the momentum tail past any closed edge, and an
    event outside the grid is not binned at all. Truth keeps the published finite edges -- true signal
    outside the truth grid has no template to scale it and is background by construction.
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


