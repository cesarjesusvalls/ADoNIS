"""Separable VEGAS importance grid (frozen-after-warm-up), the learned cousin of
`spectral.SpectralImportanceSampler`.

The grid remaps a uniform hypercube point y in [0,1]^d to x in [0,1]^d through a per-axis
piecewise-linear inverse-CDF (one table of bin edges per axis), concentrating samples where the
integrand is large.  It is built ONLY from the integrand during a short warm-up (accumulate -> refine,
a few iterations), then FROZEN -- so it is a fixed proposal: per-event weights stay exact (the grid
Jacobian is folded into the weight) and ADoNIS differentiability is untouched (the sampler is detached;
gradients flow through the JAX reweights, not the proposal).

Pure numpy (generation is detached).  Default-OFF in the generators: no grid -> bit-identical sampling.

Algorithm (Lepage VEGAS, simplified-but-faithful):
  map(y):      per axis, y in bin i=floor(y*N), x = edge[i] + frac*(edge[i+1]-edge[i]);
               dx/dy = N*(edge[i+1]-edge[i]); jac = prod over axes.
  accumulate:  add f^2 into the x-bin it fell in, per axis (f = the full event weight = integrand*jac).
  refine:      smooth + damp (^alpha) the per-bin f^2, then redistribute edges so every new bin carries
               equal cumulative importance (exact piecewise-linear inverse of the cumulative).
"""
from __future__ import annotations
import numpy as np


class VegasGrid:
    def __init__(self, ndim, nbins=50):
        self.ndim = int(ndim); self.nbins = int(nbins)
        # per-axis edges, uniform to start: shape (ndim, nbins+1), edge[:,0]=0, edge[:,-1]=1
        self.edges = np.tile(np.linspace(0.0, 1.0, self.nbins + 1), (self.ndim, 1))
        self.acc = np.zeros((self.ndim, self.nbins))   # accumulated f^2 per bin
        self.frozen = False

    # ---- sampling ----
    def map(self, y):
        """y:(n,ndim) ~U[0,1] -> x:(n,ndim) in [0,1], jac:(n,) = prod_axis dx/dy (>0)."""
        y = np.asarray(y, float)
        n, d = y.shape
        assert d == self.ndim, f"grid ndim {self.ndim} != y dim {d}"
        N = self.nbins
        x = np.empty_like(y); jac = np.ones(n)
        s = np.clip(y * N, 0.0, N - 1e-12)
        i = np.floor(s).astype(np.int64); frac = s - i
        for ax in range(d):
            e = self.edges[ax]
            lo = e[i[:, ax]]; hi = e[i[:, ax] + 1]
            x[:, ax] = lo + frac[:, ax] * (hi - lo)
            jac *= N * (hi - lo)
        return x, jac

    def density(self, x):
        """Proposal density p_vegas(x) at arbitrary x in [0,1]^ndim (relative to uniform): per axis the
        local 1/(N*bin_width) of the bin containing x; product over axes.  This is 1/jac at x, and is
        what a defensive mixture needs to weight events drawn from EITHER component by the mixture
        density q(x) = alpha*p_vegas(x) + (1-alpha)."""
        x = np.asarray(x, float); n = x.shape[0]; N = self.nbins
        dens = np.ones(n)
        for ax in range(self.ndim):
            e = self.edges[ax]
            i = np.clip(np.searchsorted(e, x[:, ax], side="right") - 1, 0, N - 1)
            dens *= 1.0 / (N * np.clip(e[i + 1] - e[i], 1e-300, None))
        return dens

    # ---- adaptation ----
    def accumulate(self, x, fval):
        """Add f^2 (f = full weight including the current grid jac) into the x-bin per axis."""
        x = np.asarray(x, float); f2 = np.asarray(fval, float) ** 2
        good = np.isfinite(f2) & (f2 > 0)
        if not good.any():
            return
        x = x[good]; f2 = f2[good]
        for ax in range(self.ndim):
            i = np.clip(np.searchsorted(self.edges[ax], x[:, ax], side="right") - 1, 0, self.nbins - 1)
            np.add.at(self.acc[ax], i, f2)

    def refine(self, alpha=1.5):
        """Rebin each axis exactly as ACHILLES AdaptiveMap::Adapt: 3-point smooth the accumulated f^2,
        normalize to r_i = sm_i/sum(sm), apply the Lepage damping importance fac_i = ((r_i-1)/ln r_i)^alpha
        (-> 1 as r->1, -> 0 as r->0), then place new edges so every bin carries equal cumulative `fac`.
        alpha=1.5 matches ACHILLES VegasParams::alpha_default.  Resets the accumulator."""
        if self.frozen:
            return
        for ax in range(self.ndim):
            d = self.acc[ax].astype(float).copy()
            if d.sum() <= 0:
                continue                                    # no info this axis -> leave edges
            # 3-point smoothing (ACHILLES Adapt), edge-aware
            sm = d.copy()
            if self.nbins > 1:
                sm[1:-1] = (d[:-2] + d[1:-1] + d[2:]) / 3.0
                sm[0] = (d[0] + d[1]) / 2.0
                sm[-1] = (d[-1] + d[-2]) / 2.0
            norm = sm.sum()
            if norm <= 0:
                continue
            r = sm / norm                                   # per-bin importance fraction
            with np.errstate(divide="ignore", invalid="ignore"):
                fac = (r - 1.0) / np.log(r)                 # Lepage damping kernel
            fac = np.where(np.abs(r - 1.0) < 1e-12, 1.0, fac)   # (r-1)/ln r -> 1 at r=1
            fac = np.where(r > 0, np.clip(fac, 1e-300, None), 1e-300)
            imp = fac ** alpha
            cum = np.concatenate([[0.0], np.cumsum(imp)])    # cumulative importance at OLD edges
            tot = cum[-1]
            if tot <= 0:
                continue
            targets = np.linspace(0.0, tot, self.nbins + 1)
            e_new = np.interp(targets, cum, self.edges[ax])  # invert: x where cum = target
            e_new[0] = 0.0; e_new[-1] = 1.0
            e_new = np.maximum.accumulate(e_new)             # guard monotonicity
            self.edges[ax] = e_new
        self.acc[:] = 0.0

    def freeze(self):
        self.frozen = True
        return self

    # ---- persistence ----
    def save(self, path):
        np.savez(path, edges=self.edges, nbins=self.nbins, ndim=self.ndim)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        g = cls(int(d["ndim"]), int(d["nbins"]))
        g.edges = d["edges"]; g.frozen = True
        return g
