"""The binning primitive: per-event weights -> per-bin observables, for every sample type.

`_device()` memoises the on-device index arrays on the instance. adonis/fit/kernels.py warms
these caches eagerly, before tracing, and relies on getting back the same cached objects
afterward; rebuilding them inside a jit trace raises UnexpectedTracerError.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

def _bin(sig_mask, values, edges):
    """(sel_idx, binidx, nbin): keep signal events with values in [edges[0], edges[-1]); assign bin."""
    nbin = len(edges) - 1
    idx = np.searchsorted(edges, values) - 1
    inb = sig_mask & (values >= edges[0]) & (values < edges[-1])
    sel = np.where(inb)[0]
    return sel.astype(np.int64), idx[sel].astype(np.int64), nbin

class BinSpec:
    """The fixed sparse map w -> per-bin observable, shared by every sample type.

    Two index orderings are kept:
      HOST   events in BANK order -- the gather w[sel] favors sequential reads over bin-sorted ones.
      DEVICE events sorted by bin, so segment_sum uses a segmented reduction instead of contended
             atomics. Built lazily; nothing pays for it unless S4_JAX_BIN=1.
    """

    __slots__ = ("sel", "coef", "binidx", "nbin", "scale", "offset", "_dev", "_chunks")

    def __init__(self, binidx, nbin, scale, sel=None, coef=None, offset=None):
        n = len(np.asarray(binidx))
        self.binidx = np.asarray(binidx, np.int64)
        self.sel = np.arange(n, dtype=np.int64) if sel is None else np.asarray(sel, np.int64)
        self.coef = None if coef is None else np.asarray(coef, float)
        self.nbin = int(nbin)
        self.scale = np.asarray(scale, float)
        self.offset = None if offset is None else np.asarray(offset, float)
        self._dev = None
        self._chunks = None

    def apply(self, w):
        """Host (numpy) reference path -- bank-ordered gather."""
        v = np.asarray(w)[self.sel]
        if self.coef is not None:
            v = self.coef * v
        out = self.scale * np.bincount(self.binidx, weights=v, minlength=self.nbin)
        return out if self.offset is None else out + self.offset

    def _device(self):
        if self._dev is None:
            import jax.numpy as jnp
            o = np.argsort(self.binidx, kind="stable")
            self._dev = dict(binidx=jnp.asarray(self.binidx[o]), sel=jnp.asarray(self.sel[o]),
                             coef=None if self.coef is None else jnp.asarray(self.coef[o]),
                             scale=jnp.asarray(self.scale),
                             offset=None if self.offset is None else jnp.asarray(self.offset))
        return self._dev

    def chunks(self, n_events, C):
        """Per-chunk gather tables for a scan over event blocks of size C.

        Chunks the event (bank) axis; `sel` picks an arbitrary subset of it, so entries in a chunk
        are ragged and are padded to a common length with a trash bin at index nbin, dropped after
        the segment_sum. Built once and cached.

        Returns (starts, loc, binidx, coef, n_chunk, L, C): `loc` is the event index within its
        window, `binidx` is nbin on padding.
        """
        key = (int(n_events), int(C))
        if getattr(self, "_chunks", None) is None:
            self._chunks = {}
        got = self._chunks.get(key)
        if got is not None:
            return got
        C = min(int(C), int(n_events))
        nch = int(np.ceil(n_events / C))
        starts = np.minimum(np.arange(nch) * C, max(int(n_events) - C, 0))
        wof = np.minimum(self.sel // C, nch - 1)
        order = np.argsort(wof, kind="stable")
        cnt = np.bincount(wof, minlength=nch)
        L = int(cnt.max()) if cnt.size else 0
        loc = np.zeros((nch, L), np.int64)
        bix = np.full((nch, L), self.nbin, np.int64)
        cof = np.zeros((nch, L), float)
        pos = 0
        for c in range(nch):
            k = int(cnt[c])
            if k:
                e = order[pos:pos + k]
                loc[c, :k] = self.sel[e] - starts[c]
                bix[c, :k] = self.binidx[e]
                cof[c, :k] = 1.0 if self.coef is None else self.coef[e]
            pos += k
        assert loc.min() >= 0 and (L == 0 or loc.max() < C), "window assignment out of range"
        self._chunks[key] = (starts, loc, bix, cof, nch, L, C)
        return self._chunks[key]

    def apply_dev(self, w):
        """Device path: identical arithmetic, stays on the GPU, reverse-mode differentiable."""
        import jax
        d = self._device()
        v = w[d["sel"]]
        if d["coef"] is not None:
            v = d["coef"] * v
        out = d["scale"] * jax.ops.segment_sum(v, d["binidx"], num_segments=self.nbin,
                                               indices_are_sorted=True)
        return out if d["offset"] is None else out + d["offset"]

def spec_of(d):
    """BinSpec for a BankSample-style dataset dict (cached on the dict)."""
    sp = d.get("_spec")
    if sp is None:
        sp = BinSpec(d["binidx"], d["nbin"], d["scale_bin"], sel=d["sel_idx"])
        d["_spec"] = sp
    return sp

def bin_w0(d, w):
    """Per-bin sum of a per-event quantity w for dataset d, scaled -- no free-H offset. For the
    Jacobian: the frozen free-H offset is theta-independent, so its derivative is zero."""
    return spec_of(d).apply(w)

def bin_w(d, w):
    """Per-bin dsigma/dx for dataset d from a full per-event weight vector w (+ frozen free-H)."""
    return bin_w0(d, w) + d["offset"]

def bin_w0_dev(d, w):
    return spec_of(d).apply_dev(w)

def bin_w_dev(d, w):
    import jax.numpy as jnp
    return bin_w0_dev(d, w) + jnp.asarray(d["offset"])
