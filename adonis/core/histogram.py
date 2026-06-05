"""Histogram utilities for differentiable predictions and oracle comparison.

Framework services used across validation/fitting (previously hand-rolled per script):
  * weighted histograms with stat errors (sum w, sum w^2);
  * EXACT integer rebinning (reshape-sum) -- avoids the center-assignment aliasing that
    makes a model/oracle ratio zig-zag when the fine:coarse bin ratio is non-integer;
  * chunked accumulation over a generator (bounded memory);
  * normalisation to dsigma/dx with propagated error, and ratio + chi2/ndf;
  * a tiny npz cache so plot-style edits don't re-pay generation.
"""
from __future__ import annotations

import os

import numpy as np


def hist_w(v, w, edges):
    """Weighted histogram: returns (sum w, sum w^2) per bin."""
    v = np.asarray(v); w = np.asarray(w); m = np.isfinite(v)
    return (np.histogram(v[m], bins=edges, weights=w[m])[0],
            np.histogram(v[m], bins=edges, weights=w[m] ** 2)[0])


def regroup(sw, sw2, factor):
    """Exact integer coarsening of fine histograms by `factor` (reshape-sum). Requires
    len % factor == 0."""
    assert len(sw) % factor == 0, f"{len(sw)} not divisible by {factor}"
    return sw.reshape(-1, factor).sum(1), sw2.reshape(-1, factor).sum(1)


def coarse_edges(fine_edges, factor):
    """Edges of the exact `factor`-coarsening of `fine_edges`."""
    assert (len(fine_edges) - 1) % factor == 0
    return fine_edges[::factor]


def chunked_accumulate(sample_weight_fn, names_edges, n_total, chunk):
    """Accumulate (sum w, sum w^2) per observable over chunked generation (bounded memory).
    `sample_weight_fn(key_index, m) -> (values_dict, w)` returns per-observable arrays + the
    weight for a chunk of m events.  Returns {name: (sw, sw2)} and the total event count."""
    nb = {n: len(e) - 1 for n, e in names_edges.items()}
    sw = {n: np.zeros(nb[n]) for n in names_edges}
    sw2 = {n: np.zeros(nb[n]) for n in names_edges}
    done = 0; c = 0
    while done < n_total:
        m = min(chunk, n_total - done)
        vals, w = sample_weight_fn(c, m)
        for n, e in names_edges.items():
            a, b = hist_w(vals[n], w, e)
            sw[n] += a; sw2[n] += b
        done += m; c += 1
    return {n: (sw[n], sw2[n]) for n in names_edges}, done


def normalize(sw, sw2, edges):
    """-> (dsigma/dx, stat error) normalised to unit area."""
    dx = np.diff(edges); S = sw.sum()
    d = sw / dx / S
    rel = np.divide(np.sqrt(sw2), sw, out=np.zeros_like(sw), where=sw > 0)
    return d, d * rel


def ratio_chi2(m_sw, m_sw2, o_sw, o_sw2):
    """Unit-area model/oracle ratio with propagated error + chi2/ndf vs unity."""
    Sm, So = m_sw.sum(), o_sw.sum()
    rm = np.divide(np.sqrt(m_sw2), m_sw, out=np.zeros_like(m_sw), where=m_sw > 0)
    ro = np.divide(np.sqrt(o_sw2), o_sw, out=np.zeros_like(o_sw), where=o_sw > 0)
    ok = (m_sw > 0) & (o_sw > 0)
    R = np.divide(m_sw / Sm, o_sw / So, out=np.full_like(m_sw, np.nan), where=ok)
    Re = R * np.sqrt(rm ** 2 + ro ** 2)
    good = np.isfinite(R) & (Re > 0)
    chi2 = float(np.sum(((R[good] - 1) / Re[good]) ** 2)); ndf = int(good.sum())
    return R, Re, chi2, ndf


def save_cache(path, key, **arrays):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, key=key, **arrays)


def load_cache(path, key):
    """Return the cached npz dict if present and the key matches, else None."""
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    if str(d["key"]) != str(key):
        return None
    return d
