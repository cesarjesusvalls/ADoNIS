"""Faithful port of ACHILLES's cubic-spline interpolation (amp_dcc_sl.f::spline+seval),
the Forsythe-Malcolm-Moler cubic spline, called on 4-point stencils in interpolate_amp.

ACHILLES interpolates the DCC amplitude table in (W, Q^2) with this spline (NOT bilinear);
matching it removes the residual ~2-3% deficit at the lowest Q^2, where the tabulated
amplitude is irregular and linear vs cubic genuinely diverge.

The spline value seval(xout) is LINEAR in the tabulated ordinates y (the knots x are
fixed), so this is a linear operator -> fully differentiable, and we apply it to the
(real) hadron-tensor grid the same way ACHILLES applies it to the amplitude. n=4 is
unrolled. End conditions: third derivative at the ends from divided differences (FMM).
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def spline4_eval(x, y, xout):
    """FMM cubic spline through 4 knots (x[...,4], y[...,4]) evaluated at xout[...].
    Batched over the leading dims. Faithful unrolled port of spline()+seval for n=4."""
    x0, x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    y0, y1, y2, y3 = y[..., 0], y[..., 1], y[..., 2], y[..., 3]

    h0, h1, h2 = x1 - x0, x2 - x1, x3 - x2

    c1 = (y1 - y0) / h0
    c2 = (y2 - y1) / h1
    c3 = (y3 - y2) / h2
    cc1 = c2 - c1
    cc2 = c3 - c2
    b1 = 2.0 * (h0 + h1)
    b2 = 2.0 * (h1 + h2)

    b0 = -h0
    b3 = -h2
    c0 = cc2 / (x3 - x1) - cc1 / (x2 - x0)
    c3e = cc2 / (x3 - x1) - cc1 / (x2 - x0)
    c0 = c0 * h0 ** 2 / (x3 - x0)
    c3e = -c3e * h2 ** 2 / (x3 - x0)

    t = h0 / b0
    b1 = b1 - t * h0
    cc1 = cc1 - t * c0
    t = h1 / b1
    b2 = b2 - t * h1
    cc2 = cc2 - t * cc1
    t = h2 / b2
    b3 = b3 - t * h2
    c3e = c3e - t * cc2

    s3 = c3e / b3
    s2 = (cc2 - h2 * s3) / b2
    s1 = (cc1 - h1 * s2) / b1
    s0 = (c0 - h0 * s1) / b0

    B0 = (y1 - y0) / h0 - h0 * (s1 + 2.0 * s0)
    B1 = (y2 - y1) / h1 - h1 * (s2 + 2.0 * s1)
    B2 = (y3 - y2) / h2 - h2 * (s3 + 2.0 * s2)
    D0 = (s1 - s0) / h0
    D1 = (s2 - s1) / h1
    D2 = (s3 - s2) / h2
    C0, C1, C2 = 3.0 * s0, 3.0 * s1, 3.0 * s2

    in0 = xout <= x1
    in2 = xout >= x2
    xi = jnp.where(in0, x0, jnp.where(in2, x2, x1))
    yi = jnp.where(in0, y0, jnp.where(in2, y2, y1))
    Bi = jnp.where(in0, B0, jnp.where(in2, B2, B1))
    Ci = jnp.where(in0, C0, jnp.where(in2, C2, C1))
    Di = jnp.where(in0, D0, jnp.where(in2, D2, D1))
    dx = xout - xi
    return yi + Bi * dx + Ci * dx ** 2 + Di * dx ** 3


def stencil_start(n, i1):
    """0-based start of the 4-point stencil for the lower grid index i1 (jnp int),
    matching interpolate_amp's iw_start logic.  ACHILLES's iw is the 1-based UPPER bracket
    (first wcms(iw) >= W); with i1 the 0-based LOWER bracket (searchsorted-1), that is
    iw = i1 + 2 (NOT i1+1 -- the off-by-one put W in the stencil's edge interval and
    under-shot the amplitude at the low-W/low-Q^2 features that dominate sigma_RES)."""
    iw = i1 + 2
    start = jnp.where(iw <= 2, 1, jnp.where(iw >= n - 1, n - 3, iw - 2)) - 1
    return jnp.clip(start, 0, n - 4)


def _spline4_eval_np(x, y, xout):
    """NumPy twin of spline4_eval (identical FMM cubic math) for the forward generator
    (no autodiff needed) -- avoids per-op JAX dispatch.  x,y (...,4), xout (...)."""
    x0, x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    y0, y1, y2, y3 = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
    h0, h1, h2 = x1 - x0, x2 - x1, x3 - x2
    c1 = (y1 - y0) / h0; c2 = (y2 - y1) / h1; c3 = (y3 - y2) / h2
    cc1 = c2 - c1; cc2 = c3 - c2
    b1 = 2.0 * (h0 + h1); b2 = 2.0 * (h1 + h2)
    b0 = -h0; b3 = -h2
    c0 = cc2 / (x3 - x1) - cc1 / (x2 - x0); c3e = c0
    c0 = c0 * h0 ** 2 / (x3 - x0); c3e = -c3e * h2 ** 2 / (x3 - x0)
    t = h0 / b0; b1 = b1 - t * h0; cc1 = cc1 - t * c0
    t = h1 / b1; b2 = b2 - t * h1; cc2 = cc2 - t * cc1
    t = h2 / b2; b3 = b3 - t * h2; c3e = c3e - t * cc2
    s3 = c3e / b3; s2 = (cc2 - h2 * s3) / b2; s1 = (cc1 - h1 * s2) / b1; s0 = (c0 - h0 * s1) / b0
    B0 = (y1 - y0) / h0 - h0 * (s1 + 2.0 * s0)
    B1 = (y2 - y1) / h1 - h1 * (s2 + 2.0 * s1)
    B2 = (y3 - y2) / h2 - h2 * (s3 + 2.0 * s2)
    D0 = (s1 - s0) / h0; D1 = (s2 - s1) / h1; D2 = (s3 - s2) / h2
    C0, C1, C2 = 3.0 * s0, 3.0 * s1, 3.0 * s2
    in0 = xout <= x1; in2 = xout >= x2
    xi = np.where(in0, x0, np.where(in2, x2, x1))
    yi = np.where(in0, y0, np.where(in2, y2, y1))
    Bi = np.where(in0, B0, np.where(in2, B2, B1))
    Ci = np.where(in0, C0, np.where(in2, C2, C1))
    Di = np.where(in0, D0, np.where(in2, D2, D1))
    dx = xout - xi
    return yi + Bi * dx + Ci * dx ** 2 + Di * dx ** 3


def _stencil_start_np(n, i1):
    iw = i1 + 2
    start = np.where(iw <= 2, 1, np.where(iw >= n - 1, n - 3, iw - 2)) - 1
    return np.clip(start, 0, n - 4)


def interp2d_spline_np(grid, Wg, Q2g, W, Q2):
    """NumPy twin of interp2d_spline (bit-identical math) for the forward generator."""
    Wg = np.asarray(Wg); Q2g = np.asarray(Q2g); grid = np.asarray(grid)
    W = np.asarray(W); Q2 = np.asarray(Q2)
    nw, nq = Wg.shape[0], Q2g.shape[0]
    iw1 = np.clip(np.searchsorted(Wg, W) - 1, 0, nw - 2)
    iq1 = np.clip(np.searchsorted(Q2g, Q2) - 1, 0, nq - 2)
    iW = _stencil_start_np(nw, iw1)[:, None] + np.arange(4)
    iQ = _stencil_start_np(nq, iq1)[:, None] + np.arange(4)
    xW = Wg[iW]; xQ = Q2g[iQ]
    block = grid[iQ[:, :, None], iW[:, None, :]]
    yW = np.moveaxis(block, 2, -1)
    rowW = _spline4_eval_np(np.broadcast_to(xW[:, None, None, :], yW.shape), yW, W[:, None, None])
    yQ = np.moveaxis(rowW, 1, -1)
    return _spline4_eval_np(np.broadcast_to(xQ[:, None, :], yQ.shape), yQ, Q2[:, None])


def interp2d_spline(grid, Wg, Q2g, W, Q2):
    """Bicubic (separable FMM-spline) interpolation of grid[nq, nw, C] at (W, Q2),
    batched over N events; fully JAX (works under grad with detached/traced W,Q2).
    Mirrors interpolate_amp: spline in W for each of the 4 Q2-stencil rows, then Q2."""
    Wg = jnp.asarray(Wg); Q2g = jnp.asarray(Q2g); grid = jnp.asarray(grid)
    nw, nq = Wg.shape[0], Q2g.shape[0]
    iw1 = jnp.clip(jnp.searchsorted(Wg, W) - 1, 0, nw - 2)
    iq1 = jnp.clip(jnp.searchsorted(Q2g, Q2) - 1, 0, nq - 2)
    iW = stencil_start(nw, iw1)[:, None] + jnp.arange(4)
    iQ = stencil_start(nq, iq1)[:, None] + jnp.arange(4)
    xW = Wg[iW]; xQ = Q2g[iQ]
    block = grid[iQ[:, :, None], iW[:, None, :]]

    yW = jnp.moveaxis(block, 2, -1)
    rowW = spline4_eval(jnp.broadcast_to(xW[:, None, None, :], yW.shape), yW,
                        W[:, None, None])
    yQ = jnp.moveaxis(rowW, 1, -1)
    out = spline4_eval(jnp.broadcast_to(xQ[:, None, :], yQ.shape), yQ, Q2[:, None])
    return out
