"""Differentiable JAX access to the DCC electroweak amplitudes.

Wraps the parsed `dcc_EW.dat` table (see `dcc_loader`) as JAX constants and provides
differentiable interpolation in (W, Q^2), with a layer of tunable knobs:

  * `axial_strength`  -- overall factor on the axial current block (the closest
                         differentiable proxy for an axial-mass / axial-coupling tune)
  * `pw_norm[14]`     -- per-partial-wave multiplicative strength (1 + pw_norm), applied
                         to every block (normalization knobs on the DCC partial waves)

The physical amplitude per (W,Q^2,current-component,partial-wave) is the sum of the
table's three contributions (bare + dressed-N* + non-resonant), matching ACHILLES
(`idata=(1,1,1)` in amp_dcc_sl.f). The interpolation is differentiable both in the
kinematics (W,Q^2) -- for a reparameterised phase space -- and in the knobs.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from adonis.channels.dcc.loader import load_cached, PW_LABELS
from adonis.core.params import PhysicsParams, DCCKnobs    # DCCKnobs is an alias of PhysicsParams


class DCCAmplitudes:
    """JAX-backed, differentiable DCC amplitude interpolator."""

    def __init__(self, table=None):
        t = table or load_cached()
        self.W = jnp.asarray(t.W)                       # (n_w,) MeV
        self.Q2 = jnp.asarray(t.Q2)                     # (n_q2,) MeV^2
        # sum the 3 za components (bare+dressed+nonres); squeeze n_gmb=1 -> (n_q2,n_w,n_idx,n_pw)
        def full(a):
            return jnp.asarray(a.sum(axis=-1)[:, :, :, :, 0])
        self.vec = full(t.vec)
        self.isv = full(t.isv)
        self.axial = full(t.axial)
        self.n_pw = self.vec.shape[-1]
        self.labels = t.labels

    # --- bilinear interpolation on the (Q2, W) grid, batched over events ----- #
    def _interp(self, vals, W, Q2):
        """vals: (n_q2, n_w, ...) complex; W,Q2: scalars -> (...) complex."""
        nw, nq = self.W.shape[0], self.Q2.shape[0]
        iw = jnp.clip(jnp.searchsorted(self.W, W) - 1, 0, nw - 2)
        iq = jnp.clip(jnp.searchsorted(self.Q2, Q2) - 1, 0, nq - 2)
        tw = (W - self.W[iw]) / (self.W[iw + 1] - self.W[iw])
        tq = (Q2 - self.Q2[iq]) / (self.Q2[iq + 1] - self.Q2[iq])
        v00, v01 = vals[iq, iw], vals[iq, iw + 1]
        v10, v11 = vals[iq + 1, iw], vals[iq + 1, iw + 1]
        return ((1 - tq) * ((1 - tw) * v00 + tw * v01)
                + tq * ((1 - tw) * v10 + tw * v11))

    def amplitudes_spline(self, W, Q2, knobs: DCCKnobs = DCCKnobs()):
        """Batched (N,) spline amplitude interpolation matching ACHILLES (FMM cubic in
        W and Q2). Returns vec, isv, axial each (N, n_idx, n_pw) complex, knobs applied."""
        from adonis.channels.dcc.spline import interp2d_spline
        ni, npw = self.vec.shape[2], self.vec.shape[3]
        def sp(block):
            flat = block.reshape(block.shape[0], block.shape[1], ni * npw)
            return interp2d_spline(flat, self.W, self.Q2, W, Q2).reshape(-1, ni, npw)
        vec, isv, axial = sp(self.vec), sp(self.isv), sp(self.axial) * knobs.axial_strength
        if knobs.pw_norm != ():
            scale = 1.0 + jnp.asarray(knobs.pw_norm)
            vec = vec * scale; isv = isv * scale; axial = axial * scale
        return vec, isv, axial

    def amplitudes_spline_np(self, W, Q2, knobs: DCCKnobs = DCCKnobs()):
        """NumPy (no-autodiff) twin of amplitudes_spline -- bit-identical FMM-cubic interp,
        ~30x faster for the forward generator (avoids per-op JAX dispatch). Returns numpy arrays."""
        from adonis.channels.dcc.spline import interp2d_spline_np
        if not hasattr(self, "_Wnp"):
            self._Wnp = np.asarray(self.W); self._Q2np = np.asarray(self.Q2)
            self._vecnp = np.asarray(self.vec); self._isvnp = np.asarray(self.isv)
            self._axialnp = np.asarray(self.axial)
        ni, npw = self.vec.shape[2], self.vec.shape[3]
        def sp(block):
            flat = block.reshape(block.shape[0], block.shape[1], ni * npw)
            return interp2d_spline_np(flat, self._Wnp, self._Q2np, W, Q2).reshape(-1, ni, npw)
        vec, isv = sp(self._vecnp), sp(self._isvnp)
        axial = sp(self._axialnp) * knobs.axial_strength
        if knobs.pw_norm != ():
            scale = 1.0 + np.asarray(knobs.pw_norm)
            vec = vec * scale; isv = isv * scale; axial = axial * scale
        return vec, isv, axial

    def amplitudes_bilinear(self, W, Q2, knobs: DCCKnobs = DCCKnobs()):
        """Batched (N,) bilinear amplitude interpolation -- faster / lighter than the spline.
        NOT W-faithful: the dsigma/dW shape (especially the high-W tail) deviates from the ACHILLES
        spline by well over 1%. Never use unless explicitly requested for a specific purpose.
        Returns vec, isv, axial each (N, n_idx, n_pw), knobs applied."""
        gw, gq = self.W, self.Q2
        nw, nq = gw.shape[0], gq.shape[0]
        iw = jnp.clip(jnp.searchsorted(gw, W) - 1, 0, nw - 2)
        iq = jnp.clip(jnp.searchsorted(gq, Q2) - 1, 0, nq - 2)
        tw = ((W - gw[iw]) / (gw[iw + 1] - gw[iw]))[:, None, None]
        tq = ((Q2 - gq[iq]) / (gq[iq + 1] - gq[iq]))[:, None, None]

        def bil(block):                                    # block (nq,nw,ni,npw)
            v00, v01 = block[iq, iw], block[iq, iw + 1]
            v10, v11 = block[iq + 1, iw], block[iq + 1, iw + 1]
            return (1 - tq) * ((1 - tw) * v00 + tw * v01) + tq * ((1 - tw) * v10 + tw * v11)

        vec, isv, axial = bil(self.vec), bil(self.isv), bil(self.axial) * knobs.axial_strength
        if knobs.pw_norm != ():
            scale = 1.0 + jnp.asarray(knobs.pw_norm)
            vec = vec * scale; isv = isv * scale; axial = axial * scale
        return vec, isv, axial

    def amplitudes_bilinear_np(self, W, Q2, knobs: DCCKnobs = DCCKnobs()):
        """NumPy twin of amplitudes_bilinear (fastest path, same W-unfaithfulness caveat)."""
        if not hasattr(self, "_Wnp"):
            self._Wnp = np.asarray(self.W); self._Q2np = np.asarray(self.Q2)
            self._vecnp = np.asarray(self.vec); self._isvnp = np.asarray(self.isv)
            self._axialnp = np.asarray(self.axial)
        gw, gq = self._Wnp, self._Q2np
        nw, nq = gw.shape[0], gq.shape[0]
        W = np.asarray(W); Q2 = np.asarray(Q2)
        iw = np.clip(np.searchsorted(gw, W) - 1, 0, nw - 2)
        iq = np.clip(np.searchsorted(gq, Q2) - 1, 0, nq - 2)
        tw = ((W - gw[iw]) / (gw[iw + 1] - gw[iw]))[:, None, None]
        tq = ((Q2 - gq[iq]) / (gq[iq + 1] - gq[iq]))[:, None, None]
        def bil(block):
            v00, v01 = block[iq, iw], block[iq, iw + 1]
            v10, v11 = block[iq + 1, iw], block[iq + 1, iw + 1]
            return (1 - tq) * ((1 - tw) * v00 + tw * v01) + tq * ((1 - tw) * v10 + tw * v11)
        vec, isv = bil(self._vecnp), bil(self._isvnp)
        axial = bil(self._axialnp) * knobs.axial_strength
        if knobs.pw_norm != ():
            scale = 1.0 + np.asarray(knobs.pw_norm)
            vec = vec * scale; isv = isv * scale; axial = axial * scale
        return vec, isv, axial

    def amplitudes(self, W, Q2, knobs: DCCKnobs = DCCKnobs()):
        """Interpolate the (vec, isv, axial) amplitudes at (W [MeV], Q2 [MeV^2]),
        with knobs applied. Each returned array is (n_idx, n_pw) complex.

        W, Q2 may be scalars; use jax.vmap for a batch of events."""
        vec = self._interp(self.vec, W, Q2)
        isv = self._interp(self.isv, W, Q2)
        axial = self._interp(self.axial, W, Q2) * knobs.axial_strength
        if knobs.pw_norm != ():
            scale = 1.0 + jnp.asarray(knobs.pw_norm)            # (n_pw,)
            vec = vec * scale; isv = isv * scale; axial = axial * scale
        return vec, isv, axial


def numpy_bilinear(table, vals_block, W, Q2):
    """Reference NumPy bilinear interpolation for validation (vals_block: numpy)."""
    gw, gq = table.W, table.Q2
    iw = np.clip(np.searchsorted(gw, W) - 1, 0, gw.size - 2)
    iq = np.clip(np.searchsorted(gq, Q2) - 1, 0, gq.size - 2)
    tw = (W - gw[iw]) / (gw[iw + 1] - gw[iw])
    tq = (Q2 - gq[iq]) / (gq[iq + 1] - gq[iq])
    v00, v01 = vals_block[iq, iw], vals_block[iq, iw + 1]
    v10, v11 = vals_block[iq + 1, iw], vals_block[iq + 1, iw + 1]
    return ((1 - tq) * ((1 - tw) * v00 + tw * v01)
            + tq * ((1 - tw) * v10 + tw * v11))
