"""Structure functions from the full hadron tensor (Phase-2.5, milestone 6c->fold).

Wraps hadron_assembly into a dcc_xsec-style provider: computes the angle-integrated
transverse (W_T) and longitudinal (W_L) response on the amplitude's native (Q^2, W)
grid -- summed over the physical isospin channels -- then bilinearly interpolates onto
event kinematics, with the M_A / pw_norm knobs flowing differentiably.

CC nu on a (p,n) nucleus sums INCOHERENTLY over the final-state channels (different
final particles don't interfere); within each channel the partial waves add coherently
(that coherence is exactly the same-L multipole interference the diagonal dcc_xsec drops).

Channels (CC nu, W^+ absorbed, tcrz=+1):
    struck n -> p pi0   (tiz=-1/2, tpinz=+1/2, tpiz=0)
    struck n -> n pi+   (tiz=-1/2, tpinz=+1/2, tpiz=+1)
    struck p -> p pi+   (tiz=+1/2, tpinz=+3/2, tpiz=+1)   [pure I=3/2]
weighted by the nucleon counts (12C: 6 p, 6 n).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

from .dcc import DCCAmplitudes, DCCKnobs
from .dcc_loader import load_cached
from .hadron_assembly import build_zmtx, angular_kernel, current_and_tensor
from .form_factors import axial_reweight_dipole
from .achilles_const import MQE as M_N, M_PI    # exact ACHILLES masses (938.919, 138.04)


@dataclass(frozen=True)
class Channel:
    itiz: int          # 2*target-nucleon isospin_z (+1 p, -1 n)
    tiz: float         # target-nucleon isospin_z
    tpinz: float       # total piN isospin_z
    tpiz: float        # final-pion isospin_z
    mult: float        # nucleon-count weight
    tcrz: float = 1.0  # current isospin_z (CC nu = +1)
    mode: int = 1      # 1 = CC nu


CC_CHANNELS = (
    Channel(itiz=-1, tiz=-0.5, tpinz=0.5, tpiz=0.0, mult=6.0),   # n -> p pi0
    Channel(itiz=-1, tiz=-0.5, tpinz=0.5, tpiz=1.0, mult=6.0),   # n -> n pi+
    Channel(itiz=+1, tiz=+0.5, tpinz=1.5, tpiz=1.0, mult=6.0),   # p -> p pi+
)


class HadronStructure:
    """Full-hadron-tensor structure functions W_T, W_L on the (Q2,W) grid."""

    def __init__(self, amp: DCCAmplitudes | None = None, channels=CC_CHANNELS,
                 n_theta=12, n_phi=12, subsample_w=1):
        t = load_cached()
        self.amp = amp or DCCAmplitudes(t)
        self.twoJ = np.asarray(t.pw_2J); self.twoL = np.asarray(t.pw_2L)
        self.twoI = np.asarray(t.pw_2I)
        # native grid (optionally W-subsampled for speed); Q2 kept full
        self.Wg = np.asarray(t.W)[::subsample_w]
        self.Q2g = np.asarray(t.Q2)
        self.channels = channels
        self.kers = [angular_kernel(self.twoJ, self.twoL, self.twoI,
                                    tcrz=c.tcrz, tiz=c.tiz, tpinz=c.tpinz, tpiz=c.tpiz,
                                    n_theta=n_theta, n_phi=n_phi)
                     for c in channels]
        # mesh of grid points (flattened) for a single vmap
        QQ, WW = np.meshgrid(self.Q2g, self.Wg, indexing="ij")
        self._Wf = jnp.asarray(WW.ravel()); self._Q2f = jnp.asarray(QQ.ravel())
        self._shape = WW.shape

    def _full_grid(self, knobs: DCCKnobs):
        """Full complex hadron tensor W^{mu,nu}[Q2,W,4,4] (summed over channels)."""
        def per_point(Wv, Q2v):
            vec, isv, axial = self.amp.amplitudes(Wv, Q2v, knobs)
            r_ax = axial_reweight_dipole(Q2v, knobs.axial_MA)
            Wmn = jnp.zeros((4, 4), jnp.complex128)
            for c, ker in zip(self.channels, self.kers):
                zmtx = build_zmtx(vec, isv, axial, Wv, Q2v, self.twoJ, self.twoL,
                                  self.twoI, mode=c.mode, itiz=c.itiz, m_N=M_N,
                                  m_pi=M_PI, r_axial=r_ax)
                Wmn = Wmn + c.mult * current_and_tensor(zmtx, ker)
            return Wmn
        Wmn = jax.vmap(per_point)(self._Wf, self._Q2f)
        return Wmn.reshape(self._shape + (4, 4))

    def _tensor_grid(self, knobs: DCCKnobs):
        """W_T, W_L on the (Q2,W) grid for the given knobs (summed over channels)."""
        Wmn = self._full_grid(knobs)
        WT = (Wmn[..., 1, 1] + Wmn[..., 2, 2]).real
        WL = Wmn[..., 3, 3].real
        return WT, WL

    def structure_grid(self, knobs: DCCKnobs):
        """Return (W_T, W_L) grids (Q2,W) plus the axes for interpolation."""
        WT, WL = self._tensor_grid(knobs)
        return WT, WL

    def _interp(self, grid, W, Q2):
        gw, gq = jnp.asarray(self.Wg), jnp.asarray(self.Q2g)
        nw, nq = gw.shape[0], gq.shape[0]
        iw = jnp.clip(jnp.searchsorted(gw, W) - 1, 0, nw - 2)
        iq = jnp.clip(jnp.searchsorted(gq, Q2) - 1, 0, nq - 2)
        tw = (W - gw[iw]) / (gw[iw + 1] - gw[iw])
        tq = (Q2 - gq[iq]) / (gq[iq + 1] - gq[iq])
        v00, v01 = grid[iq, iw], grid[iq, iw + 1]
        v10, v11 = grid[iq + 1, iw], grid[iq + 1, iw + 1]
        return ((1 - tq) * ((1 - tw) * v00 + tw * v01)
                + tq * ((1 - tw) * v10 + tw * v11))

    def structures_at(self, W, Q2, knobs: DCCKnobs):
        """(W_T, W_L) at event kinematics W,Q2 (batched), differentiable in knobs."""
        WTg, WLg = self._tensor_grid(knobs)
        WT = jax.vmap(lambda w, q: self._interp(WTg, w, q))(W, Q2)
        WL = jax.vmap(lambda w, q: self._interp(WLg, w, q))(W, Q2)
        return WT, WL

    def tensor_at(self, W, Q2, knobs: DCCKnobs):
        """Full complex W^{mu,nu}[event,4,4] at event kinematics (batched),
        differentiable in knobs. For contraction with the lepton tensor."""
        Wg = self._full_grid(knobs)                       # (Q2,W,4,4)
        flat = Wg.reshape(Wg.shape[0], Wg.shape[1], 16)
        out = jax.vmap(lambda w, q: self._interp(flat, w, q))(W, Q2)
        return out.reshape(-1, 4, 4)
