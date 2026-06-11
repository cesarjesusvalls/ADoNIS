"""DCC meson-baryon cross sections for the intranuclear cascade (the ACHILLES Virtual-
Resonances `MesonBaryonInteraction`).  Builds, from the Phase-E ANL-Osaka partial-wave
amplitudes (`anl_xsec`), the pi N -> pi' N' scattering cross sections sigma(W) [mb] for every
charge channel (elastic + charge exchange), precomputed on the amplitude W grid and exposed
as fast interpolators for the cascade.  These are the sharply Delta-peaked piN cross sections
(validated to the 9.3:2.2:1 isospin ratio in Phase E) -- the RIGHT scatter rates for the
cascade (vs the broad Oset QE).

Channels (pion_in, nucleon) -> list of (pion_out, nucleon_out, isospin-cg) with
cg = {3: c_{3/2}, 1: c_{1/2}} (the product of the initial+final meson-baryon Clebsches):
the partial-wave sum sigma = pref * sum_{L,J} (2J+1) |sum_I cg_I A^I_{L,J}|^2 (anl_xsec).
"""
from __future__ import annotations

import numpy as np

from adonis.fsi.mb.anl_xsec import load_anl, _channel_sigma

_R2 = np.sqrt(2.0) / 3.0
# (pion_in_idx, nucleon: 'p'/'n') -> [(pion_out_idx, nucleon_out, cg_dict), ...]
# pion idx: 0 pi+, 1 pi0, 2 pi-
_CHANNELS = {
    (0, "p"): [(0, "p", {3: 1.0})],                                   # pi+ p -> pi+ p (I=3/2)
    (0, "n"): [(0, "n", {3: 1.0 / 3, 1: 2.0 / 3}),                    # pi+ n -> pi+ n (elastic)
               (1, "p", {3: _R2, 1: -_R2})],                         # pi+ n -> pi0 p (cex)
    (1, "p"): [(1, "p", {3: 2.0 / 3, 1: 1.0 / 3}),                    # pi0 p -> pi0 p
               (0, "n", {3: _R2, 1: -_R2})],                         # pi0 p -> pi+ n (cex)
    (1, "n"): [(1, "n", {3: 2.0 / 3, 1: 1.0 / 3}),                    # pi0 n -> pi0 n
               (2, "p", {3: _R2, 1: -_R2})],                         # pi0 n -> pi- p (cex)
    (2, "p"): [(2, "p", {3: 1.0 / 3, 1: 2.0 / 3}),                    # pi- p -> pi- p (elastic)
               (1, "n", {3: _R2, 1: -_R2})],                         # pi- p -> pi0 n (cex)
    (2, "n"): [(2, "n", {3: 1.0})],                                  # pi- n -> pi- n (I=3/2)
}

_TABLE = {}


def _build_table():
    if _TABLE:
        return _TABLE
    Wt, amps = load_anl(0, 0)                       # piN -> piN amplitudes, W grid [MeV]
    _TABLE["W"] = Wt
    # sigma(W) for each (pion_in, nucleon, pion_out) channel
    grid = {}
    for (pin, nuc), outs in _CHANNELS.items():
        for (pout, nout, cg) in outs:
            grid[(pin, nuc, pout)] = _channel_sigma(amps, Wt, cg)   # mb
    _TABLE["grid"] = grid
    return _TABLE


def channel_sigmas(W, pion_in_idx):
    """sigma [mb] to each out-pion (pi+,pi0,pi-) for an incoming pion of charge `pion_in_idx`,
    ISOSPIN-AVERAGED over a proton/neutron target (12C: equal p/n).  W [MeV] array.
    Returns (3,)+W array sig_out[out_idx, ...]."""
    t = _build_table(); Wt = t["W"]; grid = t["grid"]
    W = np.atleast_1d(np.asarray(W, float))
    out = np.zeros((3,) + W.shape)
    for nuc in ("p", "n"):
        for (pout, nout, cg) in _CHANNELS[(pion_in_idx, nuc)]:
            s = np.interp(W, Wt, grid[(pion_in_idx, nuc, pout)], left=0.0, right=0.0)
            out[pout] += 0.5 * np.clip(s, 0.0, None)    # 0.5: average over p/n target
    return out


# --- JAX interface for the cascade (precomputed grids + jnp.interp) ----------- #
import jax.numpy as _jnp

_JGRID = {}


def _jax_grids():
    """Precompute, as jnp arrays: W grid and sig_out[pion_in, pion_out](W) averaged over a
    p/n target (12C).  Shape: jW (nW,), jsig (3 in, 3 out, nW)."""
    if _JGRID:
        return _JGRID["W"], _JGRID["sig"]
    t = _build_table(); Wt = t["W"]; grid = t["grid"]
    sig = np.zeros((3, 3, len(Wt)))
    for (pin, nuc), outs in _CHANNELS.items():
        for (pout, nout, cg) in outs:
            sig[pin, pout] += 0.5 * np.clip(grid[(pin, nuc, pout)], 0.0, None)
    _JGRID["W"] = _jnp.asarray(Wt); _JGRID["sig"] = _jnp.asarray(sig)
    return _JGRID["W"], _JGRID["sig"]


# --- DCC angular distribution sampler (replaces isotropic CM scatter in the cascade) -------- #
# Precompute the inverse CDF of dsigma/dOmega(cos) over a W grid, PER (pi_in, nucleon, pi_out)
# channel -- ACHILLES MesonBaryonInteraction::GenerateMomentum samples cos_CMS from the channel-
# specific partial-wave dsigma/dOmega (Get_CSpoly_W(W, ichan, fchan)).  Each channel's isospin
# Clebsch weights cg = {3: c_{3/2}, 1: c_{1/2}} come from _CHANNELS.  Channel index is
# chan = pi_in*6 + nuc*3 + pi_out  (pi in/out in {0:pi+,1:pi0,2:pi-}, nuc 0=proton 1=neutron).
_ANG = {}
_NCHAN = 18                                          # 3 pi_in x 2 nuc x 3 pi_out


def _cg_for_channel(pin, nuc_idx, pout):
    """Isospin cg for (pi_in, nucleon, pi_out); falls back to pure I=3/2 for non-physical combos
    (those have sigma=0 so the angular table is never sampled there)."""
    nuc = "p" if nuc_idx == 0 else "n"
    for (po, no, cg) in _CHANNELS.get((pin, nuc), []):
        if po == pout:
            return cg
    return {3: 1.0}


def _build_angular():
    if _ANG:
        return _ANG
    from adonis.fsi.mb.anl_xsec import dsigma_dOmega
    Wg = np.linspace(1085.0, 1700.0, 96)             # W grid [MeV]
    cg_grid = np.linspace(-1.0, 1.0, 181)            # cos(theta_cm) grid
    ug = np.linspace(0.0, 1.0, 64)                   # uniform grid for the inverse CDF
    inv = np.zeros((_NCHAN, Wg.size, ug.size))
    for pin in range(3):
        for nuc in range(2):
            for pout in range(3):
                ci = pin * 6 + nuc * 3 + pout
                cg = _cg_for_channel(pin, nuc, pout)
                for i, W in enumerate(Wg):
                    d = np.clip(np.asarray(dsigma_dOmega(float(W), cg_grid, cg)), 0.0, None)
                    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(cg_grid))])
                    cdf = cdf / cdf[-1] if cdf[-1] > 0 else np.linspace(0, 1, cg_grid.size)
                    inv[ci, i] = np.interp(ug, cdf, cg_grid)    # cos as a function of the CDF value
    _ANG["inv"] = _jnp.asarray(inv)                  # (NCHAN, nW, nu)
    _ANG["W0"] = float(Wg[0]); _ANG["dW"] = float(Wg[1] - Wg[0]); _ANG["nW"] = Wg.size
    _ANG["nu"] = ug.size
    return _ANG


def jax_sample_cos_cm(W, u, chan=0):
    """Sample cos(theta_cm) (N,) from the channel-specific DCC angular distribution at invariant
    mass W (N,), u (N,) ~ U[0,1].  chan (N,) or scalar = pi_in*6 + nuc*3 + pi_out (default 0 =
    pi+ p -> pi+ p, pure I=3/2).  Bilinear inverse-CDF lookup over the precomputed (chan, W, u) table."""
    t = _build_angular()
    inv = t["inv"]; nW = t["nW"]; nu = t["nu"]
    ch = _jnp.broadcast_to(_jnp.asarray(chan, _jnp.int32), W.shape)
    wf = _jnp.clip((W - t["W0"]) / t["dW"], 0.0, nW - 1.0001)
    iw = wf.astype(_jnp.int32); fw = wf - iw
    uf = _jnp.clip(u * (nu - 1), 0.0, nu - 1.0001)
    iu = uf.astype(_jnp.int32); fu = uf - iu
    c00 = inv[ch, iw, iu]; c01 = inv[ch, iw, iu + 1]; c10 = inv[ch, iw + 1, iu]; c11 = inv[ch, iw + 1, iu + 1]
    c0 = c00 * (1 - fu) + c01 * fu
    c1 = c10 * (1 - fu) + c11 * fu
    return _jnp.clip(c0 * (1 - fw) + c1 * fw, -1.0, 1.0)


def jax_channel_sigmas(W, pion_in_idx_arr):
    """sig_out (N,3) [mb] for a batch of pions: W (N,), pion_in_idx_arr (N,) in {0,1,2}.
    Isospin-averaged over a p/n target.  Pure jnp (interp); the amplitudes are constants."""
    jW, jsig = _jax_grids()                          # (nW,), (3,3,nW)
    # interp each (in,out) channel, then select the row for each pion's incoming charge
    all_io = _jnp.stack([_jnp.stack([_jnp.interp(W, jW, jsig[i, o], left=0.0, right=0.0)
                                     for o in range(3)], axis=-1)
                         for i in range(3)], axis=0)  # (3 in, N, 3 out)
    return _jnp.take_along_axis(all_io, pion_in_idx_arr[None, :, None], axis=0)[0]  # (N,3)
