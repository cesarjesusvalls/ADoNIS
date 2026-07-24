"""Differential (un-integrated) DCC hadron current as a function of the pion emission
angle (theta_pi, phi_pi) -- the per-event ingredient for a full final state.

`hadron_assembly.current_and_tensor` builds the ANGLE-INTEGRATED hadron tensor
W^{mu,nu} = sum_g w[g] sum_{isf,lam} conj(zj_g) zj_g, a Gauss-Legendre x uniform
quadrature of  int dcos(theta_pi) dphi_pi  of the per-grid-point DIFFERENTIAL current
zj[g, isf, lam, mu].  For a full final state we must instead evaluate that differential
current at the ACTUAL sampled pion angle of each event.

The angle dependence factorizes exactly.  In `angular_kernel`, for every
(isf, igm1, lam, pw) there is a SINGLE term

    K[g, ...] = coeff[isf,igm1,lam,pw] * Y_L^{llz}(theta_g) * e^{i llz phi_g}

with L = pw's orbital (twoL//2) and llz = (2*M_J - isf)//2 both FIXED by the indices,
and coeff = sqrt(2J+1) * <isospin CGs> * <spin-orbit CG> ANGLE-INDEPENDENT.  So we
precompute coeff, L, llz once (mirroring the exact `angular_kernel` selection rules),
then evaluate Y_L^{llz}(theta_pi) e^{i llz phi_pi} per event.

The interpolated amplitude `zmtx` depends only on (W, Q^2), NOT on the pion angle, so
the whole knob-dependent / differentiable path (build_zmtx) is reused unchanged; the
angle enters only through the detached, precomputed angular basis.

Consistency (see `_grid_consistency` / validate_final_state.py): summing the differential
tensor over the SAME quadrature grid with the SAME weights reproduces
`current_and_tensor` to ~1e-12.  Sampling the angle uniformly over the solid angle and
weighting by 4*pi reproduces it in expectation.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from adonis.primary.dcc.angular import cbg
from adonis.primary.dcc.assembly import (IGM1_LIST, LAM_LIST, ISF_LIST, _IXI1_OF, pw_phase)

_SQHF = 1.0 / np.sqrt(2.0)


# --------------------------------------------------------------------------- #
#  Angle-INDEPENDENT coefficients (mirror of angular_kernel's selection rules) #
# --------------------------------------------------------------------------- #
def precompute_diff_coeffs(two_J, two_L, two_I, *, tcrz, tiz, tpinz, tpiz,
                           tm_f=1.0, lmax=5, eps_tpin=1e-3, tmax=None):
    """Precompute the angle-independent pieces of the differential current for one
    isospin channel.  Returns a dict with:
        coeff   (n_isf, n_igm1, n_lam, n_pw) complex  -- zfac * cg_so
        Lpw     (n_pw,) int                            -- orbital L per partial wave
        llz     (n_isf, n_igm1, n_lam, n_pw) int       -- azimuthal index
        mask    (n_isf, n_igm1, n_lam, n_pw) float     -- selection-rule survival
        ixi1_map(n_igm1, n_lam) int                    -- (igm1,lam) -> zmtx row
    The (igm1, lam, isf) orderings match IGM1_LIST / LAM_LIST / ISF_LIST exactly.
    """
    npw = len(two_J)
    if tmax is None:
        tmax = tm_f + 0.5 + eps_tpin
    n_igm1, n_lam, n_isf = len(IGM1_LIST), len(LAM_LIST), len(ISF_LIST)

    coeff = np.zeros((n_isf, n_igm1, n_lam, npw), dtype=np.complex128)
    llz = np.zeros((n_isf, n_igm1, n_lam, npw), dtype=np.int64)
    mask = np.zeros((n_isf, n_igm1, n_lam, npw), dtype=np.float64)
    Lpw = np.asarray(two_L, dtype=np.int64) // 2
    ixi1_map = np.full((n_igm1, n_lam), -1, dtype=int)

    for ig1, igm1 in enumerate(IGM1_LIST):
        igm1x = igm1 if igm1 in (-1, 0, 1) else 0
        for il, lam in enumerate(LAM_LIST):
            ixi1_map[ig1, il] = _IXI1_OF[(igm1, lam)]
            Lambda_i = 2 * igm1x - lam                       # = 2*M_J
            for ipw in range(npw):
                twoJ, twoL, twoI = int(two_J[ipw]), int(two_L[ipw]), int(two_I[ipw])
                J, L, tpin = twoJ / 2, twoL / 2, twoI / 2
                if not (tpin + eps_tpin > abs(tpinz) and tmax > tpin
                        and twoJ >= abs(Lambda_i)):
                    continue
                zfac = (np.sqrt(twoJ + 1.0)
                        * cbg(1.0, tcrz, 0.5, tiz, tpin, tpinz)
                        * cbg(tm_f, tpiz, 0.5, tpinz - tpiz, tpin, tpinz))
                if zfac == 0.0:
                    continue
                for iisf, isf in enumerate(ISF_LIST):
                    mj = Lambda_i
                    z = (mj - isf) // 2
                    if abs(z) > L:
                        continue
                    cg_so = cbg(L, mj / 2 - isf / 2, 0.5, isf / 2, J, mj / 2)
                    if cg_so == 0.0:
                        continue
                    coeff[iisf, ig1, il, ipw] = zfac * cg_so
                    llz[iisf, ig1, il, ipw] = z
                    mask[iisf, ig1, il, ipw] = 1.0
    return {"coeff": coeff, "Lpw": Lpw, "llz": llz, "mask": mask,
            "ixi1_map": ixi1_map, "lmax": lmax}


# --------------------------------------------------------------------------- #
#  Vectorised real associated Legendre  Y_l^m(theta, 0)  (port of legendre_ylm) #
# --------------------------------------------------------------------------- #
from math import factorial as _fact, sqrt as _sqrt, pi as _pi


def _bb(n, k):
    return _fact(n) / _fact(k)


def legendre_ylm_batch(lmax, z):
    """Vectorised `hadron_tensor.legendre_ylm` over an array `z=cos(theta)` (shape (N,)).
    Returns bleg[N, l, m_index] with m = m_index - lmax (m in -lmax..lmax)."""
    z = np.asarray(z, dtype=np.float64)
    N = z.shape[0]
    bleg = np.zeros((N, lmax + 1, 2 * lmax + 1))

    def setb(l, m, v):
        bleg[:, l, m + lmax] = v

    def getb(l, m):
        return bleg[:, l, m + lmax]

    z1 = np.sqrt(np.clip(1.0 - z ** 2, 0.0, None)) + 1e-20
    z2 = z / z1
    setb(0, 0, np.ones(N))
    for l in range(1, lmax + 1):
        setb(l, l, z1 ** l / (2 ** l) * _bb(2 * l, l))
        setb(l, l - 1, z2 * getb(l, l))
        if l == 1:
            continue
        for m in range(l - 2, -1, -1):
            setb(l, m, (-getb(l, m + 2) + 2 * (m + 1) * z2 * getb(l, m + 1))
                 / ((l - m) * (l + m + 1)))
    for l in range(lmax + 1):
        for m in range(l + 1):
            fac = _sqrt((2 * l + 1) / (4.0 * _pi) * _bb(l - m, l + m)) * (-1) ** m
            pos = getb(l, m) * fac
            setb(l, m, pos)
            setb(l, -m, pos * (-1) ** m)
    return bleg


def angular_factor(theta, phi, pre):
    """Per-event angular kernel  K_e[isf,igm1,lam,pw] = coeff * Y_L^{llz}(theta) e^{i llz phi}.

    theta, phi : (N,) sampled pion angles (DETACHED -- the weight's knob-gradient does
    not flow through them).  Returns a complex numpy array (N, n_isf, n_igm1, n_lam, n_pw).
    """
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    lmax = pre["lmax"]
    coeff, Lpw, llz, mask = pre["coeff"], pre["Lpw"], pre["llz"], pre["mask"]
    bleg = legendre_ylm_batch(lmax, np.cos(theta))           # (N, lmax+1, 2lmax+1)

    # gather Y_{L[pw]}^{llz[isf,igm1,lam,pw]} per event via advanced indexing
    S = llz.shape                                            # (isf,igm1,lam,pw)
    L_idx = np.broadcast_to(Lpw[None, None, None, :], S)
    m_idx = llz + lmax
    Y = bleg[:, L_idx, m_idx]                                # (N,) + S
    az = np.exp(1j * llz[None] * phi[:, None, None, None, None])
    return coeff[None] * Y * az * mask[None]                 # (N, isf,igm1,lam,pw)


# --------------------------------------------------------------------------- #
#  Batched build_zmtx (the angle-independent, knob-dependent amplitude matrix)  #
# --------------------------------------------------------------------------- #
IDXP_START = True    # ACHILLES-faithful: skip the idxp=1 (0,5) current pair for J=1/2 waves (default).
#                      Set False only to MEASURE the size of that term vs the pre-fix behavior.


def build_zmtx_batched(vec, isv, axial, W, Q2, two_J, two_L, two_I, *, mode, itiz,
                       m_N, m_pi, r_axial=None, vfac=1.0, pion_pole=1.0):
    """Batched port of `hadron_assembly.build_zmtx`.

    vec, isv, axial : (N, 8, n_pw) complex (interpolated amplitude components).
    W, Q2           : (N,) real.  Returns zmtx (N, 8, n_pw) complex.
    Differentiable in the amplitude / r_axial (the knobs); equals the scalar build_zmtx
    event-by-event (asserted in validate_final_state.py).
    """
    npw = vec.shape[-1]
    phv = jnp.asarray([pw_phase(int(two_J[i]), int(two_L[i])) for i in range(npw)])  # (npw,)
    pha = -phv
    is_I32 = jnp.asarray([1.0 if int(two_I[i]) == 3 else 0.0 for i in range(npw)])   # (npw,)
    # ACHILLES idxp_start (amp_dcc_sl_module.f:809-813,873-877,920-924): for J=1/2 waves (two_J==1) the
    # vector AND axial current loops start at idxp=2, i.e. the idxp=1 pair (idx 1,6 -> 0-based (0,5)) is
    # NEVER populated.  keep_idxp1 (npw,) = 0 for J=1/2 waves, 1 otherwise -> zeroes that pair per wave.
    if IDXP_START:
        keep_idxp1 = jnp.asarray([0.0 if int(two_J[i]) == 1 else 1.0 for i in range(npw)])[None, :]
    else:
        keep_idxp1 = jnp.ones((1, npw))

    qc0 = (W ** 2 - m_N ** 2 - Q2) / (2.0 * W)               # (N,)
    qc = jnp.sqrt(Q2 + qc0 ** 2)
    xxx = (qc0 / qc)[:, None]                                # (N,1)
    qc0c, qcc = qc0[:, None], qc[:, None]

    zmtx = jnp.zeros((W.shape[0], 8, npw), dtype=jnp.complex128)

    if mode < 10:                                            # axial current (weak)
        a = -axial
        if r_axial is not None:
            r = jnp.asarray(r_axial)
            a = a * (r if r.ndim == 0 else r[:, None, None])
        for idxp, (src, dst) in enumerate(((0, 5), (1, 4), (2, 3), (6, 7)), start=1):
            av = a[:, src] * keep_idxp1 if idxp == 1 else a[:, src]   # skip (0,5) for J=1/2 waves
            zmtx = zmtx.at[:, src].set(av)
            zmtx = zmtx.at[:, dst].set(av * pha)
        if mode > 0:                                         # pion pole (CC only)
            facpp = (jnp.asarray(pion_pole) / (-Q2 - m_pi ** 2))[:, None]   # pion_pole: F_P reweight knob
            zp = (qc0c * zmtx[:, 2] - qcc * zmtx[:, 6]) * facpp
            zm = (qc0c * zmtx[:, 3] - qcc * zmtx[:, 7]) * facpp
            zmtx = zmtx.at[:, 2].add(-qc0c * zp)
            zmtx = zmtx.at[:, 6].add(-qcc * zp)
            zmtx = zmtx.at[:, 3].add(-qc0c * zm)
            zmtx = zmtx.at[:, 7].add(-qcc * zm)

    # vector current with EW isospin rotation.
    i32 = is_I32[None, None, :]
    if 0 < mode < 10:                                        # CC: I=1/2 -> (V-IS)/2; I=3/2 raw
        src_block = i32 * vec + (1.0 - i32) * 0.5 * (vec - isv)
    elif mode <= -1:                                         # NC: sin^2(theta_W) couplings
        # vector coupling (amp_dcc_sl_module.f:288-294, 1004-1050): isovector x (1-2 sw2)
        # for all waves; isoscalar x vvfac(itiz) added for I=1/2 (vvfac(+1)=-2sw2 proton,
        # vvfac(-1)=+2sw2 neutron). isovector=0.5(vec-isv) [zm], isoscalar=0.5(vec+isv) [zp].
        sw2 = 0.2312
        VFAC = 1.0 - 2.0 * sw2
        VVFAC = -2.0 * sw2 if itiz == 1 else 2.0 * sw2
        iso_v = 0.5 * (vec - isv)
        iso_s = 0.5 * (vec + isv)
        src_block = i32 * (VFAC * vec) + (1.0 - i32) * (VFAC * iso_v + VVFAC * iso_s)
    elif itiz == -1:                                         # EM neutron: I=1/2 -> isoscalar
        # isign=-1: the neutron-amplitude phase (amp_dcc_sl_module.f:644, applied to the
        # zampv_is block for EM only; our loader stores isv without it).
        src_block = is_I32[None, None, :] * vec - (1.0 - is_I32[None, None, :]) * isv
    else:                                                    # EM proton, or EM I=3/2
        src_block = vec
    for idxp, (src, dst) in enumerate(((0, 5), (1, 4), (2, 3)), start=1):
        vz = vfac * src_block[:, src]                        # (N, npw)
        if idxp == 1:
            vz = vz * keep_idxp1                             # skip (0,5) pair for J=1/2 waves (ACHILLES)
        zmtx = zmtx.at[:, src].add(vz)
        zmtx = zmtx.at[:, dst].add(vz * phv)
        if idxp == 3:
            zmtx = zmtx.at[:, 6].add(vz * xxx)
            zmtx = zmtx.at[:, 7].add(vz * xxx * phv)
    return zmtx


# --------------------------------------------------------------------------- #
#  Differential current + per-event hadron tensor                              #
# --------------------------------------------------------------------------- #
def differential_current(zmtx, Kfac, ixi1_map):
    """zmtx (N,8,npw) + per-event angular kernel Kfac (N,isf,igm1,lam,npw) ->
    Cartesian hadronic current zj (N, isf, lam, mu)  (mu = 0,1,2,3)."""
    ix = jnp.asarray(ixi1_map)                               # (igm1,lam)
    amp = zmtx[:, ix, :]                                     # (N, igm1, lam, npw)
    # zcrnt[e,isf,igm1,lam] = sum_pw Kfac * amp
    zcrnt = jnp.einsum("esilp,eilp->esil", jnp.asarray(Kfac), amp)
    z_m1, z_0, z_p1, z_2 = (zcrnt[..., 0, :], zcrnt[..., 1, :],
                            zcrnt[..., 2, :], zcrnt[..., 3, :])   # each (N,isf,lam)
    zj0 = z_0
    zj3 = z_2
    zj1 = (z_m1 - z_p1) * _SQHF
    zj2 = (z_m1 + z_p1) * (_SQHF * 1j)
    return jnp.stack([zj0, zj1, zj2, zj3], axis=-1)          # (N, isf, lam, mu)


def differential_tensor(zj):
    """Per-event differential hadron tensor W[e,mu,nu] = sum_{isf,lam} conj(zj_mu) zj_nu."""
    return jnp.einsum("eslm,esln->emn", jnp.conj(zj), zj)
