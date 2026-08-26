"""Full DCC hadron-tensor assembly -- the port of amp_dcc_sl.f::amplitude() + interpolate_amp,
replacing the angle-integrated diagonal bilinear approximation (dcc_xsec) with the real
helicity current zj_mu and hadron tensor W^{mu,nu}.

Frame choice: this works entirely in the piN centre-of-mass with the momentum transfer q
along +z (the Fortran `irot_q=0` branch). For the spin-summed, angle-integrated response
this is exact:
  * the nucleon spin rotations (rspin, irot_spin) are unitary on the spin indices and
    cancel in sum_{spins}|zj|^2 -> skipped;
  * dsigma/dW,dQ^2 is a Lorentz scalar, so the 2CM->lab boost (lorentz_trans, xlr) is
    unneeded as long as the lepton tensor is contracted in the same 2CM frame.

The knob-dependent physics lives entirely in `zmtx[ixi1, pw]` (the interpolated DCC
amplitude, with the form-factor / pw-norm knobs). The angular assembly (isospin CG,
spin-orbit CG, Legendre, azimuth) is a knob- and (W,Q^2)-independent linear map,
precomputed once as the kernel K so that

    zcrnt[g, isf, igm1, lam] = sum_pw  K[g, isf, igm1, lam, pw] * zmtx[ixi1(igm1,lam), pw]

and the hadron tensor is the bilinear  W^{mu,nu} = sum_g w[g] sum_{isf,lam} zj*_mu zj_nu,
which JAX differentiates exactly (fixed quadrature grid, zero variance).

See angular.py for the angular primitives (cbg, legendre_ylm) and the index-convention decode.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from adonis.channels.probes import probe_for_mode
from adonis.channels.dcc.angular import cbg, legendre_ylm, ISMI, ISMIX, ISBI

_ID1 = np.array([1, 2, 3, 7])
_ID2 = np.array([6, 5, 4, 8])

_IXI1_OF = {}
for _ixi1 in range(1, 9):
    _IXI1_OF[(int(ISMI[_ixi1]), -int(ISBI[_ixi1]))] = _ixi1 - 1

IGM1_LIST = (-1, 0, 1, 2)
LAM_LIST = (-1, 1)
ISF_LIST = (-1, 1)

DBG = {
    "pion_pole": 1.0,
    "axial_z": 1.0,
    "axial_time": 1.0,
    "vec_cc_z": 1.0,
    "idxp_start": 1,
    "axial_sign": -1.0,
}
import os as _os
for _k in list(DBG):
    _v = _os.environ.get("ADONIS_DBG_" + _k.upper())
    if _v is not None:
        DBG[_k] = float(_v) if ("." in _v or "e" in _v.lower()) else int(_v)


def pw_phase(two_J, two_L):
    """phv = (-1)**((2J-1)/2 + L + 1)  (interpolate_amp); pha (axial) = -phv."""
    return (-1.0) ** ((two_J - 1) // 2 + two_L // 2 + 1)


def build_zmtx(vec, isv, axial, W, Q2, two_J, two_L, two_I, *, mode, itiz,
               m_N, m_pi, r_axial=None, vfac=1.0):
    """Assemble the full 8-component current matrix zmtx[ixi1, pw] at one (W,Q^2).

    vec, isv, axial : (8, n_pw) complex -- interpolated amplitude components, idx
        (0-based Fortran idx-1) populated at {0,1,2} (vec/isv) and {0,1,2,6} (axial).
    mode  : 1 (CC nu), 10 (EM electron), -1 (NC); selects axial inclusion + pion pole.
    itiz  : 2*(target nucleon isospin_z); +1 proton, -1 neutron.
    r_axial : optional (n_pw,) or scalar real reweight of the axial block (M_A knob,
              applied as amplitude factor so it squares into the rate downstream).
    """
    probe_for_mode(mode)
    npw = vec.shape[1]
    phv = jnp.asarray([pw_phase(int(two_J[i]), int(two_L[i])) for i in range(npw)])
    pha = -phv

    qc0 = (W ** 2 - m_N ** 2 - Q2) / (2.0 * W)
    qc = jnp.sqrt(Q2 + qc0 ** 2)
    xxx = qc0 / qc

    zmtx = jnp.zeros((8, npw), dtype=jnp.complex128)

    if mode < 10:
        a = DBG["axial_sign"] * axial
        if r_axial is not None:
            a = a * jnp.asarray(r_axial)
        _ax_scale = {(0, 5): 1.0, (1, 4): 1.0,
                     (2, 3): DBG["axial_time"], (6, 7): DBG["axial_z"]}
        keep_idxp1 = jnp.asarray([0.0 if (DBG["idxp_start"] and int(two_J[i]) == 1) else 1.0
                                  for i in range(npw)])
        for src, dst in ((0, 5), (1, 4), (2, 3), (6, 7)):
            sc = _ax_scale[(src, dst)]
            av = a[src] * keep_idxp1 if (src, dst) == (0, 5) else a[src]
            zmtx = zmtx.at[src].set(av * sc)
            zmtx = zmtx.at[dst].set(av * pha * sc)
        if mode > 0:
            facpp = DBG["pion_pole"] / (-Q2 - m_pi ** 2)
            zp = (qc0 * zmtx[2] - qc * zmtx[6]) * facpp
            zm = (qc0 * zmtx[3] - qc * zmtx[7]) * facpp
            zmtx = zmtx.at[2].add(-qc0 * zp)
            zmtx = zmtx.at[6].add(-qc * zp)
            zmtx = zmtx.at[3].add(-qc0 * zm)
            zmtx = zmtx.at[7].add(-qc * zm)

    for ipw in range(npw):
        is_I32 = int(two_I[ipw]) == 3
        if mode < 10:
            src_block = vec if is_I32 else 0.5 * (vec - isv)
        elif itiz == -1 and not is_I32:
            src_block = -isv
        else:
            src_block = vec
        idxp_start = 2 if (DBG["idxp_start"] and int(two_J[ipw]) == 1) else 1
        for idxp, (src, dst) in enumerate(((0, 5), (1, 4), (2, 3)), start=1):
            if idxp < idxp_start:
                continue
            vz = vfac * src_block[src, ipw]
            zmtx = zmtx.at[src, ipw].add(vz)
            zmtx = zmtx.at[dst, ipw].add(vz * phv[ipw])
            if idxp == 3:
                zmtx = zmtx.at[6, ipw].add(vz * xxx * DBG["vec_cc_z"])
                zmtx = zmtx.at[7, ipw].add(vz * xxx * phv[ipw] * DBG["vec_cc_z"])
    return zmtx


def angular_kernel(two_J, two_L, two_I, *, tcrz, tiz, tpinz, tpiz, tm_f=1.0,
                   n_theta=16, n_phi=16, lmax=5, eps_tpin=1e-3, tmax=None):
    """Precompute K[g, isf, igm1, lam, pw] and quadrature weights w[g], plus the
    (igm1,lam)->ixi1 index map, for the irot_q=0 assembly.

    zcrnt[g,isf,igm1,lam] = sum_pw K[...,pw] * zmtx[ixi1_map[igm1,lam], pw]
    with zfac_a = sqrt(2J+1) * <1 tcrz;1/2 tiz|tpin tpinz> * <tm_f tpiz;1/2 .|tpin tpinz>
    and zzz = <L,mj/2-isf/2;1/2,isf/2|J,mj/2> * Y_L^{llz}(theta) * e^{i llz phi}.
    """
    npw = len(two_J)
    if tmax is None:
        tmax = tm_f + 0.5 + eps_tpin

    ct, wct = np.polynomial.legendre.leggauss(n_theta)
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    wphi = (2 * np.pi / n_phi)
    G = n_theta * n_phi
    w = np.empty(G)
    bleg = np.empty((G, lmax + 1, 2 * lmax + 1))
    azim_phi = np.empty((G, n_phi if False else 1))
    cphi = np.empty(G)
    for it in range(n_theta):
        bl = legendre_ylm(lmax, ct[it])
        for ip in range(n_phi):
            g = it * n_phi + ip
            w[g] = wct[it] * wphi
            bleg[g] = bl
    ms = np.arange(-lmax, lmax + 1)
    eim = np.empty((G, 2 * lmax + 1), dtype=np.complex128)
    for it in range(n_theta):
        for ip in range(n_phi):
            g = it * n_phi + ip
            eim[g] = np.exp(1j * ms * phi[ip])

    n_igm1, n_lam, n_isf = len(IGM1_LIST), len(LAM_LIST), len(ISF_LIST)
    K = np.zeros((G, n_isf, n_igm1, n_lam, npw), dtype=np.complex128)
    ixi1_map = np.full((n_igm1, n_lam), -1, dtype=int)

    for ig1, igm1 in enumerate(IGM1_LIST):
        igm1x = igm1 if igm1 in (-1, 0, 1) else 0
        for il, lam in enumerate(LAM_LIST):
            ixi1_map[ig1, il] = _IXI1_OF[(igm1, lam)]
            Lambda_i = 2 * igm1x - lam
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
                llpin = twoL // 2
                for iisf, isf in enumerate(ISF_LIST):
                    mj = Lambda_i
                    llz = (mj - isf) // 2
                    if abs(llz) > L:
                        continue
                    cg_so = cbg(L, mj / 2 - isf / 2, 0.5, isf / 2, J, mj / 2)
                    if cg_so == 0.0:
                        continue
                    Ylm = bleg[:, llpin, llz + lmax]
                    az = eim[:, llz + lmax]
                    K[:, iisf, ig1, il, ipw] += zfac * cg_so * Ylm * az

    return {
        "K": jnp.asarray(K), "w": jnp.asarray(w), "ixi1_map": ixi1_map,
        "igm1_list": IGM1_LIST,
    }


_SQHF = 1.0 / np.sqrt(2.0)


def current_and_tensor(zmtx, ker, fac=1.0):
    """zmtx[8,npw] + kernel -> hadron tensor W[mu,nu] (4x4 complex, 2CM frame).

    W^{mu,nu} = sum_g w[g] sum_{isf,lam} conj(zj[g,isf,lam,mu]) zj[g,isf,lam,nu]
    (initial-spin average factor left to the caller / lepton contraction)."""
    K, w, ixi1_map = ker["K"], ker["w"], ker["ixi1_map"]
    n_igm1, n_lam = ixi1_map.shape
    amp = jnp.stack([jnp.stack([zmtx[ixi1_map[i, l]] for l in range(n_lam)])
                     for i in range(n_igm1)])
    zcrnt = jnp.einsum("gsilp,ilp->gsil", K, amp)

    z_m1, z_0, z_p1, z_2 = (zcrnt[..., 0, :], zcrnt[..., 1, :],
                            zcrnt[..., 2, :], zcrnt[..., 3, :])
    zj0 = z_0
    zj3 = z_2
    zj1 = (z_m1 - z_p1) * _SQHF
    zj2 = (z_m1 + z_p1) * (_SQHF * 1j)
    zj = jnp.stack([zj0, zj1, zj2, zj3], axis=-1) * fac

    W = jnp.einsum("g,gslm,gsln->mn", w, jnp.conj(zj), zj)
    return W
