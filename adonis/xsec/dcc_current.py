"""Exclusive ACHILLES DCC RES hadron current (amp_dcc_sl.f::amplitude, irot_q=1) -> amps2.

Computes the piN current zj_mu(isf, lam) at the SAMPLED pion momentum (vs the inclusive
angle-integrated tensor in adonis/primary/dcc), then contracts with the exact leptonic current.
The two unitary spin-rotation blocks of amplitude() cancel in amps2 (Frobenius norm over the 2x2
spin indices) and are skipped.  The /fm scaling cancels in the kinematics (angles, wcm, Q2 all in
MeV) and in fac*(2xmn/hbarc); the residual overall constant (table normalisation x 2sqrt2 mN x
coupling) is validated as CONSTANT across events vs the instrumented-ACHILLES RESDUMP.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from adonis.xsec import constants as C
from adonis.xsec.leptonic import lepton_current
from adonis.xsec.dcc_kinematics import boost_matrix, setdfun
from adonis.primary.dcc.angular import cbg, legendre_ylm, ISMI, ISMIX, ISBI
from adonis.primary.dcc.assembly import build_zmtx
from adonis.primary.dcc.amplitudes import DCCAmplitudes, DCCKnobs
from adonis.primary.dcc.loader import load_cached

_AMP = DCCAmplitudes()
_T = load_cached()
_PW_2J = np.asarray(_T.pw_2J); _PW_2L = np.asarray(_T.pw_2L); _PW_2I = np.asarray(_T.pw_2I)
_NPW = len(_PW_2J)
_JMAX = int(_PW_2J.max())
_LMAX = 5
_ISP = {-1: 1, 1: 0}                      # spin index: up(+1)->0, down(-1)->1  (0-based)
_EPS_TPIN = 1e-3
_METRIC = np.array([1.0, -1.0, -1.0, -1.0])
# ONE universal constant (table-normalisation x fac x 2xmn/hbarc x |FResV|^2) putting the exclusive
# DCC amps2 on the ACHILLES absolute scale -- validated CHANNEL-INDEPENDENT (pi+ 3.771e-5, pi0
# 3.769e-5) and CONSTANT per-event to ~3.7% (the residual = amplitude-table interpolation spline
# vs ACHILLES interpolate_amp).  amps2_absolute = amps2_raw / _NORM.
_NORM = 3.77040e-05


def _ang(vec3):
    r = np.linalg.norm(vec3)
    cz = vec3[2] / r
    sz = np.sqrt(max(1 - cz ** 2, 0.0))
    if sz > 1e-12:
        zphi = (vec3[0] + 1j * vec3[1]) / (sz * r)
    else:
        zphi = 1.0 + 0j
    return cz, zphi


def exclusive_H(k_nu, k_mu, p_struck, p_outN, p_pi, itiz, mode=1, tcrz=1.0, tpiz=0.0, tm_f=1.0):
    """Hadron current H[combo(isf,lam), mu] (4,4) complex up to the overall constant, lab frame."""
    mN = C.mN
    q = np.asarray(k_nu) - np.asarray(k_mu)
    p3 = np.asarray(p_struck)[1:]
    E_on = np.sqrt(p3 @ p3 + mN ** 2)
    qE = q[0] + p_struck[0] - E_on
    qsh = np.array([qE, q[1], q[2], q[3]])
    xpnuc = np.asarray(p_outN, float); xk = np.asarray(p_pi, float)
    pcm = xpnuc + xk
    px = xpnuc + xk - qsh; px = np.array([np.sqrt(mN ** 2 + px[1:] @ px[1:]), *px[1:]])
    xlrs = boost_matrix(pcm, to_cm=True); xlr = boost_matrix(pcm, to_cm=False)
    xk2 = xlrs @ xk; qx2 = xlrs @ qsh
    xz_pin, zphi_pin = _ang(xk2[1:])
    xz_q, zphi_q = _ang(qx2[1:])
    wcm = np.sqrt(max(pcm[0] ** 2 - pcm[1:] @ pcm[1:], 1.0))
    Q2 = qsh[1:] @ qsh[1:] - qsh[0] ** 2
    dfun, off = setdfun(xz_q, _JMAX)
    bleg = np.asarray(legendre_ylm(_LMAX, xz_pin))           # (L+1, 2L+1)
    zphi_pins = {l: zphi_pin ** l for l in range(-_LMAX, _LMAX + 1)}

    vec, isv, axial = _AMP.amplitudes_spline(jnp.array([wcm]), jnp.array([Q2]), DCCKnobs())
    zmtx = np.asarray(build_zmtx(vec[0], isv[0], axial[0], wcm, Q2, _PW_2J, _PW_2L, _PW_2I,
                                 mode=mode, itiz=itiz, m_N=mN, m_pi=C.mpi0))   # (8, npw)
    tiz = itiz / 2.0; tpinz = tcrz + tiz
    tmax = tm_f + 0.5 + _EPS_TPIN

    zcrnt = np.zeros((2, 2, 4), complex)                     # [isf_idx, lam_idx, igm1_idx(0:-1,1:0,2:1,3:2)]
    IGM1 = (-1, 0, 1, 2)
    for ixi1 in range(1, 9):
        igm1 = int(ISMI[ixi1]); igm1x = int(ISMIX[ixi1]); lambda_N = -int(ISBI[ixi1])
        lam_idx = _ISP[lambda_N]; Lambda_i = 2 * igm1x - lambda_N
        ig_idx = IGM1.index(igm1)
        for pw in range(_NPW):
            jpin = int(_PW_2J[pw]); Lpin = int(_PW_2L[pw]); itpin = int(_PW_2I[pw])
            tpin = itpin / 2.0; xlpin = Lpin / 2.0; xjpin = jpin / 2.0; llpin = Lpin // 2
            if not (tpin + _EPS_TPIN > abs(tpinz) and tmax > tpin and jpin >= abs(Lambda_i)):
                continue
            zfac = (np.sqrt(jpin + 1.0) * cbg(1.0, tcrz, 0.5, tiz, tpin, tpinz)
                    * cbg(tm_f, tpiz, 0.5, tpinz - tpiz, tpin, tpinz) * zmtx[ixi1 - 1, pw])
            if zfac == 0:
                continue
            for isf in (-1, 1):
                isf_idx = _ISP[isf]; xs = isf / 2.0
                zzz = 0j
                for mj in range(max(-Lpin + isf, -jpin), min(Lpin + isf, jpin) + 1, 2):
                    xmj = mj / 2.0; llz = (mj - isf) // 2
                    if abs(llz) > llpin:
                        continue
                    zzz += (cbg(xlpin, xmj - xs, 0.5, xs, xjpin, xmj) * bleg[llpin, llz + _LMAX]
                            * zphi_pins[llz] * dfun[jpin, mj + off, Lambda_i + off]
                            * (zphi_q ** ((-mj + Lambda_i) // 2)))
                zcrnt[isf_idx, lam_idx, ig_idx] += zfac * zzz

    # helicity -> Cartesian (igm1 idx: 0:-1, 1:0, 2:1, 3:2)
    sq = 1.0 / np.sqrt(2.0)
    zjx = np.zeros((2, 2, 4), complex)
    zjx[:, :, 0] = zcrnt[:, :, 1]                            # time (igm1=0)
    zjx[:, :, 3] = zcrnt[:, :, 3]                            # z    (igm1=2)
    zjx[:, :, 1] = (zcrnt[:, :, 0] - zcrnt[:, :, 2]) * sq
    zjx[:, :, 2] = (zcrnt[:, :, 0] + zcrnt[:, :, 2]) * sq * 1j
    # boost 2CM -> lab
    zj = np.einsum('mn,abn->abm', xlr, zjx)                  # (isf, lam, mu)
    return zj.reshape(4, 4)                                  # combo=(isf,lam) flattened


def exclusive_amps2(k_nu, k_mu, p_struck, p_outN, p_pi, itiz, hPID):
    """amps2 up to the overall constant: sum_{isf,lam} |L.H|^2."""
    tpiz = {211: 1.0, 111: 0.0, -211: -1.0}[int(hPID)]
    H = exclusive_H(k_nu, k_mu, p_struck, p_outN, p_pi, itiz, tpiz=tpiz)
    L = np.asarray(lepton_current(jnp.asarray(k_nu)[None], jnp.asarray(k_mu)[None]))[0]  # (4,4)
    LH = np.einsum('am,bm,m->ab', L, H, _METRIC)
    return float(np.sum(np.abs(LH) ** 2)) / _NORM
