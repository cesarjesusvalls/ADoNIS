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

from adonis.channels import constants as C
from adonis.primary.dcc import conventions as _conv
from adonis.channels.leptonic import lepton_current
from adonis.channels.dcc_kinematics import boost_matrix, setdfun
from adonis.primary.dcc.angular import cbg, legendre_ylm, legendre_ylm_batch, ISMI, ISMIX, ISBI
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
# DCC amplitude table validity (currents_pi_dcc.f90:109-120): outside -> J_mu = 0.  ESSENTIAL --
# without it the spline EXTRAPOLATES to garbage on high-Enu/high-Q2 events (1e8x spurious amps2).
_W_LO = 1076.957; _W_HI = 2000.0; _Q2_HI = 5.0e6
# First-principles RES normalisation (NO fit).  ACHILLES builds the hadron current as
#   H = FResV * zj_raw * fac * (2*xmn/hbarc)   (res_spec_currents; amp_dcc_sl.f:417 fac;
#   currents_pi_dcc.f90:130  J_mu*=2*xmn/hbarc),  with FResV = Vud*ee/(sw*sqrt2*2) the hadronic
#   EW coupling (LeptonicCurrent.cc:63; resV=1).  My zj_raw omits FResV, fac and the fm-unit
#   scalings; assembling those constants (fac^2 = 2/(fnuc^2 4pi) with the fm->MeV scaling fnuc^2,
#   and (2 m_N/hbarc)^2) gives  1/_NORM = |FResV|^2 * (2 m_N)^2 / (2 pi).
# Per-event amps2 audit vs the ACHILLES free-proton RESDUMP (scripts/amps2_resdump_audit.py): the
# ratio amps2_ACH/amps2_ADO is FLAT at 1.0011 -- a pure constant => the only offset is _NORM's mass.
# ACHILLES's 2*xmn/hbarc uses xmn = the NEUTRON mass (one_body.f90:3, 939.566), not the (mp+mn)/2
# average -- so use C.mn here.  This removes the 0.11% (= (mn/mN_avg)^2) amps2 deficit; the absolute
# scale is DERIVED, not tuned.
_FRESV = C.Vud * C.ee / (C.sw * np.sqrt(2.0) * 2.0)          # |hadronic CC coupling|
_NORM = 2.0 * np.pi / (_FRESV ** 2 * (2.0 * _conv.norm_m_N()) ** 2)   # neutron mass via conventions
# EM (e,e') RES: the hadronic vertex couples with the PHOTON charge ee (the N->Delta transition FFs are
# in the DCC vector amplitude), replacing the CC hadronic coupling FResV.  TWO differences vs _NORM:
#   1. FResV -> ee   (photon coupling; leptonic side carries -ee*i and i/q^2 via lepton_current kind="EM")
#   2. the CC isospin factor fac*=sqrt(2) (amp_dcc_sl_module.f:275, mode 1-4 ONLY) is ABSENT for EM
#      (mode=10) -> the CC fac^2 baked into _NORM carries an extra 2 that EM must NOT have -> *2 on _NORM_EM.
# Net: _NORM_EM = 2 * _NORM * (FResV^2/ee^2).  Validated against oracle_ee_C_res.
_NORM_EM = 2.0 * (2.0 * np.pi) / (C.ee ** 2 * (2.0 * _conv.norm_m_N()) ** 2)

# Amplitude(W,Q2) interpolation.  DEFAULT = "spline" (bit-faithful to ACHILLES interpolate_amp) -- the
# SAFE default; every reported result must use it.  "bilinear" is NOT W-FAITHFUL (the dsigma/dW shape,
# esp. the high-W tail, deviates well beyond 1%).  bilinear must NEVER be used unless the user has
# EXPLICITLY requested it for a specific purpose.
BATCH_INTERP = "spline"
# Diagnostic override for the AMPLITUDE-INTERNAL pion mass (build_zmtx: qc, pion-pole facpp).
# None -> use the per-channel hPID mass.  Used to determine which m_pi ACHILLES uses in the amplitude.
AMP_MPI_OVERRIDE = None
# The DCC partial-wave amplitude is built with the momentum transfer q as the quantization (z) axis
# (ACHILLES does this via TransformQZ before computing the current). amps2 is a Lorentz scalar but
# this implementation is only correct when q is along +z, so we rotate every event into that frame.
# Callers that already pass q-along-z momenta (e.g. ACHILLES RESDUMP) are unaffected (rotation ~ identity).
ROTATE_QZ = True


def _rotate_q_to_z(mom_list, q):
    """Rotate every event's spatial momenta so q (spatial) points along +z. Returns rotated list."""
    N = q.shape[0]
    qs = q[:, 1:]
    qmag = np.linalg.norm(qs, axis=1, keepdims=True)
    qn = qs / np.where(qmag > 0, qmag, 1.0)
    nx, ny, nz = qn[:, 0], qn[:, 1], qn[:, 2]
    denom = 1.0 + nz
    safe = denom > 1e-9
    d = np.where(safe, denom, 1.0)
    R = np.empty((N, 3, 3))
    R[:, 0, 0] = 1 - nx ** 2 / d; R[:, 0, 1] = -nx * ny / d; R[:, 0, 2] = -nx
    R[:, 1, 0] = -nx * ny / d;    R[:, 1, 1] = 1 - ny ** 2 / d; R[:, 1, 2] = -ny
    R[:, 2, 0] = nx;              R[:, 2, 1] = ny;              R[:, 2, 2] = nz
    R[~safe] = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])  # q along -z -> flip
    return [np.concatenate([p[:, :1], np.einsum('nij,nj->ni', R, p[:, 1:])], axis=1) for p in mom_list]


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
                                 mode=mode, itiz=itiz, m_N=mN, m_pi=_conv.amp_m_pi()))   # (8, npw); centralized convention
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
    # DCC table validity gate
    q = np.asarray(k_nu) - np.asarray(k_mu); Q2 = q[1:] @ q[1:] - q[0] ** 2
    pcm = np.asarray(p_outN) + np.asarray(p_pi); W = np.sqrt(max(pcm[0] ** 2 - pcm[1:] @ pcm[1:], 0.0))
    if W < _W_LO or W > _W_HI or Q2 < 0 or Q2 > _Q2_HI:
        return 0.0
    L = np.asarray(lepton_current(jnp.asarray(k_nu)[None], jnp.asarray(k_mu)[None]))[0]  # (4,4)
    LH = np.einsum('am,bm,m->ab', L, H, _METRIC)
    return float(np.sum(np.abs(LH) ** 2)) / _NORM


# ---- vectorised (batched) version for high-N MC ------------------------------------------ #
import jax
from adonis.channels.dcc_kinematics import boost_matrix_batch, setdfun_batch

_BUILD_ZMTX_V = None


def _build_zmtx_vmapped(vec, isv, axial, W, Q2, itiz, mpi, r_axial=None, pion_pole=1.0, mode=1):
    """Unified amplitude matrix: the batched differential.build_zmtx_batched (the single
    build_zmtx; proven == vmapped assembly.build_zmtx for CC in tests/test_build_zmtx_equiv.py).
    m_N via conventions (amplitude-internal avg mass); m_pi passed in (conventions.amp_m_pi()).
    r_axial (N,) scales the axial amplitudes (M_A reweight hook; None = nominal).
    mode: 1 CC (default) | 10 EM (e,e': axial off, EM isospin) | -1 NC."""
    from adonis.primary.dcc.differential import build_zmtx_batched as _bzb
    return _bzb(vec, isv, axial, W, Q2, _PW_2J, _PW_2L, _PW_2I,
                mode=mode, itiz=itiz, m_N=_conv.amp_m_N(), m_pi=mpi, r_axial=r_axial, pion_pole=pion_pole)


def exclusive_amps2_batch(k_nu, k_mu, p_struck, p_outN, p_pi, itiz, hPID, tcrz=1.0, tm_f=1.0,
                          return_zj=False, q_direct=None, r_axial=None, return_q2=False, knobs=None,
                          pion_pole=1.0, probe="CC"):
    """Vectorised exclusive amps2 over a batch of events (all SAME channel: itiz, hPID).
    Returns amps2 (N,) on the ACHILLES absolute scale (/_NORM).
    return_zj=True returns the lab hadron current zj (N,4_combo,4_mu) instead (DIAGNOSTIC).
    q_direct (N,4): use this q verbatim as the (already de-Forest-shifted) transfer.
    probe: "CC" (default; nu N, weak, mode=1, _NORM) is BIT-IDENTICAL to the pre-EM path.  "EM"
    (inclusive (e,e'): photon leptonic current + i/q^2, DCC mode=10 EM isospin, _NORM_EM)."""
    _mode = 10 if probe == "EM" else 1
    _lep_kind = "EM" if probe == "EM" else "CC_nu"
    _norm = _NORM_EM if probe == "EM" else _NORM
    k_nu = np.asarray(k_nu, float); k_mu = np.asarray(k_mu, float)
    p_struck = np.asarray(p_struck, float); p_outN = np.asarray(p_outN, float); p_pi = np.asarray(p_pi, float)
    N = k_nu.shape[0]; mN = C.mN
    tpiz = {211: 1.0, 111: 0.0, -211: -1.0}[int(hPID)]
    mpi = _conv.amp_m_pi()    # amplitude-internal pion mass = isospin-avg fpio 138.04 (Risk-1 study)
    q = k_nu - k_mu
    if ROTATE_QZ:
        mlist = [k_nu, k_mu, p_struck, p_outN, p_pi]
        if q_direct is not None:
            mlist.append(np.asarray(q_direct, float))
        rot = _rotate_q_to_z(mlist, q)
        k_nu, k_mu, p_struck, p_outN, p_pi = rot[0], rot[1], rot[2], rot[3], rot[4]
        if q_direct is not None:
            q_direct = rot[5]
        q = k_nu - k_mu
    E_on = np.sqrt(np.sum(p_struck[:, 1:] ** 2, axis=1) + mN ** 2)
    if q_direct is not None:
        qsh = np.asarray(q_direct, float)
    else:
        qsh = q.copy(); qsh[:, 0] = q[:, 0] + p_struck[:, 0] - E_on
    pcm = p_outN + p_pi
    xlrs = boost_matrix_batch(pcm, to_cm=True); xlr = boost_matrix_batch(pcm, to_cm=False)
    xk2 = np.einsum('nij,nj->ni', xlrs, p_pi); qx2 = np.einsum('nij,nj->ni', xlrs, qsh)
    def ang(v3):
        r = np.linalg.norm(v3, axis=1); cz = v3[:, 2] / r
        sz = np.sqrt(np.clip(1 - cz ** 2, 0, None))
        zphi = np.where(sz > 1e-12, (v3[:, 0] + 1j * v3[:, 1]) / (np.where(sz > 1e-12, sz, 1) * r), 1.0 + 0j)
        return cz, zphi
    xz_pin, zphi_pin = ang(xk2[:, 1:]); xz_q, zphi_q = ang(qx2[:, 1:])
    wcm = np.sqrt(np.clip(pcm[:, 0] ** 2 - np.sum(pcm[:, 1:] ** 2, axis=1), 1.0, None))
    Q2 = np.sum(qsh[:, 1:] ** 2, axis=1) - qsh[:, 0] ** 2
    dfun, off = setdfun_batch(xz_q, _JMAX)
    bleg = legendre_ylm_batch(_LMAX, xz_pin)                                  # (N, L+1, 2L+1)
    # interp switch (default "spline" = bit-matches ACHILLES interpolate_amp).  "bilinear" is NOT
    # W-faithful (W-SHAPE offender, >1% in the high-W tail) -- NEVER use unless explicitly requested.
    _kn = knobs if knobs is not None else DCCKnobs()    # pw_norm / axial_strength reweight hook (record-build)
    if BATCH_INTERP == "spline":
        vec, isv, axial = _AMP.amplitudes_spline_np(wcm, Q2, _kn)
    else:
        vec, isv, axial = _AMP.amplitudes_bilinear_np(wcm, Q2, _kn)
    amp_mpi = AMP_MPI_OVERRIDE if AMP_MPI_OVERRIDE is not None else mpi
    r_ax = None if r_axial is None else jnp.asarray(r_axial)
    zmtx = np.asarray(_build_zmtx_vmapped(vec, isv, axial, jnp.asarray(wcm), jnp.asarray(Q2), itiz, amp_mpi,
                                          r_axial=r_ax, pion_pole=pion_pole, mode=_mode))  # (N,8,npw)
    tiz = itiz / 2.0; tpinz = tcrz + tiz; tmax = tm_f + 0.5 + _EPS_TPIN
    IGM1 = (-1, 0, 1, 2)
    zcrnt = np.zeros((N, 2, 2, 4), complex)
    for ixi1 in range(1, 9):
        igm1 = int(ISMI[ixi1]); igm1x = int(ISMIX[ixi1]); lambda_N = -int(ISBI[ixi1])
        lam_idx = _ISP[lambda_N]; Lambda_i = 2 * igm1x - lambda_N; ig_idx = IGM1.index(igm1)
        for pw in range(_NPW):
            jpin = int(_PW_2J[pw]); Lpin = int(_PW_2L[pw]); itpin = int(_PW_2I[pw])
            tpin = itpin / 2.0; xlpin = Lpin / 2.0; xjpin = jpin / 2.0; llpin = Lpin // 2
            if not (tpin + _EPS_TPIN > abs(tpinz) and tmax > tpin and jpin >= abs(Lambda_i)):
                continue
            cgi = cbg(1.0, tcrz, 0.5, tiz, tpin, tpinz) * cbg(tm_f, tpiz, 0.5, tpinz - tpiz, tpin, tpinz)
            if cgi == 0:
                continue
            zfac = np.sqrt(jpin + 1.0) * cgi * zmtx[:, ixi1 - 1, pw]          # (N,)
            for isf in (-1, 1):
                isf_idx = _ISP[isf]; xs = isf / 2.0
                zzz = np.zeros(N, complex)
                for mj in range(max(-Lpin + isf, -jpin), min(Lpin + isf, jpin) + 1, 2):
                    xmj = mj / 2.0; llz = (mj - isf) // 2
                    if abs(llz) > llpin:
                        continue
                    zzz += (cbg(xlpin, xmj - xs, 0.5, xs, xjpin, xmj) * bleg[:, llpin, llz + _LMAX]
                            * (zphi_pin ** llz) * dfun[:, jpin, mj + off, Lambda_i + off]
                            * (zphi_q ** ((-mj + Lambda_i) // 2)))
                zcrnt[:, isf_idx, lam_idx, ig_idx] += zfac * zzz
    sq = 1.0 / np.sqrt(2.0)
    zjx = np.zeros((N, 2, 2, 4), complex)
    zjx[:, :, :, 0] = zcrnt[:, :, :, 1]; zjx[:, :, :, 3] = zcrnt[:, :, :, 3]
    zjx[:, :, :, 1] = (zcrnt[:, :, :, 0] - zcrnt[:, :, :, 2]) * sq
    zjx[:, :, :, 2] = (zcrnt[:, :, :, 0] + zcrnt[:, :, :, 2]) * sq * 1j
    # DIAGNOSTIC frame toggle: amps2 is a Lorentz scalar, so contracting in the 2CM frame
    # (no boost on zj, boost the leptons in instead) must equal the lab contraction.
    import os as _os
    if _os.environ.get("ADONIS_CONTRACT_FRAME", "lab") == "cm":
        xlrs = boost_matrix_batch(pcm, to_cm=True)
        knu_c = np.einsum('nmk,nk->nm', xlrs, k_nu); kmu_c = np.einsum('nmk,nk->nm', xlrs, k_mu)
        zj = zjx.reshape(N, 4, 4)                                            # current in 2CM
        L = np.asarray(lepton_current(jnp.asarray(knu_c), jnp.asarray(kmu_c), kind=_lep_kind))
    else:
        zj = np.einsum('nmk,nabk->nabm', xlr, zjx).reshape(N, 4, 4)          # (N, combo, mu)
        L = np.asarray(lepton_current(jnp.asarray(k_nu), jnp.asarray(k_mu), kind=_lep_kind))  # (N,4,4)
    if return_zj:
        return zj                                                            # (N, combo, mu) lab current
    LH = np.einsum('ncm,nbm,m->ncb', L, zj, _METRIC)
    a2 = np.sum(np.abs(LH) ** 2, axis=(1, 2)) / _norm
    gate = (wcm >= _W_LO) & (wcm <= _W_HI) & (Q2 >= 0) & (Q2 <= _Q2_HI)   # DCC table validity
    out = np.where(gate, a2, 0.0)
    if return_q2:
        return out, Q2          # the amplitude-evaluation Q2 [MeV^2] (de-Forest-shifted)
    return out
