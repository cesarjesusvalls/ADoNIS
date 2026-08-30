"""Exclusive ACHILLES DCC RES hadron current (amp_dcc_sl.f::amplitude, irot_q=1) -> amps2.

Computes the piN current zj_mu(isf, lam) at the sampled pion momentum (vs. the inclusive
angle-integrated tensor in adonis/primary/dcc), then contracts with the exact leptonic current.
The two unitary spin-rotation blocks of amplitude() cancel in amps2 (Frobenius norm over the 2x2
spin indices) and are skipped. The /fm scaling cancels in the kinematics (angles, wcm, Q2 all in
MeV) and in fac*(2xmn/hbarc); the residual overall constant (table normalisation x 2sqrt2 mN x
coupling) is constant across events.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from adonis.channels import constants as C
from adonis.channels.probes import probe_spec
from adonis.channels.dcc import conventions as _conv
from adonis.channels.currents.leptonic import lepton_current
from adonis.channels.dcc.wigner import boost_matrix, setdfun
from adonis.channels.dcc.angular import cbg, legendre_ylm, legendre_ylm_batch, ISMI, ISMIX, ISBI
from adonis.channels.dcc.assembly import build_zmtx
from adonis.channels.dcc.amplitudes import DCCAmplitudes, DCCKnobs
from adonis.channels.dcc.loader import load_cached

class _Tables:
    __slots__ = ("amp", "T", "pw_2J", "pw_2L", "pw_2I", "npw", "jmax")


_TBL = None


def _tbl():
    global _TBL
    if _TBL is None:
        t = load_cached()
        n = _Tables()
        n.amp = DCCAmplitudes(t); n.T = t
        n.pw_2J = np.asarray(t.pw_2J); n.pw_2L = np.asarray(t.pw_2L); n.pw_2I = np.asarray(t.pw_2I)
        n.npw = len(n.pw_2J); n.jmax = int(n.pw_2J.max())
        _TBL = n
    return _TBL

_DELTA_WAVE = 5
_LMAX = 5
_ISP = {-1: 1, 1: 0}
_EPS_TPIN = 1e-3
_METRIC = np.array([1.0, -1.0, -1.0, -1.0])
_W_LO = 1076.957; _W_HI = 2000.0; _Q2_HI = 5.0e6
_FRESV = C.Vud * C.ee / (C.sw * np.sqrt(2.0) * 2.0)
_NORM = 2.0 * np.pi / (_FRESV ** 2 * (2.0 * _conv.norm_m_N()) ** 2)
_NORM_EM = 2.0 * (2.0 * np.pi) / (C.ee ** 2 * (2.0 * _conv.norm_m_N()) ** 2)
_FRESV_NC = C.ee / (2.0 * C.sw * C.cw)
_NORM_NC = 2.0 * (2.0 * np.pi) / (_FRESV_NC ** 2 * (2.0 * _conv.norm_m_N()) ** 2)
_NORMS = {"CC": _NORM, "EM": _NORM_EM, "NC": _NORM_NC}

BATCH_INTERP = "spline"
AMP_MPI_OVERRIDE = None
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
    R[~safe] = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
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


def exclusive_H(k_nu, k_lep, p_struck, p_outN, p_pi, itiz, mode=1, tcrz=1.0, tpiz=0.0, tm_f=1.0):
    """Hadron current H[combo(isf,lam), mu] (4,4) complex up to the overall constant, lab frame."""
    mN = C.mN
    q = np.asarray(k_nu) - np.asarray(k_lep)
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
    dfun, off = setdfun(xz_q, _tbl().jmax)
    bleg = np.asarray(legendre_ylm(_LMAX, xz_pin))
    zphi_pins = {l: zphi_pin ** l for l in range(-_LMAX, _LMAX + 1)}

    vec, isv, axial = _tbl().amp.amplitudes_spline(jnp.array([wcm]), jnp.array([Q2]), DCCKnobs())
    zmtx = np.asarray(build_zmtx(vec[0], isv[0], axial[0], wcm, Q2, _tbl().pw_2J, _tbl().pw_2L, _tbl().pw_2I,
                                 mode=mode, itiz=itiz, m_N=mN, m_pi=_conv.amp_m_pi()))
    tiz = itiz / 2.0; tpinz = tcrz + tiz
    tmax = tm_f + 0.5 + _EPS_TPIN

    zcrnt = np.zeros((2, 2, 4), complex)
    IGM1 = (-1, 0, 1, 2)
    for ixi1 in range(1, 9):
        igm1 = int(ISMI[ixi1]); igm1x = int(ISMIX[ixi1]); lambda_N = -int(ISBI[ixi1])
        lam_idx = _ISP[lambda_N]; Lambda_i = 2 * igm1x - lambda_N
        ig_idx = IGM1.index(igm1)
        for pw in range(_tbl().npw):
            jpin = int(_tbl().pw_2J[pw]); Lpin = int(_tbl().pw_2L[pw]); itpin = int(_tbl().pw_2I[pw])
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

    sq = 1.0 / np.sqrt(2.0)
    zjx = np.zeros((2, 2, 4), complex)
    zjx[:, :, 0] = zcrnt[:, :, 1]
    zjx[:, :, 3] = zcrnt[:, :, 3]
    zjx[:, :, 1] = (zcrnt[:, :, 0] - zcrnt[:, :, 2]) * sq
    zjx[:, :, 2] = (zcrnt[:, :, 0] + zcrnt[:, :, 2]) * sq * 1j
    zj = np.einsum('mn,abn->abm', xlr, zjx)
    return zj.reshape(4, 4)


def exclusive_amps2(k_nu, k_lep, p_struck, p_outN, p_pi, itiz, hPID):
    """amps2 up to the overall constant: sum_{isf,lam} |L.H|^2."""
    tpiz = {211: 1.0, 111: 0.0, -211: -1.0}[int(hPID)]
    H = exclusive_H(k_nu, k_lep, p_struck, p_outN, p_pi, itiz, tpiz=tpiz)
    q = np.asarray(k_nu) - np.asarray(k_lep); Q2 = q[1:] @ q[1:] - q[0] ** 2
    pcm = np.asarray(p_outN) + np.asarray(p_pi); W = np.sqrt(max(pcm[0] ** 2 - pcm[1:] @ pcm[1:], 0.0))
    if W < _W_LO or W > _W_HI or Q2 < 0 or Q2 > _Q2_HI:
        return 0.0
    L = np.asarray(lepton_current(jnp.asarray(k_nu)[None], jnp.asarray(k_lep)[None]))[0]
    LH = np.einsum('am,bm,m->ab', L, H, _METRIC)
    return float(np.sum(np.abs(LH) ** 2)) / _NORM


import jax
from adonis.channels.dcc.wigner import boost_matrix_batch, setdfun_batch

_BUILD_ZMTX_V = None


def _build_zmtx_vmapped(vec, isv, axial, W, Q2, itiz, mpi, r_axial=None, pion_pole=1.0, mode=1):
    """Batched amplitude matrix via differential.build_zmtx_batched (matches assembly.build_zmtx
    event-by-event for CC). m_N via conventions (amplitude-internal avg mass); m_pi passed in
    (conventions.amp_m_pi()). r_axial (N,) scales the axial amplitudes (M_A reweight hook; None =
    nominal). mode: 1 CC (default) | 10 EM (e,e': axial off, EM isospin) | -1 NC."""
    from adonis.channels.dcc.differential import build_zmtx_batched as _bzb
    return _bzb(vec, isv, axial, W, Q2, _tbl().pw_2J, _tbl().pw_2L, _tbl().pw_2I,
                mode=mode, itiz=itiz, m_N=_conv.amp_m_N(), m_pi=mpi, r_axial=r_axial, pion_pole=pion_pole)


def _zmtx_to_zjx(zmtx, N, itiz, tcrz, tm_f, tpiz, bleg, zphi_pin, dfun, off, zphi_q):
    """The LINEAR zmtx -> 2CM current zjx (N,2,2,4) map -- the angular assembly, shared by the summed
    current and the per-structure unit currents (return_structures), so the two cannot drift."""
    tiz = itiz / 2.0; tpinz = tcrz + tiz; tmax = tm_f + 0.5 + _EPS_TPIN
    IGM1 = (-1, 0, 1, 2)
    zcrnt = np.zeros((N, 2, 2, 4), complex)
    for ixi1 in range(1, 9):
        igm1 = int(ISMI[ixi1]); igm1x = int(ISMIX[ixi1]); lambda_N = -int(ISBI[ixi1])
        lam_idx = _ISP[lambda_N]; Lambda_i = 2 * igm1x - lambda_N; ig_idx = IGM1.index(igm1)
        for pw in range(_tbl().npw):
            jpin = int(_tbl().pw_2J[pw]); Lpin = int(_tbl().pw_2L[pw]); itpin = int(_tbl().pw_2I[pw])
            tpin = itpin / 2.0; xlpin = Lpin / 2.0; xjpin = jpin / 2.0; llpin = Lpin // 2
            if not (tpin + _EPS_TPIN > abs(tpinz) and tmax > tpin and jpin >= abs(Lambda_i)):
                continue
            cgi = cbg(1.0, tcrz, 0.5, tiz, tpin, tpinz) * cbg(tm_f, tpiz, 0.5, tpinz - tpiz, tpin, tpinz)
            if cgi == 0:
                continue
            zfac = np.sqrt(jpin + 1.0) * cgi * zmtx[:, ixi1 - 1, pw]
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
    return zjx


def exclusive_amps2_batch(k_nu, k_lep, p_struck, p_outN, p_pi, itiz, hPID, tcrz=1.0, tm_f=1.0,
                          return_zj=False, q_direct=None, r_axial=None, return_q2=False, knobs=None,
                          pion_pole=1.0, probe="CC", return_structures=False):
    """Vectorised exclusive amps2 over a batch of events (all same channel: itiz, hPID).
    Returns amps2 (N,) on the ACHILLES absolute scale (/_NORM).
    return_zj=True returns the lab hadron current zj (N,4_combo,4_mu) instead (diagnostic).
    q_direct (N,4): use this q verbatim as the (already de-Forest-shifted) transfer.
    probe: "CC" (default; nu N, weak, mode=1, _NORM). "EM" (inclusive (e,e'): photon leptonic
    current + i/q^2, DCC mode=10 EM isospin, _NORM_EM)."""
    _spec = probe_spec(probe)
    _mode = _spec.dcc_mode
    _lep_kind = _spec.lep_kind
    _norm = _NORMS[probe]
    k_nu = np.asarray(k_nu, float); k_lep = np.asarray(k_lep, float)
    p_struck = np.asarray(p_struck, float); p_outN = np.asarray(p_outN, float); p_pi = np.asarray(p_pi, float)
    N = k_nu.shape[0]; mN = C.mN
    tpiz = {211: 1.0, 111: 0.0, -211: -1.0}[int(hPID)]
    mpi = _conv.amp_m_pi()
    q = k_nu - k_lep
    if ROTATE_QZ:
        mlist = [k_nu, k_lep, p_struck, p_outN, p_pi]
        if q_direct is not None:
            mlist.append(np.asarray(q_direct, float))
        rot = _rotate_q_to_z(mlist, q)
        k_nu, k_lep, p_struck, p_outN, p_pi = rot[0], rot[1], rot[2], rot[3], rot[4]
        if q_direct is not None:
            q_direct = rot[5]
        q = k_nu - k_lep
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
    dfun, off = setdfun_batch(xz_q, _tbl().jmax)
    bleg = legendre_ylm_batch(_LMAX, xz_pin)
    _kn = knobs if knobs is not None else DCCKnobs()
    if BATCH_INTERP == "spline":
        vec, isv, axial = _tbl().amp.amplitudes_spline_np(wcm, Q2, _kn)
    else:
        vec, isv, axial = _tbl().amp.amplitudes_bilinear_np(wcm, Q2, _kn)
    amp_mpi = AMP_MPI_OVERRIDE if AMP_MPI_OVERRIDE is not None else mpi
    r_ax = None if r_axial is None else jnp.asarray(r_axial)
    zmtx = np.asarray(_build_zmtx_vmapped(vec, isv, axial, jnp.asarray(wcm), jnp.asarray(Q2), itiz, amp_mpi,
                                          r_axial=r_ax, pion_pole=pion_pole, mode=_mode))
    ang = (N, itiz, tcrz, tm_f, tpiz, bleg, zphi_pin, dfun, off, zphi_q)

    if return_structures:
        w5 = _DELTA_WAVE if _DELTA_WAVE < _tbl().npw else None
        def _wmask(a, keep5):
            out = np.array(a)
            if w5 is not None:
                if keep5:
                    out[:, :, [i for i in range(_tbl().npw) if i != w5]] = 0
                else:
                    out[:, :, w5] = 0
            return out
        zero = np.zeros_like(vec)
        def _zmtx(v, s, ax, pole):
            return np.asarray(_build_zmtx_vmapped(v, s, ax, jnp.asarray(wcm), jnp.asarray(Q2), itiz, amp_mpi,
                                                  r_axial=jnp.ones(N), pion_pole=pole, mode=_mode))
        atoms = {}
        for tag, keep5 in (("rest", False), ("w5", True)):
            vv, ss, aa = _wmask(vec, keep5), _wmask(isv, keep5), _wmask(axial, keep5)
            zV = _zmtx(vv, ss, zero, 0.0)
            zA = _zmtx(zero, zero, aa, 0.0)
            zAP = _zmtx(zero, zero, aa, 1.0)
            atoms[f"V_{tag}"] = _zmtx_to_zjx(zV, *ang)
            atoms[f"A_{tag}"] = _zmtx_to_zjx(zA, *ang)
            atoms[f"P_{tag}"] = _zmtx_to_zjx(zAP - zA, *ang)
        L = np.asarray(lepton_current(jnp.asarray(k_nu), jnp.asarray(k_lep), kind=_lep_kind))
        gate = (wcm >= _W_LO) & (wcm <= _W_HI) & (Q2 >= 0) & (Q2 <= _Q2_HI)
        zjs = {k: np.einsum('nmk,nabk->nabm', xlr, v).reshape(N, 4, 4) for k, v in atoms.items()}
        return dict(zj=zjs, L=L, gate=gate, Q2=Q2, norm=_norm)

    zjx = _zmtx_to_zjx(zmtx, *ang)
    zj = np.einsum('nmk,nabk->nabm', xlr, zjx).reshape(N, 4, 4)
    L = np.asarray(lepton_current(jnp.asarray(k_nu), jnp.asarray(k_lep), kind=_lep_kind))
    if return_zj:
        return zj
    LH = np.einsum('ncm,nbm,m->ncb', L, zj, _METRIC)
    a2 = np.sum(np.abs(LH) ** 2, axis=(1, 2)) / _norm
    gate = (wcm >= _W_LO) & (wcm <= _W_HI) & (Q2 >= 0) & (Q2 <= _Q2_HI)
    out = np.where(gate, a2, 0.0)
    if return_q2:
        return out, Q2
    return out
