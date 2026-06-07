"""Frame + angular primitives for the exclusive ACHILLES DCC current (amp_dcc_sl_module.f):
the Lorentz boost `lorentz_trans` and the half-integer Wigner d-functions `setdfun`/`fblmmx`.

These feed the irot_q=1 exclusive zcrnt assembly (the angle-dependent piN current at a specific
pion momentum), which the inclusive (angle-integrated) port in adonis/primary/dcc skips.  Pure
NumPy (detached kinematics; the DCC knobs enter via the amplitude table, not these rotations).
"""
from __future__ import annotations

import numpy as np
from math import factorial

_H = np.array([np.sqrt(float(factorial(n))) for n in range(60)])    # h(n) = sqrt(n!)


def boost_matrix(pcm, to_cm=True):
    """lorentz_trans: boost matrix xlr[mu,nu], pcm (0:3)=(E,px,py,pz).  to_cm=True (i=1, ifac=-1)
    Lab->2CM; to_cm=False (i=2, ifac=+1) 2CM->Lab.  Index 0=time."""
    xm = np.sqrt(pcm[0] ** 2 - pcm[1] ** 2 - pcm[2] ** 2 - pcm[3] ** 2)
    ifac = -1.0 if to_cm else 1.0
    xlr = np.zeros((4, 4))
    rxm = 1.0 / xm
    xlr[0, 1] = pcm[1] * rxm * ifac; xlr[0, 2] = pcm[2] * rxm * ifac; xlr[0, 3] = pcm[3] * rxm * ifac
    xlr[0, 0] = pcm[0] * rxm
    xxx = 1.0 / (xm * (xm + pcm[0]))
    xlr[1, 1] = 1 + pcm[1] ** 2 * xxx; xlr[1, 2] = pcm[1] * pcm[2] * xxx; xlr[1, 3] = pcm[1] * pcm[3] * xxx
    xlr[2, 2] = 1 + pcm[2] ** 2 * xxx; xlr[2, 3] = pcm[2] * pcm[3] * xxx
    xlr[3, 3] = 1 + pcm[3] ** 2 * xxx
    for n in range(4):
        for m in range(n + 1, 4):
            xlr[m, n] = xlr[n, m]
    return xlr


def boost_matrix_batch(pcm, to_cm=True):
    """Vectorised lorentz_trans over a batch: pcm (N,4) -> xlr (N,4,4)."""
    pcm = np.asarray(pcm, float)
    xm = np.sqrt(np.clip(pcm[:, 0] ** 2 - np.sum(pcm[:, 1:] ** 2, axis=1), 1e-9, None))
    ifac = -1.0 if to_cm else 1.0
    N = pcm.shape[0]
    xlr = np.zeros((N, 4, 4))
    rxm = 1.0 / xm
    xlr[:, 0, 1] = pcm[:, 1] * rxm * ifac; xlr[:, 0, 2] = pcm[:, 2] * rxm * ifac
    xlr[:, 0, 3] = pcm[:, 3] * rxm * ifac; xlr[:, 0, 0] = pcm[:, 0] * rxm
    xxx = 1.0 / (xm * (xm + pcm[:, 0]))
    for a in range(1, 4):
        for b in range(a, 4):
            xlr[:, a, b] = (1.0 if a == b else 0.0) + pcm[:, a] * pcm[:, b] * xxx
    for n in range(4):
        for m in range(n + 1, 4):
            xlr[:, m, n] = xlr[:, n, m]
    return xlr


def fblmmx_batch(l, mf, mi, cc, ss):
    """Vectorised integer Wigner small-d over arrays cc, ss (N,).  Returns (N,)."""
    out = np.zeros_like(cc)
    if l < 0 or abs(mf) > l or abs(mi) > l:
        return out
    jmip = l + mi; jmim = l - mi; jmfp = l + mf; jmfm = l - mf
    mfmim = mf - mi; iicos = 2 * l - mfmim; iisin = mfmim
    kmax = min(jmip, jmfm); kmin = max(0, -mfmim)
    if kmax < kmin:
        return out
    for kx in range(kmin, kmax + 1):
        phase = (-1.0) ** (mfmim + kx)
        factor = (_H[jmip] * _H[jmim] * _H[jmfp] * _H[jmfm]
                  / (_H[jmip - kx] * _H[kx] * _H[jmfm - kx] * _H[kx + mfmim]) ** 2)
        out += phase * factor * cc ** (iicos - 2 * kx) * ss ** (iisin + 2 * kx)
    return out


def setdfun_batch(x, jmax):
    """Vectorised setdfun over x (N,): dfun (N, n+1, 2n+1, 2n+1), off.  Half-integer Wigner-d."""
    x = np.asarray(x, float); N = x.shape[0]
    n = 2 * 5 - 1; off = n
    dfun = np.zeros((N, n + 1, 2 * n + 1, 2 * n + 1))
    ss = np.sqrt((1.0 - x) / 2.0); cc = np.sqrt((1.0 + x) / 2.0)
    for lx in range(1, jmax + 1, 2):
        for mf in range(-lx, lx + 1, 2):
            for mi in range(-lx, lx + 1, 2):
                jj = (lx - 1) // 2
                mfm = (mf - 1) // 2; mfp = (mf + 1) // 2; mim = (mi - 1) // 2; mip = (mi + 1) // 2
                df = np.zeros(N)
                if jj >= abs(mfm) and jj >= abs(mim):
                    df += fblmmx_batch(jj, mfm, mim, cc, ss) * np.sqrt((lx + mf) * (lx + mi)) / (2 * lx) * cc
                if jj >= abs(mfp) and jj >= abs(mip):
                    df += fblmmx_batch(jj, mfp, mip, cc, ss) * np.sqrt((lx - mf) * (lx - mi)) / (2 * lx) * cc
                if jj >= abs(mfm) and jj >= abs(mip):
                    df += -fblmmx_batch(jj, mfm, mip, cc, ss) * np.sqrt((lx + mf) * (lx - mi)) / (2 * lx) * ss
                if jj >= abs(mfp) and jj >= abs(mim):
                    df += fblmmx_batch(jj, mfp, mim, cc, ss) * np.sqrt((lx - mf) * (lx + mi)) / (2 * lx) * ss
                dfun[:, lx, mf + off, mi + off] = df
    return dfun, off


def fblmmx(l, mf, mi, cc, ss):
    """Integer Wigner small-d d^l_{mf,mi} via the sqrt(n!) factorial sum (amp_dcc_sl_module.f)."""
    if l < 0 or abs(mf) > l or abs(mi) > l:
        return 0.0
    jmip = l + mi; jmim = l - mi; jmfp = l + mf; jmfm = l - mf
    mfmim = mf - mi
    iicos = 2 * l - mfmim; iisin = mfmim
    kmax = min(jmip, jmfm); kmin = max(0, -mfmim)
    if kmax < kmin:
        return 0.0
    s = 0.0
    for kx in range(kmin, kmax + 1):
        phase = (-1.0) ** (mfmim + kx)
        factor = (_H[jmip] * _H[jmim] * _H[jmfp] * _H[jmfm]
                  / (_H[jmip - kx] * _H[kx] * _H[jmfm - kx] * _H[kx + mfmim]) ** 2)
        s += phase * factor * cc ** (iicos - 2 * kx) * ss ** (iisin + 2 * kx)
    return s


def setdfun(x, jmax):
    """Half-integer Wigner d: returns dfun[lx, mf, mi] for odd lx in 1..jmax, mf,mi in -lx..lx step
    2 (i.e. 2*half-integer indices).  x = cos(theta).  Faithful to setdfun (df1..df4 build)."""
    n = 2 * 5 - 1                                       # 2*njmx-1 = 9
    dfun = np.zeros((n + 1, 2 * n + 1, 2 * n + 1))      # index [lx, mf+off, mi+off]
    off = n
    ss = np.sqrt((1.0 - x) / 2.0); cc = np.sqrt((1.0 + x) / 2.0)
    for lx in range(1, jmax + 1, 2):
        maxm = lx
        for mf in range(-maxm, maxm + 1, 2):
            for mi in range(-maxm, maxm + 1, 2):
                jj = (lx - 1) // 2
                mfm = (mf - 1) // 2; mfp = (mf + 1) // 2
                mim = (mi - 1) // 2; mip = (mi + 1) // 2
                df = 0.0
                if jj >= abs(mfm) and jj >= abs(mim):
                    df += fblmmx(jj, mfm, mim, cc, ss) * np.sqrt((lx + mf) * (lx + mi)) / (2 * lx) * cc
                if jj >= abs(mfp) and jj >= abs(mip):
                    df += fblmmx(jj, mfp, mip, cc, ss) * np.sqrt((lx - mf) * (lx - mi)) / (2 * lx) * cc
                if jj >= abs(mfm) and jj >= abs(mip):
                    df += -fblmmx(jj, mfm, mip, cc, ss) * np.sqrt((lx + mf) * (lx - mi)) / (2 * lx) * ss
                if jj >= abs(mfp) and jj >= abs(mim):
                    df += fblmmx(jj, mfp, mim, cc, ss) * np.sqrt((lx - mf) * (lx + mi)) / (2 * lx) * ss
                dfun[lx, mf + off, mi + off] = df
    return dfun, off
