"""Bit-exact port of ACHILLES SpectralFunction.cc (the QE/RES InitialStateWeight S(p,E)).

norm = sum_i mom[i]^2 dp_p[i] 4pi hp,  dp_p[i] = sum_j S[i,j] he ;  S(p,E)=func(p,E)/norm clamped
>=0, 0 outside grid.  func = Interp2D polynomial order (3,1): cubic (4-pt) in p, linear (2-pt) in
E, via Polint (Numerical-Recipes Neville).  InitialStateWeight = n_nucleon * S(|p|, removal),
removal = mqe - E_in (mqe = 0.5(mp+mn)).  Evaluated in NumPy (detached: the nuclear input is not a
differentiable knob).  Validated bit-exact vs the instrumented-ACHILLES initwgt dump.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from adonis.xsec import constants as C

_ACH = Path(__file__).resolve().parents[2].parent / "Achilles"


def _polint(xa, ya, x):
    """Numerical-Recipes Polint (Neville), faithful to Interpolation.cc::Polint."""
    n = len(xa)
    c = np.array(ya, float); d = np.array(ya, float)
    dif = abs(x - xa[0]); ns = 0
    for i in range(n):
        dift = abs(x - xa[i])
        if dift < dif:
            ns = i; dif = dift
    y = ya[ns]; ns -= 1
    for m in range(n - 1):
        for i in range(n - m - 1):
            ho = xa[i] - x; hp = xa[i + m + 1] - x
            w = c[i + 1] - d[i]; den = ho - hp
            if den == 0:
                raise RuntimeError("Polint: zero denominator")
            den = w / den
            d[i] = hp * den; c[i] = ho * den
        if 2 * ns + 1 < (n - m - 1):
            y += c[ns + 1]
        else:
            y += d[ns]; ns -= 1
    return y


def _neville_batch(xa, ya, x):
    """Vectorised Neville (Polint) over the last axis (n points) for a batch.  xa,ya (N,n)."""
    n = xa.shape[1]
    c = ya.astype(float).copy(); d = ya.astype(float).copy()
    dif = np.abs(x[:, None] - xa); ns = np.argmin(dif, axis=1)          # nearest point
    y = ya[np.arange(len(x)), ns].astype(float); ns = ns - 1
    for m in range(n - 1):
        for i in range(n - m - 1):
            ho = xa[:, i] - x; hp = xa[:, i + m + 1] - x
            w = c[:, i + 1] - d[:, i]; den = ho - hp
            den = w / den
            d[:, i] = hp * den; c[:, i] = ho * den
        take_c = (2 * ns + 1) < (n - m - 1)
        idx = np.clip(ns + 1, 0, n - 1)
        y = y + np.where(take_c, c[np.arange(len(x)), idx], d[np.arange(len(x)), np.clip(ns, 0, n - 1)])
        ns = np.where(take_c, ns, ns - 1)
    return y


class SpectralFunction:
    """Faithful port; __call__(p, E) returns S(p,E) (>=0, 0 outside grid)."""

    def __init__(self, filename, polyX=3, polyY=1):
        path = _ACH / filename if not Path(filename).is_absolute() else Path(filename)
        toks = path.read_text().split()
        it = iter(toks)
        ne = int(next(it)); np_ = int(next(it))
        mom = np.zeros(np_); energy = np.zeros(ne); spec = np.zeros(np_ * ne)
        for j in range(np_):
            mom[j] = float(next(it))
            for i in range(ne):
                energy[i] = float(next(it)); spec[j * ne + i] = float(next(it))
        self.mom = mom; self.energy = energy; self.spec = spec  # spec[j*ne+i] = (mom j, energy i)
        self.ne = ne; self.np = np_
        hp = mom[1] - mom[0]; he = energy[1] - energy[0]
        dp_p = np.array([spec[i * ne:(i + 1) * ne].sum() * he for i in range(np_)])
        self.norm = float((mom ** 2 * dp_p * 4 * np.pi * hp).sum())
        self.pox = polyX + 1; self.poy = polyY + 1                # 4, 2

    def _interp2d(self, x, y):
        kx, ky, kz = self.mom, self.energy, self.spec
        nY = self.ne; pox, poy = self.pox, self.poy
        idxX = int(np.searchsorted(kx, x, side="left"))
        while idxX < pox // 2: idxX += 1
        while len(kx) - idxX < pox // 2 + pox % 2: idxX -= 1
        xI = kx[idxX - pox // 2: idxX - pox // 2 + pox]
        idxY = int(np.searchsorted(ky, y, side="left"))
        while idxY < poy // 2: idxY += 1
        while len(ky) - idxY < poy // 2 + poy % 2: idxY -= 1
        yI = ky[idxY - poy // 2: idxY - poy // 2 + poy]
        tmp2 = np.zeros(pox)
        for i in range(pox):
            tmp = np.array([kz[idxY + j - poy // 2 + nY * (idxX - pox // 2 + i)] for j in range(poy)])
            tmp2[i] = _polint(yI, tmp, y)
        return _polint(xI, tmp2, x)

    def __call__(self, p, E):
        if p < self.mom[0] or p > self.mom[-1] or E < self.energy[0] or E > self.energy[-1]:
            return 0.0
        r = self._interp2d(p, E) / self.norm
        return r if r > 0 else 0.0

    def batch(self, p, E):
        """Vectorised S(p,E) over arrays (faithful to the per-event Polint, same clamps).
        Cubic (4-pt) in p, linear (2-pt) in E."""
        p = np.asarray(p, float); E = np.asarray(E, float)
        kx, ky, kz = self.mom, self.energy, self.spec
        nY = self.ne; pox, poy = self.pox, self.poy           # 4, 2
        in_grid = (p >= kx[0]) & (p <= kx[-1]) & (E >= ky[0]) & (E <= ky[-1])
        pc = np.clip(p, kx[0], kx[-1]); Ec = np.clip(E, ky[0], ky[-1])
        ix = np.searchsorted(kx, pc, side="left")
        ix = np.clip(ix, pox // 2, len(kx) - (pox // 2 + pox % 2))     # window fits -> start ix-2
        iy = np.searchsorted(ky, Ec, side="left")
        iy = np.clip(iy, poy // 2, len(ky) - (poy // 2 + poy % 2))     # start iy-1
        x0 = ix - pox // 2; y0 = iy - poy // 2                          # window starts
        # gather z[ p in x0..x0+3, E in y0..y0+1 ] -> (N,4,2)
        xi = x0[:, None] + np.arange(pox)[None, :]                      # (N,4)
        yi = y0[:, None] + np.arange(poy)[None, :]                      # (N,2)
        z = kz[(xi[:, :, None] * nY) + yi[:, None, :]]                  # (N,4,2)
        xk = kx[xi]; yk = ky[yi]                                        # (N,4),(N,2)
        # linear Polint over E (2 pts): tmp2[i] = z0 + (z1-z0)(E-y0)/(y1-y0)
        t = (Ec - yk[:, 0]) / (yk[:, 1] - yk[:, 0])
        tmp2 = z[:, :, 0] + (z[:, :, 1] - z[:, :, 0]) * t[:, None]      # (N,4)
        # cubic Polint over p (4 pts), Neville
        val = _neville_batch(xk, tmp2, pc)
        r = val / self.norm
        return np.where(in_grid & (r > 0), r, 0.0)


def initial_state_weight(p_mag, E_in, sf: SpectralFunction, n_nucleon):
    """n_nucleon * S(|p|, removal), removal = mqe - E_in  (QESpectral::InitialStateWeight)."""
    removal = C.mN - E_in
    return n_nucleon * sf(p_mag, removal)


def _trapz_cdf(rho):
    rho = np.clip(np.asarray(rho, float), 0, None)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (rho[:-1] + rho[1:]))])
    tot = cdf[-1]
    return cdf / tot if tot > 0 else np.linspace(0, 1, len(rho))


class SpectralImportanceSampler:
    """IMPORTANCE sampler over S(p,E) built from a SpectralFunction's own table -- draws
    (|p|, removal E) ~ |p|^2 S(p,E) (the physical distribution).  Used to flatten the dominant
    flat-MC weight variance in the QE/RES integrator (the ADoNIS analog of ACHILLES Vegas): the
    |p|^2 S factor cancels in the importance weight, so the struck-nucleon contributes only a
    constant.  norm = sf.norm (S normalized so int |p|^2 n_p 4pi dp = 1)."""

    def __init__(self, sf: SpectralFunction):
        mom = np.asarray(sf.mom); ne = sf.ne; np_ = sf.np
        S = (np.asarray(sf.spec).reshape(np_, ne)) / sf.norm     # S(p_j, E_i), normalized
        S = np.clip(S, 0, None)
        n_p = S.sum(axis=1) * (sf.energy[1] - sf.energy[0])       # int S dE  (momentum marginal)
        self.mom = mom; self.energy = np.asarray(sf.energy)
        self.p_cdf = _trapz_cdf(mom ** 2 * n_p)                   # |p| ~ |p|^2 n_p
        self.e_cdf = np.stack([_trapz_cdf(S[j]) for j in range(np_)])   # E | p

    def sample(self, n, rng):
        up = rng.random(n)
        pj = np.clip(np.searchsorted(self.p_cdf, up), 1, len(self.mom) - 1)
        c0, c1 = self.p_cdf[pj - 1], self.p_cdf[pj]
        fr = np.clip((up - c0) / (c1 - c0 + 1e-30), 0, 1)
        pmag = self.mom[pj - 1] + fr * (self.mom[pj] - self.mom[pj - 1])
        ue = rng.random(n)
        def einterp(rows):
            ei = np.clip(np.array([np.searchsorted(rows[k], ue[k]) for k in range(n)]), 1, len(self.energy) - 1)
            d0 = rows[np.arange(n), ei - 1]; d1 = rows[np.arange(n), ei]
            ef = np.clip((ue - d0) / (d1 - d0 + 1e-30), 0, 1)
            return self.energy[ei - 1] + ef * (self.energy[ei] - self.energy[ei - 1])
        E_lo = einterp(self.e_cdf[pj - 1]); E_hi = einterp(self.e_cdf[pj])
        E_rm = (1 - fr) * E_lo + fr * E_hi
        ct = 2 * rng.random(n) - 1; st = np.sqrt(np.clip(1 - ct ** 2, 0, None)); ph = 2 * np.pi * rng.random(n)
        pvec = np.stack([pmag * st * np.cos(ph), pmag * st * np.sin(ph), pmag * ct], axis=1)
        return pvec, E_rm
