"""Spectral function S(p, E): the initial-state nucleon momentum/removal-energy distribution.

THE single home for the spectral function (was split across adonis/channels/spectral.py [numpy
production] and this module [jax eager]).  Bit-exact port of ACHILLES SpectralFunction.cc.

S(p,E) is the probability density of a struck nucleon with momentum |p| [MeV] and removal energy
E [MeV].  The struck-nucleon proposal is DETACHED (a fixed kind-1 proposal; the knobs never enter
the sampling, only the elementary-xsec weight, so gradients stay exact).  There is ONE sampler --
`SpectralImportanceSampler` (numpy, |p|,E ~ |p|^2 S) -- used by every stack; the eager/jax callers
(`SpectralSampler.sample(key, n)`, `SpectralFunction.sample_nucleon(key, n)`) wrap its numpy draw in
`jnp.asarray` + `stop_gradient` (the proposal is detached, so the backend is free -- one shared
implementation, no numpy/jax duplication).

Table parse resolves two path conventions: a path with a directory ("data/Spectral_Functions/pke..")
is taken relative to the sibling Achilles/ checkout (the production convention); a bare filename
("pke12p_tot.data") is taken under $ACHILLES_DATA/Spectral_Functions (the eager convention).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from adonis.io import achilles_data_root, achilles_sibling_root
from adonis.nuclear.base import NuclearModel

from adonis.channels import constants as C

_ACH = achilles_sibling_root()   # sibling checkout (production paths); single source: adonis.io
SF_DIR = achilles_data_root() / "Spectral_Functions"             # $ACHILLES_DATA (bare-name paths)


def _resolve_sf_path(filename) -> Path:
    """Two conventions: absolute -> as-is; a path WITH a directory -> Achilles sibling (production,
    e.g. 'data/Spectral_Functions/pke12n_tot.data'); a BARE filename -> $ACHILLES_DATA (eager)."""
    p = Path(filename)
    if p.is_absolute():
        return p
    if p.parent != Path("."):            # has a directory component -> production convention
        return _ACH / filename
    return SF_DIR / filename             # bare name -> eager convention


# interpolation + CDF helpers are generic numerics (not spectral-specific) -> adonis.numerics
from adonis.numerics import polint as _polint, neville_batch as _neville_batch, trapz_cdf as _trapz_cdf


class SpectralFunction(NuclearModel):
    """Faithful ACHILLES SpectralFunction port + NuclearModel.  `__call__(p, E)` / `batch(p, E)` return
    S(p,E) (>=0, 0 outside grid); `sample_nucleon(key, n)` is the detached struck-nucleon proposal (via
    the one `SpectralImportanceSampler`, wrapped for jax)."""

    def __init__(self, filename="pke12p_tot.data", polyX=3, polyY=1):
        path = _resolve_sf_path(filename)
        toks = path.read_text().split()
        it = iter(toks)
        ne = int(next(it)); np_ = int(next(it))
        mom = np.zeros(np_); energy = np.zeros(ne); spec = np.zeros(np_ * ne)
        for j in range(np_):
            mom[j] = float(next(it))
            for i in range(ne):
                energy[i] = float(next(it)); spec[j * ne + i] = float(next(it))
        self.name = str(filename)
        self.mom = mom; self.energy = energy; self.spec = spec  # spec[j*ne+i] = (mom j, energy i)
        self.ne = ne; self.np = np_
        hp = mom[1] - mom[0]; he = energy[1] - energy[0]
        dp_p = np.array([spec[i * ne:(i + 1) * ne].sum() * he for i in range(np_)])
        self.norm = float((mom ** 2 * dp_p * 4 * np.pi * hp).sum())
        self.pox = polyX + 1; self.poy = polyY + 1                # 4, 2
        self._sampler = None                                     # lazy SpectralImportanceSampler

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
        xi = x0[:, None] + np.arange(pox)                              # (N,4)
        yi = y0[:, None] + np.arange(poy)                              # (N,2)
        z = kz[(xi[:, :, None] * nY) + yi[:, None, :]]                 # (N,4,2)
        xk = kx[xi]; yk = ky[yi]                                        # (N,4),(N,2)
        t = (Ec - yk[:, 0]) / (yk[:, 1] - yk[:, 0])
        tmp2 = z[:, :, 0] + (z[:, :, 1] - z[:, :, 0]) * t[:, None]      # (N,4)
        val = _neville_batch(xk, tmp2, pc)
        r = val / self.norm
        return np.where(in_grid & (r > 0), r, 0.0)

    # -- NuclearModel: the detached struck-nucleon proposal (jax-facing, one numpy sampler) --------- #
    def sample_nucleon(self, key, n):
        if self._sampler is None:
            self._sampler = SpectralImportanceSampler(self)
        seed = int(jax.random.randint(key, (), 0, 2 ** 31 - 1))
        pvec, E_rm = self._sampler.sample(n, np.random.default_rng(seed))
        return jax.lax.stop_gradient(jnp.asarray(pvec)), jax.lax.stop_gradient(jnp.asarray(E_rm))

    def oracle_test(self, oracle=None, key=None, n=200_000, ks_max=0.02, **kw):
        """|p| KS distance of the sampled momentum marginal vs the table's inverse-CDF target."""
        from adonis.core.validation import TestResult
        key = jax.random.PRNGKey(0) if key is None else key
        p_vec, E_rm = self.sample_nucleon(key, n)
        pmag = np.sort(np.linalg.norm(np.asarray(p_vec), axis=1))
        F_tab = np.interp(pmag, np.asarray(self._sampler.mom), np.asarray(self._sampler.p_cdf))
        F_emp = np.arange(1, n + 1) / n
        ks = float(np.max(np.abs(F_emp - F_tab)))
        return TestResult(
            f"{type(self).__name__}.oracle", "oracle", bool(ks <= ks_max), False,
            f"|p| KS distance {ks:.4f} vs table (max {ks_max})  "
            f"<E_rm>={float(np.mean(np.asarray(E_rm))):.1f} MeV",
            {"ks_pmag": ks, "ks_max": ks_max, "mean_E_removal": float(np.mean(np.asarray(E_rm)))},
        )


def initial_state_weight(p_mag, E_in, sf: SpectralFunction, n_nucleon):
    """n_nucleon * S(|p|, removal), removal = mqe - E_in  (QESpectral::InitialStateWeight)."""
    removal = C.mN - E_in
    return n_nucleon * sf(p_mag, removal)


class SpectralImportanceSampler:
    """THE spectral sampler (numpy).  Draws (|p|, removal E) ~ |p|^2 S(p,E) from a SpectralFunction's own
    table, so the |p|^2 S flat-MC weight peak cancels in the importance weight (the ADoNIS analog of the
    ACHILLES Vegas struck-nucleon proposal).  Fine inverse-CDF grids (~0.25 MeV E, ~1 MeV |p|) built with
    the SF's OWN Polint (sf.batch) == ACHILLES interpolation."""

    def __init__(self, sf: SpectralFunction):
        mom = np.asarray(sf.mom); ne = sf.ne; np_ = sf.np
        Eraw = np.asarray(sf.energy)
        Ef = np.linspace(Eraw[0], Eraw[-1], max(int((Eraw[-1] - Eraw[0]) / 0.25) + 1, len(Eraw)))
        pf = np.linspace(mom[0], mom[-1], max(int((mom[-1] - mom[0]) / 1.0) + 1, np_))
        PP, EE = np.meshgrid(pf, Ef, indexing="ij")
        Sff = np.clip(sf.batch(PP.ravel(), EE.ravel()).reshape(len(pf), len(Ef)), 0, None)
        n_p = Sff.sum(axis=1) * (Ef[1] - Ef[0])                  # int S dE  (momentum marginal)
        self.mom = pf; self.energy = Ef
        self.p_cdf = _trapz_cdf(pf ** 2 * n_p)                    # |p| ~ |p|^2 n_p (fine grid)
        self.e_cdf = np.stack([_trapz_cdf(Sff[j]) for j in range(len(pf))])  # E | p (fine grid)

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


# --------------------------------------------------------------------------- #
#  Back-compat shims for the eager/jax callers (one numpy sampler underneath)
# --------------------------------------------------------------------------- #
def load_spectral(name: str = "pke12p_tot.data") -> SpectralFunction:
    """A SpectralFunction IS the parsed table now (kept for the DCC eager callers)."""
    return SpectralFunction(name)


class SpectralSampler:
    """Jax-facing wrapper over the ONE numpy SpectralImportanceSampler: `sample(key, n)` seeds a numpy
    rng from `key`, draws the detached proposal, wraps in jnp + stop_gradient (proposal is detached)."""

    def __init__(self, sf: SpectralFunction):
        self._imp = SpectralImportanceSampler(sf)
        self.mom = jnp.asarray(self._imp.mom); self.energy = jnp.asarray(self._imp.energy)
        self.p_cdf = jnp.asarray(self._imp.p_cdf)

    def sample(self, key, n):
        seed = int(jax.random.randint(key, (), 0, 2 ** 31 - 1))
        pvec, E_rm = self._imp.sample(n, np.random.default_rng(seed))
        return jax.lax.stop_gradient(jnp.asarray(pvec)), jax.lax.stop_gradient(jnp.asarray(E_rm))
