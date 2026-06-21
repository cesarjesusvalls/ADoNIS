"""Spectral function S(p, E): initial-state nucleon momentum/removal-energy
distribution (Phase-2 step 5).

Parses the ACHILLES `pke*` tables (format per src/Achilles/SpectralFunction.cc):
    ne np                                     # n_energy, n_momentum
    for j in range(np):                       # momentum grid
        mom[j]
        for i in range(ne): energy[i] S[j,i]  # energy grid + spectral density

S(p,E) is the probability density of finding a nucleon with momentum |p| [MeV]
and removal energy E [MeV]. The momentum-magnitude pdf is rho(p) = 4 pi p^2 n(p)
with n(p) = int S(p,E) dE; the total normalises to the nucleon count convention.

Provides a DETACHED sampler (struck-nucleon |p|, isotropic direction, removal
energy E) used as the fixed proposal in the reweighting fold -- the knobs never
enter the sampling, only the elementary-xsec weight, so gradients stay exact.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from adonis.paths import achilles_data_root
from adonis.xsec.spectral import _neville_batch          # the validated ACHILLES Polint (Neville)

SF_DIR = achilles_data_root() / "Spectral_Functions"


def _polint_SE(mom, energy, S, pf, Ef):
    """S(pf, Ef) via the ACHILLES Polint -- cubic (4pt) in |p|, linear (2pt) in E -- on the fine
    output grids.  S is the raw (np, ne) table.  Mirrors adonis/xsec/spectral.py SpectralFunction.batch
    (same window selection + Neville), so the importance sampler reproduces ACHILLES's S exactly."""
    mom = np.asarray(mom); energy = np.asarray(energy); S = np.asarray(S)
    nx, ny = len(mom), len(energy); pox, poy = 4, 2
    PP, EE = np.meshgrid(pf, Ef, indexing="ij"); p = PP.ravel(); E = EE.ravel()
    pc = np.clip(p, mom[0], mom[-1]); Ec = np.clip(E, energy[0], energy[-1])
    ix = np.clip(np.searchsorted(mom, pc, side="left"), pox // 2, nx - (pox // 2 + pox % 2))
    iy = np.clip(np.searchsorted(energy, Ec, side="left"), poy // 2, ny - (poy // 2 + poy % 2))
    xi = (ix - pox // 2)[:, None] + np.arange(pox); yi = (iy - poy // 2)[:, None] + np.arange(poy)
    z = S[xi[:, :, None], yi[:, None, :]]                       # (N,4,2)
    xk = mom[xi]; yk = energy[yi]
    t = (Ec - yk[:, 0]) / (yk[:, 1] - yk[:, 0])
    tmp2 = z[:, :, 0] + (z[:, :, 1] - z[:, :, 0]) * t[:, None]  # linear in E
    val = _neville_batch(xk, tmp2, pc)                          # cubic in |p|
    return np.clip(val, 0, None).reshape(len(pf), len(Ef))


@dataclass(frozen=True)
class SpectralTable:
    mom: np.ndarray        # (np,) momentum grid [MeV]
    energy: np.ndarray     # (ne,) removal-energy grid [MeV]
    S: np.ndarray          # (np, ne) spectral density
    n_p: np.ndarray        # (np,) momentum distribution n(p) = int S dE
    norm: float            # 4pi int p^2 n(p) dp


def load_spectral(name: str = "pke12p_tot.data") -> SpectralTable:
    toks = np.fromstring(open(SF_DIR / name).read().replace("E", "e"), sep=" ")
    ne, npm = int(toks[0]), int(toks[1])
    rest = toks[2:]
    mom = np.empty(npm)
    energy = np.empty(ne)
    S = np.empty((npm, ne))
    k = 0
    for j in range(npm):
        mom[j] = rest[k]; k += 1
        for i in range(ne):
            energy[i] = rest[k]; S[j, i] = rest[k + 1]; k += 2
    he = energy[1] - energy[0]
    hp = mom[1] - mom[0]
    n_p = S.sum(axis=1) * he
    norm = float(np.sum(mom ** 2 * n_p) * 4 * np.pi * hp)
    return SpectralTable(mom, energy, S, n_p, norm)


def _trapz_cdf(rho):
    """Normalised cumulative-trapezoid CDF of a tabulated density `rho` on a uniform
    grid: N points, CDF[0]=0, CDF[-1]=1 (grid spacing cancels in the normalisation)."""
    rho = np.clip(np.asarray(rho, dtype=float), 0.0, None)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (rho[:-1] + rho[1:]))])
    tot = cdf[-1]
    return cdf / tot if tot > 0 else np.linspace(0.0, 1.0, len(rho))


class SpectralSampler:
    """Differentiable-friendly (detached) sampler over the spectral function."""

    def __init__(self, table: SpectralTable):
        self.t = table
        # FINE grids for BOTH inverse-CDFs.  The raw table is coarse (E 5 MeV around the sharp shell
        # removal peak ~15 MeV; |p| 20 MeV); a trapz CDF + linear inverse-interp on it assumes CONSTANT
        # density within each coarse bin while S / |p|^2 n_p are sloped, so it smears the peak (under-
        # samples low E_rm) and biases |p|.  Rebuild on ~0.25 MeV (E) and ~1 MeV (|p|) grids, linear-
        # interpolating S onto them, so the inverse-CDFs reproduce |p|^2 S(p,E) -- unbiased, matching
        # ACHILLES's per-point S-weighted draw.  (Same fix as adonis/xsec/spectral.py.)
        mom = np.asarray(table.mom); Eraw = np.asarray(table.energy); S = np.clip(table.S, 0, None)
        Ef = np.linspace(Eraw[0], Eraw[-1], max(int((Eraw[-1] - Eraw[0]) / 0.25) + 1, len(Eraw)))
        pf = np.linspace(mom[0], mom[-1], max(int((mom[-1] - mom[0]) / 1.0) + 1, len(mom)))
        Sff = _polint_SE(mom, Eraw, S, pf, Ef)                  # cubic-p (4pt) + linear-E (2pt) == ACHILLES Polint
        n_p = Sff.sum(axis=1) * (Ef[1] - Ef[0])
        self.mom = jnp.asarray(pf); self.energy = jnp.asarray(Ef)
        self.p_cdf = jnp.asarray(_trapz_cdf(pf ** 2 * n_p))                          # |p| ~ |p|^2 n_p
        self.e_cdf = jnp.asarray(np.stack([_trapz_cdf(Sff[j]) for j in range(len(pf))]))  # E | p

    def sample(self, key, n):
        """Return (p_vec [n,3] MeV, E_removal [n] MeV), all detached draws."""
        kp, kth, kph, ke = jax.random.split(key, 4)
        # |p| via trapezoidal inverse-CDF interpolation
        up = jax.random.uniform(kp, (n,))
        p_idx = jnp.clip(jnp.searchsorted(self.p_cdf, up), 1, self.mom.shape[0] - 1)
        c0, c1 = self.p_cdf[p_idx - 1], self.p_cdf[p_idx]
        frac = jnp.clip((up - c0) / (c1 - c0 + 1e-30), 0, 1)
        p_mag = self.mom[p_idx - 1] + frac * (self.mom[p_idx] - self.mom[p_idx - 1])
        # isotropic direction
        cth = 2 * jax.random.uniform(kth, (n,)) - 1
        sth = jnp.sqrt(jnp.clip(1 - cth ** 2, 0, 1))
        phi = 2 * jnp.pi * jax.random.uniform(kph, (n,))
        p_vec = jnp.stack([p_mag * sth * jnp.cos(phi),
                           p_mag * sth * jnp.sin(phi),
                           p_mag * cth], axis=-1)
        # removal energy E | p: interpolate the CONDITIONAL between the two bracketing
        # momentum rows (p_idx-1, p_idx) by the same `frac`. The p-E correlation is steep
        # in the high-|p| tail, so snapping to the upper row biases E high there (quantile
        # interpolation: blend the two inverse-CDF draws -> <E|p> linear in |p|).
        ue = jax.random.uniform(ke, (n,))
        E_lo = self._einterp(self.e_cdf[p_idx - 1], ue)
        E_hi = self._einterp(self.e_cdf[p_idx], ue)
        E_rm = (1 - frac) * E_lo + frac * E_hi
        return jax.lax.stop_gradient(p_vec), jax.lax.stop_gradient(E_rm)

    def _einterp(self, cdf_rows, ue):
        """Interpolated inverse-CDF draw of removal energy on the per-event rows."""
        ei = jax.vmap(lambda c, u: jnp.searchsorted(c, u))(cdf_rows, ue)
        ei = jnp.clip(ei, 1, self.energy.shape[0] - 1)
        d0 = jnp.take_along_axis(cdf_rows, (ei - 1)[:, None], 1)[:, 0]
        d1 = jnp.take_along_axis(cdf_rows, ei[:, None], 1)[:, 0]
        ef = jnp.clip((ue - d0) / (d1 - d0 + 1e-30), 0, 1)
        return self.energy[ei - 1] + ef * (self.energy[ei] - self.energy[ei - 1])


# --------------------------------------------------------------------------- #
#  NuclearModel wrapper (swappable interface)
# --------------------------------------------------------------------------- #
from adonis.nuclear.base import NuclearModel       # noqa: E402


class SpectralFunction(NuclearModel):
    """Spectral-function nuclear model S(p,E); caches the parsed table so repeated
    sampling does not re-read the file.  Draws are bit-identical to the bare
    SpectralSampler used previously (same key -> same nucleon)."""

    def __init__(self, name: str = "pke12p_tot.data"):
        self.name = name
        self.table = load_spectral(name)
        self.sampler = SpectralSampler(self.table)

    def sample_nucleon(self, key, n):
        return self.sampler.sample(key, n)

    # -- per-module self-tests ------------------------------------------------- #
    # closure_test inherited (skipped: a detached sampler has no differentiable
    # parameter).  oracle_test checks the sampled momentum marginal reproduces the
    # input spectral-function table -- the table IS the reference physics here.
    def oracle_test(self, oracle=None, key=None, n=200_000, ks_max=0.02, **kw):
        from adonis.core.validation import TestResult
        key = jax.random.PRNGKey(0) if key is None else key
        p_vec, E_rm = self.sample_nucleon(key, n)
        pmag = np.sort(np.linalg.norm(np.asarray(p_vec), axis=1))
        # KS distance between the empirical |p| CDF and the table's inverse-CDF target
        F_tab = np.interp(pmag, np.asarray(self.table.mom), np.asarray(self.sampler.p_cdf))
        F_emp = np.arange(1, n + 1) / n
        ks = float(np.max(np.abs(F_emp - F_tab)))
        passed = ks <= ks_max
        return TestResult(
            f"{type(self).__name__}.oracle", "oracle", bool(passed), False,
            f"|p| KS distance {ks:.4f} vs table (max {ks_max})  "
            f"<E_rm>={float(np.mean(np.asarray(E_rm))):.1f} MeV",
            {"ks_pmag": ks, "ks_max": ks_max, "mean_E_removal": float(np.mean(np.asarray(E_rm)))},
        )
