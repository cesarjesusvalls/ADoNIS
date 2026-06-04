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

SF_DIR = Path("/Users/cjesus/Software/DiffSinglePiProd/Achilles/data/Spectral_Functions")


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
        self.mom = jnp.asarray(table.mom)
        self.energy = jnp.asarray(table.energy)
        # momentum-magnitude pdf rho(p) ~ p^2 n(p). Build the CDF by the TRAPEZOIDAL
        # rule (mass per interval = (rho_i+rho_{i+1})/2), so inverse-CDF interpolation
        # samples |p| unbiased -- a point-mass cumsum biases |p| ~half a grid-spacing
        # low (the grid is coarse, 20 MeV).
        rho = table.mom ** 2 * np.clip(table.n_p, 0, None)
        self.p_cdf = jnp.asarray(_trapz_cdf(rho))
        # per-momentum energy CDF, trapezoidal; sampled WITH interpolation (the old code
        # snapped E to the nearest grid node -> biased E ~half a 5 MeV spacing high).
        Sclip = np.clip(table.S, 0, None)
        ecdf = np.stack([_trapz_cdf(Sclip[j]) for j in range(Sclip.shape[0])])
        self.e_cdf = jnp.asarray(ecdf)                       # (np, ne), each 0..1

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
        # removal energy E | p via the per-momentum energy CDF (nearest p row), INTERPOLATED
        ue = jax.random.uniform(ke, (n,))
        row = jnp.clip(p_idx, 0, self.e_cdf.shape[0] - 1)
        e_cdf_rows = self.e_cdf[row]                         # (n, ne)
        ei = jax.vmap(lambda c, u: jnp.searchsorted(c, u))(e_cdf_rows, ue)
        ei = jnp.clip(ei, 1, self.energy.shape[0] - 1)
        d0 = jnp.take_along_axis(e_cdf_rows, (ei - 1)[:, None], 1)[:, 0]
        d1 = jnp.take_along_axis(e_cdf_rows, ei[:, None], 1)[:, 0]
        efrac = jnp.clip((ue - d0) / (d1 - d0 + 1e-30), 0, 1)
        E_rm = self.energy[ei - 1] + efrac * (self.energy[ei] - self.energy[ei - 1])
        return jax.lax.stop_gradient(p_vec), jax.lax.stop_gradient(E_rm)
