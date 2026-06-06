"""Histogram (spectrum) neutrino flux -- loads the experiment flux tables shipped with the
paper release (ACHILLES `Type: Spectrum`, `flux/*.dat`): T2K_nu.dat, MINERvA_cc1pi0_flux.dat,
microboone_NCpi0_*.dat.  Samples E_nu by inverse-CDF of (flux x bin width), matching ACHILLES's
flux-folded event generation; the per-event cross-section weighting is carried by the matrix
element downstream (this only provides the beam-energy proposal).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from adonis.flux.base import FluxModel


def _parse_flux_dat(path):
    """Parse a paper-release flux .dat into (edges[MeV] (nbin+1,), value (nbin,)).

    Handles both layouts seen in the release:
      * T2K_nu.dat:           idx  lo  hi  value     (energies in GeV)
      * microboone/MINERvA:   lo   hi  value         (energies in GeV)
    Non-numeric header lines are skipped.  Bins are assumed contiguous (hi[i]==lo[i+1])."""
    los, his, vals = [], [], []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s[0] in "#":
            continue
        tok = s.split()
        try:
            nums = [float(t) for t in tok]
        except ValueError:
            continue
        if len(nums) >= 4:                      # idx lo hi value
            lo, hi, v = nums[1], nums[2], nums[3]
        elif len(nums) == 3:                    # lo hi value
            lo, hi, v = nums
        else:
            continue
        los.append(lo); his.append(hi); vals.append(v)
    los, his, vals = np.array(los), np.array(his), np.array(vals)
    edges = np.concatenate([los, his[-1:]]) * 1000.0     # GeV -> MeV
    return edges, np.clip(vals, 0.0, None)


class Spectrum(FluxModel):
    """Inverse-CDF sampler over a histogram flux table [MeV]."""

    def __init__(self, path):
        self.path = str(path)
        self.edges, self.value = _parse_flux_dat(path)
        widths = np.diff(self.edges)
        w = self.value * widths                          # probability mass per bin (flux*dE)
        self._p = w / w.sum()
        self._cum = np.concatenate([[0.0], np.cumsum(self._p)])
        # flux-weighted mean energy (representative scalar)
        cen = 0.5 * (self.edges[:-1] + self.edges[1:])
        self._emean = float(np.sum(self._p * cen))
        self._jedges = jnp.asarray(self.edges)
        self._jcum = jnp.asarray(self._cum)

    @property
    def e_nu_nominal(self) -> float:
        return self._emean

    def sample_enu(self, key, n):
        """Sample E_nu (n,) [MeV] ~ flux: pick a bin by its mass, then uniform within the bin."""
        u = jax.random.uniform(key, (n,))
        # bin index b with cum[b] <= u < cum[b+1]
        b = jnp.clip(jnp.searchsorted(self._jcum, u, side="right") - 1, 0, self.edges.size - 2)
        lo = self._jedges[b]; hi = self._jedges[b + 1]
        ku = jax.random.fold_in(key, 1)
        return lo + (hi - lo) * jax.random.uniform(ku, (n,))

    def self_test(self):
        """Mean of sampled energies matches the flux-weighted mean within MC error."""
        key = jax.random.PRNGKey(0)
        e = np.asarray(self.sample_enu(key, 200_000))
        rel = abs(e.mean() - self._emean) / self._emean
        assert rel < 0.02, f"Spectrum flux mean off by {rel:.3f}"
        assert e.min() >= self.edges[0] - 1 and e.max() <= self.edges[-1] + 1
        return True
