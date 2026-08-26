"""Spectrum (histogram) neutrino flux -- the single spectrum-flux implementation.

Bit-exact ACHILLES Spectrum flux (Beams.cc Spectrum + T2KHeader), reading ANY paper-release
`flux/*.dat` table (T2K_nu.dat, minerva_numu_fhc.dat, microboone_numu.dat, ...): 2 header lines,
then "bin elo ehi flux" with energies in GeV.  m_energy_units = 1/GeV.  flux_integral =
sum_i (edges[i+1]-edges[i]) * heights[i].  m_flux = linear (Interp1D poly order 1) on padded bin
centres.  m_min/max_energy = edges.front()/back() (GeV).

Beam sampling (single ran u): min = max(seed_GeV, edges[0]); E_GeV = u*(max-min)+min; lab E = 1000*E.
J_beam (the event.Weight() beam factor) = (delta * m_flux(E_GeV)) / flux_integral, delta=max-min.
"""
from __future__ import annotations

import os as _os
from pathlib import Path
import numpy as np

from adonis.numerics import polint as _polint

from adonis.io import achilles_sibling_root
_ACH = achilles_sibling_root()
from adonis.constants import MASS_PDG_MUON as M_MU, MASS_PDG_PROTON as M_P


def _default_flux():
    return _os.environ.get("ADONIS_FLUX_FILE", "flux/T2K_nu.dat")


def _parse_spectrum(path):
    """Parse an ACHILLES Spectrum flux table -> (edges[GeV] (nbin+1,), heights (nbin,)).  Handles the
    release layouts (header/non-numeric lines skipped).  Column count doesn't disambiguate: both
    T2K and MINERvA have 4 cols.  The tell is column 0 -- a sequential 0-based INDEX (T2K) vs the
    lower edge (MINERvA):
      * T2K_nu.dat:          idx  elo  ehi  flux             -> cols [1],[2],[3]
      * MINERvA_*.dat:       elo  ehi  value  error          -> cols [0],[1],[2]  (error dropped)
      * 3-col spectrum:      elo  ehi  value                 -> cols [0],[1],[2]
    Contiguous bins assumed (ehi[i] == elo[i+1]); the final edge is the last ehi."""
    rows = []
    for ln in path.read_text().splitlines():
        try:
            nums = [float(x) for x in ln.split()]
        except ValueError:
            continue
        if len(nums) >= 3:
            rows.append(nums)
    if not rows:
        raise ValueError(f"no numeric flux rows parsed from {path}")
    col0_is_index = all(abs(r[0] - i) < 1e-9 for i, r in enumerate(rows))
    o = 1 if col0_is_index else 0
    los = [r[o] for r in rows]; his = [r[o + 1] for r in rows]; hts = [r[o + 2] for r in rows]
    edges = np.array(los + his[-1:])
    return edges, np.array(hts)


class SpectrumFlux:
    """Piecewise-constant Spectrum flux with ACHILLES-faithful beam sampling.  Reads ANY spectrum
    table; the no-arg constructor uses ADONIS_FLUX_FILE (T2K_nu.dat by default)."""

    def __init__(self, filename=None):
        path = _ACH / (filename if filename is not None else _default_flux())
        self.edges, self.heights0 = _parse_spectrum(path)
        self.flux_integral = float(np.sum(np.diff(self.edges) * self.heights0))
        centres = [self.edges[0]] + [(self.edges[i] + self.edges[i - 1]) / 2 for i in range(1, len(self.edges))] + [self.edges[-1]]
        hpad = [self.heights0[0]] + list(self.heights0) + [self.heights0[-1]]
        self.centres = np.array(centres); self.hpad = np.array(hpad)
        self.min_energy = self.edges[0]
        self.max_energy = self.edges[-1]

    def f(self, E_GeV):
        """m_flux(E_GeV): linear (2-pt Polint) interpolation on padded centres."""
        x = np.atleast_1d(E_GeV); out = np.zeros_like(x, float)
        kx = self.centres; ky = self.hpad
        for k, xv in enumerate(x):
            idx = int(np.searchsorted(kx, xv, side="left"))
            idx = min(max(idx, 1), len(kx) - 1)
            out[k] = _polint(kx[idx - 1:idx + 1], ky[idx - 1:idx + 1], xv)
        return out if out.size > 1 else float(out[0])

    def seed_min_GeV(self, m_lep=M_MU):
        """beam min_energy seed = (Smin - mp^2)/(2 mp) [MeV] -> GeV, clamped to edges[0].

        m_lep is the outgoing lepton mass, i.e. the threshold this beam has to clear; defaults to
        M_MU so existing CC callers are unaffected.  It is a parameter because NC has no threshold
        (the outgoing lepton is a massless neutrino), and using the CC value there would silently
        truncate the low-energy flux."""
        Smin = (m_lep + M_P) ** 2
        seed = (Smin - M_P ** 2) / (2 * M_P)
        return max(seed / 1000.0, self.min_energy)

    def sample_beam(self, u, minE, mode=None):
        """Map the single beam uniform u[:,4] -> (E_GeV, J_beam), in one of two modes that estimate
        the same flux-weighted integral (unbiased, equal in expectation):

          'flat' (legacy ACHILLES BeamMapper): E ~ Uniform[minE, maxE]; J_beam = dE * f(E) /
                 flux_integral.  Most draws land in the high-E flux tail with near-zero weight, so
                 effective sample size is small.

          'is' (importance, default): E drawn from the nominal flux shape via the exact inverse-CDF
                 of the piecewise-constant flux (raw edges/heights) on [minE, maxE]; proposal density
                 q(E) = h_bin/Z, Z = int_minE^maxE flux.  J_beam = f(E) * Z / (flux_integral * h_bin)
                 -- same numerator f as 'flat', so the estimator is identical in expectation, with
                 J_beam roughly constant and a much larger effective sample size.

        Sampling from the nominal (frozen) flux keeps the kind-1 contract: flux gradients would be
        recovered by an extra smooth per-event factor f_theta(E)/f_nom(E) (not applied here).
        mode=None resolves to the module-level BEAM_MODE toggle."""
        mode = mode or BEAM_MODE
        u = np.atleast_1d(np.asarray(u, float))
        maxE = self.max_energy
        if mode == "flat":
            dE = maxE - minE
            E = u * dE + minE
            J = (dE * np.asarray(self.f(E))) / self.flux_integral
            return E, J
        if mode != "is":
            raise ValueError(f"unknown beam mode {mode!r} (expected 'is' or 'flat')")
        lo = np.maximum(self.edges[:-1], minE)
        hi = np.minimum(self.edges[1:], maxE)
        width = np.clip(hi - lo, 0.0, None)
        mass = width * self.heights0
        Z = float(mass.sum())
        cdf = np.concatenate([[0.0], np.cumsum(mass)]) / Z
        bi = np.clip(np.searchsorted(cdf, u, side="right") - 1, 0, len(self.heights0) - 1)
        frac = (u - cdf[bi]) / np.clip(cdf[bi + 1] - cdf[bi], 1e-30, None)
        E = lo[bi] + frac * width[bi]
        J = np.asarray(self.f(E)) * Z / (self.flux_integral * np.clip(self.heights0[bi], 1e-30, None))
        return E, J


BEAM_MODE = _os.environ.get("ADONIS_BEAM_MODE", "is")
