"""Bit-exact ACHILLES T2K Spectrum flux (Beams.cc Spectrum + T2KHeader).

T2K_nu.dat: 2 header lines, then "bin elo ehi flux" with energies in GeV, flux in cm^-2/50MeV.
m_energy_units = 1/GeV.  flux_integral = sum_i (edges[i+1]-edges[i]) * heights[i].  m_flux = linear
(Interp1D poly order 1) on padded bin centres: centres = [e0, (e1+e0)/2, ..., e_last], heights
padded by h[0] front and h[-1] back.  m_min/max_energy = edges.front()/back() (GeV).

Beam sampling (single ran u): min = max(seed_GeV, edges[0]); E_GeV = u*(max-min)+min; lab E = 1000*E.
J_beam (the event.Weight() beam factor) = (delta * m_flux(E_GeV)) / flux_integral, delta=max-min.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from adonis.xsec.spectral import _polint

_ACH = Path(__file__).resolve().parents[2].parent / "Achilles"
M_MU = 105.7
M_P = 938.27


class T2KFlux:
    def __init__(self, filename="flux/T2K_nu.dat"):
        path = _ACH / filename
        lines = path.read_text().splitlines()
        edges, heights = [], []
        for ln in lines[2:]:                       # 2 header lines
            t = ln.split()
            if len(t) < 4:
                continue
            edges.append(float(t[1])); heights.append(float(t[3]))
        edges.append(float(lines[-1].split()[2]))  # final ehi
        self.edges = np.array(edges); self.heights0 = np.array(heights)
        self.flux_integral = float(np.sum(np.diff(self.edges) * self.heights0))
        # padded bin centres + heights for the linear interpolator
        centres = [self.edges[0]] + [(self.edges[i] + self.edges[i - 1]) / 2 for i in range(1, len(self.edges))] + [self.edges[-1]]
        hpad = [self.heights0[0]] + list(self.heights0) + [self.heights0[-1]]
        self.centres = np.array(centres); self.hpad = np.array(hpad)
        self.min_energy = self.edges[0]            # GeV
        self.max_energy = self.edges[-1]           # GeV

    def f(self, E_GeV):
        """m_flux(E_GeV): linear (2-pt Polint) interpolation on padded centres."""
        x = np.atleast_1d(E_GeV); out = np.zeros_like(x, float)
        kx = self.centres; ky = self.hpad
        for k, xv in enumerate(x):
            idx = int(np.searchsorted(kx, xv, side="left"))
            idx = min(max(idx, 1), len(kx) - 1)
            out[k] = _polint(kx[idx - 1:idx + 1], ky[idx - 1:idx + 1], xv)
        return out if out.size > 1 else float(out[0])

    def seed_min_GeV(self):
        """beam min_energy seed = (Smin - mp^2)/(2 mp) [MeV] -> GeV, clamped to edges[0]."""
        Smin = (M_MU + M_P) ** 2
        seed = (Smin - M_P ** 2) / (2 * M_P)       # MeV
        return max(seed / 1000.0, self.min_energy)  # GeV
