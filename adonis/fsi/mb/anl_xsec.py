"""ANL-Osaka meson-baryon partial-wave cross sections (Phase E1).

Parses the ANL tables (`data/MesonBaryonAmplitudes/ANL/ANL_i-f.dat`) and forms the
physical piN -> piN total cross section by the partial-wave sum used in ACHILLES
(`MesonBaryonAmplitudes.cc::CalcCrossSectionW_grid`):

    sigma_if(W) = (hbar c)^2 * 10 * 2 pi * 4 W^2 / PF
                  * sum_{L,J} (2J+1) | sum_I CG_I^{if} A^{I}_{L,J}(W) |^2     [mb]

with PF = (W^2 - m_M^2 - m_B^2)^2 - 4 m_M^2 m_B^2 (Kallen; p_cm = sqrt(PF)/(2W)).
The table columns are the 20 waves L_{2I,2J} x (Re, Im); the label gives (L, I, J)
directly (e.g. P33 = L=1, I=3/2, J=3/2 = the Delta(1232)).

This forward sigma(W) IS the ANL-Osaka model, so it self-validates against the known
piN cross section (the Delta peak at W~1232) -- no cascade binary needed (which is a
confirmed showstopper here; see docs/phases/README.md).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from adonis.paths import achilles_data_root
from adonis.primary.dcc.form_factors import M_PI_GEV  # GeV; we work in MeV here

HBARC = 197.32              # MeV fm
M_PI = 139.57018           # charged pion [MeV]
M_N = 938.27208816         # proton [MeV]

# wave order in the ANL files; label L_{2I,2J} -> (L, twoI, twoJ)
WAVES = ["S11", "S31", "P11", "P13", "P31", "P33", "D13", "D15", "D33", "D35",
         "F15", "F17", "F35", "F37", "G17", "G19", "G37", "G39", "H19", "H39"]
_L = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4, "H": 5}


def wave_qn(name):
    """(L, twoI, twoJ) from a wave label like 'P33'."""
    return _L[name[0]], int(name[1]), int(name[2])


def load_anl(i=0, f=0, root=None):
    """Parse ANL_i-f.dat -> (W[MeV], amps[nW, 20] complex) in WAVES order."""
    root = achilles_data_root() if root is None else Path(root)
    path = root / "MesonBaryonAmplitudes" / "ANL" / f"ANL_{i}-{f}.dat"
    rows = []
    for line in open(path):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        rows.append([float(x) for x in s.split()])
    arr = np.array(rows)
    W = arr[:, 0]
    vals = arr[:, 1:]                                    # (nW, 40) = 20 x (Re, Im)
    amps = vals[:, 0::2] + 1j * vals[:, 1::2]            # (nW, 20)
    return W, amps


def _pcm2(W, mM=M_PI, mB=M_N):
    PF = (W ** 2 - mM ** 2 - mB ** 2) ** 2 - 4.0 * mM ** 2 * mB ** 2
    return PF                                            # ACHILLES "PF" (= 4 W^2 p_cm^2)


def pip_p_total(W=None, norm=1.0):
    """pi+ p -> pi+ p total cross section [mb] (pure I=3/2). `norm` is a differentiable
    overall sigma-normalisation knob (=1 nominal). Returns (W[MeV], sigma[mb])."""
    Wt, amps = load_anl(0, 0)
    if W is not None:
        # linear interp of the complex amps onto requested W
        amps = np.stack([np.interp(W, Wt, amps[:, k]) for k in range(amps.shape[1])], axis=1)
        Wt = np.asarray(W)
    s = np.zeros(len(Wt))
    for k, name in enumerate(WAVES):
        L, twoI, twoJ = wave_qn(name)
        if twoI != 3:                                   # pi+ p is pure I=3/2
            continue
        s += (twoJ + 1.0) * np.abs(amps[:, k]) ** 2
    PF = _pcm2(Wt)
    pref = HBARC ** 2 * 10.0 * 2.0 * np.pi * 4.0 * Wt ** 2 / PF
    return Wt, norm * pref * s
