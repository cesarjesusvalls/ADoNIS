"""Parsers for published measurement releases (T2K STV, via the NUISANCE data tree).

Format knowledge -- a ROOT histogram layout, a text covariance layout -- which the fit layer needs to
build real-data fits (adonis.fit.stages.build_fakedata already loads CC1pi from here).  It belongs in
the package for the same reason the fit does; what does NOT belong in the package is an assumption
about where the files sit on disk, so the data root is a parameter with the previous hardcoded relative
path as its default.

Nothing in the physics engine imports this subpackage -- keep it that way, so `adonis` stays publishable
without the measurement tree.  Only fit stages and drivers may import it.
"""
from __future__ import annotations

import os

import numpy as np
import uproot

# Previously "../nuisance/data" hardcoded at each call site.  Overridable, default unchanged.
DATA_ROOT = os.environ.get("ADONIS_MEASUREMENT_ROOT", "../nuisance/data")

def load_cc0pi(obs):
    """T2K CC0pi-Np STV data (edges, per-bin conv, data[1e-38], cov[1e-38^2]) -- as in tune.py."""
    r = uproot.open(f"{DATA_ROOT}/T2K/CC0pi/STV/{'dpt' if obs == 'dpt' else 'dat'}Results.root")
    if obs == "dpt":
        edges = np.asarray(r["Result"].axis().edges()) * 1000.0             # MeV
        conv = 1e-33 / 12.0 * 1000.0 * 1e38
    else:
        edges = np.asarray(r["Result"].axis().edges())                     # rad
        conv = 1e-33 / 12.0 * 1e38
    data = np.asarray(r["Result"].values()) * 1e38
    cov = np.asarray(r["Covariance_Matrix"].values())
    return edges, conv, data, cov

NB_PER_CM2 = 1e33          # 1 cm^2 in nb
NUCLEONS_CH = 13           # a CH unit: 12 C + 1 H.  The CC0pi release above is quoted on carbon and
                           # uses 12 instead -- the two T2K releases do not share a convention.


def load_cc1pi(name, nucleons=NUCLEONS_CH):
    """T2K CC1pi+Np STV data: (edges, data, cov).

    The release is per nucleon in cm^2; `nucleons` converts it to the per-CH nb units the CC1pi
    model side works in.  Pass the count your model is normalised to -- it is not a property of
    the file.
    """
    scale = NB_PER_CM2 * nucleons
    lines = open(f"{DATA_ROOT}/T2K/CC1pipNp_STV/xsec_{name}.txt").read().splitlines()
    edges = np.array([float(x) for x in lines[0].split(":")[1].split()])
    vals = np.array([float(x) for x in lines[1].split(":")[1].split()]); nb = len(vals)
    cov = np.array([[float(x) for x in lines[3 + i].split()] for i in range(nb)])
    return edges, vals * scale, cov * scale ** 2
