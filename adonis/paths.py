"""Resolve ACHILLES input-data paths.

The JAX loaders read two ACHILLES tables at runtime -- the DCC electroweak
amplitudes (`dcc_EW.dat`) and the spectral function
(`Spectral_Functions/pke12p_tot.data`).  Both live in the ACHILLES `data/`
directory.  Override that directory with the `ACHILLES_DATA` environment variable
(e.g. in CI, where the tables are extracted from the published container image
`ghcr.io/cesarjesusvalls/achilles:oracle`); it defaults to the local clone so
local runs are unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ACHILLES_DATA = "/Users/cjesus/Software/DiffSinglePiProd/Achilles/data"


def achilles_data_root() -> Path:
    return Path(os.environ.get("ACHILLES_DATA", DEFAULT_ACHILLES_DATA))
