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

# Default to a repo-local (git-ignored) directory so a fresh clone works on any
# machine: populate it with `python scripts/fetch_achilles_data.py` (pulls the
# tables from the public oracle image) or point ACHILLES_DATA at a local ACHILLES
# `data/` directory.  See docs/CONTAINER.md.
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ACHILLES_DATA = str(_REPO_ROOT / "data" / "achilles")


def achilles_data_root() -> Path:
    return Path(os.environ.get("ACHILLES_DATA", DEFAULT_ACHILLES_DATA))
