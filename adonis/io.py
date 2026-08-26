"""Where the ACHILLES input tables live.

ADoNIS reads ACHILLES' tabulated inputs (DCC electroweak amplitudes, spectral functions, QMC
nucleon configurations, flux files) from two roots, since upstream splits them the same way:

    ACHILLES_DATA   the `data/` directory itself -- tables addressed by bare name
    ACHILLES_HOME   an ACHILLES checkout -- for inputs addressed by a relative 'data/...' path

ACHILLES_DATA defaults to `<repo>/data/achilles` (or extract one from the published container
image `ghcr.io/cesarjesusvalls/achilles:oracle`). ACHILLES_HOME defaults to a sibling
`../Achilles` checkout.

`require()` is the accessor to use when opening a file: on failure it names the missing file and
which env var to set.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ACHILLES_DATA = str(_REPO_ROOT / "data" / "achilles")


def achilles_data_root() -> Path:
    """The ACHILLES `data/` directory ($ACHILLES_DATA, else <repo>/data/achilles)."""
    return Path(os.environ.get("ACHILLES_DATA", DEFAULT_ACHILLES_DATA))


def achilles_sibling_root() -> Path:
    """An ACHILLES checkout ($ACHILLES_HOME, else a sibling ../Achilles).

    Only the inputs that upstream addresses by a relative 'data/...' path need this -- the QMC
    nucleon configurations and the flux tables.  Everything else resolves under achilles_data_root().
    """
    return Path(os.environ.get("ACHILLES_HOME", str(_REPO_ROOT.parent / "Achilles")))


def require(*parts, home: bool = False) -> Path:
    """Resolve an input file and fail with an actionable message if it is not there.

    `home=True` resolves under an ACHILLES checkout rather than under the data directory.
    """
    root = achilles_sibling_root() if home else achilles_data_root()
    p = root.joinpath(*parts)
    if not p.exists():
        var = "ACHILLES_HOME" if home else "ACHILLES_DATA"
        raise FileNotFoundError(
            f"missing ACHILLES input: {Path(*parts)}\n"
            f"  looked in : {root}\n"
            f"  set       : {var}=<path to an ACHILLES {'checkout' if home else 'data/ directory'}>\n"
            f"  or extract it from ghcr.io/cesarjesusvalls/achilles:oracle")
    return p
