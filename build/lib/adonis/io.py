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

_ABOVE_PACKAGE = Path(__file__).resolve().parent.parent


def _checkout_root():
    """The source checkout this package sits in, or None when it is installed.

    Installed, the directory above the package is site-packages, which holds no configs/ and must
    never be written to; paths then default relative to the working directory instead.
    """
    return _ABOVE_PACKAGE if (_ABOVE_PACKAGE / "pyproject.toml").exists() else None


_REPO_ROOT = _ABOVE_PACKAGE
DEFAULT_ACHILLES_DATA = str((_checkout_root() or Path.cwd()) / ("data/achilles" if _checkout_root()
                                                               else "achilles_data"))
PATHS_CONFIG = (_checkout_root() or _ABOVE_PACKAGE) / "configs" / "paths.yaml"


def _configured(key):
    """One value from configs/paths.yaml, or None when the file is absent."""
    if not PATHS_CONFIG.exists():
        return None
    import yaml
    return (yaml.safe_load(PATHS_CONFIG.read_text()) or {}).get(key)


def output_root() -> Path:
    """Everything this project writes.

    $ADONIS_OUT, then configs/paths.yaml, then output/ under the checkout -- or under the working
    directory when the package is installed rather than run from a checkout.
    """
    return Path(os.environ.get("ADONIS_OUT") or _configured("output_root")
                or (_checkout_root() or Path.cwd()) / "output")


def bank_root() -> Path:
    """Where generated event banks live.  $ADONIS_BANKS, then configs/paths.yaml, else <output>/banks."""
    return Path(os.environ.get("ADONIS_BANKS") or _configured("bank_root") or output_root() / "banks")


def bank_path(name) -> Path:
    """Resolve a bank named by a config: absolute as given, otherwise relative to bank_root()."""
    p = Path(name)
    return p if p.is_absolute() else bank_root() / p


def achilles_data_root() -> Path:
    """The ACHILLES `data/` directory ($ACHILLES_DATA, else <repo>/data/achilles)."""
    return Path(os.environ.get("ACHILLES_DATA", DEFAULT_ACHILLES_DATA))


def achilles_sibling_root() -> Path:
    """An ACHILLES checkout ($ACHILLES_HOME, else a sibling ../Achilles).

    Only the inputs that upstream addresses by a relative 'data/...' path need this -- the QMC
    nucleon configurations and the flux tables.  Everything else resolves under achilles_data_root().
    """
    root = _checkout_root() or Path.cwd()
    return Path(os.environ.get("ACHILLES_HOME", str(root.parent / "Achilles")))


def achilles_flux_root() -> Path:
    """The ACHILLES flux tables ($ACHILLES_FLUX, else flux/ beside the data directory)."""
    return Path(os.environ.get("ACHILLES_FLUX") or achilles_data_root() / "flux")


def checkout_path(relative) -> Path:
    """Resolve a path written the way ACHILLES addresses its own tree, e.g. "data/Spectral_Functions/x".

    A checkout is used when one is available.  Otherwise the leading component is mapped to the
    directory that holds it: "data/..." under achilles_data_root(), "flux/..." under
    achilles_flux_root().  Those denote the same files, so a data directory extracted from the
    reference image is sufficient and no source checkout is required.
    """
    rel = Path(relative)
    in_checkout = achilles_sibling_root() / rel
    if in_checkout.exists():
        return in_checkout
    head = rel.parts[0] if rel.parts else ""
    if head == "data":
        return achilles_data_root().joinpath(*rel.parts[1:])
    if head == "flux":
        return achilles_flux_root().joinpath(*rel.parts[1:])
    return in_checkout


def require(*parts, home: bool = False) -> Path:
    """Resolve an input file and fail with an actionable message if it is not there.

    `home=True` resolves under an ACHILLES checkout rather than under the data directory.
    """
    p = checkout_path(Path(*parts)) if home else achilles_data_root().joinpath(*parts)
    root = p.parent
    if not p.exists():
        var = "ACHILLES_HOME" if home else "ACHILLES_DATA"
        raise FileNotFoundError(
            f"missing ACHILLES input: {Path(*parts)}\n"
            f"  looked in : {root}\n"
            f"  set       : {var}, or ACHILLES_DATA to a data/ directory from the image\n"
            f"  or extract it from ghcr.io/cesarjesusvalls/achilles:oracle")
    return p
