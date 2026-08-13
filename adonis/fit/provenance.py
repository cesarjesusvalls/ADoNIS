"""What produced this array: config digest, git commit, and which slice of the work it is.

The provenance travels INSIDE the npz, not beside it.  A sibling `.manifest.json` is written by the
runner as a run log, but it cannot be the thing a merge validates against: manifests and shards are
associated only by filename convention, so a renamed, copied or hand-merged shard silently loses its
manifest and the merge sees an artefact with no claims attached.  Stamped fields cannot be orphaned from
the data they describe.

    np.savez(out, **provenance.stamp(rows=(row_base, n_row), grid=N), chi2_abs=..., ...)

Shards written before this existed carry nothing; `read()` returns None for them and the merge degrades
to a warning rather than refusing to plot the campaign that motivated the check.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

_PREFIX = "prov_"
_GIT = None


def git_sha() -> str:
    """Short HEAD, cached.  'unknown' when git is unavailable -- never raises: provenance must not be
    able to break a run that would otherwise succeed."""
    global _GIT
    if _GIT is None:
        try:
            r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2])
            _GIT = r.stdout.strip() or "unknown"
        except Exception:
            _GIT = "unknown"
    return _GIT


def stamp(**extra) -> dict:
    """Fields to splat into `np.savez`.  Keys are prefixed so they cannot collide with physics arrays.

    `extra` records how this shard slices the work -- e.g. rows=(base, n), grid=N for a 2-D scan.  Those
    are what let a merge state exactly which rows are missing instead of measuring a NaN fraction.
    """
    cfg_path = os.environ.get("ADONIS_FIT_CONFIG", "")
    digest = ""
    if cfg_path:
        try:
            from adonis.fit.config import FitConfig
            digest = FitConfig.load(cfg_path).digest()
        except Exception as e:                      # a broken config must fail in the stage, not here
            digest = f"unreadable:{type(e).__name__}"
    out = {f"{_PREFIX}config": os.path.basename(cfg_path), f"{_PREFIX}digest": digest,
           f"{_PREFIX}stage": os.environ.get("ADONIS_FIT_STAGE", ""),
           f"{_PREFIX}git": git_sha(), f"{_PREFIX}time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out.update({f"{_PREFIX}{k}": v for k, v in extra.items()})
    return out


def read(z) -> dict | None:
    """Provenance of a loaded npz, or None if it predates stamping."""
    keys = [k for k in getattr(z, "files", ()) if k.startswith(_PREFIX)]
    if not keys:
        return None
    out = {}
    for k in keys:
        v = z[k]
        out[k[len(_PREFIX):]] = v.item() if getattr(v, "ndim", 1) == 0 else v.tolist()
    return out
