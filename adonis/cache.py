"""On-disk memo for an expensive reduction, fingerprinted on its inputs.  No physics.

Memoises a pure function of (input files, parameters) to <CACHE>/<name>.npz. The fingerprint
covers each dependency's (path, size, mtime) plus `params`; a mismatch rebuilds the
entry. `ADONIS_PLOT_REFRESH=1` forces a rebuild (needed if a file is rewritten with its mtime
preserved, which fingerprinting cannot see). Cache root: `ADONIS_PLOT_CACHE`, else a cache
directory under the output root.
"""
import os
import json
import hashlib
from pathlib import Path

import numpy as np

from adonis.io import output_root

CACHE = Path(os.environ.get("ADONIS_PLOT_CACHE") or output_root() / "cache" / "plot")
_FP = "__fingerprint__"


def _iter_files(dep):
    p = Path(dep)
    if p.is_dir():
        yield from sorted(q for q in p.rglob("*") if q.is_file())
    elif p.is_file():
        yield p


def fingerprint(deps=(), params=None):
    """Stable digest of the inputs a cached reduction depends on (stat only -- never reads content)."""
    h = hashlib.sha256()
    h.update(json.dumps(params or {}, sort_keys=True, default=str).encode())
    for dep in deps:
        for f in _iter_files(dep):
            st = f.stat()
            h.update(f"{f}|{st.st_size}|{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def cached(name, build, *, deps=(), params=None, refresh=None):
    """Return build() -- a dict of ndarrays -- memoised to <CACHE>/<name>.npz."""
    if refresh is None:
        refresh = os.environ.get("ADONIS_PLOT_REFRESH", "") == "1"
    fp = fingerprint(deps, params)
    path = CACHE / f"{name}.npz"
    if path.exists() and not refresh:
        try:
            z = np.load(path, allow_pickle=False)
            if str(z[_FP].item()) == fp:
                return {k: z[k] for k in z.files if k != _FP}
        except Exception:
            pass
    out = build()
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **{_FP: np.array(fp), **{k: np.asarray(v) for k, v in out.items()}})
    os.replace(tmp, path)
    return out
