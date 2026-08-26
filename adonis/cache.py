"""On-disk memo for an expensive reduction, fingerprinted on its inputs.  No physics.

A reduction that is a pure function of (input files, parameters) can be memoised here so that
restyling a downstream plot, or any other re-use, costs only the import plus the read.

    from adonis import cache as plotcache
    d = plotcache.cached("fig03_pip_Ar",
                         lambda: {"edges": e, "y": y, "yerr": ye},     # -> dict of ndarrays
                         deps=[bank_dir, *hepmc_paths],
                         params={"nbins": 30})

Correctness: the entry stores a fingerprint of every dependency's (relative path, size, mtime) plus
the params, and is ignored unless it matches -- so regenerating a bank or changing nbins rebuilds
automatically.  `ADONIS_PLOT_REFRESH=1` forces a rebuild anyway; use it if a file is ever rewritten
with its mtime preserved, which fingerprinting cannot see.
Cache root: `ADONIS_PLOT_CACHE`, default `<repo>/data/cache/plot`.
"""
import os
import json
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("ADONIS_PLOT_CACHE", str(ROOT / "data" / "cache" / "plot")))
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
