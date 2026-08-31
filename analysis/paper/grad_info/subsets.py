"""Resolve column axes (configs/paper/fisher_subsets.yaml) against a Jacobian npz's `dskeys`, turning
declarative groups into concrete bin-row slices.  Degrades gracefully when the sample composition
changes:

  * a glob that matches nothing            -> reported, dropped
  * a group left with no keys              -> reported, dropped as a column
  * a dskey no group in the axis picks up  -> reported as `uncovered`
"""
from pathlib import Path
import fnmatch
import os

import numpy as np
import yaml

CONFIG = Path(os.environ.get(
    "ADONIS_FISHER_CONFIG",
    Path(__file__).resolve().parents[3] / "configs" / "paper" / "fisher_subsets.yaml"))

SCHEMA = ("J", "sigma", "prior", "pnames", "dskeys", "row0")


def load_config(path=CONFIG):
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    if not cfg.get("axes"):
        raise ValueError(f"{path}: no `axes` defined")
    return cfg


def check_schema(d, label=""):
    """Validate a loaded npz against the physfit Jacobian contract; return (J, sigma, prior, pnames)."""
    missing = [k for k in SCHEMA if k not in d.files]
    if missing:
        raise KeyError(f"{label}: not a physfit-schema Jacobian npz (missing {missing}; "
                       f"has {sorted(d.files)[:8]}...)")
    J, sigma, prior = np.asarray(d["J"]), np.asarray(d["sigma"]), np.asarray(d["prior"])
    pnames = [str(x) for x in d["pnames"]]
    dskeys = [str(x) for x in d["dskeys"]]
    row0 = np.asarray(d["row0"])
    if J.ndim != 2:
        raise ValueError(f"{label}: J must be 2-D (bins x knobs), got {J.shape}")
    if J.shape[0] != sigma.shape[0]:
        raise ValueError(f"{label}: J has {J.shape[0]} bin rows but sigma has {sigma.shape[0]}")
    if J.shape[1] != prior.shape[0] or J.shape[1] != len(pnames):
        raise ValueError(f"{label}: J has {J.shape[1]} knob columns but prior/pnames have "
                         f"{prior.shape[0]}/{len(pnames)}")
    if len(row0) != len(dskeys) + 1:
        raise ValueError(f"{label}: row0 has {len(row0)} offsets for {len(dskeys)} datasets "
                         f"(expected {len(dskeys) + 1})")
    if row0[0] != 0 or row0[-1] != J.shape[0]:
        raise ValueError(f"{label}: row0 spans {row0[0]}..{row0[-1]}, J has {J.shape[0]} rows")
    if np.any(np.diff(row0) <= 0):
        raise ValueError(f"{label}: row0 is not strictly increasing ({row0})")
    return J, sigma, prior, pnames, dskeys, row0


def resolve_axis(axis, dskeys, log=print):
    """[(name, label, [dskeys]), ...] for one axis spec, in declaration order.

    Items are literal dskeys, globs over dskeys, or <ref> to an earlier group in this axis.  Keys are
    de-duplicated but keep first-seen order, so a ladder rung reads parent-first.
    """
    out, by_name, declared = [], {}, []
    for name, spec in (axis.get("groups") or {}).items():
        spec = spec if isinstance(spec, dict) else {"keys": spec}
        keys, dropped = [], []
        declared.append(name)
        for item in spec.get("keys") or []:
            item = str(item)
            if item.startswith("<") and item.endswith(">"):
                ref = item[1:-1]
                if ref not in declared:
                    raise KeyError(f"group '{name}': <{ref}> is not an earlier group in this axis "
                                   f"(declared so far: {declared[:-1]})")
                hit = by_name.get(ref, [])
            elif any(c in item for c in "*?["):
                hit = [k for k in dskeys if fnmatch.fnmatch(k, item)]
                if not hit:
                    dropped.append(item)
            else:
                hit = [item] if item in dskeys else []
                if not hit:
                    dropped.append(item)
            keys += [k for k in hit if k not in keys]
        if dropped:
            log(f"  [subsets] {axis.get('_name', '?')}/{name}: no dskey matches {dropped}")
        if not keys:
            log(f"  [subsets] {axis.get('_name', '?')}/{name}: DROPPED (no datasets present)")
            continue
        by_name[name] = keys
        out.append((name, spec.get("label", name), keys))

    covered = {k for _n, _l, ks in out for k in ks}
    uncovered = [k for k in dskeys if k not in covered]
    if uncovered:
        log(f"  [subsets] {axis.get('_name', '?')}: UNCOVERED datasets (in the npz, in no column): "
            f"{uncovered}")
    return out


def rows_for(keys, dskeys, row0):
    """Bin-row indices for a set of dataset keys (sorted, so the slice is contiguous where it can be)."""
    idx = sorted(dskeys.index(k) for k in keys)
    return np.concatenate([np.arange(row0[j], row0[j + 1]) for j in idx])
