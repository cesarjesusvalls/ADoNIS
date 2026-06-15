"""Experimental-data overlay loaders for the analysis API.

Two sources (matching the existing scripts):
  - npz          : data/oracle/t2k_cc0pi_stv_data.npz (keys '<obs>_edges','<obs>','<obs>_err'),
                   per-nucleon cm^2 -> per-A nb via x 1e33 (nb/cm^2) x A.
  - nuisance_txt : nuisance release txt (edges:/values:/covariance) -- cc1pi_fig_tki.load_data format.
Returns per-observable {ctr, val, err} for chi2_ratio_panel's optional data overlay.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

NB_PER_CM2 = 1e33


def _from_npz(cfg, specs):
    d = dict(np.load(cfg.data.path))
    scale = (NB_PER_CM2 * cfg.data.A) if cfg.data.per_nucleon_cm2 else 1.0
    out = {}
    for key, edges, _ in specs:
        dk = cfg.data.names.get(key, key)
        if f"{dk}_edges" not in d:
            continue
        e = np.asarray(d[f"{dk}_edges"]); ctr = 0.5 * (e[1:] + e[:-1])
        val = np.asarray(d[dk]) * scale
        err = np.asarray(d.get(f"{dk}_err", np.zeros_like(val))) * scale
        out[key] = {"ctr": ctr, "val": val, "err": err}
    return out


def _from_nuisance_txt(cfg, specs):
    out = {}
    for key, edges, _ in specs:
        name = cfg.data.names.get(key, key)
        path = Path(cfg.data.path) / f"xsec_{name}.txt"
        if not path.exists():
            continue
        lines = path.read_text().splitlines()
        e = np.array([float(x) for x in lines[0].split(":")[1].split()])
        v = np.array([float(x) for x in lines[1].split(":")[1].split()])
        cov = np.array([[float(x) for x in lines[3 + i].split()] for i in range(len(v))])
        out[key] = {"ctr": 0.5 * (e[1:] + e[:-1]), "val": v, "err": np.sqrt(np.diag(cov))}
    return out


def load_overlay(cfg, specs):
    if not cfg.data.enabled:
        return None
    if cfg.data.source == "npz":
        return _from_npz(cfg, specs)
    if cfg.data.source == "nuisance_txt":
        return _from_nuisance_txt(cfg, specs)
    raise ValueError(f"unknown data source {cfg.data.source!r}")
