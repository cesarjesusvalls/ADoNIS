"""Shared definition of the final-state oracle histograms.

The oracle stores, per observable, a FIXED fine binning plus the weighted sums
sum(w) and sum(w^2) per bin (so it carries per-bin statistical errors and can be
re-binned to any coarser integer multiple downstream).  Both the local generator
(scripts/make_oracle_finalstate.py) and the CI parser (scripts/oracle_from_hepmc.py)
build the npz through this module, so the edges can never drift apart.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from adonis.data.oracle.parse_hepmc import parse_events
from adonis.data.oracle.parse_hepmc_nu import event_kin_full, REF_TOTAL_NB

# Deterministic per-batch Options block (replaces the OptionDefaults include so each
# batch gets an independent, reproducible seed).
_SEED_OPTIONS = ("Options:\n  Initialize:\n    Seed: {seed}\n    Accuracy: 1e-2\n"
                 "  Unweighting:\n    Name: Percentile\n    percentile: 99")


def write_run_config(base_yml, out_yml, nevents, seed, out_hepmc):
    """Write a run config from `base_yml` with the given event count, seed, and
    output path -- used to drive the ACHILLES container for one oracle batch."""
    y = Path(base_yml).read_text()
    y = re.sub(r"NEvents:\s*\d+", f"NEvents: {nevents}", y, count=1)
    y = re.sub(r"Name:\s*res_1pi_12C_nu\.hepmc", f"Name: {out_hepmc}", y)
    y = y.replace('Options: !include "data/default/OptionDefaults.yml"',
                  _SEED_OPTIONS.format(seed=seed))
    Path(out_yml).write_text(y)

# Fine fixed binning per observable (re-binnable to coarser multiples downstream).
ORACLE_EDGES = {
    "W":               np.linspace(1076.0, 1700.0, 313),
    "Q2":              np.linspace(0.0, 2.0e6, 201),
    "ppi_mag":         np.linspace(0.0, 700.0, 141),
    "cos_theta_star":  np.linspace(-1.0, 1.0, 101),
    "phi_star":        np.linspace(-np.pi, np.pi, 73),
    "lepton_energy":   np.linspace(0.0, 1500.0, 151),
    "lepton_costheta": np.linspace(-1.0, 1.0, 201),
    "nucleon_mom":     np.linspace(0.0, 1800.0, 181),
}


def new_accumulator(edges=ORACLE_EDGES):
    return {f"{p}_{k}": np.zeros(len(edges[k]) - 1) for k in edges for p in ("sw", "sw2")}


def accumulate_hepmc(path, acc, edges=ORACLE_EDGES):
    """Histogram one hepmc file into `acc` (sum w, sum w^2 per bin).  Returns the
    number of signal events added."""
    cols = {k: [] for k in edges}
    ws = []
    for evt in parse_events(Path(path)):
        r = event_kin_full(evt)
        if r is None:
            continue
        for k in edges:
            cols[k].append(r[k])
        ws.append(r["w"])
    ws = np.asarray(ws)
    for k in edges:
        v = np.asarray(cols[k]); m = np.isfinite(v)
        acc[f"sw_{k}"] += np.histogram(v[m], bins=edges[k], weights=ws[m])[0]
        acc[f"sw2_{k}"] += np.histogram(v[m], bins=edges[k], weights=ws[m] ** 2)[0]
    return len(ws)


def save_oracle(out, n_total, acc, edges=ORACLE_EDGES):
    np.savez(out, n_events=n_total, ref_total_nb=REF_TOTAL_NB,
             **{f"edges_{k}": edges[k] for k in edges}, **acc)
