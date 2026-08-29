"""Hadron-nucleus reaction and secondary-production cross sections from a generated beam bank.

Both are binned in beam momentum and normalised by the bank's geometric pi*R^2, so the estimator is
the reacted (or secondary-producing) fraction of thrown beam particles per bin, times that area.
Uncertainties are binomial on the same fractions.  "Secondary" means absorption for a pion beam and
any outgoing pion for a nucleon beam.
"""
from __future__ import annotations

import numpy as np


def beam_cross_section(bank_dir, nbins):
    """(edges, sigma_reac, sigma_second, err_reac, err_second) in the bank's pi*R^2 units.

    `bank_dir` is a generated beam bank; `nbins` splits [pmin, pmax] from its manifest uniformly.
    """
    from adonis.workflow import records as REC
    from adonis.workflow.generate_bank import load_bank

    bank = load_bank(str(bank_dir))
    man = bank["manifest"]
    edges = np.linspace(man["pmin"], man["pmax"], nbins + 1)
    idx = np.clip(np.digitize(np.asarray(bank["beam_p"], float), edges) - 1, 0, nbins - 1)
    ntry = np.bincount(idx, minlength=nbins).astype(float)

    flags = REC.derive_flags(bank)
    reacted = flags["reacted"].astype(float)
    second = (flags["absorbed"].astype(float) if man["species"] == "PION"
              else (np.asarray(bank["n_pi_out"]) > 0).astype(float))
    nr = np.bincount(idx, weights=reacted, minlength=nbins)
    ns = np.bincount(idx, weights=second, minlength=nbins)

    pir2 = man["pir2_mb"]
    with np.errstate(divide="ignore", invalid="ignore"):
        sr, ss = pir2 * nr / ntry, pir2 * ns / ntry
        er = pir2 * np.sqrt(nr * np.clip(1.0 - nr / ntry, 0.0, 1.0)) / ntry
        es = pir2 * np.sqrt(ns * np.clip(1.0 - ns / ntry, 0.0, 1.0)) / ntry
    return edges, sr, ss, er, es
