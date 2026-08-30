"""ACHILLES/NuHepMC reading and absolute normalization.

`parse_events` streams a NuHepMC Asciiv3 text file into per-event dicts (weight, signal_process_id,
particle list). `hepmc_norm`/`weight_to_nb_of` give the absolute cross-section scale, derived entirely
from the file header (GenCrossSection) and the per-event weights -- no hardcoded constants:
    sigma_sel[nb] = (sum_w_selected / sum_w_all) * GenCrossSection[nb]
    weight_to_nb  = GenCrossSection[nb] / sum_w_all
Extractors store hepmc_norm's result in the npz so figures never re-parse or hardcode a scale.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

PB_TO_NB = 1.0e-3


def parse_events(path):
    """Yield per-event dicts {w, gen_xs, proc, parts}.  parts = list of (pid, status, p4=(E,px,py,pz) MeV).
    A lightweight streaming parser (one pass, no buffering of the whole file)."""
    evt = None
    with open(path) as fh:
        for line in fh:
            tag = line[:2]
            if tag == "E ":
                if evt is not None:
                    yield evt
                evt = {"w": None, "gen_xs": None, "proc": None, "parts": []}
            elif evt is None:
                continue
            elif tag == "W ":
                tok = line.split()[1]
                if tok != "CV":
                    evt["w"] = float(tok)
            elif tag == "A " and "GenCrossSection" in line:
                evt["gen_xs"] = float(line.split()[3])
            elif tag == "A " and "signal_process_id" in line:
                evt["proc"] = int(line.split()[3])
            elif tag == "P ":
                f = line.split()
                pid = int(f[3])
                p4 = np.array([float(f[7]), float(f[4]), float(f[5]), float(f[6])])
                evt["parts"].append((pid, int(f[9]), p4))
    if evt is not None:
        yield evt


def minkowski2(p):
    """p=(E,px,py,pz) -> p.p with (+,-,-,-)."""
    return p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2


def hepmc_norm(path):
    """One lightweight pass over a hepmc -> dict(gen_xs_pb, sum_w_all, n_events, weight_to_nb).
    weight_to_nb = GenCrossSection[nb] / sum_w_all (multiply raw event weights by it -> absolute nb)."""
    path = Path(path)
    sum_w = 0.0; n = 0; gen_xs_pb = None
    with open(path) as fh:
        for line in fh:
            t = line[:2]
            if t == "W ":
                tok = line.split()[1]
                if tok == "CV":
                    continue
                sum_w += float(tok); n += 1
            elif t == "A " and "GenCrossSection" in line:
                gen_xs_pb = float(line.split()[3])
    if gen_xs_pb is None:
        raise ValueError(f"{path}: no GenCrossSection header found")
    if sum_w <= 0:
        raise ValueError(f"{path}: sum of weights is {sum_w}")
    return dict(gen_xs_pb=gen_xs_pb, sum_w_all=sum_w, n_events=n,
                weight_to_nb=gen_xs_pb * PB_TO_NB / sum_w)


def weight_to_nb_of(npz):
    """Read the nb-per-weight factor an extractor stored in an ACHILLES npz (the ONLY way figures should
    convert ACHILLES event weights to nb -- no hardcoded constants)."""
    if "weight_to_nb" not in npz.files:
        raise KeyError("ACHILLES npz lacks 'weight_to_nb' (GenCrossSection/sum_w from the hepmc header); "
                       "re-extract with python -m analysis.oracle_tools.extract")
    return float(np.asarray(npz["weight_to_nb"]))
