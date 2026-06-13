"""HepMC absolute normalization, derived ENTIRELY from the file (no hardcoded constants).

A weighted ACHILLES/NuHepMC sample carries two things needed for an absolute cross section:
  * per-event ``W <weight>`` lines (arbitrary internal units; their sum is run-dependent), and
  * the run total in the header ``A 0 GenCrossSection <xs> ...`` (pb).
The cross section of ANY selection is the HepMC-standard
    sigma_sel[nb] = (sum_w_selected / sum_w_all) * GenCrossSection[nb]
so the per-event "nb per unit weight" factor is ``weight_to_nb = GenCrossSection[nb] / sum_w_all``.
Both inputs come from the file -- this replaces the hand-pasted SCALE_ACH magic numbers.

`hepmc_norm` is a LIGHTWEIGHT single pass: it only scans ``W`` and ``GenCrossSection`` lines
(no particle parsing), so it is cheap even on multi-GB hepmc.  The extractors store its result
in the npz (`gen_xs_pb`, `sum_w_all`, `weight_to_nb`) so figures never re-parse or hardcode.
"""
from __future__ import annotations

from pathlib import Path

PB_TO_NB = 1.0e-3


def hepmc_norm(path):
    """One lightweight pass over a hepmc: return dict(gen_xs_pb, sum_w_all, n_events,
    weight_to_nb).  weight_to_nb = GenCrossSection[nb] / sum_w_all (multiply raw event
    weights by it to get absolute nb)."""
    path = Path(path)
    sum_w = 0.0
    n = 0
    gen_xs_pb = None
    with open(path) as fh:
        for line in fh:
            t = line[:2]
            if t == "W ":
                tok = line.split()[1]
                if tok == "CV":
                    continue                           # HepMC3 weight-NAMES declaration (header)
                sum_w += float(tok); n += 1
            elif t == "A " and "GenCrossSection" in line:
                gen_xs_pb = float(line.split()[3])     # running estimate; last value = final
    if gen_xs_pb is None:
        raise ValueError(f"{path}: no GenCrossSection header found")
    if sum_w <= 0:
        raise ValueError(f"{path}: sum of weights is {sum_w}")
    return dict(gen_xs_pb=gen_xs_pb, sum_w_all=sum_w, n_events=n,
                weight_to_nb=gen_xs_pb * PB_TO_NB / sum_w)


if __name__ == "__main__":
    import sys
    d = hepmc_norm(sys.argv[1])
    print(f"GenCrossSection = {d['gen_xs_pb']:.6e} pb   sum_w_all = {d['sum_w_all']:.6e}   "
          f"N = {d['n_events']}\n  weight_to_nb = {d['weight_to_nb']:.6e} nb/weight")


def weight_to_nb_of(npz):
    """Read the nb-per-weight factor an extractor stored in an ACHILLES npz.  Raises a clear
    error if absent (npz predates the from-header normalization -> re-extract).  This is the
    ONLY way figures should convert ACHILLES event weights to nb -- no hardcoded constants."""
    import numpy as _np
    if "weight_to_nb" not in npz.files:
        raise KeyError("ACHILLES npz lacks 'weight_to_nb' (GenCrossSection/sum_w from the hepmc "
                       "header); re-extract with scripts/extract_t2k_{cc0pi,cc1pi}_tki.py")
    return float(_np.asarray(npz["weight_to_nb"]))
