"""High-statistics neutrino oracle with the FULL final state, as accumulated histograms.

Like make_oracle_highstats.py, but extracts and accumulates the full final-state
observables (cos theta*_pi, phi*_pi, lepton energy/angle, recoil-nucleon momentum, plus
W, Q2, |p_pi|) needed to validate the differentiable full-final-state fold
(fold_final_state).  Runs ACHILLES in batches, histograms each batch (sum w, sum w^2 per
bin), deletes each hepmc, and saves the combined histograms with per-bin errors.

Run (from the repo root):  python scripts/make_oracle_finalstate.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from analysis.utils.finalstate import (ORACLE_EDGES as EDGES, new_accumulator,
                                           accumulate_hepmc, save_oracle)  # noqa: E402

# Local ACHILLES checkout/build (for the high-statistics oracle). Override with
# ACHILLES_ROOT; defaults to a sibling clone. CI uses the container instead
# (scripts/oracle_from_hepmc.py + .github/workflows/oracle.yml) -- see docs/CONTAINER.md.
ACHILLES = Path(os.environ.get(
    "ACHILLES_ROOT", Path(__file__).resolve().parents[2] / "Achilles"))
BASE_YML = Path("data/oracle/res_1pi_12C_nu.yml").resolve()
OUT = Path("data/oracle/oracle_finalstate.npz")

N_PER_BATCH = 250_000
N_BATCHES = 8
BATCH_HEPMC = "/tmp/oracle_fs_batch.hepmc"

_OPTIONS = ("Options:\n  Initialize:\n    Seed: {seed}\n    Accuracy: 1e-2\n"
            "  Unweighting:\n    Name: Percentile\n    percentile: 99")


def write_yml(seed, out_hepmc, path):
    y = BASE_YML.read_text()
    y = y.replace("NEvents: 1000000", f"NEvents: {N_PER_BATCH}")
    y = y.replace("Name: res_1pi_12C_nu.hepmc", f"Name: {out_hepmc}")
    y = y.replace('Options: !include "data/default/OptionDefaults.yml"',
                  _OPTIONS.format(seed=seed))
    Path(path).write_text(y)


acc = new_accumulator()
n_total = 0
for b in range(N_BATCHES):
    yml = f"/tmp/oracle_fs_{b}.yml"
    write_yml(7000 + b, BATCH_HEPMC, yml)
    subprocess.run([str(ACHILLES / "build/bin/achilles"), yml], cwd=ACHILLES,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    n = accumulate_hepmc(BATCH_HEPMC, acc)
    n_total += n
    os.remove(BATCH_HEPMC)
    print(f"  batch {b+1}/{N_BATCHES}: {n:,} signal events, total {n_total:,}", flush=True)

save_oracle(OUT, n_total, acc)
print(f"saved -> {OUT}  ({n_total:,} total events)")
