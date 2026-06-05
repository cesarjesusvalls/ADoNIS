"""High-statistics neutrino oracle with the FULL final state, as accumulated histograms.

Like make_oracle_highstats.py, but extracts and accumulates the full final-state
observables (cos theta*_pi, phi*_pi, lepton energy/angle, recoil-nucleon momentum, plus
W, Q2, |p_pi|) needed to validate the differentiable full-final-state fold
(fold_final_state).  Runs ACHILLES in batches, histograms each batch (sum w, sum w^2 per
bin), deletes each hepmc, and saves the combined histograms with per-bin errors.

Run (from the diffpi dir):  python make_oracle_finalstate.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from adonis.data.oracle.parse_hepmc import parse_events             # noqa: E402
from adonis.data.oracle.parse_hepmc_nu import event_kin_full, REF_TOTAL_NB  # noqa: E402

ACHILLES = Path("/Users/cjesus/Software/DiffSinglePiProd/Achilles")
BASE_YML = Path("data/oracle/res_1pi_12C_nu.yml").resolve()
OUT = Path("data/oracle/oracle_finalstate.npz")

N_PER_BATCH = 250_000
N_BATCHES = 8
BATCH_HEPMC = "/tmp/oracle_fs_batch.hepmc"

# fine fixed binning per observable (re-binnable to coarser multiples downstream)
EDGES = {
    "W":               np.linspace(1076.0, 1700.0, 313),
    "Q2":              np.linspace(0.0, 2.0e6, 201),
    "ppi_mag":         np.linspace(0.0, 700.0, 141),
    "cos_theta_star":  np.linspace(-1.0, 1.0, 101),
    "phi_star":        np.linspace(-np.pi, np.pi, 73),
    "lepton_energy":   np.linspace(0.0, 1500.0, 151),
    "lepton_costheta": np.linspace(-1.0, 1.0, 201),
    "nucleon_mom":     np.linspace(0.0, 1800.0, 181),
}

_OPTIONS = ("Options:\n  Initialize:\n    Seed: {seed}\n    Accuracy: 1e-2\n"
            "  Unweighting:\n    Name: Percentile\n    percentile: 99")


def write_yml(seed, out_hepmc, path):
    y = BASE_YML.read_text()
    y = y.replace("NEvents: 1000000", f"NEvents: {N_PER_BATCH}")
    y = y.replace("Name: res_1pi_12C_nu.hepmc", f"Name: {out_hepmc}")
    y = y.replace('Options: !include "data/default/OptionDefaults.yml"',
                  _OPTIONS.format(seed=seed))
    Path(path).write_text(y)


acc = {f"{p}_{k}": np.zeros(len(EDGES[k]) - 1) for k in EDGES for p in ("sw", "sw2")}
n_total = 0
for b in range(N_BATCHES):
    yml = f"/tmp/oracle_fs_{b}.yml"
    write_yml(7000 + b, BATCH_HEPMC, yml)
    subprocess.run([str(ACHILLES / "build/bin/achilles"), yml], cwd=ACHILLES,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cols = {k: [] for k in EDGES}
    ws = []
    for evt in parse_events(Path(BATCH_HEPMC)):
        r = event_kin_full(evt)
        if r is None:
            continue
        for k in EDGES:
            cols[k].append(r[k])
        ws.append(r["w"])
    ws = np.asarray(ws)
    for k in EDGES:
        v = np.asarray(cols[k]); m = np.isfinite(v)
        acc[f"sw_{k}"] += np.histogram(v[m], bins=EDGES[k], weights=ws[m])[0]
        acc[f"sw2_{k}"] += np.histogram(v[m], bins=EDGES[k], weights=ws[m] ** 2)[0]
    n_total += len(ws)
    os.remove(BATCH_HEPMC)
    print(f"  batch {b+1}/{N_BATCHES}: {len(ws):,} signal events, total {n_total:,}", flush=True)

np.savez(OUT, n_events=n_total, ref_total_nb=REF_TOTAL_NB,
         **{f"edges_{k}": EDGES[k] for k in EDGES}, **acc)
print(f"saved -> {OUT}  ({n_total:,} total events)")
