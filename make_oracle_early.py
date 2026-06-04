"""Independent EARLY oracle sample (parallel to the main 10M run) with INCREMENTAL
saving, so we can plot after each batch instead of waiting for the full result.

Distinct seeds (5000+) and temp paths so it never touches the main make_oracle_highstats
run. Saves oracle/oracle_early.npz after every batch.

Run:  python make_oracle_early.py
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "oracle")
from parse_hepmc import parse_events                      # noqa: E402
from parse_hepmc_nu import event_kin, REF_TOTAL_NB        # noqa: E402

ACHILLES = Path("/Users/cjesus/Software/DiffSinglePiProd/Achilles")
BASE_YML = Path("oracle/res_1pi_12C_nu.yml").resolve()
OUT = Path("oracle/oracle_early.npz")

N_PER_BATCH = 500_000
N_BATCHES = 6
BATCH_HEPMC = "/tmp/oracle_early_batch.hepmc"

We = np.linspace(1076.0, 1700.0, 313)
Q2e = np.linspace(0.0, 2.0e6, 201)
ppie = np.linspace(0.0, 700.0, 141)

_OPTIONS = ("Options:\n  Initialize:\n    Seed: {seed}\n    Accuracy: 1e-2\n"
            "  Unweighting:\n    Name: Percentile\n    percentile: 99")


def write_yml(seed, out_hepmc, path):
    y = BASE_YML.read_text()
    y = y.replace("NEvents: 1000000", f"NEvents: {N_PER_BATCH}")
    y = y.replace("Name: res_1pi_12C_nu.hepmc", f"Name: {out_hepmc}")
    y = y.replace('Options: !include "data/default/OptionDefaults.yml"',
                  _OPTIONS.format(seed=seed))
    Path(path).write_text(y)


def hist2(v, w, edges):
    return (np.histogram(v, bins=edges, weights=w)[0],
            np.histogram(v, bins=edges, weights=w ** 2)[0])


acc = {k: np.zeros(n) for k, n in
       [("swW", len(We) - 1), ("sw2W", len(We) - 1), ("swQ", len(Q2e) - 1),
        ("sw2Q", len(Q2e) - 1), ("swP", len(ppie) - 1), ("sw2P", len(ppie) - 1)]}
n_total = 0
for b in range(N_BATCHES):
    yml = f"/tmp/oracle_early_{b}.yml"
    write_yml(5000 + b, BATCH_HEPMC, yml)
    subprocess.run([str(ACHILLES / "build/bin/achilles"), yml], cwd=ACHILLES,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    Ws, Q2s, Ps, ws = [], [], [], []
    for evt in parse_events(Path(BATCH_HEPMC)):
        kin = event_kin(evt)
        if kin is None:
            continue
        Q2, W, ppi = kin
        Q2s.append(Q2); Ws.append(W); Ps.append(ppi); ws.append(evt["w"])
    Ws, Q2s, Ps, ws = map(np.array, (Ws, Q2s, Ps, ws))
    m = np.isfinite(Ws)
    hw, h2w = hist2(Ws[m], ws[m], We); acc["swW"] += hw; acc["sw2W"] += h2w
    hq, h2q = hist2(Q2s, ws, Q2e); acc["swQ"] += hq; acc["sw2Q"] += h2q
    hp, h2p = hist2(Ps, ws, ppie); acc["swP"] += hp; acc["sw2P"] += h2p
    n_total += len(ws)
    os.remove(BATCH_HEPMC)
    np.savez(OUT, We=We, Q2e=Q2e, ppie=ppie, n_events=n_total,
             ref_total_nb=REF_TOTAL_NB, **acc)       # INCREMENTAL save
    print(f"  batch {b+1}/{N_BATCHES}: total {n_total:,} -> {OUT}", flush=True)
print("done")
