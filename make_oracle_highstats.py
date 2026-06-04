"""Generate a HIGH-STATISTICS neutrino oracle as accumulated HISTOGRAMS (not a giant
hepmc): run ACHILLES in batches with distinct seeds, histogram each batch (sum w and
sum w^2 per bin -> per-bin weighted statistical error), accumulate, DELETE each batch's
hepmc, and save the combined histograms with errors.

Per-event weights are on the same absolute scale across batches (same generator/xsec),
so pooling sum(w), sum(w^2) per bin over all batches is the 10M-event weighted estimate;
the per-bin error is sqrt(sum w^2), the relative error shrinking as 1/sqrt(N).

Stored on a FINE fixed binning so the comparison can re-bin down to any coarser multiple.

Run (from the diffpi dir):  python make_oracle_highstats.py
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
OUT = Path("oracle/oracle_highstats.npz")

N_PER_BATCH = 500_000     # 20 x 500k = 10M events (each hepmc ~670MB, deleted after)
N_BATCHES = 20
BATCH_HEPMC = "/tmp/oracle_batch.hepmc"

# fine fixed binning (re-binnable down to coarser multiples)
We = np.linspace(1076.0, 1700.0, 313)        # ~2 MeV bins
Q2e = np.linspace(0.0, 2.0e6, 201)           # 0.01 GeV^2 bins
ppie = np.linspace(0.0, 700.0, 141)          # 5 MeV bins

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
    yml = f"/tmp/oracle_batch_{b}.yml"
    write_yml(1000 + b, BATCH_HEPMC, yml)
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
    print(f"  batch {b+1}/{N_BATCHES}: {len(ws):,} events, total {n_total:,}")

np.savez(OUT, We=We, Q2e=Q2e, ppie=ppie, n_events=n_total,
         ref_total_nb=REF_TOTAL_NB, **acc)
print(f"saved -> {OUT}  ({n_total:,} total events)")
