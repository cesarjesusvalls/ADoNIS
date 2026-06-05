"""Re-parse the neutrino CC oracle to EVENT-LEVEL arrays (not a frozen histogram), so
dsigma/dW etc. can be re-binned to any granularity without re-parsing.

Source: res_1pi_12C_nu_200k.hepmc (200k raw events; the original 1M run that made
oracle_distributions_nu.npz was not retained, but 200k supports ~60-90 bins at a few-%
statistical noise). Saves W, Q2, p_pi and per-event nb weights to a fine npz.

Run:  python oracle/reparse_nu_events.py
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_hepmc import parse_events, minkowski2
from parse_hepmc_nu import event_kin, REF_TOTAL_NB

HEPMC = Path("/Users/cjesus/Software/DiffSinglePiProd/Achilles/res_1pi_12C_nu_200k.hepmc")
OUT = Path(__file__).resolve().parent / "oracle_nu_events.npz"


def main():
    Q2s, Ws, ppis, weights = [], [], [], []
    n = 0
    for evt in parse_events(HEPMC):
        n += 1
        kin = event_kin(evt)
        if kin is None:
            continue
        Q2, W, p_pi = kin
        Q2s.append(Q2); Ws.append(W); ppis.append(p_pi); weights.append(evt["w"])
    Q2s, Ws, ppis, weights = map(np.array, (Q2s, Ws, ppis, weights))
    w_nb = weights * (REF_TOTAL_NB / weights.sum())
    print(f"parsed {n} events, {len(Q2s)} with FS pion; "
          f"W {np.nanmin(Ws):.0f}..{np.nanpercentile(Ws,99):.0f} MeV")
    np.savez(OUT, Q2=Q2s, W=Ws, ppi=ppis, w_nb=w_nb, ref_total_nb=REF_TOTAL_NB)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
