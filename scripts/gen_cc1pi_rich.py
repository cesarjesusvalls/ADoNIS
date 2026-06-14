"""BANK the ADoNIS CC1pi RICH per-event record from the CENTRAL res_C chain (return_raw=True):
event-matched PRE- and POST-FSI 4-vectors + every proton candidate (recoil + knockouts) with species.
One generation -> ANY signal definition is then pure re-binning (scripts/cc1pi_signal.py).

Usage: python -u scripts/gen_cc1pi_rich.py [NRES=150000] [NSEED=4]
Output: data/oracle/t2k_cc1pi_rich_adonis.npz
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import scripts.cc1pi_fig_tki as F

NRES = int(sys.argv[1]) if len(sys.argv) > 1 else 150000
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 4


def main():
    parts = []
    for sd in range(NSEED):
        d = F.res_C(NRES, sd, return_raw=True)
        d["w"] = d["w"] / NSEED
        parts.append(d)
        print(f"  seed {sd+1}/{NSEED}: +{len(d['w'])} events", flush=True)
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez("data/oracle/t2k_cc1pi_rich_adonis.npz", **out)
    import scripts.cc1pi_signal as S
    for fsi in (True, False):
        sd = dict(S.DEFAULT, fsi=fsi)
        r = S.ado_select(out, sd)
        print(f"  [{'FSI ' if fsi else 'noFSI'} carbon signal] sigma={r['w'].sum():.4e} nb  n={len(r['w'])}", flush=True)
    print(f"banked {len(out['w'])} events -> data/oracle/t2k_cc1pi_rich_adonis.npz", flush=True)


if __name__ == "__main__":
    main()
