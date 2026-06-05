"""Reaction cross section sigma(p_pi) from an achilles-cascade CrossSection-mode hepmc.

In CrossSection mode (RunCascade.cc::InitCrossSection) each event is a test pion fired at
a random impact parameter in a disc of radius R; the event weight is pi*R^2 (in nb) if the
pion interacted (a "hit") and 0 otherwise.  So the reaction cross section in a pion-momentum
bin is just the MEAN event weight in that bin:

    sigma_reaction(p) = < weight >_p = pi R^2 * P(interact | p)   [nb].

Usage:  python scripts/cascade_xsec_from_hepmc.py cascade.hepmc [nbins]
Prints  p_pi[MeV]  sigma[mb]  per bin (nb -> mb via /1e6).
"""
import sys
from pathlib import Path

import numpy as np

PI_PID = {211, 111, -211}


def events(path):
    """Yield (incoming_pion_|p|[MeV], weight) per event."""
    w = None
    p_in = None
    with open(path) as fh:
        for line in fh:
            t = line[:2]
            if t == "E ":
                if w is not None:
                    yield p_in, w
                w, p_in = None, None
            elif t == "W ":
                w = float(line.split()[1])
            elif t == "P ":
                f = line.split()
                pid, status = int(f[3]), int(f[9])
                # the beam test pion: a pion with an incoming/beam status (4) or the
                # largest-|pz| pion; take the first beam-status pion.
                if abs(pid) in (211, 111) and status in (4, 11) and p_in is None:
                    px, py, pz = float(f[4]), float(f[5]), float(f[6])
                    p_in = (px * px + py * py + pz * pz) ** 0.5
    if w is not None:
        yield p_in, w


def sigma_of_p(path, nbins=20):
    P, W = [], []
    for p_in, w in events(path):
        if p_in is None:
            continue
        P.append(p_in); W.append(w)
    P, W = np.array(P), np.array(W)
    lo, hi = P.min(), P.max()
    edges = np.linspace(lo, hi, nbins + 1)
    idx = np.clip(np.digitize(P, edges) - 1, 0, nbins - 1)
    cen = 0.5 * (edges[:-1] + edges[1:])
    sig = np.array([W[idx == b].mean() if np.any(idx == b) else 0.0 for b in range(nbins)])
    return cen, sig, len(P)


if __name__ == "__main__":
    path = Path(sys.argv[1])
    nb = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    cen, sig, n = sigma_of_p(path, nb)
    print(f"# reaction sigma(p_pi) from {path} ({n} events)")
    print("# p_pi[MeV]  sigma[mb]")
    for c, s in zip(cen, sig):
        print(f"{c:8.1f}  {s/1e6:8.2f}")
