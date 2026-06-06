"""Reaction AND absorption sigma(p_pi) from an achilles-cascade CrossSection-mode hepmc.

CrossSection mode fires a test pion at a random impact parameter in a disc of radius R; the
event weight is pi R^2 [nb] if the pion interacted ("hit"), else 0.  So:
    sigma_reaction(p) = <weight>_p                                  (any interaction)
    sigma_abs(p)      = <weight * [no final-state pion]>_p          (pion absorbed/removed)
The incoming test pion is status 29 (external_test); final-state particles are status 1.

Usage:  python scripts/cascade_abs_from_hepmc.py file1.hepmc [file2 ...]  --nbins 14
Prints  p_pi[MeV]  sigma_reaction[mb]  sigma_abs[mb]  per bin (nb -> mb via /1e6).
"""
import sys
from pathlib import Path
import numpy as np

PI_PIDS = {211, 111, -211}


def events(path):
    """Yield (p_in[MeV], has_final_pion) per ACCEPTED (interacted) event, plus expose the
    per-file (pi R^2 [mb], n_accepted, n_attempted) via attributes on the generator's return.

    In CrossSection mode every WRITTEN event interacted; the per-event weight is the constant
    pi R^2.  The reaction cross section is sigma = pi R^2 * (accepted/attempted), with
    accepted/attempted carried in the final HepMC `GenCrossSection acc att` fields.  Beam is
    uniform in p over [80,500], so attempts are uniform in p -> per-bin sigma is recoverable."""
    w = p_in = None
    has_pi = False
    acc = att = None
    with open(path) as fh:
        for line in fh:
            t = line[:2]
            if t == "E ":
                if p_in is not None:
                    yield p_in, has_pi
                p_in, has_pi = None, False
            elif t == "W ":
                tok = line.split()
                try:
                    w = float(tok[1])
                except (IndexError, ValueError):
                    pass
            elif line.startswith("A 0 GenCrossSection"):
                tok = line.split()
                try:
                    acc, att = int(tok[5]), int(tok[6])   # accepted, attempted (running)
                except (IndexError, ValueError):
                    pass
            elif t == "P ":
                f = line.split()
                try:
                    pid, status = int(f[3]), int(f[9])
                except (IndexError, ValueError):
                    continue
                px, py, pz = float(f[4]), float(f[5]), float(f[6])
                if abs(pid) in (211, 111) and status == 29 and p_in is None:
                    p_in = (px * px + py * py + pz * pz) ** 0.5
                elif status == 1 and pid in PI_PIDS:
                    has_pi = True
    if p_in is not None:
        yield p_in, has_pi
    events.last = (w / 1e12 if w else None, acc, att)    # (pi R^2 [mb], accepted, attempted)


def sigmas(paths, nbins=14, lo=80.0, hi=500.0):
    """Absolute sigma_reaction(p) and sigma_abs(p) [mb].  Attempts are uniform in p, so the
    per-bin attempt count = total_attempts * (binwidth / range)."""
    P, ABSV = [], []
    piR2 = 0.0
    tot_att = 0
    for path in paths:
        for p_in, has_pi in events(path):
            if p_in is None:
                continue
            P.append(p_in); ABSV.append(0 if has_pi else 1)
        a, acc, att = events.last
        if a:
            piR2 = a
        if att:
            tot_att += att
    P, ABSV = np.array(P), np.array(ABSV)
    edges = np.linspace(lo, hi, nbins + 1)
    idx = np.clip(np.digitize(P, edges) - 1, 0, nbins - 1)
    cen = 0.5 * (edges[:-1] + edges[1:])
    att_per_bin = tot_att * (1.0 / nbins)                 # attempts uniform in p
    nacc = np.array([int(np.sum(idx == b)) for b in range(nbins)])
    nabs = np.array([int(np.sum((idx == b) & (ABSV == 1))) for b in range(nbins)])
    sig_r = piR2 * nacc / att_per_bin
    sig_a = piR2 * nabs / att_per_bin
    return cen, sig_r, sig_a, nacc, len(P)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    nb = 14
    if "--nbins" in sys.argv:
        nb = int(sys.argv[sys.argv.index("--nbins") + 1])
    cen, sr, sa, n, ntot = sigmas(args, nb)
    print(f"# reaction+absorption sigma(p_pi) from {len(args)} file(s), {ntot} events")
    print(f"# {'p_pi[MeV]':>9} {'sig_reac[mb]':>12} {'sig_abs[mb]':>11} {'n':>6}")
    for c, r, a, ni in zip(cen, sr, sa, n):
        print(f"{c:11.1f} {r:12.2f} {a:11.2f} {ni:6d}")
