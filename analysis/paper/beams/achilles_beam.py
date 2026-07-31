"""ACHILLES CrossSection-mode hepmc -> sigma(p) for a TAGGED beam (pi+ / proton / neutron on 12C).

ACHILLES writes ONLY the events in which the beam particle REACTED; `NEvents` is therefore the number of
written (reacted) events, and the attempt count comes from the running GenCrossSection counter (its 4th
field).  With the beam's impact parameter uniform in a disk of radius R (card Cascade/Params/radius) and
|p| uniform in [P_LO, P_HI], the per-bin attempts are n_tried * (bin_width / range), so

    sigma_X(bin) = pi R^2 * n_X(bin) / n_tried(bin)          [mb]

The beam particle is the one with status 29 and px = py = 0 exactly (fired along +z).  That matters for a
NUCLEON beam: the struck nucleon is also status 29 with the same PID, so selecting on the PID alone would
pick the wrong particle.

Observables:
  pi+ beam : reaction   (every written event)  and  absorption   (reacted, NO surviving pion)
  p/n beam : reaction   (every written event)  and  pion production (any final-state pion = the
             NN -> N Delta -> NN pi inelastic channel -- the handle on s_NN_inelastic)
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from adonis.flux.hadron import BEAMS, PIR2_MB                      # noqa: E402

# ACHILLES cascade hepmc dir; ACHILLES_BEAM_DIR overrides (e.g. a repro tree with per-shard subdirs).
ACH_DIR = Path(os.environ.get("ACHILLES_BEAM_DIR",
                              str(Path(__file__).resolve().parents[3] / "output" / "achilles")))
PION_PIDS = (211, 111, -211)


def _scan(path, beam_pid):
    """Per WRITTEN (=reacted) event: (beam |p|, has_final_pion).  Returns (list, n_tried)."""
    out = []
    p_in = None
    has_pi = False
    n_tried = 0
    started = False
    with open(path) as fh:
        for line in fh:
            t = line[:2]
            if t == "E ":
                if started:
                    out.append((p_in, has_pi))
                started = True
                p_in = None
                has_pi = False
            elif t == "A " and "GenCrossSection" in line:
                n_tried = max(n_tried, int(line.split()[-1]))          # running attempt counter
            elif t == "P ":
                f = line.split()
                pid = int(f[3]); px = float(f[4]); py = float(f[5]); pz = float(f[6])
                status = int(f[9])
                # THE BEAM: status 29, our PID, fired exactly along +z (px = py = 0).  The struck nucleon
                # is also status 29 with the same PID for a nucleon beam -- hence the px/py test.
                if p_in is None and status == 29 and pid == beam_pid and px == 0.0 and py == 0.0:
                    p_in = abs(pz)
                if status == 1 and pid in PION_PIDS:
                    has_pi = True
    if started:
        out.append((p_in, has_pi))
    return [o for o in out if o[0] is not None], n_tried


_STEM = {
    "C":  {"pip": "run_cascade_pip_C_broad",  "prot": "run_cascade_prot_C",  "neut": "run_cascade_neut_C"},
    "Ar": {"pip": "run_cascade_pip_Ar_broad", "prot": "run_cascade_prot_Ar", "neut": "run_cascade_neut_Ar"},
}


def source_paths(beam, target="C"):
    """The hepmc files sigma_of_p will scan.  Public so a caller can fingerprint them for a cache
    without re-deriving the stem mapping (which would then be able to drift out of sync)."""
    return sorted(ACH_DIR.glob(f"**/{_STEM[target][beam]}*.hepmc"))   # flat OR per-shard subdirs


def sigma_of_p(beam, edges, target="C"):
    """Combine all seed batches -> (sigma_reaction, sigma_second, err_r, err_s, n_r, n_s, n_tried).
    `second` = absorption for a pion beam, pion production for a nucleon beam."""
    pid, species, _q = BEAMS[beam]
    paths = source_paths(beam, target)
    if not paths:
        raise FileNotFoundError(f"no ACHILLES hepmc for {beam} ({_STEM[target][beam]}*)")
    nb = len(edges) - 1
    n_r = np.zeros(nb); n_s = np.zeros(nb); n_tried = 0
    for p in paths:
        evts, nt = _scan(p, pid)
        n_tried += nt
        if not evts:
            continue
        pin = np.array([e[0] for e in evts]); hpi = np.array([e[1] for e in evts])
        idx = np.clip(np.digitize(pin, edges) - 1, 0, nb - 1)
        second = (~hpi) if species == "PION" else hpi      # absorbed (no pion) | pion-produced
        for b in range(nb):
            m = idx == b
            n_r[b] += int(m.sum())
            n_s[b] += int((m & second).sum())
    ntot = n_tried * (np.diff(edges) / (edges[-1] - edges[0]))          # uniform-in-p attempts per bin
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = PIR2_MB * n_r / ntot
        ss = PIR2_MB * n_s / ntot
        # BINOMIAL error on the reacted/tried efficiency (was Poisson sqrt(n_r)); = Poisson * sqrt(1-eps)
        er = PIR2_MB * np.sqrt(n_r * np.clip(1.0 - n_r / ntot, 0.0, 1.0)) / ntot
        es = PIR2_MB * np.sqrt(n_s * np.clip(1.0 - n_s / ntot, 0.0, 1.0)) / ntot
    return sr, ss, er, es, n_r, n_s, n_tried


if __name__ == "__main__":
    for beam, (lo, hi) in (("pip", (50, 1000)), ("prot", (300, 1400)), ("neut", (300, 1400))):
        edges = np.linspace(lo, hi, 16)
        try:
            sr, ss, er, es, nr, ns, nt = sigma_of_p(beam, edges)
        except FileNotFoundError as e:
            print(f"{beam}: {e}"); continue
        cen = 0.5 * (edges[:-1] + edges[1:])
        lab = "absorption" if beam == "pip" else "pion-production"
        print(f"\n== ACHILLES {beam} on C | {int(nr.sum()):,} reacted / {nt:,} tried ==")
        print(f"{'p [MeV/c]':>10} {'sigma_react [mb]':>18} {lab + ' [mb]':>22}")
        for i in range(len(cen)):
            print(f"{cen[i]:10.0f} {sr[i]:12.1f} +/- {er[i]:4.1f} {ss[i]:14.1f} +/- {es[i]:4.1f}")
