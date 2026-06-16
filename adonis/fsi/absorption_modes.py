"""Pion s-wave absorption charge modes -- a faithful transcription of ACHILLES
DeltaInteraction (src/Achilles/CascadeInteractions/DeltaInteractions.cc).

ACHILLES enumerates, per (pion, struck-nucleon) pair, the allowed (absorption_partner, {out1,out2})
modes (absorption_modes table, DeltaInteractions.cc:17-36) and competes them with the Nucl.Phys.
A568 isospin partition (DeltaInteractions.cc:152-185):
  opposite-isospin-partner modes share (5/6)*oset / opp_count
  same-isospin-partner     modes share (1/6)*oset / same_count
where opp/same_count = number of LOCALLY-PRESENT modes of each type (a mode whose required partner
species is absent nearby is dropped, FindClosest -> SIZE_MAX).  The emitted proton count is the literal
charge sum of the sampled mode -- there is NO isospin averaging on the charge.  ADoNIS previously chose
the partner GEOMETRICALLY (nearest allowed nucleon), which only coincides with this isospin branching
for N=Z (carbon); for neutron-rich targets (Ar) it systematically under-counts 2p / over-counts 1p.

This module derives, for any (pion, struck, partner-availability), the proton-count distribution
P(0p),P(1p),P(2p) and the partner species realizing each outcome -- the single source of truth the
discrete cascade samples from.  No fitted constants: the 5/6:1/6 split is the A568 reaction physics.
"""
from __future__ import annotations

PROTON, NEUTRON = 2212, 2112
PIP, PIM, PI0 = 211, -211, 111

# (pion, struck) -> list of (absorption_partner_pid, (out1_pid, out2_pid))   [VERBATIM ACHILLES]
ABSORPTION_MODES = {
    (PIP, PROTON):  [(NEUTRON, (PROTON, PROTON))],
    (PIP, NEUTRON): [(NEUTRON, (PROTON, NEUTRON)), (NEUTRON, (NEUTRON, PROTON)),
                     (PROTON, (PROTON, PROTON))],
    (PIM, PROTON):  [(NEUTRON, (NEUTRON, NEUTRON)), (PROTON, (NEUTRON, PROTON)),
                     (PROTON, (PROTON, NEUTRON))],
    (PIM, NEUTRON): [(PROTON, (NEUTRON, NEUTRON))],
    (PI0, PROTON):  [(PROTON, (PROTON, PROTON)), (NEUTRON, (PROTON, NEUTRON)),
                     (NEUTRON, (NEUTRON, PROTON))],
    (PI0, NEUTRON): [(NEUTRON, (NEUTRON, NEUTRON)), (PROTON, (NEUTRON, PROTON)),
                     (PROTON, (PROTON, NEUTRON))],
}


def mode_weights(pion, struck, has_p=True, has_n=True):
    """Replay ACHILLES's partition for one (pion, struck) pair given which partner species are present
    locally.  Returns list of (weight, n_protons_out, partner_pid), weights summing to 1 (or all-zero
    if no mode is realizable).  Mirrors DeltaInteractions.cc:152-185."""
    modes = ABSORPTION_MODES[(pion, struck)]
    present = [m for m in modes if (m[0] == PROTON and has_p) or (m[0] == NEUTRON and has_n)]
    same = [m for m in present if m[0] == struck]
    opp = [m for m in present if m[0] != struck]
    out = []
    for partner, outgoing in present:
        if partner == struck:
            w = (1.0 / 6.0) / len(same)
        else:
            w = (5.0 / 6.0) / len(opp)
        nprot = sum(1 for q in outgoing if q == PROTON)
        out.append((w, nprot, partner))
    tot = sum(w for w, _, _ in out)
    return [(w / tot, npr, p) for (w, npr, p) in out] if tot > 0 else []


def proton_count_dist(pion, struck, has_p=True, has_n=True):
    """P(0p), P(1p), P(2p) for one (pion, struck) pair."""
    p = [0.0, 0.0, 0.0]
    for w, npr, _ in mode_weights(pion, struck, has_p, has_n):
        p[npr] += w
    return tuple(p)


# cascade charge index -> pion PID (0:pi+, 1:pi0, 2:pi-), matching cascade_discrete `ch`.
_CHMAP = {0: PIP, 1: PI0, 2: PIM}


def kernel_tables():
    """(6,3) numpy tables indexed by `ch*2 + struck_p` (ch 0:pi+ 1:pi0 2:pi-; struck_p 1=proton):
      W[idx]    = [P(0p), P(1p), P(2p)] with BOTH partner species present locally,
      PART[idx,k] = partner species realizing exactly k protons (1=proton, 0=neutron, -1=unrealizable).
    Each proton-count maps to a unique partner species per (pion, struck), so PART is well-defined.
    The discrete cascade masks W by local partner availability and renormalizes per event."""
    import numpy as np
    W = np.zeros((6, 3)); PART = -np.ones((6, 3), np.int32)
    for ch in range(3):
        for sp in range(2):
            pion = _CHMAP[ch]; struck = PROTON if sp == 1 else NEUTRON
            idx = ch * 2 + sp
            for w, npr, partner in mode_weights(pion, struck):
                W[idx, npr] += w
                PART[idx, npr] = 1 if partner == PROTON else 0
    return W, PART
