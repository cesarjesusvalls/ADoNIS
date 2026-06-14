"""DECISIVE check: what fraction of the ACHILLES CC1pi+Np signal has a STRUCK NEUTRON
(= n->n pi+, whose proton can only come from FSI knockout) vs a struck proton (p->p pi+)?
This decides whether removing the n->n pi+ recoil-neutron-as-proton (the recoil-pid fix) is
correct (ACHILLES really has few such signal events) or an over-removal (ACHILLES has many).

Applies the EXACT tight signal selection of extract_t2k_cc1pi_tki.py and tags each signal event
by the status-2 struck nucleon pid.  Streams the running fraction; stop once stable.
Usage: python -u scripts/cc1pi_channel_fraction.py [hepmc=_oracle_out/T2K_CH_virt.hepmc] [maxsig=8000]
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adonis.data.oracle.parse_hepmc import parse_events

HEPMC = sys.argv[1] if len(sys.argv) > 1 else "_oracle_out/T2K_CH_virt.hepmc"
MAXSIG = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
MU, PIP, PROT = 13, 211, 2212
MESONS = {111, 211, -211, 221, 130, 310, 311, 321, -321, -311}
COS70 = np.cos(70.0 * np.pi / 180.0)
MU_LO, MU_HI = 250.0, 7000.0
PI_LO, PI_HI = 150.0, 1200.0
P_LO, P_HI = 450.0, 1200.0


def mom(p4):
    return (p4[1] ** 2 + p4[2] ** 2 + p4[3] ** 2) ** 0.5


def inacc(p4, lo, hi):
    m = mom(p4)
    return lo < m < hi and (p4[3] / m) > COS70


w_n = w_p = 0.0          # struck-neutron / struck-proton signal weight (n->n pi+ / p->p pi+)
nsig = nstruck_carbon = 0
for evt in parse_events(Path(HEPMC)):
    mu = pip = None; protons = []; n_other = 0; n_pip = 0; struck_pid = None; struck_p = None
    for pid, status, p4 in evt["parts"]:
        if status == 2 and pid in (2112, 2212) and struck_pid is None:
            struck_pid = pid; struck_p = mom(p4)
        if status != 1:
            continue
        if pid == MU: mu = p4
        elif pid == PIP: n_pip += 1; pip = p4
        elif pid in MESONS: n_other += 1
        elif pid == PROT: protons.append(p4)
    if mu is None or pip is None or n_pip != 1 or n_other != 0 or not protons or struck_pid is None:
        continue
    if not (inacc(mu, MU_LO, MU_HI) and inacc(pip, PI_LO, PI_HI)):
        continue
    if not any(inacc(p, P_LO, P_HI) for p in protons):
        continue
    if struck_p < 1.0:          # free hydrogen (struck proton at rest) -> p->p pi+, but skip for the carbon split
        continue
    wt = evt["w"] or 1.0
    nsig += 1
    if struck_pid == 2112: w_n += wt          # n->n pi+ (proton is an FSI knockout)
    else: w_p += wt                            # p->p pi+ (native proton)
    if nsig % 1000 == 0:
        tot = w_n + w_p
        print(f"  signal(carbon)={nsig}  n->n pi+ frac = {w_n/max(tot,1e-30):.4f}  "
              f"(w_n={w_n:.3e} w_p={w_p:.3e})", flush=True)
    if nsig >= MAXSIG:
        break

tot = w_n + w_p
print(f"\nACHILLES CC1pi signal (carbon): {nsig} events")
print(f"  n->n pi+ (struck neutron, proton via FSI knockout) fraction = {w_n/max(tot,1e-30):.4f}")
print(f"  p->p pi+ (struck proton, native)                   fraction = {w_p/max(tot,1e-30):.4f}")
