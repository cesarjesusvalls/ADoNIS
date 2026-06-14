"""Extract the ACHILLES CC1pi RICH per-event bank from a hepmc: full final-state 4-vectors so ANY
signal definition is pure re-binning (scripts/cc1pi_signal.py).  Run once per hepmc:
  FSI    : _oracle_out/T2K_CH_virt.hepmc        -> data/oracle/t2k_cc1pi_rich_ach_FSI.npz
  no-FSI : _oracle_out/T2K_CH_virt_nofsi.hepmc  -> data/oracle/t2k_cc1pi_rich_ach_nofsi.npz

Per event (NO signal cut applied -- the cut is the re-binning step): mu 4-vec, ALL pions (4-vec+pid,
padded to K), ALL protons (4-vec, padded to M), struck nucleon 4-vec + pid, nu 4-vec, n_other_meson,
weight.  Stores the hepmc GenCrossSection/sum_w (absolute nb) via hepmc_norm.
Usage: python -u scripts/extract_cc1pi_rich.py <hepmc> <out.npz> [Kpi=4] [Mprot=10]
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adonis.data.oracle.parse_hepmc import parse_events
from adonis.data.oracle.normalization import hepmc_norm

HEPMC, OUT = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 4
M = int(sys.argv[4]) if len(sys.argv) > 4 else 10
MU, PIP, PROT = 13, 211, 2212
PIONS = {111, 211, -211}
MESONS = {111, 211, -211, 221, 130, 310, 311, 321, -321, -311}

mu_l, nu_l, st_l, sp_l, w_l, nom_l = [], [], [], [], [], []
pip4_l, pipid_l, pr4_l = [], [], []
n_evt = nkept = 0
for evt in parse_events(Path(HEPMC)):
    n_evt += 1
    mu = nu = struck = None; struck_pid = 0; pions = []; protons = []; n_other = 0
    for pid, status, p4 in evt["parts"]:
        if pid == 14 and (nu is None or p4[0] > nu[0]):
            nu = p4
        if status == 2 and pid in (2112, 2212) and struck is None:
            struck = p4; struck_pid = pid
        if status != 1:
            continue
        if pid == MU:
            mu = p4
        elif pid in PIONS:
            pions.append((pid, p4))
            if pid != PIP:
                n_other += 1
        elif pid in MESONS:
            n_other += 1
        elif pid == PROT:
            protons.append(p4)
    if mu is None or nu is None or struck is None:
        continue
    nkept += 1
    # pad pions (4-vec + pid) to K, protons (4-vec) to M
    p4p = np.zeros((K, 4)); pidp = np.zeros(K, np.int64)
    for i, (pp, q) in enumerate(sorted(pions, key=lambda t: -np.linalg.norm(t[1][1:]))[:K]):
        p4p[i] = q; pidp[i] = pp
    prp = np.zeros((M, 4))
    for i, q in enumerate(sorted(protons, key=lambda v: -np.linalg.norm(v[1:]))[:M]):
        prp[i] = q
    mu_l.append(mu); nu_l.append(nu); st_l.append(struck); sp_l.append(struck_pid)
    pip4_l.append(p4p); pipid_l.append(pidp); pr4_l.append(prp)
    n_other_pi0_minus = n_other
    nom_l.append(n_other_pi0_minus); w_l.append(evt["w"] or 1.0)
    if n_evt % 200000 == 0:
        print(f"  {n_evt} parsed, {nkept} kept", flush=True)

nrm = hepmc_norm(HEPMC)
np.savez(OUT, mu=np.array(mu_l), nu=np.array(nu_l), struck=np.array(st_l),
         struck_pid=np.array(sp_l), pi_p4=np.array(pip4_l), pi_pid=np.array(pipid_l),
         prot_p4=np.array(pr4_l), n_other_meson=np.array(nom_l), w=np.array(w_l),
         gen_xs_pb=nrm["gen_xs_pb"], sum_w_all=nrm["sum_w_all"], weight_to_nb=nrm["weight_to_nb"])
print(f"kept {nkept}/{n_evt}; wrote {OUT}  weight_to_nb={nrm['weight_to_nb']:.4e}", flush=True)
