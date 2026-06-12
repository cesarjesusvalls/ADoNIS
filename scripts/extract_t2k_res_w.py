"""Extract the UNSELECTED RES-event vertex-W spectrum from an ACHILLES no-FSI hepmc:
every event with >=1 primary pion in the final state (no-FSI -> the produced pion) is RES.
Stores (W, Q2, Enu, pi_p of the leading pion, struck |p| for the C/H split, weight).
Usage: python scripts/extract_t2k_res_w.py <nofsi.hepmc> <out.npz>"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adonis.data.oracle.parse_hepmc import parse_events

PIONS = {211, 111, -211}
W_l, Q2_l, E_l, pp_l, ps_l, w_l = [], [], [], [], [], []
pcth_l, np_l, npc_l, mu_p_l, mu_c_l, ppid_l = [], [], [], [], [], []
npi_l, pok_l = [], []
n_evt = 0
for evt in parse_events(Path(sys.argv[1])):
    n_evt += 1
    nu = mu = pstr = None; pis = []; nucs = []; nuc_pid = []
    for pid, status, p4 in evt["parts"]:
        if pid == 14 and (nu is None or p4[0] > nu[0]):
            nu = np.asarray(p4)
        if status == 2 and pid in (2112, 2212) and pstr is None:
            pstr = np.asarray(p4)
        if status != 1:
            continue
        if pid == 13:
            mu = np.asarray(p4)
        elif pid in PIONS:
            pis.append((pid, np.asarray(p4)))
        elif pid in (2112, 2212):
            nucs.append(np.asarray(p4))
            nuc_pid.append((pid, np.asarray(p4)))
    if not pis or mu is None or nu is None or pstr is None:
        continue
    lpid, lpi = max(pis, key=lambda t: np.linalg.norm(t[1][1:]))
    lnuc = max(nucs, key=lambda v: np.linalg.norm(v[1:])) if nucs else np.zeros(4)
    npi_l.append(len(pis))
    # leading IN-WINDOW proton present? (the FULL-signal proton leg)
    prot_ok = False
    for pidn, p4n in nuc_pid:
        pm = np.linalg.norm(p4n[1:])
        if pidn == 2212 and 450 < pm < 1200 and p4n[3]/max(pm,1e-9) > 0.342:
            prot_ok = True
    pok_l.append(prot_ok)
    pim = np.linalg.norm(lpi[1:]); num_ = np.linalg.norm(lnuc[1:]); mum = np.linalg.norm(mu[1:])
    pcth_l.append(float(lpi[3]/max(pim,1e-9))); ppid_l.append(int(lpid))
    np_l.append(float(num_)); npc_l.append(float(lnuc[3]/max(num_,1e-9)))
    mu_p_l.append(float(mum)); mu_c_l.append(float(mu[3]/max(mum,1e-9)))
    q = nu - mu; tot = q + pstr
    W_l.append(float(np.sqrt(max(tot[0]**2 - tot[1]**2 - tot[2]**2 - tot[3]**2, 0.0))))
    Q2_l.append(float(np.sum(q[1:]**2) - q[0]**2))
    E_l.append(float(nu[0]))
    pp_l.append(float(np.linalg.norm(lpi[1:])))
    ps_l.append(float(np.linalg.norm(pstr[1:])))
    w_l.append(evt["w"] or 1.0)
    if n_evt % 200000 == 0:
        print(f"  {n_evt} events parsed, {len(w_l)} RES", flush=True)
np.savez(sys.argv[2], W=np.array(W_l), Q2=np.array(Q2_l), Enu=np.array(E_l),
         pi_p=np.array(pp_l), pstr=np.array(ps_l), w=np.array(w_l),
         pi_cth=np.array(pcth_l), pi_pid=np.array(ppid_l), nuc_p=np.array(np_l),
         nuc_cth=np.array(npc_l), mu_p=np.array(mu_p_l), mu_cth=np.array(mu_c_l),
         n_pi=np.array(npi_l), prot_ok=np.array(pok_l))
print(f"RES events: {len(w_l)} / {n_evt} total;  wrote {sys.argv[2]}", flush=True)
