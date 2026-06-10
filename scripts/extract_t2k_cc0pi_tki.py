"""Extract T2K CC0pi-Np STV observables (delta_pT, delta_alphaT) from an ACHILLES full-event
hepmc, with the T2K signal phase-space cuts (arXiv:1802.05078):
  muon:   p_mu > 250 MeV/c,  cos(theta_mu) > -0.6
  proton: 450 < p_p < 1000 MeV/c,  cos(theta_p) > 0.4
  no mesons in the final state (CC0pi).  Leading proton = highest momentum within the window.

Usage:  python scripts/extract_t2k_cc0pi_tki.py <hepmc> [out.npz]
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adonis.data.oracle.parse_hepmc import parse_events

MU = 13; PROT = 2212
MESONS = {111, 211, -211, 221, 130, 310, 311, 321, -321, -311}


def _mom(p4):
    return np.sqrt(p4[1] ** 2 + p4[2] ** 2 + p4[3] ** 2)


NU = 14    # nu_mu (T2K); the incoming-beam neutrino (max-E pid14, any status) sets Q^2


NUCLEONS = {2112, 2212}    # struck nucleon recorded with status 2 (initial-state, off-shell)


def cc0pi_events(path):
    for evt in parse_events(Path(path)):
        mu = None; protons = []; n_meson = 0; nu = None; pstr = None
        for pid, status, p4 in evt["parts"]:
            if pid == NU and (nu is None or p4[0] > nu[0]):
                nu = np.asarray(p4)                 # beam neutrino (max energy)
            if status == 2 and pid in NUCLEONS and pstr is None:
                pstr = np.asarray(p4)               # struck (initial-state) nucleon for vertex W
            if status != 1:
                continue
            if pid == MU:
                mu = np.asarray(p4)
            elif pid in MESONS:
                n_meson += 1
            elif pid == PROT:
                protons.append(np.asarray(p4))
        if mu is None or nu is None or n_meson != 0 or not protons:
            continue
        pmu = _mom(mu)
        if pmu < 250.0 or mu[3] / pmu < -0.6:
            continue
        # NUISANCE T2K_CC0pi_STV: the HIGHEST-momentum proton must itself pass 450-1000, cos>0.4
        lead = max(protons, key=_mom); pl = _mom(lead)
        if not (450.0 < pl < 1000.0 and lead[3] / pl > 0.4):
            continue
        yield nu, mu, lead, pstr, (evt["w"] or 1.0), (evt["proc"] or 0)


def observables(path):
    dpt, dat, q2, w, wvar, proc = [], [], [], [], [], []
    for nu, mu, p, pstr, wt, pc in cc0pi_events(path):
        lt = mu[1:3]; pt = p[1:3]; dvec = lt + pt
        dptmag = np.linalg.norm(dvec)
        dpt.append(float(dptmag))
        c = -np.dot(lt, dvec) / (np.linalg.norm(lt) * dptmag + 1e-9)
        dat.append(float(np.arccos(np.clip(c, -1, 1))))
        qv = nu - mu
        q2.append(float((qv[1] ** 2 + qv[2] ** 2 + qv[3] ** 2 - qv[0] ** 2) / 1e6))   # [GeV^2]
        if pstr is not None:
            tot = qv + pstr                       # vertex hadronic 4-mom = q + struck nucleon
            w2 = tot[0] ** 2 - (tot[1] ** 2 + tot[2] ** 2 + tot[3] ** 2)
            wvar.append(float(np.sqrt(max(w2, 0.0))))   # vertex W [MeV]
        else:
            wvar.append(np.nan)
        w.append(wt); proc.append(pc)
    return (np.array(dpt), np.array(dat), np.array(q2), np.array(wvar),
            np.array(w), np.array(proc))


if __name__ == "__main__":
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "data/oracle/t2k_cc0pi_tki_achilles.npz"
    dpt, dat, q2, wvar, w, proc = observables(path)
    np.savez(out, dpt=dpt, dalphat=dat, Q2=q2, W=wvar, w=w, proc=proc)
    qe = proc == 200; res = proc >= 400
    print(f"CC0pi-Np signal events: {len(w)}   sum_w={w.sum():.4e} nb")
    print(f"  by process: QE(200) {int(qe.sum())} ev sum_w={w[qe].sum():.4e}   "
          f"RES(401/402) {int(res.sum())} ev sum_w={w[res].sum():.4e}   "
          f"other {int((~qe & ~res).sum())}")
    print(f"  <dpT>={np.average(dpt, weights=w):.1f} MeV   <daT>={np.degrees(np.average(dat, weights=w)):.1f} deg"
          f"   <Q2>={np.average(q2, weights=w):.3f} GeV^2")
    nW = int(np.isfinite(wvar).sum())
    if nW:
        m = np.isfinite(wvar)
        print(f"  <W>={np.average(wvar[m], weights=w[m]):.1f} MeV   ({nW}/{len(w)} have struck nucleon)")
    print("wrote", out)
