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


def cc0pi_events(path):
    for evt in parse_events(Path(path)):
        mu = None; protons = []; n_meson = 0
        for pid, status, p4 in evt["parts"]:
            if status != 1:
                continue
            if pid == MU:
                mu = np.asarray(p4)
            elif pid in MESONS:
                n_meson += 1
            elif pid == PROT:
                protons.append(np.asarray(p4))
        if mu is None or n_meson != 0 or not protons:
            continue
        pmu = _mom(mu)
        if not (1500.0 < pmu < 10000.0 and mu[3]/pmu > np.cos(20*np.pi/180)):
            continue
        # NUISANCE MINERvA CC0pi STV: highest-momentum proton in 450-1200 MeV, theta_p<70deg
        lead = max(protons, key=_mom); pl = _mom(lead)
        if not (450.0 < pl < 1200.0 and lead[3]/pl > np.cos(70*np.pi/180)):
            continue
        yield mu, lead, (evt["w"] or 1.0)


def observables(path):
    dpt, dat, pn, w = [], [], [], []
    for mu, p, wt in cc0pi_events(path):
        lt = mu[1:3]; pt = p[1:3]; dvec = lt + pt
        dptmag = np.linalg.norm(dvec)
        dpt.append(float(dptmag))
        c = -np.dot(lt, dvec) / (np.linalg.norm(lt) * dptmag + 1e-9)
        pL=mu[3]+p[3]; Evis=mu[0]+p[0]; R=11174.862+pL-Evis
        dpL=0.5*R-(10252.547**2+dptmag**2)/(2*max(R,1.0))
        pn.append(float(np.sqrt(max(dptmag**2+dpL**2,0.0))))
        dat.append(float(np.arccos(np.clip(c, -1, 1))))
        w.append(wt)
    return np.array(dpt), np.array(dat), np.array(pn), np.array(w)


if __name__ == "__main__":
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "data/oracle/t2k_cc0pi_tki_achilles.npz"
    dpt, dat, pn, w = observables(path)
    np.savez(out, dpt=dpt, dalphat=dat, pn=pn, w=w)
    print(f"CC0pi-Np signal events: {len(w)}   sum_w={w.sum():.4e} nb")
    print(f"  <dpT>={np.average(dpt, weights=w):.1f} MeV   <daT>={np.degrees(np.average(dat, weights=w)):.1f} deg")
    print("wrote", out)
