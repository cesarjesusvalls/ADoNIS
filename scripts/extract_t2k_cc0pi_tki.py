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
        if pmu < 250.0 or mu[3] / pmu < -0.6:
            continue
        # NUISANCE T2K_CC0pi_STV: the HIGHEST-momentum proton must itself pass 450-1000, cos>0.4
        lead = max(protons, key=_mom); pl = _mom(lead)
        if not (450.0 < pl < 1000.0 and lead[3] / pl > 0.4):
            continue
        yield mu, lead, (evt["w"] or 1.0)


def observables(path):
    dpt, dat, w = [], [], []
    for mu, p, wt in cc0pi_events(path):
        lt = mu[1:3]; pt = p[1:3]; dvec = lt + pt
        dptmag = np.linalg.norm(dvec)
        dpt.append(float(dptmag))
        c = -np.dot(lt, dvec) / (np.linalg.norm(lt) * dptmag + 1e-9)
        dat.append(float(np.arccos(np.clip(c, -1, 1))))
        w.append(wt)
    return np.array(dpt), np.array(dat), np.array(w)


if __name__ == "__main__":
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "data/oracle/t2k_cc0pi_tki_achilles.npz"
    dpt, dat, w = observables(path)
    np.savez(out, dpt=dpt, dalphat=dat, w=w)
    print(f"CC0pi-Np signal events: {len(w)}   sum_w={w.sum():.4e} nb")
    print(f"  <dpT>={np.average(dpt, weights=w):.1f} MeV   <daT>={np.degrees(np.average(dat, weights=w)):.1f} deg")
    print("wrote", out)
