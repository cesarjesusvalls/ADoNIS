"""Extract T2K CC1pi+ STV observables (delta_pTT, p_N, delta_pT) from an ACHILLES full-event
hepmc (the paper run_T2K_virtresonances generation).

CC1pi+ signal (T2K, arXiv:2102.03346): exactly one mu-, exactly one pi+, at least one proton,
no other mesons.  The leading proton is the highest-momentum proton.  Per-event weights (nb)
are used; histograms are normalised to area for the shape comparison.

Usage:  python scripts/extract_t2k_cc1pi_tki.py <hepmc> [out.npz]
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adonis.data.oracle.parse_hepmc import parse_events

MU = 13; PIP = 211; PROT = 2212
MESONS = {111, 211, -211, 221, 130, 310, 311, 321, -321, -311}
M_A = 11174.862; M_A1 = 10252.547        # 12C, 11B [MeV]


def _cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def cc1pip_events(path):
    """Yield (p_mu4, p_pi4, p_lead_p4, weight) for CC1pi+ signal events (full 4-vectors)."""
    for evt in parse_events(Path(path)):
        mu = pip = None
        protons = []
        n_other_meson = 0
        n_pip = 0
        for pid, status, p4 in evt["parts"]:
            if status != 1:
                continue
            if pid == MU:
                mu = np.asarray(p4)
            elif pid == PIP:
                n_pip += 1; pip = np.asarray(p4)
            elif pid in MESONS:
                n_other_meson += 1
            elif pid == PROT:
                protons.append(np.asarray(p4))
        if mu is None or pip is None or n_pip != 1 or n_other_meson != 0 or not protons:
            continue
        lead = max(protons, key=lambda p: p[1] ** 2 + p[2] ** 2 + p[3] ** 2)
        yield mu, pip, lead, (evt["w"] or 1.0)


def observables(path):
    """delta_pTT, p_N, delta_alphaT, delta_pT for CC1pi+.  Beam = +z; the same definitions as
    adonis.observables.kinematics (p_N via Furmanski-Sobczyk with 12C->11B)."""
    beam = np.array([0.0, 0.0, 1.0])
    dptt, pn, dat, dpt, w = [], [], [], [], []
    for mu, pip, p, wt in cc1pip_events(path):
        mu3, pi3, p3 = mu[1:], pip[1:], p[1:]
        zhat = _cross(beam, mu3); zhat = zhat / (np.linalg.norm(zhat) + 1e-9)
        had3 = pi3 + p3
        dptt.append(float(np.dot(had3, zhat)))
        lt = mu3[:2]; dpt_vec = lt + had3[:2]; dptmag = np.linalg.norm(dpt_vec)
        dpt.append(float(dptmag))
        c = -np.dot(lt, dpt_vec) / (np.linalg.norm(lt) * dptmag + 1e-9)
        dat.append(float(np.arccos(np.clip(c, -1, 1))))
        # p_N (Furmanski-Sobczyk)
        pL = mu[3] + pip[3] + p[3]; Evis = mu[0] + pip[0] + p[0]
        R = M_A + pL - Evis
        dpL = 0.5 * R - (M_A1 ** 2 + dptmag ** 2) / (2.0 * max(R, 1.0))
        pn.append(float(np.sqrt(max(dptmag ** 2 + dpL ** 2, 0.0))))
        w.append(wt)
    return (np.array(dptt), np.array(pn), np.array(dat), np.array(dpt), np.array(w))


if __name__ == "__main__":
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "data/oracle/t2k_cc1pi_tki_achilles.npz"
    dptt, pn, dat, dpt, w = observables(path)
    np.savez(out, dptt=dptt, pn=pn, dalphat=dat, dpt=dpt, w=w)
    print(f"CC1pi+ signal events: {len(w)}   sum_w={w.sum():.4e} nb")
    print(f"  rms(dpTT)={np.sqrt(np.average(dptt**2, weights=w)):.1f}  <p_N>={np.average(pn, weights=w):.1f}  "
          f"<dpT>={np.average(dpt, weights=w):.1f}  <daT>={np.degrees(np.average(dat, weights=w)):.1f}deg")
    print("wrote", out)
