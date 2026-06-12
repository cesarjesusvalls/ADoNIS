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
COS70 = np.cos(70.0 * np.pi / 180.0)     # 0.342, T2K forward acceptance

# T2K CC1pi+ STV "tight" signal phase space (NUISANCE T2K_CC1pipNp_CH_XSec_1DSTV_nu, PRD 103 112009):
# mu 250-7000, pi+ 150-1200, leading proton 450-1200 MeV/c, all theta<70deg wrt the beam.
MU_LO, MU_HI = 250.0, 7000.0
PI_LO, PI_HI = 150.0, 1200.0
P_LO, P_HI = 450.0, 1200.0


def _mom(p4):
    return np.sqrt(p4[1] ** 2 + p4[2] ** 2 + p4[3] ** 2)


def _in_accept(p4, lo, hi):
    m = _mom(p4)
    return lo < m < hi and (p4[3] / m) > COS70          # momentum window + theta<70 wrt +z beam


def cc1pip_events(path):
    """Yield (p_mu4, p_pi4, p_lead_p4, weight, is_hydrogen) for CC1pi+ TIGHT-signal events.
    is_hydrogen = struck nucleon at rest (free proton, delta_pTT==0) -> NUISANCE flat daT throw."""
    for evt in parse_events(Path(path)):
        mu = pip = None
        protons = []; n_other_meson = 0; n_pip = 0; struck_p = None
        for pid, status, p4 in evt["parts"]:
            if status != 1:
                if abs(pid) in (2112, 2212) and status in (4, 11, 21) and struck_p is None:
                    struck_p = (p4[1] ** 2 + p4[2] ** 2 + p4[3] ** 2) ** 0.5   # initial nucleon |p|
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
        # tight acceptance: mu, pi, and >=1 proton in the windows; leading accepted proton
        if not (_in_accept(mu, MU_LO, MU_HI) and _in_accept(pip, PI_LO, PI_HI)):
            continue
        acc_p = [p for p in protons if _in_accept(p, P_LO, P_HI)]
        if not acc_p:
            continue
        lead = max(acc_p, key=_mom)
        is_h = struck_p is not None and struck_p < 1.0      # free proton at rest
        yield mu, pip, lead, (evt["w"] or 1.0), is_h


def observables(path, seed=0):
    """delta_pTT, p_N, delta_alphaT, delta_pT for the TIGHT CC1pi+ sample.  delta_alphaT follows
    NUISANCE exactly: acos(-ptmu.dpt/(|ptmu||dpt|)) for bound (carbon), but a FLAT random throw in
    [0,pi] for free-hydrogen events (delta_pT=0 -> undefined; NUISANCE issue 92 / PRD 103 112009)."""
    beam = np.array([0.0, 0.0, 1.0])
    rng = np.random.default_rng(seed)
    dptt, pn, dat, dpt, w, ish = [], [], [], [], [], []
    for mu, pip, p, wt, is_h in cc1pip_events(path):
        ish.append(bool(is_h))
        mu3, pi3, p3 = mu[1:], pip[1:], p[1:]
        zhat = np.cross(beam, mu3); zhat = zhat / (np.linalg.norm(zhat) + 1e-9)
        had3 = pi3 + p3
        dptt_v = float(np.dot(had3, zhat)); dptt.append(dptt_v)
        lt = mu3[:2]; dpt_vec = lt + had3[:2]; dptmag = np.linalg.norm(dpt_vec)
        dpt.append(float(dptmag))
        # hydrogen tag = delta_pTT == 0 (free proton; carbon has Fermi spread) -> NUISANCE flat daT
        if is_h or abs(dptt_v) < 0.5:
            dat.append(float(rng.uniform(0.0, np.pi)))      # NUISANCE hydrogen prescription
        else:
            c = -np.dot(lt, dpt_vec) / (np.linalg.norm(lt) * dptmag + 1e-9)
            dat.append(float(np.arccos(np.clip(c, -1, 1))))
        pL = mu[3] + pip[3] + p[3]; Evis = mu[0] + pip[0] + p[0]
        R = M_A + pL - Evis
        dpL = 0.5 * R - (M_A1 ** 2 + dptmag ** 2) / (2.0 * max(R, 1.0))
        pn.append(float(np.sqrt(max(dptmag ** 2 + dpL ** 2, 0.0))))
        w.append(wt)
    return (np.array(dptt), np.array(pn), np.array(dat), np.array(dpt), np.array(w), np.array(ish))


if __name__ == "__main__":
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "data/oracle/t2k_cc1pi_tki_achilles.npz"
    dptt, pn, dat, dpt, w, ish = observables(path)
    # NOTE: the status-code is_h tag fails on this hepmc (0 tagged of an 18k H component);
    # the saved is_h uses the EXACT kinematic tag instead: free-proton events have dptt == 0
    # (no Fermi motion; only 0.24% of carbon events fall within |dptt|<0.5).
    np.savez(out, dptt=dptt, pn=pn, dalphat=dat, dpt=dpt, w=w, is_h=(np.abs(dptt) < 0.5))
    print(f"CC1pi+ signal events: {len(w)}   sum_w={w.sum():.4e} nb")
    print(f"  rms(dpTT)={np.sqrt(np.average(dptt**2, weights=w)):.1f}  <p_N>={np.average(pn, weights=w):.1f}  "
          f"<dpT>={np.average(dpt, weights=w):.1f}  <daT>={np.degrees(np.average(dat, weights=w)):.1f}deg")
    print("wrote", out)
