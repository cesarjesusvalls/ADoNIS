"""Topology-only CC1pi+ extraction storing the SUPERSET needed for the 2x2 diagnostic
(signal-def / no-signal-def) x (FSI / no-FSI).  ONE pass per hepmc.

Topology (no acceptance windows): exactly one mu-, exactly one pi+, no other meson, >=1 proton.
Per event we store mu/pi kinematics, vertex W/Q2/Enu, and the STV observables computed BOTH
with the leading-OVERALL proton (for the no-signal-def cell) and the leading-ACCEPTED proton
(450-1200 MeV, theta<70; for the signal-def cell).  The figure applies the four masks downstream.

Usage: python scripts/extract_t2k_cc1pi_4way.py <hepmc> <out.npz>
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adonis.data.oracle.parse_hepmc import parse_events
from adonis.data.oracle.normalization import hepmc_norm

MU = 13; PIP = 211; PROT = 2212
MESONS = {111, 211, -211, 221, 130, 310, 311, 321, -321, -311}
M_A = 11174.862; M_A1 = 10252.547
COS70 = np.cos(70.0 * np.pi / 180.0)
MU_LO, MU_HI = 250.0, 7000.0
PI_LO, PI_HI = 150.0, 1200.0
P_LO, P_HI = 450.0, 1200.0
BEAM = np.array([0.0, 0.0, 1.0])


def _mom(p4):
    return float(np.sqrt(p4[1] ** 2 + p4[2] ** 2 + p4[3] ** 2))


def _in_accept(p4, lo, hi):
    m = _mom(p4)
    return (lo < m < hi) and (p4[3] / max(m, 1e-9)) > COS70


def _stv(mu, pip, p, is_h, rng):
    """(dptt, pn, dat, dpt) for muon, pi+, proton 4-vectors -- mirrors extract_t2k_cc1pi_tki."""
    mu3, pi3, p3 = mu[1:], pip[1:], p[1:]
    zhat = np.cross(BEAM, mu3); zhat = zhat / (np.linalg.norm(zhat) + 1e-9)
    had3 = pi3 + p3
    dptt = float(np.dot(had3, zhat))
    lt = mu3[:2]; dpt_vec = lt + had3[:2]; dptmag = float(np.linalg.norm(dpt_vec))
    if is_h or abs(dptt) < 0.5:
        dat = float(rng.uniform(0.0, np.pi))
    else:
        c = -np.dot(lt, dpt_vec) / (np.linalg.norm(lt) * dptmag + 1e-9)
        dat = float(np.arccos(np.clip(c, -1, 1)))
    pL = mu[3] + pip[3] + p[3]; Evis = mu[0] + pip[0] + p[0]
    R = M_A + pL - Evis
    dpL = 0.5 * R - (M_A1 ** 2 + dptmag ** 2) / (2.0 * max(R, 1.0))
    pn = float(np.sqrt(max(dptmag ** 2 + dpL ** 2, 0.0)))
    return dptt, pn, dat, dptmag


def main():
    path = sys.argv[1]; out = sys.argv[2]
    rng = np.random.default_rng(0)
    cols = {k: [] for k in (
        "w", "is_h", "mu_p", "mu_cth", "pi_p", "pi_cth", "W", "Q2", "Enu",
        "dptt_all", "pn_all", "dat_all", "dpt_all", "lp_p_all", "lp_cth_all",
        "has_acc", "dptt_acc", "pn_acc", "dat_acc", "dpt_acc", "lp_p_acc",
        "mu_acc", "pi_acc")}
    n_evt = 0
    for evt in parse_events(Path(path)):
        n_evt += 1
        if n_evt % 200000 == 0:
            print(f"  parsed {n_evt} events, kept {len(cols['w'])}", flush=True)
        mu = pip = None; protons = []; n_other = 0; n_pip = 0
        nu = None; pstr = None; struck_p = None
        for pid, status, p4 in evt["parts"]:
            if pid == 14 and (nu is None or p4[0] > nu[0]):
                nu = np.asarray(p4)
            if status == 2 and pid in (2112, 2212) and pstr is None:
                pstr = np.asarray(p4); struck_p = _mom(p4)
            if status != 1:
                continue
            if pid == MU:
                mu = np.asarray(p4)
            elif pid == PIP:
                n_pip += 1; pip = np.asarray(p4)
            elif pid in MESONS:
                n_other += 1
            elif pid == PROT:
                protons.append(np.asarray(p4))
        if mu is None or pip is None or n_pip != 1 or n_other != 0 or not protons:
            continue
        is_h = struck_p is not None and struck_p < 1.0
        lead_all = max(protons, key=_mom)
        acc_p = [p for p in protons if _in_accept(p, P_LO, P_HI)]
        lead_acc = max(acc_p, key=_mom) if acc_p else None
        # vertex W/Q2/Enu
        if nu is not None and pstr is not None:
            q = nu - mu; tot = q + pstr
            W = float(np.sqrt(max(tot[0] ** 2 - tot[1] ** 2 - tot[2] ** 2 - tot[3] ** 2, 0.0)))
            Q2 = float((q[1] ** 2 + q[2] ** 2 + q[3] ** 2) - q[0] ** 2)
            Enu = float(nu[0])
        else:
            W = Q2 = Enu = 0.0
        d_all = _stv(mu, pip, lead_all, is_h, rng)
        cols["w"].append(float(evt["w"] or 1.0)); cols["is_h"].append(bool(is_h))
        cols["mu_p"].append(_mom(mu)); cols["mu_cth"].append(float(mu[3] / max(_mom(mu), 1e-9)))
        cols["pi_p"].append(_mom(pip)); cols["pi_cth"].append(float(pip[3] / max(_mom(pip), 1e-9)))
        cols["W"].append(W); cols["Q2"].append(Q2); cols["Enu"].append(Enu)
        cols["dptt_all"].append(d_all[0]); cols["pn_all"].append(d_all[1])
        cols["dat_all"].append(d_all[2]); cols["dpt_all"].append(d_all[3])
        cols["lp_p_all"].append(_mom(lead_all))
        cols["lp_cth_all"].append(float(lead_all[3] / max(_mom(lead_all), 1e-9)))
        cols["mu_acc"].append(_in_accept(mu, MU_LO, MU_HI))
        cols["pi_acc"].append(_in_accept(pip, PI_LO, PI_HI))
        if lead_acc is not None:
            d_acc = _stv(mu, pip, lead_acc, is_h, rng)
            cols["has_acc"].append(True); cols["dptt_acc"].append(d_acc[0])
            cols["pn_acc"].append(d_acc[1]); cols["dat_acc"].append(d_acc[2])
            cols["dpt_acc"].append(d_acc[3]); cols["lp_p_acc"].append(_mom(lead_acc))
        else:
            cols["has_acc"].append(False)
            for k in ("dptt_acc", "pn_acc", "dat_acc", "dpt_acc", "lp_p_acc"):
                cols[k].append(np.nan)
    nrm = hepmc_norm(path)
    arrs = {k: np.array(v) for k, v in cols.items()}
    np.savez(out, gen_xs_pb=nrm["gen_xs_pb"], sum_w_all=nrm["sum_w_all"],
             weight_to_nb=nrm["weight_to_nb"], n_parsed=n_evt, **arrs)
    print(f"parsed {n_evt}; kept {len(arrs['w'])} CC1pi+ topology events; "
          f"acc-proton {int(arrs['has_acc'].sum())}; wrote {out}", flush=True)


if __name__ == "__main__":
    main()
