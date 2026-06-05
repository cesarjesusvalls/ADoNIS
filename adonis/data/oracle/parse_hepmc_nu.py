"""Parse the NEUTRINO CC single-pion oracle (res_1pi_12C_nu.hepmc) into weak
differential distributions dsigma/dQ^2, dsigma/dW, pion |p| -- the bin-by-bin
targets to validate the WEAK (vector+axial) DCC assembly and the M_A knob.

Mirrors parse_hepmc.py but the incoming lepton is the neutrino (nu_e, pid 12,
status 4) and the outgoing lepton is the charged lepton (e-, pid 11, status 1).
Reuses parse_events / minkowski2 from parse_hepmc.

Run:  python oracle/parse_hepmc_nu.py
"""
from pathlib import Path
import sys

import numpy as np

from adonis.data.oracle.parse_hepmc import parse_events, minkowski2

HEPMC = Path("/Users/cjesus/Software/DiffSinglePiProd/Achilles/res_1pi_12C_nu.hepmc")
OUT = Path("data/oracle/oracle_distributions_nu.npz")
REF_TOTAL_NB = 4.943360e-05      # /tmp/nu1M.log Total xsec (CC, 1500 MeV nu_e on 12C)

PI_PIDS = {111, 211, -211}
NU_PIDS = {12, 14, 16, -12, -14, -16}
CHG_LEP = {11, 13, -11, -13}


def event_kin(evt):
    k_in = k_out = p_pi = p_struck = None
    for pid, status, p4 in evt["parts"]:
        if pid in NU_PIDS and status == 4:
            k_in = p4
        elif pid in CHG_LEP and status == 1:
            k_out = p4
        elif pid in PI_PIDS and status == 1:
            p_pi = p4
        elif abs(pid) in (2112, 2212) and status == 2:
            p_struck = p4
    if k_in is None or k_out is None or p_pi is None:
        return None
    q = k_in - k_out
    Q2 = -minkowski2(q)
    W = np.sqrt(max(minkowski2(q + p_struck), 0.0)) if p_struck is not None else np.nan
    p_pi_mag = np.sqrt(np.sum(p_pi[1:] ** 2))
    return Q2, W, p_pi_mag


def _boost_to_rest(P, a):
    """Boost 4-vector a=(E,px,py,pz) into the rest frame of P (matches
    diffpi.lepton_tensor.boost_to_rest / observables.cos_theta_star)."""
    M = np.sqrt(max(minkowski2(P), 1e-9))
    gamma = P[0] / M
    beta = P[1:] / P[0]
    b2 = max(np.sum(beta ** 2), 1e-30)
    bda = np.dot(beta, a[1:])
    a0 = gamma * (a[0] - bda)
    avec = a[1:] + ((gamma - 1.0) * bda / b2 - gamma * a[0]) * beta
    return np.concatenate([[a0], avec])


def event_kin_full(evt):
    """Full final-state kinematics for the differentiable-fold comparison.  Returns a
    dict of observables (matching diffpi.observables) or None.  Final recoil nucleon is
    the status-1 nucleon; initial struck nucleon is the status-2 nucleon."""
    k_in = k_out = p_pi = p_struck = p_N = None
    for pid, status, p4 in evt["parts"]:
        if pid in NU_PIDS and status == 4:
            k_in = p4
        elif pid in CHG_LEP and status == 1:
            k_out = p4
        elif pid in PI_PIDS and status == 1:
            p_pi = p4
        elif abs(pid) in (2112, 2212) and status == 2:
            p_struck = p4
        elif abs(pid) in (2112, 2212) and status == 1:
            p_N = p4
    if k_in is None or k_out is None or p_pi is None or p_struck is None:
        return None
    q = k_in - k_out
    P = q + p_struck
    Q2 = -minkowski2(q)
    W = np.sqrt(max(minkowski2(P), 0.0))
    ppi_mag = np.sqrt(np.sum(p_pi[1:] ** 2))
    # pion in the piN-CM (rest frame of q+p_struck), relative to q
    pi_r = _boost_to_rest(P, p_pi)[1:]
    q_r = _boost_to_rest(P, q)[1:]
    kp_r = _boost_to_rest(P, k_out)[1:]
    e3 = q_r / (np.linalg.norm(q_r) + 1e-12)
    cos_ts = float(np.dot(pi_r, e3) / (np.linalg.norm(pi_r) + 1e-12))
    kp_perp = kp_r - np.dot(kp_r, e3) * e3
    e1 = kp_perp / (np.linalg.norm(kp_perp) + 1e-12)
    e2 = np.cross(e3, e1)
    phi_s = float(np.arctan2(np.dot(pi_r, e2), np.dot(pi_r, e1)))
    lep_E = float(k_out[0])
    lep_cth = float(np.dot(k_out[1:], k_in[1:]) / (np.linalg.norm(k_out[1:]) * np.linalg.norm(k_in[1:]) + 1e-12))
    nuc_mom = float(np.sqrt(np.sum(p_N[1:] ** 2))) if p_N is not None else np.nan
    return {"Q2": Q2, "W": W, "ppi_mag": ppi_mag, "cos_theta_star": cos_ts,
            "phi_star": phi_s, "lepton_energy": lep_E, "lepton_costheta": lep_cth,
            "nucleon_mom": nuc_mom, "w": evt["w"]}


def main():
    if not HEPMC.exists():
        sys.exit(f"hepmc not found: {HEPMC}")
    Q2s, Ws, ppis, weights = [], [], [], []
    n = 0
    for evt in parse_events(HEPMC):
        n += 1
        kin = event_kin(evt)
        if kin is None:
            continue
        Q2, W, p_pi = kin
        Q2s.append(Q2); Ws.append(W); ppis.append(p_pi); weights.append(evt["w"])
    Q2s, Ws, ppis, weights = map(np.array, (Q2s, Ws, ppis, weights))
    print(f"events parsed:        {n}")
    print(f"with FS pion:         {len(Q2s)}")
    print(f"Q2 [MeV^2] range:     {Q2s.min():.3e} .. {Q2s.max():.3e}")
    print(f"W  [MeV]   range:     {np.nanmin(Ws):.0f} .. {np.nanmax(Ws):.0f}")
    w_nb = weights * (REF_TOTAL_NB / weights.sum())

    Q2_edges = np.linspace(0.0, np.percentile(Q2s, 99.0), 31)
    W_edges = np.linspace(1080.0, np.nanpercentile(Ws, 99.0), 31)
    ppi_edges = np.linspace(0.0, np.percentile(ppis, 99.0), 31)

    def dsig(v, edges):
        m = np.isfinite(v)
        h, _ = np.histogram(v[m], bins=edges, weights=w_nb[m])
        return h / np.diff(edges)

    np.savez(OUT,
             Q2_edges=Q2_edges, dsig_Q2=dsig(Q2s, Q2_edges),
             W_edges=W_edges, dsig_W=dsig(Ws, W_edges),
             ppi_edges=ppi_edges, dsig_ppi=dsig(ppis, ppi_edges),
             ref_total_nb=REF_TOTAL_NB, n_signal=len(Q2s))
    print(f"integral check (nb):  W={np.nansum(dsig(Ws,W_edges)*np.diff(W_edges)):.3e} "
          f"(ref {REF_TOTAL_NB:.3e})")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
