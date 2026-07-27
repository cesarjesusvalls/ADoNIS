"""Final-state oracle histograms (ACHILLES neutrino single-pion full final state).

Stores, per observable, a FIXED fine binning plus weighted sums sum(w) and sum(w^2) per bin (per-bin
stat errors; re-binnable to coarser integer multiples downstream).  Both the local generator
(scripts/make_oracle_finalstate.py) and the CI parser (scripts/oracle_from_hepmc.py) build the npz
through this module so the edges never drift apart.

Consolidates the former adonis/data/oracle/{finalstate,parse_hepmc_nu}.py; the full-final-state event
kinematics (event_kin_full) are folded in here (their only consumer was finalstate accumulation).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from adonis.oracle.hepmc import parse_events, minkowski2

REF_TOTAL_NB = 4.943360e-05          # Total xsec (CC, 1500 MeV nu_e on 12C)
NU_PIDS = {12, 14, 16, -12, -14, -16}
CHG_LEP = {11, 13, -11, -13}
PI_PIDS = {111, 211, -211}


def _boost_to_rest(P, a):
    """Boost 4-vector a=(E,px,py,pz) into the rest frame of P."""
    M = np.sqrt(max(minkowski2(P), 1e-9))
    gamma = P[0] / M
    beta = P[1:] / P[0]
    b2 = max(np.sum(beta ** 2), 1e-30)
    bda = np.dot(beta, a[1:])
    a0 = gamma * (a[0] - bda)
    avec = a[1:] + ((gamma - 1.0) * bda / b2 - gamma * a[0]) * beta
    return np.concatenate([[a0], avec])


def event_kin_full(evt):
    """Full final-state kinematics (matching diffpi.observables) or None.  Final recoil nucleon is the
    status-1 nucleon; initial struck nucleon is the status-2 nucleon."""
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
    q = k_in - k_out; P = q + p_struck
    Q2 = -minkowski2(q); W = np.sqrt(max(minkowski2(P), 0.0))
    ppi_mag = np.sqrt(np.sum(p_pi[1:] ** 2))
    pi_r = _boost_to_rest(P, p_pi)[1:]; q_r = _boost_to_rest(P, q)[1:]; kp_r = _boost_to_rest(P, k_out)[1:]
    e3 = q_r / (np.linalg.norm(q_r) + 1e-12)
    cos_ts = float(np.dot(pi_r, e3) / (np.linalg.norm(pi_r) + 1e-12))
    kp_perp = kp_r - np.dot(kp_r, e3) * e3
    e1 = kp_perp / (np.linalg.norm(kp_perp) + 1e-12); e2 = np.cross(e3, e1)
    phi_s = float(np.arctan2(np.dot(pi_r, e2), np.dot(pi_r, e1)))
    lep_E = float(k_out[0])
    lep_cth = float(np.dot(k_out[1:], k_in[1:]) / (np.linalg.norm(k_out[1:]) * np.linalg.norm(k_in[1:]) + 1e-12))
    nuc_mom = float(np.sqrt(np.sum(p_N[1:] ** 2))) if p_N is not None else np.nan
    return {"Q2": Q2, "W": W, "ppi_mag": ppi_mag, "cos_theta_star": cos_ts, "phi_star": phi_s,
            "lepton_energy": lep_E, "lepton_costheta": lep_cth, "nucleon_mom": nuc_mom, "w": evt["w"]}


# Deterministic per-batch Options block (replaces the OptionDefaults include so each batch is reproducible).
_SEED_OPTIONS = ("Options:\n  Initialize:\n    Seed: {seed}\n    Accuracy: 1e-2\n"
                 "  Unweighting:\n    Name: Percentile\n    percentile: 99")


def write_run_config(base_yml, out_yml, nevents, seed, out_hepmc):
    """Write a run config from `base_yml` with the given event count, seed, and output path."""
    y = Path(base_yml).read_text()
    y = re.sub(r"NEvents:\s*\d+", f"NEvents: {nevents}", y, count=1)
    y = re.sub(r"Name:\s*res_1pi_12C_nu\.hepmc", f"Name: {out_hepmc}", y)
    y = y.replace('Options: !include "data/default/OptionDefaults.yml"', _SEED_OPTIONS.format(seed=seed))
    Path(out_yml).write_text(y)


# Fine fixed binning per observable (re-binnable to coarser multiples downstream).
ORACLE_EDGES = {
    "W":               np.linspace(1076.0, 1700.0, 313),
    "Q2":              np.linspace(0.0, 2.0e6, 201),
    "ppi_mag":         np.linspace(0.0, 700.0, 141),
    "cos_theta_star":  np.linspace(-1.0, 1.0, 101),
    "phi_star":        np.linspace(-np.pi, np.pi, 73),
    "lepton_energy":   np.linspace(0.0, 1500.0, 151),
    "lepton_costheta": np.linspace(-1.0, 1.0, 201),
    "nucleon_mom":     np.linspace(0.0, 1800.0, 181),
}


def new_accumulator(edges=ORACLE_EDGES):
    return {f"{p}_{k}": np.zeros(len(edges[k]) - 1) for k in edges for p in ("sw", "sw2")}


def accumulate_hepmc(path, acc, edges=ORACLE_EDGES):
    """Histogram one hepmc into `acc` (sum w, sum w^2 per bin); returns #signal events added."""
    cols = {k: [] for k in edges}; ws = []
    for evt in parse_events(Path(path)):
        r = event_kin_full(evt)
        if r is None:
            continue
        for k in edges:
            cols[k].append(r[k])
        ws.append(r["w"])
    ws = np.asarray(ws)
    for k in edges:
        v = np.asarray(cols[k]); m = np.isfinite(v)
        acc[f"sw_{k}"] += np.histogram(v[m], bins=edges[k], weights=ws[m])[0]
        acc[f"sw2_{k}"] += np.histogram(v[m], bins=edges[k], weights=ws[m] ** 2)[0]
    return len(ws)


def save_oracle(out, n_total, acc, edges=ORACLE_EDGES):
    np.savez(out, n_events=n_total, ref_total_nb=REF_TOTAL_NB,
             **{f"edges_{k}": edges[k] for k in edges}, **acc)
