"""Numpy-batch observable formulas -- the SINGLE source reused by both channels' selection.

These are verbatim the formulas in scripts/cc1pi_fig_tki.observables (CC1pi TKI: dptt, pN, dalphat,
dpt) and scripts/cc0pi_engine_combined._obs (CC0pi: dpt, dalphat, Q2, W, p_mu, cos_mu, lp_p), plus the
shared vertex W/Q2 used by cc1pi_engine_plot.  The carbon TKI masses come from
adonis.observables.kinematics (single source) so the numpy twin and the JAX OBSERVABLES registry
agree bit-for-bit.  Inputs are (n,4) lab 4-vectors [MeV]; outputs per-event arrays.
"""
from __future__ import annotations
import numpy as np
from adonis.observables.kinematics import M_A_12C, M_A_11B

M_A12, M_A11 = M_A_12C, M_A_11B


def mom(p4):
    return np.linalg.norm(np.atleast_2d(p4)[:, 1:], axis=1)


def vertex_W_Q2(nu, mu, struck):
    """Vertex W = |q + struck| [MeV] and leptonic Q2 = (|q3|^2 - q0^2)/1e6 [GeV^2]."""
    q = nu - mu
    tot = q + struck
    W = np.sqrt(np.clip(tot[:, 0] ** 2 - np.sum(tot[:, 1:] ** 2, axis=1), 0.0, None))
    Q2 = (np.sum(q[:, 1:] ** 2, axis=1) - q[:, 0] ** 2) / 1e6
    return W, Q2


def tki(kmu, ppi, lead, is_h, seed=0):
    """CC1pi TKI -- (dptt, pN, dalphat, dpt).  Verbatim cc1pi_fig_tki.observables (single source).
    pN uses the 12C->11B longitudinal inference; the NUISANCE hydrogen prescription randomizes
    dalphat where is_h or |dptt|<0.5."""
    rng = np.random.default_rng(seed)
    mu3, pi3, p3 = kmu[:, 1:], ppi[:, 1:], lead[:, 1:]
    beam = np.array([0.0, 0.0, 1.0])
    zhat = np.cross(np.broadcast_to(beam, mu3.shape), mu3)
    zhat = zhat / (np.linalg.norm(zhat, axis=1, keepdims=True) + 1e-9)
    had3 = pi3 + p3
    dptt = np.sum(had3 * zhat, axis=1)
    lt = mu3[:, :2]; dpt_vec = lt + had3[:, :2]
    dpt = np.linalg.norm(dpt_vec, axis=1)
    c = -np.sum(lt * dpt_vec, axis=1) / (np.linalg.norm(lt, axis=1) * dpt + 1e-9)
    dat = np.arccos(np.clip(c, -1, 1))
    flat = is_h | (np.abs(dptt) < 0.5)                       # NUISANCE hydrogen prescription
    dat = np.where(flat, rng.uniform(0.0, np.pi, len(dat)), dat)
    pL = kmu[:, 3] + ppi[:, 3] + lead[:, 3]; Evis = kmu[:, 0] + ppi[:, 0] + lead[:, 0]
    R = M_A12 + pL - Evis
    dpL = 0.5 * R - (M_A11 ** 2 + dpt ** 2) / (2.0 * np.clip(R, 1.0, None))
    pN = np.sqrt(np.clip(dpt ** 2 + dpL ** 2, 0.0, None))
    return dptt, pN, dat, dpt


def cc0pi_obs(mu, lead, struck, nu):
    """CC0pi observables from muon + leading proton (no pion).  Verbatim cc0pi_engine_combined._obs."""
    lt = mu[:, 1:3]; pt = lead[:, 1:3]; dv = lt + pt
    dpt = np.linalg.norm(dv, axis=1)
    c = -np.sum(lt * dv, axis=1) / (np.linalg.norm(lt, axis=1) * dpt + 1e-12)
    dat = np.arccos(np.clip(c, -1, 1))
    W, Q2 = vertex_W_Q2(nu, mu, struck)
    pmu = np.linalg.norm(mu[:, 1:], axis=1); cmu = mu[:, 3] / np.clip(pmu, 1e-9, None)
    lpp = np.linalg.norm(lead[:, 1:], axis=1)
    return dict(dpt=dpt, dalphat=dat, Q2=Q2, W=W, p_mu=pmu, cos_mu=cmu, lp_p=lpp)
