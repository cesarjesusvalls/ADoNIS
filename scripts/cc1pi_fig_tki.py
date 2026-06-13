"""T2K CC1pi+Np STV (arXiv:2102.03346 / PRD 103 112009): ADoNIS vs ACHILLES vs T2K data for
p_N and delta_pTT (the paper's T2K_CC1pipNp figure pair) + delta_alphaT.

Blueprint chain (CLAUDE.md): RES-C events from the bit-exact res_xsec generator, pion through
the discrete cascade (SURVIVAL branch: pid_pi==211 after FSI -- absorption/charge-exchange
removes the event), leading proton through the discrete nucleon cascade; RES-H from the free-
proton 3-body generator (pi+ always survives; no FSI).  QE contributes nothing (no pion).
Selection/observables mirror scripts/extract_t2k_cc1pi_tki.py exactly (tight windows, theta<70deg;
NUISANCE hydrogen prescription: flat delta_alphaT throw, carbon-mass p_N formula for all events).

Pion-scatter RECOIL protons are emitted and re-cascaded (ACHILLES FinalizeMomentum/UpdateKicked
mirrored via DiscreteCascadeFSI.last_scat_ko): the leading proton is the highest-momentum
IN-WINDOW candidate among {RES nucleon, scatter knockout, its secondary knockout}.  Declared
approximation: only the leading PROTON recoil per pion is tracked (neutron recoils' secondary
knockouts neglected).

Usage: python scripts/cc1pi_fig_tki.py [N_per_seed]   (default 200000; NSEED=4)
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from adonis.data.oracle.normalization import weight_to_nb_of
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

import adonis.xsec.dcc_current as dcc; dcc.BATCH_INTERP = "spline"
from adonis.xsec import res_xsec
from adonis.core.event import EventRecord
from adonis.fsi.cascade_discrete import (DiscreteCascadeFSI, DiscreteNucleonFSI, DiscreteCascadeConfig,
                                         sample_nucleons, propagate_nucleon_discrete)
from scripts.h_cc0pi import generate_H

NRES = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
NH, NSEED = 50000, 4
_CFG = lambda **k: DiscreteCascadeConfig(cylinder=True, step=0.04, max_steps=260, **k)  # T2K run-card

# tight signal phase space (PRD 103 112009; == extract_t2k_cc1pi_tki.py)
MU_LO, MU_HI = 250.0, 7000.0
PI_LO, PI_HI = 150.0, 1200.0
P_LO, P_HI = 450.0, 1200.0
COS70 = np.cos(np.deg2rad(70.0))
M_A12, M_A11 = 11174.862, 10252.547                  # 12C, 11B [MeV]
NB_PER_CM2 = 1e33
A_CH = 13.0                                          # data is per NUCLEON of CH (12 C + 1 H)
# ABLATION knob (default "full" = production, unchanged): which proton candidates feed the signal.
#   "full"        = {RES nucleon, leading pion-scatter knockout, its secondary knockout} (faithful set)
#   "no_secondary"= drop the secondary knockout (tests the bounded-recursion approximation)
#   "res_only"    = RES nucleon only, no pion-scatter knockouts (removes the n->n pi+-via-knockout path)
KNOCKOUT_MODE = "full"


def _acc(p4, lo, hi):
    m = np.linalg.norm(p4[:, 1:], axis=1)
    return (m > lo) & (m < hi) & (p4[:, 3] / np.clip(m, 1e-9, None) > COS70)


def observables(kmu, ppi, lead, is_h, seed=0):
    """Vectorized (dptt, pN, dalphat, dpt) -- formulas identical to extract_t2k_cc1pi_tki.py."""
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


def res_C(n, seed, return_raw=False):
    e = res_xsec.generate(n, seed=seed, return_events=True)["events"]
    knu, kmu, pstr = (np.asarray(e[k]) for k in ("k_nu", "k_mu", "p_struck"))
    ppi, pN, w = np.asarray(e["p_pi"]), np.asarray(e["p_N"]), np.asarray(e["w"])
    m = len(w)
    ev = EventRecord(k=jnp.asarray(knu), kp=jnp.asarray(kmu), p_struck=jnp.asarray(pstr),
                     p_pi=jnp.asarray(ppi), p_N=jnp.asarray(pN), w=jnp.asarray(w),
                     channel=jnp.zeros(m, jnp.int32), pid_pi=jnp.asarray(e["ppid"], jnp.int32),
                     pid_N=jnp.asarray(e["Npid"], jnp.int32), pid_Ni=jnp.asarray(e["ipid"], jnp.int32),
                     W=jnp.zeros(m), Q2_adj=jnp.zeros(m))
    pion = DiscreteCascadeFSI(_CFG(seed=1)); ev = pion.apply(None, ev, key=jax.random.PRNGKey(seed + 11))
    ko, ko_pos, ko_fz = pion.last_scat_ko                        # leading pi-scatter recoil proton
    nf = DiscreteNucleonFSI(_CFG(seed=2)); ev = nf.apply(None, ev, key=jax.random.PRNGKey(seed + 13))
    ppi_f = np.asarray(ev.p_pi); pid_pi = np.asarray(ev.pid_pi)
    # re-cascade the scatter knockout through the nucleon transport (ACHILLES UpdateKicked);
    # its own (proton) knockout is a further candidate.
    has_ko = np.linalg.norm(np.asarray(ko)[:, 1:], axis=1) > 1.0
    cfg = _CFG(seed=3)
    kn = jax.random.PRNGKey(seed + 17)
    npos2, nmom2, nisp2 = sample_nucleons(jax.random.fold_in(kn, 1), m, cfg)
    ko_in = jnp.where(jnp.asarray(has_ko)[:, None], jnp.asarray(ko), ev.p_N)   # dummy where none
    ko_f, _, ko_ko, _, _, _, _, ko_made_pi = propagate_nucleon_discrete(
        jnp.asarray(ko_pos), ko_in, jnp.ones(m, bool), npos2, nmom2, nisp2, cfg,
        jax.random.fold_in(kn, 2), fz0=jnp.asarray(ko_fz))
    ko_f = np.where(has_ko[:, None], np.asarray(ko_f), 0.0)
    ko_ko = np.where(has_ko[:, None], np.asarray(ko_ko), 0.0)
    # leading proton = highest-momentum IN-WINDOW candidate among
    # {nucleon-cascade leading PROTON (primary-if-proton OR its NN knockout proton -- the only n->n pi+
    #  signal-proton path, species threaded from the cascade), re-cascaded pion-scatter knockout, its
    #  secondary knockout}
    lead0 = np.asarray(nf.last_lead_prot)
    if KNOCKOUT_MODE == "res_only":
        cands = lead0[:, None, :]                                # RES nucleon only (ablation)
    elif KNOCKOUT_MODE == "no_secondary":
        cands = np.stack([lead0, ko_f], axis=1)                  # drop the secondary knockout (ablation)
    else:
        cands = np.stack([lead0, ko_f, ko_ko], axis=1)           # (m, 3, 4) faithful set
    inwin = np.stack([_acc(cands[:, i], P_LO, P_HI) for i in range(cands.shape[1])], axis=1)
    mom = np.linalg.norm(cands[:, :, 1:], axis=2) * inwin
    lead = cands[np.arange(m), np.argmax(mom, axis=1)]
    has_p = inwin.any(axis=1)
    no_extra_pi = ~(np.asarray(nf.last_made_pion) | (has_ko & np.asarray(ko_made_pi)))
    if return_raw:
        # PRE-selection per-event bank (ALL m events): the ADoNIS analog of the ACHILLES
        # res_w_FSI extraction, so the signal cut ladder can be applied identically off-line.
        mu_m = np.linalg.norm(kmu[:, 1:], axis=1)
        pi_m = np.linalg.norm(ppi_f[:, 1:], axis=1)
        lp_m = np.linalg.norm(lead[:, 1:], axis=1)
        qv = knu - kmu; totv = qv + pstr
        Wv = np.sqrt(np.clip(totv[:, 0] ** 2 - np.sum(totv[:, 1:] ** 2, axis=1), 0, None))
        Q2v = (np.sum(qv[:, 1:] ** 2, axis=1) - qv[:, 0] ** 2) / 1e6
        return dict(
            mu_p=mu_m, mu_cth=kmu[:, 3] / np.clip(mu_m, 1e-9, None),
            pi_p=pi_m, pi_cth=ppi_f[:, 3] / np.clip(pi_m, 1e-9, None), pid_pi=pid_pi,
            lp_p=lp_m, lp_cth=lead[:, 3] / np.clip(lp_m, 1e-9, None),
            has_p=has_p, no_extra_pi=no_extra_pi, W=Wv, Q2=Q2v, w=w,
            ppid0=np.asarray(e["ppid"]))                    # primary RES pion charge (channel tag)
    sel = ((pid_pi == 211) & has_p & no_extra_pi & (w > 0)
           & _acc(kmu, MU_LO, MU_HI) & _acc(ppi_f, PI_LO, PI_HI))
    dptt, pN_o, dat, dpt = observables(kmu[sel], ppi_f[sel], lead[sel], np.zeros(sel.sum(), bool), seed)
    pim = np.linalg.norm(ppi_f[sel][:, 1:], axis=1)
    qv = (knu - kmu)[sel]; totv = qv + pstr[sel]                 # vertex hadronic 4-mom
    Wv = np.sqrt(np.clip(totv[:, 0] ** 2 - np.sum(totv[:, 1:] ** 2, axis=1), 0, None))
    Q2v = (np.sum(qv[:, 1:] ** 2, axis=1) - qv[:, 0] ** 2) / 1e6  # GeV^2
    return dict(dptt=dptt, pn=pN_o, dalphat=dat, dpt=dpt, w=w[sel],
                nsc=np.asarray(pion.last_nsc)[sel],           # pion scatter count (diagnostics)
                pi_p=pim, pi_cth=ppi_f[sel][:, 3] / np.clip(pim, 1e-9, None),
                lp_p=np.linalg.norm(lead[sel][:, 1:], axis=1), W=Wv, Q2=Q2v)


def res_H(n, seed):
    knu, kmu, pN, pPi, w = generate_H(n, seed=seed)
    knu, kmu, pN, pPi, w = (np.asarray(x) for x in (knu, kmu, pN, pPi, w))
    sel = (w > 0) & _acc(kmu, MU_LO, MU_HI) & _acc(pPi, PI_LO, PI_HI) & _acc(pN, P_LO, P_HI)
    dptt, pN_o, dat, dpt = observables(kmu[sel], pPi[sel], pN[sel], np.ones(sel.sum(), bool), seed)
    pim = np.linalg.norm(pPi[sel][:, 1:], axis=1)
    pstr = np.tile([938.27, 0, 0, 0.0], (int(sel.sum()), 1))     # free proton at rest
    qv = (knu - kmu)[sel]; totv = qv + pstr
    Wv = np.sqrt(np.clip(totv[:, 0] ** 2 - np.sum(totv[:, 1:] ** 2, axis=1), 0, None))
    Q2v = (np.sum(qv[:, 1:] ** 2, axis=1) - qv[:, 0] ** 2) / 1e6
    return dict(dptt=dptt, pn=pN_o, dalphat=dat, dpt=dpt, w=w[sel], nsc=np.zeros(int(sel.sum()), np.int32),
                pi_p=pim, pi_cth=pPi[sel][:, 3] / np.clip(pim, 1e-9, None),
                lp_p=np.linalg.norm(pN[sel][:, 1:], axis=1), W=Wv, Q2=Q2v)


def load_data(name):
    """Parse the release txt: bin edges (MeV), values (per nucleon cm^2/(MeV/c) or /rad),
    covariance."""
    lines = open(f"../nuisance/data/T2K/CC1pipNp_STV/xsec_{name}.txt").read().splitlines()
    edges = np.array([float(x) for x in lines[0].split(":")[1].split()])
    vals = np.array([float(x) for x in lines[1].split(":")[1].split()])
    nb = len(vals)
    cov = np.array([[float(x) for x in lines[3 + i].split()] for i in range(nb)])
    return edges, vals, cov


def main():
    bp = "data/oracle/t2k_cc1pi_tki_adonis_blueprint.npz"
    if os.path.exists(bp) and "--recompute" not in sys.argv:
        d = np.load(bp); ado = {k: np.asarray(d[k]) for k in ("dptt", "pn", "dalphat", "w")}
        print("loaded cached ADoNIS blueprint (use --recompute to regenerate)", flush=True)
    else:
        acc = {}
        for cell, fn, n in (("RES-C", res_C, NRES), ("RES-H", res_H, NH)):
            parts = []
            for sd in range(NSEED):
                d = fn(n, sd)
                parts.append(d)
                print(f"  {cell} seed {sd+1}/{NSEED}: +{len(d['w'])}", flush=True)
            acc[cell] = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            acc[cell]["w"] = acc[cell]["w"] / NSEED
        ado = {k: np.concatenate([acc[c][k] for c in acc]) for k in ("dptt", "pn", "dalphat", "w")}
        np.savez(bp, **ado,
                 **{f"resc_{k}": acc["RES-C"][k] for k in acc["RES-C"]},
                 **{f"resh_{k}": acc["RES-H"][k] for k in acc["RES-H"]})
        print(f"  sigma_CC1pi(tight): RES-C {acc['RES-C']['w'].sum():.4e}  RES-H {acc['RES-H']['w'].sum():.4e} nb",
              flush=True)

    ach = np.load("data/oracle/t2k_cc1pi_tki_achilles.npz")
    ach_w = np.asarray(ach["w"]) * weight_to_nb_of(ach)   # GenCrossSection/sum_w from the npz header

    ado["dalphat_deg"] = np.degrees(ado["dalphat"])          # daT release bins are in DEGREES
    VARS = [("pn", "pN", r"$p_N$ [MeV/c]", 1.0),
            ("dptt", "dpTT", r"$\delta p_{TT}$ [MeV/c]", 1.0),
            ("dalphat_deg", "daT", r"$\delta\alpha_T$ [deg]", 1.0)]
    fig, axes = plt.subplots(2, len(VARS), figsize=(6.2 * len(VARS), 7.5),
                             height_ratios=[3, 1], sharex="col")
    for c, (key, rel, xlab, xs) in enumerate(VARS):
        edges, dval, dcov = load_data(rel)
        bw = np.diff(edges); ctr = 0.5 * (edges[1:] + edges[:-1])
        d_y = dval * NB_PER_CM2 * A_CH                            # per-nucleon cm2 -> nb per CH
        d_e = np.sqrt(np.diag(dcov)) * NB_PER_CM2 * A_CH
        def hist(v, w):
            h, _ = np.histogram(v, bins=edges, weights=w)
            e2, _ = np.histogram(v, bins=edges, weights=w ** 2)
            return h / bw, np.sqrt(e2) / bw
        ach_v = np.degrees(np.asarray(ach["dalphat"])) if key == "dalphat_deg" else np.asarray(ach[key])
        da, ea = hist(ach_v, ach_w)
        dd, ed = hist(ado[key], ado["w"])
        ax, axr = axes[0, c], axes[1, c]
        ax.fill_between(edges, np.append(da - ea, (da - ea)[-1]), np.append(da + ea, (da + ea)[-1]),
                        step="post", color="0.5", alpha=0.25, lw=0)
        ax.step(edges, np.append(da, da[-1]), where="post", color="0.35", lw=1.4, label="ACHILLES")
        ax.errorbar(ctr, dd, yerr=ed, fmt="s", color="C0", ms=4, capsize=2, lw=1.0,
                    label="ADoNIS (CH)", zorder=4)                 # markers at bin centers
        ax.errorbar(ctr, d_y, yerr=d_e, fmt="o", color="k", ms=5, capsize=3, lw=1.4,
                    label="T2K data", zorder=5)
        ax.set_ylabel(r"d$\sigma$/dx [nb/unit per CH]"); ax.set_ylim(bottom=0); ax.legend(fontsize=8)
        ax.set_title(f"CC1$\\pi^+$Np   {xlab}")
        with np.errstate(divide="ignore", invalid="ignore"):
            r = da / dd; re = r * np.sqrt((ed / dd) ** 2 + (ea / da) ** 2)
        msk = (da > 0) & (dd > 0) & np.isfinite(re)
        chi2 = float(np.sum((da[msk] - dd[msk]) ** 2 / (ea[msk] ** 2 + ed[msk] ** 2)))
        axr.axhspan(0.9, 1.1, color="green", alpha=0.12); axr.axhline(1.0, ls="--", color="green")
        axr.errorbar(ctr[msk], r[msk], yerr=re[msk], fmt="o", color="C3", ms=4, capsize=2, lw=1.1)
        axr.set_ylim(0.5, 1.5); axr.set_ylabel("ACH / ADO"); axr.set_xlabel(xlab)
        axr.text(0.03, 0.85, f"$\\chi^2$/ndf = {chi2/max(int(msk.sum()),1):.2f}",
                 transform=axr.transAxes, fontsize=10)
        print(f"  {key}: integral nb  ACH={np.sum(da*bw):.3e}  ADO={np.sum(dd*bw):.3e}  "
              f"T2K={np.sum(d_y*bw):.3e}", flush=True)
    fig.suptitle("T2K CC1$\\pi^+$Np STV (tight) — ADoNIS vs ACHILLES vs T2K data (PRD 103 112009)",
                 fontsize=13)
    fig.tight_layout()
    out = "paper_figures/cc1pi_pn_dptt_data.png"; fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    main()
