"""T2K CC1pi+Np STV (arXiv:2102.03346 / PRD 103 112009): ADoNIS vs ACHILLES vs T2K data for
p_N and delta_pTT (the paper's T2K_CC1pipNp figure pair) + delta_alphaT.

Blueprint chain (CLAUDE.md): RES-C events from the bit-exact res_xsec generator, pion through
the discrete cascade (SURVIVAL branch: pid_pi==211 after FSI -- absorption/charge-exchange
removes the event), leading proton through the discrete nucleon cascade; RES-H from the free-
proton 3-body generator (pi+ always survives; no FSI).  QE contributes nothing (no pion).
Selection/observables mirror scripts/extract_t2k_cc1pi_tki.py exactly (tight windows, theta<70deg;
NUISANCE hydrogen prescription: flat delta_alphaT throw, carbon-mass p_N formula for all events).

FSI runs through the single Gaussian POOL cascade (cascade_nucleus): pion + recoil + ALL knockout
generations are tracked jointly, so the leading proton is the highest-momentum IN-WINDOW proton among
the unified pool candidate set prot[] (every escaped proton terminal across generations, origin-tagged:
0=RES/QE nucleon chain, 1=pion-knockout, 2=primary pion).  This supersedes the old approximate
{RES nucleon, scatter knockout, its secondary knockout} set (which neglected deeper knockouts).

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
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig
from adonis.fsi.pool_fsi import run_fsi, proton_candidates
from scripts.h_cc0pi import generate_H

NRES = int(sys.argv[1]) if (len(sys.argv) > 1 and sys.argv[1].isdigit()) else 200000  # robust when imported
NH, NSEED = 50000, 4
MPROT = 6                                                       # top-M proton terminals kept from the pool
_CFG = lambda **k: DiscreteCascadeConfig(step=0.04, max_steps=600, engine="pool", **k)  # Gaussian pool

# tight signal phase space (PRD 103 112009; == extract_t2k_cc1pi_tki.py)
MU_LO, MU_HI = 250.0, 7000.0
PI_LO, PI_HI = 150.0, 1200.0
P_LO, P_HI = 450.0, 1200.0
COS70 = np.cos(np.deg2rad(70.0))
M_A12, M_A11 = 11174.862, 10252.547                  # 12C, 11B [MeV]
NB_PER_CM2 = 1e33
A_CH = 13.0                                          # data is per NUCLEON of CH (12 C + 1 H)
# ABLATION knob (default "full" = production): which pool proton candidates feed the signal, by origin tag.
#   "full"        = every escaped proton terminal across all generations (the validated pool set)
#   "no_secondary"= drop pion-knockout SECONDARIES (origin==1 & gen>=2; bounded-recursion test)
#   "res_only"    = RES/QE nucleon chain only (origin==0: primary recoil + its NN knockouts; no pion ko)
KNOCKOUT_MODE = "full"


def _apply_knockout_mode(prot, org, gen):
    """Filter the unified pool proton-candidate set by the KNOCKOUT_MODE ablation (origin/gen tags)."""
    if KNOCKOUT_MODE == "full":
        return prot, org, gen
    if KNOCKOUT_MODE == "res_only":
        keep = (org == 0)
    elif KNOCKOUT_MODE == "no_secondary":
        keep = ~((org == 1) & (gen >= 2))
    else:
        raise ValueError(f"KNOCKOUT_MODE={KNOCKOUT_MODE!r}")
    return (np.where(keep[:, :, None], prot, 0.0), np.where(keep, org, -1), np.where(keep, gen, -1))


_OTHER_MESON = (111, -211, -1)            # vs a pi+ signal: pi0 / pi- / converted(eta,K)


def _signal_pip(prot, pid_pi, cr_pid, pi_post, cr_p4, p_win=(P_LO, P_HI), cth=COS70):
    """Validated CC1pi+ signal core on the pool bank: exactly one pi+ over {primary, created} with no
    other meson, and the leading IN-WINDOW proton from prot[].  Returns (lead, has_p, pi_f, meson_ok)."""
    n = len(pid_pi); ar = np.arange(n)
    prim_pip = (pid_pi == 211); cr_pip = (cr_pid == 211)
    n_pip = prim_pip.astype(int) + cr_pip.astype(int)
    n_other = np.isin(pid_pi, _OTHER_MESON).astype(int) + np.isin(cr_pid, _OTHER_MESON).astype(int)
    meson_ok = (n_pip == 1) & (n_other == 0)
    pi_f = np.where(prim_pip[:, None], pi_post, cr_p4)
    pm = np.linalg.norm(prot[:, :, 1:], axis=2); ct = prot[:, :, 3] / np.clip(pm, 1e-9, None)
    inwin = (pm > p_win[0]) & (pm < p_win[1]) & (ct > cth)
    j = np.argmax(np.where(inwin, pm, -1.0), axis=1)
    return prot[ar, j], inwin[ar, j], pi_f, meson_ok


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
    ppid, ipid, Npid = np.asarray(e["ppid"]), np.asarray(e["ipid"]), np.asarray(e["Npid"])
    # ONE joint pool cascade: primary pion + recoil nucleon + every knockout generation (origin-tagged).
    out = run_fsi(jnp.asarray(ppi), jnp.asarray(pN), jnp.asarray(ppid, jnp.int32),
                  jnp.asarray(ipid, jnp.int32), jnp.asarray(Npid, jnp.int32),
                  _CFG(seed=1), jax.random.PRNGKey(seed + 11), channel="res")
    pid_pi = np.asarray(out["pterm"]["pid"]); pi_post = np.asarray(out["pterm"]["p4"])
    nsc = np.asarray(out["pterm"]["nsc"])
    cr_pid = np.where(np.asarray(out["created"]["alive"]), np.asarray(out["created"]["pid"]), 0)
    cr_p4 = np.asarray(out["created"]["p4"])
    prot, prot_org, prot_gen = _apply_knockout_mode(*proton_candidates(out["nterms"], MPROT))
    if return_raw:
        # RICH per-event bank: full PRE-/POST-FSI 4-vectors + the unified pool proton-candidate set
        # prot[] (every escaped proton terminal across generations, origin/gen tagged) -> any signal
        # definition is pure re-binning (scripts/cc1pi_signal.ado_select).  4-vectors are (E,px,py,pz) MeV.
        return dict(
            nu=knu, mu=kmu, struck=pstr, ipid=ipid,
            pi_pre=ppi, ppid_pre=ppid, rec_pre=pN, Npid=Npid,             # pre-FSI (primary RES)
            pi_post=pi_post, pid_pi_post=pid_pi, pi_nsc=nsc,              # post-FSI primary pion
            cr_p4=cr_p4, cr_pid=cr_pid,                                   # leading cascade-created pion
            prot=prot, prot_origin=prot_org, prot_gen=prot_gen, w=w)
    # DEFAULT CC1pi+ signal on the unified pool candidates (single source: _signal_pip / observables)
    lead, has_p, pi_f, meson_ok = _signal_pip(prot, pid_pi, cr_pid, pi_post, cr_p4)
    sel = (meson_ok & has_p & (w > 0) & _acc(kmu, MU_LO, MU_HI) & _acc(pi_f, PI_LO, PI_HI))
    dptt, pN_o, dat, dpt = observables(kmu[sel], pi_f[sel], lead[sel], np.zeros(int(sel.sum()), bool), seed)
    pim = np.linalg.norm(pi_f[sel][:, 1:], axis=1)
    qv = (knu - kmu)[sel]; totv = qv + pstr[sel]                 # vertex hadronic 4-mom
    Wv = np.sqrt(np.clip(totv[:, 0] ** 2 - np.sum(totv[:, 1:] ** 2, axis=1), 0, None))
    Q2v = (np.sum(qv[:, 1:] ** 2, axis=1) - qv[:, 0] ** 2) / 1e6  # GeV^2
    return dict(dptt=dptt, pn=pN_o, dalphat=dat, dpt=dpt, w=w[sel],
                nsc=nsc[sel],                                  # pion scatter count (diagnostics)
                pi_p=pim, pi_cth=pi_f[sel][:, 3] / np.clip(pim, 1e-9, None),
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
