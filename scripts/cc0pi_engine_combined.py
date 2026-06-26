"""T2K CC0pi-Np: the CORRECT engine prediction (final-state signal = mu + >=1 proton + 0 pions),
built from BOTH engine banks -- QE (channel=qe) AND RES-with-pion-absorbed (the CC1pi RES bank, events
whose pion died in FSI) -- vs ACHILLES, with a SINGLE self-consistent selection applied identically to
both (T2K NUISANCE def: the global highest-momentum proton must itself pass 450-1000 MeV/c, cos>0.4).

ACHILLES reference = the full-event RICH bank (t2k_cc1pi_rich_ach_FSI.npz, same T2K_CH hepmc), carbon-only
(struck |p|>1), 0-pion selection -> gives ALL observables from 4-vectors.  ADoNIS = QE(12C)+RES-abs(12C),
carbon-only.  No free-H (1.7%, no 4-vectors) -> both sides 12C-only, apples-to-apples.

Observables: delta_pT, delta_alphaT, Q2, vertex-W, p_mu, cos(theta_mu), leading p_p.
Usage: python -u scripts/cc0pi_engine_combined.py
Output: paper_figures/cc0pi_ENGINE_combined.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

P_LO, P_HI, COSP = 450.0, 1000.0, 0.4          # T2K CC0pi proton window
MU_LO, COSMU = 250.0, -0.6                      # T2K CC0pi muon
_PI = (211, 111, -211)


def _obs(mu, lead, struck, nu, w):
    """CC0pi observables from muon + leading proton (had = proton; no pion)."""
    lt = mu[:, 1:3]; pt = lead[:, 1:3]; dv = lt + pt
    dpt = np.linalg.norm(dv, axis=1)
    c = -np.sum(lt * dv, axis=1) / (np.linalg.norm(lt, axis=1) * dpt + 1e-12)
    dat = np.arccos(np.clip(c, -1, 1))
    q = nu - mu; tot = q + struck
    W = np.sqrt(np.clip(tot[:, 0] ** 2 - np.sum(tot[:, 1:] ** 2, axis=1), 0, None))
    Q2 = (np.sum(q[:, 1:] ** 2, axis=1) - q[:, 0] ** 2) / 1e6
    pmu = np.linalg.norm(mu[:, 1:], axis=1); cmu = mu[:, 3] / np.clip(pmu, 1e-9, None)
    lpp = np.linalg.norm(lead[:, 1:], axis=1)
    return dict(dpt=dpt, dalphat=dat, Q2=Q2, W=W, p_mu=pmu, cos_mu=cmu, lp_p=lpp, w=w)


def _select_engine(bank, carbon_only=True):
    """CC0pi from an engine rich bank: 0 surviving pion + GLOBAL-leading proton in window."""
    b = dict(np.load(bank)); n = len(b["w"]); ar = np.arange(n)
    mu = b["mu"]; pmu = np.linalg.norm(mu[:, 1:], axis=1); cmu = mu[:, 3] / np.clip(pmu, 1e-9, None)
    prot = b["prot"]; pm = np.linalg.norm(prot[:, :, 1:], axis=2)
    j = np.argmax(pm, axis=1); lead = prot[ar, j]                         # GLOBAL leading proton
    lpm = pm[ar, j]; lcth = lead[:, 3] / np.clip(lpm, 1e-9, None)
    in_win = (lpm > P_LO) & (lpm < P_HI) & (lcth > COSP)                  # the lead must itself pass
    no_pi = (~np.isin(b["pid_pi"], _PI)) & (~np.isin(b["cr_pid"], _PI))
    sel = (b["w"] > 0) & in_win & no_pi & (pmu > MU_LO) & (cmu > COSMU)
    if carbon_only:
        sel &= np.linalg.norm(b["struck"][:, 1:], axis=1) > 1.0
    s = sel
    return _obs(mu[s], lead[s], b["struck"][s], b["nu"][s], b["w"][s])


def _select_ach(carbon_only=True):
    b = dict(np.load("data/oracle/t2k_cc1pi_rich_ach_FSI.npz")); wn = float(b["weight_to_nb"])
    n = len(b["w"]); ar = np.arange(n)
    mu = b["mu"]; pmu = np.linalg.norm(mu[:, 1:], axis=1); cmu = mu[:, 3] / np.clip(pmu, 1e-9, None)
    npi = np.isin(b["pi_pid"], list(_PI)).sum(1)
    pr = b["prot_p4"]; pm = np.linalg.norm(pr[:, :, 1:], axis=2)
    j = np.argmax(pm, axis=1); lead = pr[ar, j]                          # GLOBAL leading proton
    lpm = pm[ar, j]; lcth = lead[:, 3] / np.clip(lpm, 1e-9, None)
    in_win = (lpm > P_LO) & (lpm < P_HI) & (lcth > COSP)
    sel = (b["w"] > 0) & (npi == 0) & in_win & (pmu > MU_LO) & (cmu > COSMU)
    if carbon_only:
        sel &= np.linalg.norm(b["struck"][:, 1:], axis=1) > 1.0
    s = sel
    return _obs(mu[s], lead[s], b["struck"][s], b["nu"][s], b["w"][s] * wn)


VARS = [("dpt", np.linspace(0, 800, 21), r"$\delta p_T$ [MeV]"),
        ("dalphat", np.linspace(0, np.pi, 21), r"$\delta\alpha_T$ [rad]"),
        ("Q2", np.linspace(0, 1.4, 21), r"$Q^2$ [GeV$^2$]"),
        ("W", np.linspace(938, 1400, 21), r"vertex $W$ [MeV]"),
        ("p_mu", np.linspace(0, 2000, 21), r"$p_\mu$ [MeV]"),
        ("cos_mu", np.linspace(-0.6, 1.0, 21), r"$\cos\theta_\mu$"),
        ("lp_p", np.linspace(450, 1000, 21), r"lead $p_p$ [MeV]")]


def main():
    qe = _select_engine("data/oracle/t2k_cc0pi.npz")
    res = _select_engine("data/oracle/t2k_cc1pi.npz")            # RES with pion absorbed
    ado = {k: np.concatenate([qe[k], res[k]]) for k in qe}
    H = _select_ach()
    sA, sH = ado["w"].sum(), H["w"].sum()
    print(f"ADoNIS engine CC0pi (12C: QE {qe['w'].sum():.4e} + RES-abs {res['w'].sum():.4e}) = {sA:.4e} nb")
    print(f"ACHILLES CC0pi (12C, rich bank, SAME T2K selection) = {sH:.4e} nb   ACH/ADO {sH/sA:.4f}")
    print(f"{'var':8s} chi2/ndf  ACH/ADO")
    fig, ax = plt.subplots(2, len(VARS), figsize=(3.3 * len(VARS), 6.4), height_ratios=[3, 1],
                           squeeze=False, sharex="col")
    for c, (k, edg, xl) in enumerate(VARS):
        bw = np.diff(edg); ctr = 0.5 * (edg[1:] + edg[:-1])
        da, _ = np.histogram(H[k], edg, weights=H["w"]); ea = np.sqrt(np.histogram(H[k], edg, weights=H["w"]**2)[0]) / bw; da /= bw
        dd, _ = np.histogram(ado[k], edg, weights=ado["w"]); ed = np.sqrt(np.histogram(ado[k], edg, weights=ado["w"]**2)[0]) / bw; dd /= bw
        a0, a1 = ax[0, c], ax[1, c]
        a0.fill_between(edg, np.append(da-ea, (da-ea)[-1]), np.append(da+ea, (da+ea)[-1]), step="post", color="0.55", alpha=0.55, lw=0, label="ACH stat")
        a0.step(edg, np.append(da, da[-1]), where="post", color="0.3", lw=1.3, label="ACHILLES")
        a0.errorbar(ctr, dd, yerr=ed, fmt="s", color="C0", ms=3, capsize=2, lw=0.9, label="ENGINE QE+RES-abs")
        a0.set_title(xl, fontsize=9); a0.set_ylim(bottom=0)
        if c == 0: a0.legend(fontsize=7)
        m = (da > 0) & (dd > 0); chi2 = float(np.sum((da[m]-dd[m])**2 / (ea[m]**2+ed[m]**2))); ndf = int(m.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            r = da/dd; re = r*np.sqrt((ed/dd)**2+(ea/da)**2)
        a1.axhspan(0.9, 1.1, color="green", alpha=0.12); a1.axhline(1.0, ls="--", color="green", lw=0.7)
        a1.errorbar(ctr[m], r[m], yerr=re[m], fmt="o", color="C3", ms=3, capsize=2, lw=0.8)
        a1.set_ylim(0.6, 1.4); a1.set_xlabel(xl, fontsize=8); a1.text(0.04, 0.83, f"{chi2/max(ndf,1):.1f}", transform=a1.transAxes, fontsize=9)
        print(f"{k:8s}  {chi2/max(ndf,1):6.2f}   {np.sum(da*bw)/max(np.sum(dd*bw),1e-30):.3f}")
    fig.suptitle(f"T2K CC0$\\pi$-Np (12C)  ENGINE QE+RES-absorbed vs ACHILLES (same T2K selection)  "
                 f"ACH/ADO {sH/sA:.3f}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); out = "paper_figures/cc0pi_ENGINE_combined.png"
    fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    main()
