"""CC0pi-Np with a fixed number of EJECTED PROTONS (final-state topology), ADoNIS vs ACHILLES.

Standard CC0pi selection (muon cut, 0 pions, LEADING proton in the T2K window) PLUS require exactly
N ejected protons = final-state protons with |p| > P_EJECT.  This is a pure final-state topology cut
on the rich banks (the "2p2h-like" 2-proton sample); only the leading proton carries the std cuts.

Usage: python -u scripts/cc0pi_nproton.py [N=2] [P_EJECT=250]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import adonis.workflow.observables as O
import adonis.workflow.plotting as PL
from adonis.workflow.config import load_analysis_config

NREQ = int(sys.argv[1]) if len(sys.argv) > 1 else 2
P_EJECT = float(sys.argv[2]) if len(sys.argv) > 2 else 250.0
sd = load_analysis_config("configs/ana_cc0pi.yaml").signal
_PI = (211, 111, -211)
P_LO, P_HI, COSP = sd.p_win[0], sd.p_win[1], sd.cth
MU_LO, COSMU = sd.mu_win[0], sd.cos_mu


def base_count_lead(prot_p4, mu, struck, w, no_pi, carbon=True):
    n = len(w); ar = np.arange(n)
    pmu = np.linalg.norm(mu[:, 1:], axis=1); cmu = mu[:, 3] / np.clip(pmu, 1e-9, None)
    pm = np.linalg.norm(prot_p4[:, :, 1:], axis=2); cth = prot_p4[:, :, 3] / np.clip(pm, 1e-9, None)
    j = np.argmax(pm, axis=1); lead = prot_p4[ar, j]; lpm = pm[ar, j]
    lcth = lead[:, 3] / np.clip(lpm, 1e-9, None)
    in_win = (lpm > P_LO) & (lpm < P_HI) & (lcth > COSP)
    n_eject = (pm > P_EJECT).sum(1)
    base = (w > 0) & in_win & no_pi & (pmu > MU_LO) & (cmu > COSMU)
    if carbon:
        base = base & (np.linalg.norm(struck[:, 1:], axis=1) > 1.0)
    return base, n_eject, lead


def adonis():
    obs = []
    for f in ("data/oracle/t2k_cc0pi_engine_rich.npz", "data/oracle/t2k_cc1pi_engine_rich.npz"):
        b = dict(np.load(f))
        no_pi = (~np.isin(b["pid_pi"], _PI)) & (~np.isin(b["cr_pid"], _PI))
        base, ne, lead = base_count_lead(b["prot"], b["mu"], b["struck"], b["w"], no_pi)
        s = base & (ne == NREQ)
        o = O.cc0pi_obs(b["mu"][s], lead[s], b["struck"][s], b["nu"][s]); o["w"] = b["w"][s]
        obs.append(o)
    return {k: np.concatenate([o[k] for o in obs]) for k in obs[0]}


def achilles():
    a = dict(np.load("data/oracle/t2k_cc1pi_rich_ach_FSI.npz")); wn = float(a["weight_to_nb"])
    no_pi = (np.isin(a["pi_pid"], list(_PI)).sum(1) == 0)
    base, ne, lead = base_count_lead(a["prot_p4"], a["mu"], a["struck"], a["w"], no_pi)
    s = base & (ne == NREQ)
    o = O.cc0pi_obs(a["mu"][s], lead[s], a["struck"][s], a["nu"][s]); o["w"] = a["w"][s] * wn
    return o, a, base


VARS = [("dpt", np.linspace(0, 800, 16), r"$\delta p_T$ [MeV]"),
        ("dalphat", np.linspace(0, np.pi, 13), r"$\delta\alpha_T$ [rad]"),
        ("Q2", np.linspace(0, 1.4, 16), r"$Q^2$ [GeV$^2$]"),
        ("lp_p", np.linspace(450, 1000, 16), r"lead $p_p$ [MeV]")]


def main():
    ado = adonis(); ach, a, base = achilles()
    sa, sr = ado["w"].sum(), ach["w"].sum()
    print(f"P_EJECT={P_EJECT} MeV, require EXACTLY {NREQ} ejected protons")
    print(f"sigma: ADoNIS {sa:.4e}  ACHILLES {sr:.4e}  ACH/ADO {sr/max(sa,1e-30):.3f}  "
          f"(N: ado {len(ado['w'])}, ach {len(ach['w'])})")
    title = f"CC0$\\pi$ EXACTLY {NREQ} ejected protons (|p|>{P_EJECT:.0f} MeV); leading-proton std cuts"
    fig, res, sig = PL.make_figure(VARS, ach, ado, title=title, ratio_ylim=(0.4, 1.8),
                                   ado_label=f"ADoNIS ({NREQ}p)", ref_label=f"ACHILLES ({NREQ}p)", panel_w=4.0)
    out = f"paper_figures/cc0pi_{NREQ}proton.png"; fig.savefig(out, dpi=120); print("wrote", out)
    for k, _, _ in VARS:
        print(f"  {k:8s} chi2/ndf {res[k]['chi2']/max(res[k]['ndf'],1):.2f}  ACH/ADO {res[k]['ach_ado']:.3f}")
    pm = np.linalg.norm(a["prot_p4"][:, :, 1:], axis=2); ww = (a["w"] * float(a["weight_to_nb"]))[base]; tot = ww.sum()
    print("ejected-proton multiplicity (ACHILLES CC0pi), thr sensitivity:")
    for thr in (200, 250, 300, 400):
        nn = (pm > thr).sum(1)[base]
        print(f"  thr={thr}: ==1 {((nn==1)*ww).sum()/tot*100:4.1f}%  ==2 {((nn==2)*ww).sum()/tot*100:4.1f}%  "
              f">=2 {((nn>=2)*ww).sum()/tot*100:4.1f}%")


if __name__ == "__main__":
    main()
