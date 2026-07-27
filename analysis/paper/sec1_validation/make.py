"""Paper section 1 -- ADoNIS reproduces ACHILLES.  Overlay + ratio panel + chi2/ndf for EVERY observable
the analysis uses:

  neutrino (T2K, 12C):   CC0pi  dpt, dalphaT, p_mu, cos_mu
                         CC1pi  p_N, dpTT, dalphaT, p_pi, cos_pi
                         incl   N_p (>300 MeV/c), N_pi+-
  tagged beams (12C):    pi+ / p / n  reaction + (absorption | pion production)

The neutrino side is sourced EXACTLY as Gate I / the knob x sample Fisher source it: ADoNIS from the
frozen event bank (`build_physfit_datasets`, the same 11 datasets and the same bin edges), ACHILLES from
the rich oracle `output/achilles/t2k_cc1pi_rich_ach_FSI_proc.npz`.  The STV formulas are reused verbatim
(`bank_plot.dpt/dat/pN_1pi/dptt_1pi/dat_1pi`) on both sides; only the per-event 4-vector EXTRACTION
differs (bank ragged final state vs the oracle's top-K padded arrays).  So this figure validates precisely
the datasets section 3 measures.

CC0pi signal here is TOPOLOGICAL (no final-state pion) on BOTH sides -- the only definition the ACHILLES
oracle can express (it has no "primary pion absorbed" truth flag) and the one an experiment measures.
Gate I uses the primary-pion definition internally; the two differ only by the rare cascade-created
surviving pion in an otherwise-CC0pi event (NN->NNpi, ~1-3% at T2K), so the distributions are the same to
that level.

Everything is per-event (values, weight) fed to the ONE validated panel `plotting.chi2_ratio_panel`
(weighted dsigma/dx, sqrt(sum w^2) errors, ACH stat band + ADO points, ratio, per-bin chi2).  Weights are
absolute nb on both sides (bank w0; oracle w * weight_to_nb, = 1 here).

Usage:  python -m analysis.paper.sec1_validation.make [--no-beams]
"""
import os
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "altgen"))

from adonis.reweight import bank_plot as BP, bank_reweight as BR
from adonis.reweight.full_knobs import nominal_knobs
from adonis.workflow.plotting import chi2_ratio_panel
from analysis.paper import style

BANKDIR = os.environ.get("ADONIS_BANK", "output/event_bank_v2")   # override -> paper_banks/nu_T2K_C/merged
ORACLE = str(ROOT / "output" / "achilles" / "t2k_cc1pi_rich_ach_FSI_proc.npz")

# T2K acceptance constants (single source of truth = tune / bank_plot; repeated here only for the oracle)
MU_LO, COSMU, PL_LO, PL_HI, COSP = 250.0, -0.6, 450.0, 1000.0, 0.4          # CC0pi signal (tune._sel)
P_THR_MULT = 300.0                                                          # proton multiplicity threshold

LABELS = {"dpt": r"CC0$\pi$  $\delta p_T$ [MeV/c]", "dat": r"CC0$\pi$  $\delta\alpha_T$ [rad]",
          "pmu": r"CC0$\pi$  $p_\mu$ [MeV/c]", "cosmu": r"CC0$\pi$  $\cos\theta_\mu$",
          "pn": r"CC1$\pi$  $p_N$ [MeV/c]", "dptt": r"CC1$\pi$  $\delta p_{TT}$ [MeV/c]",
          "daT": r"CC1$\pi$  $\delta\alpha_T$ [deg]", "ppi": r"CC1$\pi$  $p_\pi$ [MeV/c]",
          "cospi": r"CC1$\pi$  $\cos\theta_\pi$", "n_p": r"incl  $N_p$ ($>$300 MeV/c)",
          "n_chpi": r"incl  $N_{\pi^\pm}$"}
CC0PI = ["dpt", "dat", "pmu", "cosmu"]
CC1PI = ["pn", "dptt", "daT", "ppi", "cospi"]
MULT = ["n_p", "n_chpi"]


# ---------------------------------------------------------------- physfit edges (the Gate-I binning) ---
def physfit_edges():
    """The exact 11-observable bin edges Gate I uses -- from build_physfit_datasets on the same bank."""
    import physical_fit as PF
    B = BP.load_bank(BANKDIR)
    w0 = np.asarray(BR.weight_jit(BR.to_jax(B), nominal_knobs(), BR.default_grids()))
    ds = PF.build_physfit_datasets(B, w0, lambda *_: None)
    return {d["key"]: np.asarray(d["edges"]) for d in ds}, B, w0


# ---------------------------------------------------------------- ADoNIS side (event bank) --------------
def adonis_obs(B, w0):
    """Per-event (values, w) for the 11 observables, from the bank -- topological CC0pi + T2K CC1pi STV."""
    out = {}
    kmu = B["k_mu"].astype(np.float64)
    pmu = np.linalg.norm(kmu[:, 1:], axis=1); cmu = kmu[:, 3] / np.maximum(pmu, 1e-9)
    # CC0pi (topological: no final-state pion)
    sig0, lead0 = BP.signal_cc0pi(B, topological=True)
    out["dpt"] = (np.asarray(BP.dpt(B, lead0))[sig0], w0[sig0])
    out["dat"] = (np.asarray(BP.dat(B, lead0))[sig0], w0[sig0])
    out["pmu"] = (pmu[sig0], w0[sig0])
    out["cosmu"] = (cmu[sig0], w0[sig0])
    # CC1pi STV
    m1, lead1, pip1 = BP.signal_cc1pi_stv(B)
    ppi = np.linalg.norm(np.asarray(pip1)[:, 1:], axis=1); cpi = np.asarray(pip1)[:, 3] / np.maximum(ppi, 1e-9)
    out["pn"] = (np.asarray(BP.pN_1pi(kmu, lead1, pip1))[m1], w0[m1])
    out["dptt"] = (np.asarray(BP.dptt_1pi(kmu, lead1, pip1))[m1], w0[m1])
    out["daT"] = (np.degrees(np.asarray(BP.dat_1pi(kmu, lead1, pip1)))[m1], w0[m1])
    out["ppi"] = (ppi[m1], w0[m1])
    out["cospi"] = (cpi[m1], w0[m1])
    # inclusive multiplicities (muon acceptance only)
    inc = (pmu > MU_LO) & (cmu > COSMU)
    npip, _npi0, npim = BP.pion_counts(B)
    out["n_p"] = (BP.n_protons(B, pmin=P_THR_MULT).astype(float)[inc], w0[inc])
    out["n_chpi"] = ((npip + npim).astype(float)[inc], w0[inc])
    return out


# ---------------------------------------------------------------- ACHILLES side (rich oracle) ----------
def _cc0pi_stv(kmu, lead):
    """dpt, dat from mu + leading proton -- verbatim tune._dpt/_dat, inlined for the oracle 4-vectors."""
    lt = kmu[:, 1:3]; dv = lt + lead[:, 1:3]
    dpt = np.linalg.norm(dv, axis=1)
    num = -np.sum(lt * dv, axis=1)
    den = np.linalg.norm(lt, axis=1) * np.clip(np.linalg.norm(dv, axis=1), 1e-9, None)
    dat = np.arccos(np.clip(num / den, -1.0, 1.0))
    return dpt, dat


def achilles_obs():
    d = np.load(ORACLE, allow_pickle=True)
    w = np.asarray(d["w"]) * float(d["weight_to_nb"])
    mu = np.asarray(d["mu"]); prot = np.asarray(d["prot_p4"]); pip4 = np.asarray(d["pi_p4"])
    pipid = np.asarray(d["pi_pid"])
    pmu = np.linalg.norm(mu[:, 1:], axis=1); cmu = mu[:, 3] / np.maximum(pmu, 1e-9)
    npi = (pipid != 0).sum(1)                                              # final-state pions (0 = pad)
    npip = (pipid == 211).sum(1); npim = (pipid == -211).sum(1); npi0 = (pipid == 111).sum(1)
    lead0 = prot[:, 0, :]                                                  # global leading proton (sorted)
    plead = np.linalg.norm(lead0[:, 1:], axis=1); clead = lead0[:, 3] / np.maximum(plead, 1e-9)

    out = {}
    # CC0pi TOPOLOGICAL: no final pion + leading proton in the CC0pi window + muon acceptance
    sel0 = ((npi == 0) & (plead > 0) & (pmu > MU_LO) & (cmu > COSMU)
            & (plead >= PL_LO) & (plead < PL_HI) & (clead > COSP))
    dpt, dat = _cc0pi_stv(mu, lead0)
    out["dpt"] = (dpt[sel0], w[sel0]); out["dat"] = (dat[sel0], w[sel0])
    out["pmu"] = (pmu[sel0], w[sel0]); out["cosmu"] = (cmu[sel0], w[sel0])

    # CC1pi STV: exactly one pi+, no other pion; leading proton in [450,1200) & forward; mu/pi forward+window
    P_LO_1, P_HI_1, PI_LO, PI_HI = 450.0, 1200.0, 150.0, 1200.0
    MU_HI = 7000.0; CTH = float(np.cos(np.deg2rad(70.0)))
    # leading pi+ 4-vector (first pi+ slot; rows are |p|-sorted so slot order is momentum order)
    is_pip = (pipid == 211)
    has_pip = is_pip.any(1)
    ipip = np.argmax(is_pip, axis=1)                                      # first pi+ slot (0 if none)
    pip = pip4[np.arange(len(mu)), ipip]                                  # (n,4); junk where ~has_pip
    ppi = np.linalg.norm(pip[:, 1:], axis=1); cpi = pip[:, 3] / np.maximum(ppi, 1e-9)
    # leading ACCEPTED proton in the CC1pi window (mirror bank_plot.leading_proton_window over the 10 slots)
    pm = np.linalg.norm(prot[:, :, 1:], axis=2); pcz = prot[:, :, 3] / np.maximum(pm, 1e-9)
    acc = (pm >= P_LO_1) & (pm < P_HI_1) & (pcz > CTH)
    key = np.where(acc, pm, -1.0)
    jlead = np.argmax(key, axis=1); has_lead1 = key[np.arange(len(mu)), jlead] > 0
    lead1 = prot[np.arange(len(mu)), jlead]
    m1 = ((npip == 1) & (npi0 == 0) & (npim == 0) & has_pip & has_lead1
          & (pmu >= MU_LO) & (pmu < MU_HI) & (cmu > CTH)
          & (ppi >= PI_LO) & (ppi < PI_HI) & (cpi > CTH))
    out["pn"] = (np.asarray(BP.pN_1pi(mu, lead1, pip))[m1], w[m1])
    out["dptt"] = (np.asarray(BP.dptt_1pi(mu, lead1, pip))[m1], w[m1])
    out["daT"] = (np.degrees(np.asarray(BP.dat_1pi(mu, lead1, pip)))[m1], w[m1])
    out["ppi"] = (ppi[m1], w[m1]); out["cospi"] = (cpi[m1], w[m1])

    # inclusive multiplicities (muon acceptance only)
    inc = (pmu > MU_LO) & (cmu > COSMU)
    n_p = ((np.linalg.norm(prot[:, :, 1:], axis=2) > P_THR_MULT)).sum(1).astype(float)
    out["n_p"] = (n_p[inc], w[inc]); out["n_chpi"] = ((npip + npim).astype(float)[inc], w[inc])
    return out


# ---------------------------------------------------------------- plotting ------------------------------
def _grid(keys, edges_by, ado, ach, fname, title):
    n = len(keys)
    fig, ax = plt.subplots(2, n, figsize=(max(3.3 * n, 7.0), 5.6), height_ratios=[3, 1],
                           squeeze=False, sharex="col")
    for c, k in enumerate(keys):
        av, aw = ach[k]; dv, dw = ado[k]
        res = chi2_ratio_panel(ax[0, c], ax[1, c], edges_by[k],
                               {"values": av, "w": aw}, {"values": dv, "w": dw},
                               label=LABELS[k], ado_label="ADoNIS", ref_label="ACHILLES",
                               ratio_band=(0.95, 1.05), ratio_ylim=(0.8, 1.2))
        ax[0, c].text(0.04, 0.90, f"$\\chi^2$/ndf {res['chi2']/max(res['ndf'],1):.2f}",
                      transform=ax[0, c].transAxes, fontsize=7.5)
        if c == 0:
            ax[0, c].legend(fontsize=7)
            ax[0, c].set_ylabel(r"$d\sigma/dx$ [nb]")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    style.save(fig, fname)


def main(beams=True):
    style.use()
    edges_by, B, w0 = physfit_edges()
    ado = adonis_obs(B, w0)
    ach = achilles_obs()
    _grid(CC0PI, edges_by, ado, ach, "sec1_cc0pi", r"ADoNIS vs ACHILLES — T2K CC0$\pi$ on $^{12}$C")
    _grid(CC1PI, edges_by, ado, ach, "sec1_cc1pi", r"ADoNIS vs ACHILLES — T2K CC1$\pi$ on $^{12}$C")
    _grid(MULT, edges_by, ado, ach, "sec1_multiplicity",
          r"ADoNIS vs ACHILLES — CC-inclusive multiplicities on $^{12}$C")
    if beams:
        from analysis.paper.beams import make_figs
        make_figs.main()                                                  # -> output/paper/beams_validation


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-beams", action="store_true")
    a = ap.parse_args()
    main(beams=not a.no_beams)
