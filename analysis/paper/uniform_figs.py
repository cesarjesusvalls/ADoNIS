"""UNIFORM ADoNIS-vs-ACHILLES final-state figures for EVERY paper_banks sample.

One code path, one oracle blueprint (fs_rich), one shared panel (chi2_ratio_panel).  For each bank we read
the SAME ragged final state (fs_off/fs_pid/fs_chg/fs_p4 -- identical across neutrino / (e,e') / beam banks)
and the SAME rich oracle (achilles/fsrich/<bank>.npz: prot_p4/pi_p4/pi_pid/neut_p4/w), compute the SAME
final-state observables (N_p, N_pi+-, leading-proton |p| & cos-theta, leading-pi |p|), and plot them
through the ONE validated ratio panel.  No per-bank special-casing.

  python -m analysis.paper.uniform_figs [bank ...]   ->  output/paper_banks_figs/fs_<bank>.pdf  (+ .png)
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from adonis.workflow.plotting import chi2_ratio_panel      # noqa: E402
from analysis.paper import style                           # noqa: E402

BANKS = ["nu_T2K_C", "nu_MINERvA_C", "nu_uBooNE_Ar",
         "beam_prot_C", "beam_pip_C", "beam_neut_C", "beam_prot_Ar", "beam_pip_Ar", "beam_neut_Ar",
         "ee_C", "ee_Ar"]
# bank -> rich-oracle npz stem(s) under output/achilles/fsrich/ (ee combines qe+res to match the bank)
ORACLE = {b: [b] for b in BANKS}
# ee uses the FSI-ON oracles (rebuilt-from-source achilles; the prebuilt image segfaulted on (e,e')+cascade)
# so it's post-FSI vs post-FSI, matching the ADoNIS ee bank.  _fsi combined by combine_fsrich.
ORACLE["ee_C"] = ["ee_C_qe_fsi", "ee_C_res_fsi"]
ORACLE["ee_Ar"] = ["ee_Ar_qe_fsi", "ee_Ar_res_fsi"]
BANKDIR = ROOT / "output" / "paper_banks"
ORADIR = ROOT / "output" / "achilles" / "fsrich"

TITLE = {"nu_T2K_C": "T2K $\\nu_\\mu$ on $^{12}$C", "nu_MINERvA_C": "MINERvA $\\nu_\\mu$ on $^{12}$C",
         "nu_uBooNE_Ar": "MicroBooNE $\\nu_\\mu$ on $^{40}$Ar",
         "beam_prot_C": "p on $^{12}$C", "beam_pip_C": "$\\pi^+$ on $^{12}$C", "beam_neut_C": "n on $^{12}$C",
         "beam_prot_Ar": "p on $^{40}$Ar", "beam_pip_Ar": "$\\pi^+$ on $^{40}$Ar", "beam_neut_Ar": "n on $^{40}$Ar",
         "ee_C": "(e,e') on $^{12}$C", "ee_Ar": "(e,e') on $^{40}$Ar"}


def _seg_max(val, seg_ids, n_events):
    """max of `val` within each event segment (seg_ids = event index per particle); -1 where empty."""
    out = np.full(n_events, -1.0)
    if len(val):
        np.maximum.at(out, seg_ids, val)
    return out


def _seg_count(mask, seg_ids, n_events):
    out = np.zeros(n_events)
    if mask.any():
        np.add.at(out, seg_ids[mask], 1.0)
    return out


def bank_obs(bank):
    """final-state observables + per-event weight from a paper_banks bank (fs_* ragged)."""
    import json
    fs = sorted((BANKDIR / bank / "merged").glob("chunk_*.npz"))
    is_ee = bank.startswith("ee"); is_beam = bank.startswith("beam")
    wk = "c" if is_ee else "w0"
    xkey = "theta" if is_ee else ("reacted" if is_beam else None)   # per-event field for the acceptance mask
    W, PID, P4, SEG, X = [], [], [], [], []
    ev_base = 0
    for f in fs:
        d = np.load(f)
        w = np.asarray(d[wk], float); ne = len(w)
        cnt = np.diff(np.asarray(d["fs_off"], np.int64))      # particles per event
        SEG.append(np.repeat(np.arange(ne) + ev_base, cnt))   # event index per particle
        W.append(w); PID.append(np.asarray(d["fs_pid"])); P4.append(np.asarray(d["fs_p4"], float))
        if xkey:
            X.append(np.asarray(d[xkey]))
        ev_base += ne
    w = np.concatenate(W); pid = np.concatenate(PID); p4 = np.concatenate(P4); seg = np.concatenate(SEG)
    n_all = len(w)
    # per-event acceptance mask to make ADoNIS the IDENTICAL measurement as the oracle:
    #  (e,e'): electron in [14,17] deg (the oracle's window; Mott is forward-peaked so "all angles" differs).
    #  beam:   REACTED events only (ACHILLES CrossSection-mode writes only reacted; ADoNIS fs_* also holds the
    #          non-reacted beam pass-through, which must be excluded).
    if is_ee or is_beam:
        x = np.concatenate(X)
        keep = ((x >= 14.0) & (x <= 17.0)) if is_ee else x.astype(bool)
        pk = keep[seg]; seg = (np.cumsum(keep) - 1)[seg[pk]]; pid = pid[pk]; p4 = p4[pk]; w = w[keep]
    # absolute-nb normalization to match the oracle sigma:
    if is_beam:
        w = w * 1e6 / n_all         # w0=PIR2 [mb] -> nb; dsigma = PIR2 * N_reacted-particles / N_total (== n_tried)
    else:
        man = json.load(open(BANKDIR / bank / "merged" / "manifest.json"))
        w = w / man.get("n_chunks", len(fs))   # independent full-flux sigma estimates averaged (== load_bank)
    return _observables(pid, p4, seg, w, len(w))


def oracle_obs(bank):
    """same observables from the rich oracle(s) (prot_p4 (M,4), pi_p4/pi_pid (K,4)/(K,))."""
    ev_w, plead, pcos, pilead, npr, npi = [], [], [], [], [], []
    import json
    bnorm = None
    if bank.startswith("beam"):
        bn = json.load(open(ORADIR / "beam_norm.json"))[bank[5:]]
        bnorm = bn["pir2_mb"] * 1e6 / bn["n_tried"]     # geometric CrossSection-mode: sigma = PIR2*N/n_tried
    for stem in ORACLE[bank]:
        d = np.load(ORADIR / f"{stem}.npz", allow_pickle=True)
        # beam ACHILLES uses PIR2*N_reacted/n_tried [mb], NOT flux*gen_xs -> constant weight per reacted event
        w = np.full(len(d["w"]), bnorm) if bnorm is not None else np.asarray(d["w"], float) * float(d["weight_to_nb"])
        pr = np.asarray(d["prot_p4"], float); pip = np.asarray(d["pi_p4"], float); pipid = np.asarray(d["pi_pid"])
        prm = np.linalg.norm(pr[:, :, 1:], axis=2)                      # (n,M)
        prc = np.where(prm > 0, pr[:, :, 3] / np.maximum(prm, 1e-9), -2.0)
        pim = np.linalg.norm(pip[:, :, 1:], axis=2)                     # (n,K)
        ispi = (pipid == 211) | (pipid == -211)
        pim_ch = np.where(ispi, pim, 0.0)
        ev_w.append(w)
        npro = (prm > 0).sum(1); npio = ispi.sum(1)
        npr.append(npro.astype(float)); npi.append(npio.astype(float))
        # leading |p|: NaN where the event has no such particle (so both sides exclude empties identically)
        pl = prm.max(1); plead.append(np.where(npro > 0, pl, np.nan))
        pcos.append(np.where(npro > 0, prc[np.arange(len(pr)), prm.argmax(1)], np.nan))
        pilead.append(np.where(npio > 0, pim_ch.max(1), np.nan))
    w = np.concatenate(ev_w)
    return {"n_p": (np.concatenate(npr), w), "n_chpi": (np.concatenate(npi), w),
            "p_lead_p": (np.concatenate(plead), w), "cos_lead_p": (np.concatenate(pcos), w),
            "p_lead_pi": (np.concatenate(pilead), w)}


def _observables(pid, p4, seg, w, n):
    mom = np.linalg.norm(p4[:, 1:], axis=1)
    cth = np.where(mom > 0, p4[:, 3] / np.maximum(mom, 1e-9), -2.0)
    isp = pid == 2212
    ispi = (pid == 211) | (pid == -211)
    n_p = _seg_count(isp, seg, n); n_pi = _seg_count(ispi, seg, n)
    p_lead_p = _seg_max(np.where(isp, mom, -1.0), seg, n)
    p_lead_pi = _seg_max(np.where(ispi, mom, -1.0), seg, n)
    # leading-proton cos-theta: cth at the proton achieving p_lead_p (approx via a second seg pass)
    lead_is = isp & np.isclose(mom, p_lead_p[seg]) & (mom > 0)
    cos_lead_p = np.full(n, -2.0)
    np.maximum.at(cos_lead_p, seg[lead_is], cth[lead_is])
    return {"n_p": (n_p, w), "n_chpi": (n_pi, w), "p_lead_p": (np.where(p_lead_p >= 0, p_lead_p, np.nan), w),
            "cos_lead_p": (cos_lead_p, w), "p_lead_pi": (np.where(p_lead_pi >= 0, p_lead_pi, np.nan), w)}


KEYS = ["n_p", "n_chpi", "p_lead_p", "p_lead_pi", "cos_lead_p"]
LABELS = {"n_p": "$N_p$", "n_chpi": "$N_{\\pi^\\pm}$", "p_lead_p": "leading $p$  $|p|$ [MeV/c]",
          "p_lead_pi": "leading $\\pi^\\pm$  $|p|$ [MeV/c]", "cos_lead_p": "leading $p$  $\\cos\\theta$"}
EDGES = {"n_p": np.arange(-0.5, 6.5), "n_chpi": np.arange(-0.5, 4.5),
         "p_lead_p": np.linspace(0, 1500, 26), "p_lead_pi": np.linspace(0, 1200, 26),
         "cos_lead_p": np.linspace(-1, 1, 26)}


def make(bank):
    ado = bank_obs(bank); ach = oracle_obs(bank)
    fig, ax = plt.subplots(2, len(KEYS), figsize=(3.1 * len(KEYS), 5.4), height_ratios=[3, 1],
                           squeeze=False, sharex="col")
    for c, k in enumerate(KEYS):
        av, aw = ach[k]; dv, dw = ado[k]
        mask_a = np.isfinite(av); mask_d = np.isfinite(dv)
        res = chi2_ratio_panel(ax[0, c], ax[1, c], EDGES[k],
                               {"values": av[mask_a], "w": aw[mask_a]}, {"values": dv[mask_d], "w": dw[mask_d]},
                               label=LABELS[k], ado_label="ADoNIS", ref_label="ACHILLES",
                               ratio_band=(0.9, 1.1), ratio_ylim=(0.5, 1.5))
        print(f"  chi2/ndf  {bank:14s} {k:12s} {res['chi2']/max(res['ndf'],1):8.2f}  (ndf={res['ndf']})", flush=True)
        ax[0, c].text(0.05, 0.88, f"$\\chi^2$/ndf {res['chi2']/max(res['ndf'],1):.2f}",
                      transform=ax[0, c].transAxes, fontsize=7.5)
        if c == 0:
            ax[0, c].legend(fontsize=7); ax[0, c].set_ylabel(r"$d\sigma/dx$ [nb]")
    fig.suptitle(f"ADoNIS vs ACHILLES — {TITLE[bank]}  (final state)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    style.save(fig, f"fs_{bank}")


if __name__ == "__main__":
    style.use()
    banks = sys.argv[1:] or BANKS
    for b in banks:
        try:
            make(b); print(f"[ok] {b}", flush=True)
        except Exception as e:
            print(f"[FAIL] {b}: {e}", flush=True)
