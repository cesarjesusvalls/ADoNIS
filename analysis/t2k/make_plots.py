"""All T2K ADoNIS-vs-ACHILLES figures in ONE driver, opt-in by block.

Blocks (default: all except --pdf):
  --matrix            CC0pi/CC1pi x {qe,res,both} x {incl,0p,1p,2p} x material grid (ratio + chi2)
  --multiplicity      dsigma vs final-state multiplicity N(p/n/pi+/pi0/pi-), qe AND res
  --nucleon-momentum  dsigma vs leading proton AND neutron |p| (QE)
  --cc1pi-stv         T2K CC1pi+Np STV (pN, dpTT, dalphaT): ADoNIS vs ACHILLES vs T2K data
  --pdf               bundle every PNG produced this run into output/figures/t2k_plots.pdf

Shared: --material {C,Ar,all}  --mode {fsi,nofsi}  --flux t2k  --tag ''  (ADoNIS bank = {flux}_{mat}_{chan}{tag}.npz)
The heavy lifting lives in adonis.workflow (config/analyze/signal/observables/plotting); this is the driver.
Usage: python -m analysis.t2k.make_plots [--matrix --multiplicity ...] [--material C] [--mode fsi] [--pdf]
"""
import argparse
import glob
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root (analysis/t2k/ -> .)

OUTDIR = "output/figures"
ADO_DIR_DEFAULT = "output/adonis"
COS70 = float(np.cos(np.deg2rad(70.0)))
MU_WIN = [250.0, 7000.0]
_PRODUCED = []                                                  # PNGs written this run (for --pdf)


def _save(fig, name):
    os.makedirs(os.path.dirname(name), exist_ok=True)
    fig.savefig(name, dpi=120); plt.close(fig); _PRODUCED.append(name)
    print("wrote", name, flush=True)


def _ado_glob(flux, mat, chan, tag, suf):
    return sorted(glob.glob(f"{ADO_DIR_DEFAULT}/{flux}_{mat}_{chan}{tag}{suf}_batch*.npz"))


# ============================================================ BLOCK: cross-section matrix
# ADoNIS banks: canonical {flux}_{material}_{chanout}.npz in output/adonis (override via env).
# ACHILLES ref filenames/location pending the separate ACHILLES-output naming cleanup.
MATERIALS = {
    "C":  ("output/adonis/t2k_C_cc1pi.npz", "output/adonis/t2k_C_cc0pi.npz",
           "output/achilles/t2k_cc1pi_rich_ach_FSI_proc.npz"),
    "Ar": ("output/adonis/t2k_Ar_cc1pi.npz", "output/adonis/t2k_Ar_cc0pi.npz",
           "output/achilles/t2k_cc1pi_rich_ach_FSI_Ar_proc.npz"),
}
MATERIALS_NOFSI = {
    "C":  ("output/adonis/t2k_C_cc1pi_nofsi.npz", "output/adonis/t2k_C_cc0pi_nofsi.npz",
           "output/achilles/t2k_cc1pi_rich_ach_C_nofsi.npz"),
    "Ar": ("output/adonis/t2k_Ar_cc1pi_nofsi.npz", "output/adonis/t2k_Ar_cc0pi_nofsi.npz",
           "output/achilles/t2k_cc1pi_rich_ach_nofsi_Ar.npz"),
}
REF_PROC = {"qe": [200], "res": [401, 402], "both": None}
_TKI = [("pn", "$p_N$", [0, 120, 240, 600, 1500]),
        ("dptt", r"$\delta p_{TT}$", [-700, -300, -100, 100, 300, 700]),
        ("dalphat", r"$\delta\alpha_T$", ("lin", 0, "pi", 7)),
        ("W", "$W$", ("lin", 1080, 1700, 13)), ("Q2", "$Q^2$", ("lin", 0, 1.5, 13)),
        ("pi_p", r"$p_\pi$", ("lin", 150, 1200, 13)), ("lp_p", "lead $p_p$", ("lin", 450, 1200, 13))]
_PIFREE = [("W", "$W$", ("lin", 1080, 1700, 13)), ("Q2", "$Q^2$", ("lin", 0, 1.5, 13)),
           ("pi_p", r"$p_\pi$", ("lin", 150, 1200, 13)), ("p_mu", r"$p_\mu$", ("lin", 0, 2000, 21)),
           ("cos_mu", r"$\cos\theta_\mu$", ("lin", -0.6, 1.0, 21))]
_CC0 = [("dpt", r"$\delta p_T$", ("lin", 0, 800, 21)), ("dalphat", r"$\delta\alpha_T$", ("lin", 0, "pi", 21)),
        ("Q2", "$Q^2$", ("lin", 0, 1.4, 21)), ("W", "$W$", ("lin", 938, 1400, 21)),
        ("p_mu", r"$p_\mu$", ("lin", 0, 2000, 21)), ("cos_mu", r"$\cos\theta_\mu$", ("lin", -0.6, 1.0, 21)),
        ("lp_p", "lead $p_p$", ("lin", 450, 1000, 21))]
_MUON = [("Q2", "$Q^2$", ("lin", 0, 1.4, 21)), ("p_mu", r"$p_\mu$", ("lin", 0, 2000, 21)),
         ("cos_mu", r"$\cos\theta_\mu$", ("lin", -0.6, 1.0, 21))]


def _obs_spec(spec):
    out = []
    for key, label, b in spec:
        if isinstance(b, tuple) and b[0] == "lin":
            out.append({"key": key, "label": label, "linspace": list(b[1:])})
        else:
            out.append({"key": key, "label": label, "edges": b})
    return out


def _signal_block(channel, contrib, pcat):
    if channel == "cc0pi":
        sd = dict(mu_win=[250.0, "inf"], cos_mu=-0.6, p_win=[450.0, 1000.0], cth=0.4, pi_win=None,
                  proton_lead="global", pion_id="none", eject_thresh=250.0, target="carbon")
        sd["ref_proc"] = REF_PROC[contrib]
        if pcat == "incl":
            sd.update(proton_count="ge1", require_proton=True); return sd, _CC0
        if pcat == "0p":
            sd.update(require_proton=False, n_ejected=0); return sd, _MUON
        sd.update(require_proton=True, n_ejected=int(pcat[0])); return sd, _CC0
    sd = dict(mu_win=[250.0, 7000.0], pi_win=[150.0, 1200.0], p_win=[450.0, 1200.0], cth="cos70",
              proton_lead="in_window", pion_id="pip", count_recoil_neutron=False, target="carbon",
              W_conv="vertex", ref_proc=REF_PROC[contrib])
    if pcat == "incl":
        sd.update(proton_count="ge1", require_proton=True); return sd, _TKI
    if pcat == "0p":
        sd.update(proton_count="eq0", require_proton=False); return sd, _PIFREE
    sd.update(proton_count="eq" + pcat[0], require_proton=True); return sd, _TKI


def _banks(contrib, res_bank, qe_bank):
    if contrib == "qe":  return {"adonis_qe": [qe_bank]}
    if contrib == "res": return {"adonis_res": [res_bank]}
    return {"adonis_res": [res_bank], "adonis_qe": [qe_bank]}


def _neff(w):
    w = np.asarray(w); w = w[w > 0]
    return (int(w.size), float(w.sum() ** 2 / np.sum(w ** 2)) if w.size else 0.0, float(w.sum()))


def _run_cell(A, load_analysis_config, material, channel, contrib, pcat, mode, outdir):
    mats = MATERIALS_NOFSI if mode == "nofsi" else MATERIALS
    res_bank, qe_bank, ref = mats[material]
    res_bank = os.environ.get("ADONIS_RES_BANK", res_bank)
    qe_bank = os.environ.get("ADONIS_QE_BANK", qe_bank)
    if mode == "nofsi" and (not os.path.exists(res_bank) or not os.path.exists(qe_bank) or not os.path.exists(ref)):
        return dict(cell=f"{channel}_{contrib}_{pcat}_{material}", ok=False, err="bank/ref missing")
    sigd, obs = _signal_block(channel, contrib, pcat)
    if mode == "nofsi":
        sigd["ref_proc"] = None
    inp = _banks(contrib, res_bank, qe_bank); inp["reference"] = [ref]
    tag = f"{channel}_{contrib}_{pcat}_{material}"
    import yaml
    cfg_d = dict(inputs=inp, signal=sigd, observables=_obs_spec(obs), data={"enabled": False},
                 out_path=f"{outdir}/{tag}.png", carbon_only=True, ratio_ylim=[0.5, 1.6],
                 title=f"{material} {channel} [{contrib}] {pcat} {'noFSI' if mode == 'nofsi' else 'FSI'}: ADoNIS vs ACHILLES")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg_d, fh); tmp = fh.name
    cfg = load_analysis_config(tmp); os.unlink(tmp)
    try:
        ado = A.build_adonis(cfg); ref_d = A.build_reference(cfg)
        Na, nea, sa = _neff(ado["w"]); Nr, ner, sr = _neff(ref_d["w"])
        if Na == 0 or Nr == 0:
            return dict(cell=tag, ok=False, err="empty selection")
        A.run_analysis(cfg)
    except Exception as e:
        return dict(cell=tag, ok=False, err=str(e)[:80])
    ratio = sr / sa if sa else float("nan")
    rel = math.sqrt((1 / nea if nea else 0) + (1 / ner if ner else 0))
    pull = (ratio - 1) / (ratio * rel) if (rel and ratio == ratio) else float("nan")
    return dict(cell=tag, ok=True, sa=sa, sr=sr, ratio=ratio, Na=Na, nea=nea, Nr=Nr, ner=ner, err=ratio * rel, pull=pull)


def block_matrix(which, mode):
    import adonis.workflow.analyze as A
    from adonis.workflow.config import load_analysis_config
    outdir = f"{OUTDIR}/matrix_nofsi" if mode == "nofsi" else f"{OUTDIR}/matrix"
    os.makedirs(outdir, exist_ok=True)
    base = MATERIALS_NOFSI if mode == "nofsi" else MATERIALS
    mats = list(base) if which == "all" else [which]
    contribs = ("both",) if mode == "nofsi" else ("qe", "res", "both")
    rows = []
    for material in mats:
        for channel in ("cc0pi", "cc1pi"):
            for contrib in contribs:
                for pcat in ("incl", "0p", "1p", "2p"):
                    r = _run_cell(A, load_analysis_config, material, channel, contrib, pcat, mode, outdir)
                    rows.append(r)
                    if r["ok"]:
                        _PRODUCED.append(f"{outdir}/{r['cell']}.png")
                        print(f"[OK] {r['cell']:26s} ACH/ADO={r['ratio']:.3f}+/-{r['err']:.3f} "
                              f"({r['pull']:+.1f}s)  N_eff ado={r['nea']:.0f} ach={r['ner']:.0f}", flush=True)
                    else:
                        print(f"[FAIL] {r['cell']:26s} {r['err']}", flush=True)
    print("\n==== MATRIX SUMMARY ====")
    print(f"{'cell':30s} {'ACH/ADO':>9s} {'pull':>6s} {'Neff_ado':>9s} {'Neff_ach':>9s}")
    for r in rows:
        if r["ok"]:
            print(f"{r['cell']:30s} {r['ratio']:9.3f} {r['pull']:+6.1f} {r['nea']:9.0f} {r['ner']:9.0f}")
        else:
            print(f"{r['cell']:30s}   FAIL: {r['err']}")


# ============================================================ BLOCK: multiplicity
_SPECIES = ["N(p)", "N(n)", "N(pi+)", "N(pi0)", "N(pi-)"]
_ADO_FIELD = {"N(p)": "n_p", "N(n)": "n_n", "N(pi+)": "n_pip", "N(pi0)": "n_pi0", "N(pi-)": "n_pim"}


def _mu_mask(mu):
    p = np.linalg.norm(mu[:, 1:], axis=1); cz = mu[:, 3] / np.clip(p, 1e-9, None)
    return (p > MU_WIN[0]) & (p < MU_WIN[1]) & (cz > COS70)


def _mult_ado(paths):
    paths = [p for p in paths if "n_p" in np.load(p, allow_pickle=True).files]
    print(f"  merging {len(paths)} batch(es)", flush=True)
    nb = len(paths); cnt = {lab: [] for lab in _SPECIES}; ws = []
    for p in paths:
        b = dict(np.load(p, allow_pickle=True)); m = _mu_mask(b["mu"]) & (b["w"] > 0)
        for lab in _SPECIES:
            cnt[lab].append(b[_ADO_FIELD[lab]][m])
        ws.append(b["w"][m])
    return {lab: np.concatenate(cnt[lab]) for lab in _SPECIES}, (np.concatenate(ws) / nb if nb else np.array([]))


def _mult_ach(ach_path, procs):
    d = dict(np.load(ach_path, allow_pickle=True))
    sel = np.isin(d["proc"], procs) & _mu_mask(d["mu"])
    w = d["w"][sel] * float(d["weight_to_nb"]); pp4 = d["prot_p4"][sel]; pip = d["pi_pid"][sel]
    out = {"N(p)": (np.linalg.norm(pp4[:, :, 1:], axis=2) > 1e-6).sum(1),
           "N(pi+)": (pip == 211).sum(1), "N(pi0)": (pip == 111).sum(1), "N(pi-)": (pip == -211).sum(1)}
    if "neut_p4" in d:
        np4 = d["neut_p4"][sel]; out["N(n)"] = (np.linalg.norm(np4[:, :, 1:], axis=2) > 1e-6).sum(1)
    return out, w


def block_multiplicity(mat, mode, flux, tag):
    from adonis.workflow.plotting import chi2_ratio_panel, hist_with_errors
    suf = "_nofsi" if mode == "nofsi" else ""
    for channel in ("qe", "res"):
        chan = "cc0pi" if channel == "qe" else "cc1pi"
        ado_paths = _ado_glob(flux, mat, chan, tag, suf)
        ach_path = os.environ.get("ACH_BANK", f"output/achilles/t2k_cc1pi_rich_ach_{'nofsi' if mode == 'nofsi' else 'FSI'}_proc.npz")
        procs = [200] if channel == "qe" else [401, 402]
        clabel = channel.upper()
        if not ado_paths:
            print(f"[multiplicity] {clabel}: no ADoNIS banks ({flux}_{mat}_{chan}{tag}{suf}_batch*) -- skip", flush=True)
            continue
        aQ, wQ = _mult_ado(ado_paths)
        hQ, hwQ = _mult_ach(ach_path, procs) if os.path.exists(ach_path) else ({}, np.array([]))
        nmax = 6; edges = np.arange(nmax + 2.0); ctr = 0.5 * (edges[1:] + edges[:-1])
        fig, axes = plt.subplots(2, 5, figsize=(23, 6), height_ratios=[3, 1], sharex="col")
        for c, lab in enumerate(_SPECIES):
            a0, a1 = axes[0, c], axes[1, c]; a0.set_xlim(edges[0], edges[-1])
            if lab in hQ:
                chi2_ratio_panel(a0, a1, edges, {"values": hQ[lab], "w": hwQ}, {"values": aQ[lab], "w": wQ},
                                 label=f"{clabel}: {lab}", ado_label="ADoNIS", ref_label="ACHILLES", logy=True)
            else:
                dd, ed = hist_with_errors(aQ[lab], wQ, edges)
                a0.errorbar(ctr, dd, yerr=ed, fmt="s", color="C0", ms=3, capsize=2, lw=0.9, label="ADoNIS")
                a0.set_title(f"{clabel}: {lab}", fontsize=9); a0.set_ylim(bottom=0)
                a1.axhline(1.0, ls="--", color="green", lw=0.7); a1.set_ylim(0.5, 1.6)
                a1.text(0.5, 0.4, "no ACHILLES", transform=a1.transAxes, ha="center", fontsize=8, color="r")
                a1.set_xlabel(f"{clabel}: {lab}", fontsize=8)
        axes[0, 0].set_ylabel(f"{clabel}  d$\\sigma$/dN [nb]"); axes[1, 0].set_ylabel("ACH/ADO"); axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"{clabel} d$\\sigma$ vs final-state multiplicity ({mat}, muon-acceptance CC-inclusive; "
                     f"{'NO FSI' if mode == 'nofsi' else 'FSI'}): ADoNIS vs ACHILLES")
        fig.tight_layout(); _save(fig, f"{OUTDIR}/multiplicity_{mat}_{channel}{suf}.png")
        if hwQ.size:
            print(f"  {clabel}: sigma ADoNIS={wQ.sum():.4e} ACH={hwQ.sum():.4e} ACH/ADO={hwQ.sum()/wQ.sum():.3f}", flush=True)


# ============================================================ BLOCK: nucleon momentum
def block_nucleon_momentum(mat, mode, flux, tag):
    from adonis.workflow.plotting import chi2_ratio_panel
    suf = "_nofsi" if mode == "nofsi" else ""
    edges = np.linspace(0.0, 2000.0, 41)
    ado_paths = _ado_glob(flux, mat, "cc0pi", tag, suf)
    ach_path = os.environ.get("ACH_BANK", f"output/achilles/t2k_cc1pi_rich_ach_{'nofsi' if mode == 'nofsi' else 'FSI'}_proc.npz")
    if not ado_paths:
        print(f"[nucleon-momentum] no ADoNIS QE banks ({flux}_{mat}_cc0pi{tag}{suf}_batch*) -- skip", flush=True)
        return
    for species in ("proton", "neutron"):
        ado_field = "neut" if species == "neutron" else "prot"
        ach_field = "neut_p4" if species == "neutron" else "prot_p4"
        paths = [p for p in ado_paths if ado_field in np.load(p, allow_pickle=True).files]
        if not paths:
            print(f"[nucleon-momentum] {species}: no bank field {ado_field} -- skip", flush=True); continue
        lp, ws = [], []
        for p in paths:
            b = dict(np.load(p, allow_pickle=True)); m = _mu_mask(b["mu"]) & (b["w"] > 0)
            plead = np.linalg.norm(b[ado_field][m][:, 0, 1:], axis=1); has = plead > 0
            lp.append(plead[has]); ws.append(b["w"][m][has])
        aP, wa = np.concatenate(lp), np.concatenate(ws) / len(paths)
        fig, (a0, a1) = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[3, 1], sharex=True)
        if os.path.exists(ach_path):
            d = dict(np.load(ach_path, allow_pickle=True)); sel = np.isin(d["proc"], [200]) & _mu_mask(d["mu"])
            hp = np.linalg.norm(d[ach_field][sel][:, 0, 1:], axis=1); has = hp > 0
            hP, wh = hp[has], d["w"][sel][has] * float(d["weight_to_nb"])
            chi2_ratio_panel(a0, a1, edges, {"values": hP, "w": wh}, {"values": aP, "w": wa},
                             label=rf"QE: lead $p_{{{species[0]}}}$ [MeV]", ado_label="ADoNIS", ref_label="ACHILLES", logy=True)
        else:
            from adonis.workflow.plotting import hist_with_errors
            dd, ed = hist_with_errors(aP, wa, edges); ctr = 0.5 * (edges[1:] + edges[:-1])
            a0.errorbar(ctr, dd, yerr=ed, fmt="s", color="C0", ms=3, lw=0.9, label="ADoNIS"); a0.set_yscale("log")
        a0.set_ylabel("QE  d$\\sigma$/dp [nb/MeV]"); a1.set_ylabel("ACH/ADO"); a0.legend(fontsize=8)
        fig.suptitle(f"QE d$\\sigma$/dp(lead {species}) ({mat}, muon-acceptance CC-inclusive; "
                     f"{'NO FSI' if mode == 'nofsi' else 'FSI'}): ADoNIS vs ACHILLES")
        fig.tight_layout(); _save(fig, f"{OUTDIR}/nucleon_momentum_{mat}_{species}{suf}.png")


# ============================================================ BLOCK: CC1pi+Np STV (ADoNIS vs ACHILLES vs data)
def block_cc1pi_stv(flux="t2k", mat="C"):
    """CC1pi+Np STV (pN/dpTT/daT) vs ACHILLES vs T2K data.  RES-C comes from the EXISTING engine bank
    (no regeneration -- the bank already carries the per-event rich final state); only the free-proton H
    contribution (pi+ always survives, no nuclear FSI) is generated."""
    import jax; jax.config.update("jax_enable_x64", True)
    import adonis.xsec.dcc_current as dcc; dcc.BATCH_INTERP = "spline"
    import adonis.workflow.observables as O
    from adonis.workflow.config import SignalDef
    from adonis.workflow.signal import select_signal
    from adonis.workflow.free_proton import generate_H
    from analysis.utils.hepmc import weight_to_nb_of

    NH, NSEED = 50000, 4
    MU_LO, MU_HI = 250.0, 7000.0; PI_LO, PI_HI = 150.0, 1200.0; P_LO, P_HI = 450.0, 1200.0
    NB_PER_CM2 = 1e33; A_CH = 13.0

    def _acc(p4, lo, hi):
        m = np.linalg.norm(p4[:, 1:], axis=1)
        return (m > lo) & (m < hi) & (p4[:, 3] / np.clip(m, 1e-9, None) > COS70)

    def load_data(name):
        lines = open(f"../nuisance/data/T2K/CC1pipNp_STV/xsec_{name}.txt").read().splitlines()
        edges = np.array([float(x) for x in lines[0].split(":")[1].split()])
        vals = np.array([float(x) for x in lines[1].split(":")[1].split()]); nb = len(vals)
        cov = np.array([[float(x) for x in lines[3 + i].split()] for i in range(nb)])
        return edges, vals, cov

    # RES-C: read the EXISTING engine bank (no regeneration); CC1pi+ tight signal via the package selector.
    bankf = f"{ADO_DIR_DEFAULT}/{flux}_{mat}_cc1pi.npz"
    if not os.path.exists(bankf):
        print(f"[cc1pi-stv] ADoNIS bank {bankf} missing (run generation first) -- skip", flush=True); return
    sd = SignalDef(mu_win=(MU_LO, MU_HI), pi_win=(PI_LO, PI_HI), p_win=(P_LO, P_HI), cth=COS70,
                   proton_lead="in_window", pion_id="pip", require_proton=True, proton_count="ge1",
                   target="carbon", W_conv="vertex")
    selC = select_signal(bankf, sd)                          # dptt/pn/dalphat/w (absolute nb) from the bank
    print(f"  RES-C (bank {os.path.basename(bankf)}): {len(selC['w'])} signal ev  sigma={selC['w'].sum():.4e} nb", flush=True)

    # RES-H: free proton (pi+ always survives, no nuclear FSI) -- the only piece generated.
    Hp = {"dptt": [], "pn": [], "dalphat": [], "w": []}
    for s in range(NSEED):
        knu, kmu, pN, pPi, w = (np.asarray(x) for x in generate_H(NH, seed=s))
        m = (w > 0) & _acc(kmu, MU_LO, MU_HI) & _acc(pPi, PI_LO, PI_HI) & _acc(pN, P_LO, P_HI)
        dptt, pn, dat, _ = O.tki(kmu[m], pPi[m], pN[m], np.ones(int(m.sum()), bool), s)
        Hp["dptt"].append(dptt); Hp["pn"].append(pn); Hp["dalphat"].append(dat); Hp["w"].append(w[m])
    H = {k: np.concatenate(v) for k, v in Hp.items()}; H["w"] = H["w"] / NSEED
    print(f"  RES-H (free p, generated): {len(H['w'])} ev  sigma={H['w'].sum():.4e} nb", flush=True)

    ado = {k: np.concatenate([selC[k], H[k]]) for k in ("dptt", "pn", "dalphat", "w")}

    ach_path = os.environ.get("ACH_CC1PI", "output/achilles/t2k_cc1pi_tki_achilles.npz")
    if not os.path.exists(ach_path):
        print(f"[cc1pi-stv] ACHILLES ref {ach_path} missing -- skip (set ACH_CC1PI)", flush=True); return
    ach = np.load(ach_path); ach_w = np.asarray(ach["w"]) * weight_to_nb_of(ach)
    ado["dalphat_deg"] = np.degrees(ado["dalphat"])
    VARS = [("pn", "pN", r"$p_N$ [MeV/c]"), ("dptt", "dpTT", r"$\delta p_{TT}$ [MeV/c]"),
            ("dalphat_deg", "daT", r"$\delta\alpha_T$ [deg]")]
    fig, axes = plt.subplots(2, len(VARS), figsize=(6.2 * len(VARS), 7.5), height_ratios=[3, 1], sharex="col")
    for c, (key, rel, xlab) in enumerate(VARS):
        edges, dval, dcov = load_data(rel); bw = np.diff(edges); ctr = 0.5 * (edges[1:] + edges[:-1])
        d_y = dval * NB_PER_CM2 * A_CH; d_e = np.sqrt(np.diag(dcov)) * NB_PER_CM2 * A_CH
        def hist(v, w):
            h, _ = np.histogram(v, bins=edges, weights=w); e2, _ = np.histogram(v, bins=edges, weights=w ** 2)
            return h / bw, np.sqrt(e2) / bw
        ach_v = np.degrees(np.asarray(ach["dalphat"])) if key == "dalphat_deg" else np.asarray(ach[key])
        da, ea = hist(ach_v, ach_w); dd, ed = hist(ado[key], ado["w"]); ax, axr = axes[0, c], axes[1, c]
        ax.fill_between(edges, np.append(da - ea, (da - ea)[-1]), np.append(da + ea, (da + ea)[-1]),
                        step="post", color="0.5", alpha=0.25, lw=0)
        ax.step(edges, np.append(da, da[-1]), where="post", color="0.35", lw=1.4, label="ACHILLES")
        ax.errorbar(ctr, dd, yerr=ed, fmt="s", color="C0", ms=4, capsize=2, lw=1.0, label="ADoNIS (CH)", zorder=4)
        ax.errorbar(ctr, d_y, yerr=d_e, fmt="o", color="k", ms=5, capsize=3, lw=1.4, label="T2K data", zorder=5)
        ax.set_ylabel(r"d$\sigma$/dx [nb/unit per CH]"); ax.set_ylim(bottom=0); ax.legend(fontsize=8)
        ax.set_title(f"CC1$\\pi^+$Np   {xlab}")
        with np.errstate(divide="ignore", invalid="ignore"):
            r = da / dd; re = r * np.sqrt((ed / dd) ** 2 + (ea / da) ** 2)
        msk = (da > 0) & (dd > 0) & np.isfinite(re)
        chi2 = float(np.sum((da[msk] - dd[msk]) ** 2 / (ea[msk] ** 2 + ed[msk] ** 2)))
        axr.axhspan(0.9, 1.1, color="green", alpha=0.12); axr.axhline(1.0, ls="--", color="green")
        axr.errorbar(ctr[msk], r[msk], yerr=re[msk], fmt="o", color="C3", ms=4, capsize=2, lw=1.1)
        axr.set_ylim(0.5, 1.5); axr.set_ylabel("ACH / ADO"); axr.set_xlabel(xlab)
        axr.text(0.03, 0.85, f"$\\chi^2$/ndf = {chi2/max(int(msk.sum()),1):.2f}", transform=axr.transAxes, fontsize=10)
    fig.suptitle("T2K CC1$\\pi^+$Np STV (tight) — ADoNIS vs ACHILLES vs T2K data (PRD 103 112009)", fontsize=13)
    fig.tight_layout(); _save(fig, f"{OUTDIR}/cc1pi_stv_pn_dptt_data.png")


# ============================================================ PDF bundle
def build_pdf():
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.image as mpimg
    figs = [f for f in _PRODUCED if os.path.exists(f)] or sorted(glob.glob(f"{OUTDIR}/**/*.png", recursive=True))
    if not figs:
        print("[pdf] no PNGs to bundle", flush=True); return
    out = f"{OUTDIR}/t2k_plots.pdf"
    with PdfPages(out) as pdf:
        for fn in figs:
            img = mpimg.imread(fn); h, w = img.shape[:2]
            fig = plt.figure(figsize=(min(11, w / 100), min(8.5, h / 100)))
            ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.axis("off"); pdf.savefig(fig, dpi=150); plt.close(fig)
    print(f"{len(figs)} figures -> {out}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="All T2K ADoNIS-vs-ACHILLES figures (opt-in blocks).")
    ap.add_argument("--matrix", action="store_true")
    ap.add_argument("--multiplicity", action="store_true")
    ap.add_argument("--nucleon-momentum", action="store_true")
    ap.add_argument("--cc1pi-stv", action="store_true")
    ap.add_argument("--pdf", action="store_true", help="bundle every PNG produced this run into one PDF")
    ap.add_argument("--material", default="C", choices=["C", "Ar", "all"])
    ap.add_argument("--mode", default="fsi", choices=["fsi", "nofsi"])
    ap.add_argument("--flux", default="t2k")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    blocks = (a.matrix, a.multiplicity, a.nucleon_momentum, a.cc1pi_stv)
    do_all = not any(blocks) and not a.pdf                    # no block flag -> all; --pdf alone -> bundle only
    mats = ["C", "Ar"] if a.material == "all" else [a.material]
    if a.matrix or do_all:
        block_matrix(a.material, a.mode)
    for mat in mats:
        if a.multiplicity or do_all:
            block_multiplicity(mat, a.mode, a.flux, a.tag)
        if a.nucleon_momentum or do_all:
            block_nucleon_momentum(mat, a.mode, a.flux, a.tag)
    if a.cc1pi_stv or do_all:
        block_cc1pi_stv(a.flux, "C")                         # T2K CC1pi is the CH (carbon+H) measurement
    if a.pdf:
        build_pdf()


if __name__ == "__main__":
    main()
