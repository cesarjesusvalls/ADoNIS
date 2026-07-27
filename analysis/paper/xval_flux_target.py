"""Cross-experiment validation: ADoNIS vs ACHILLES for a DIFFERENT flux (MINERvA on C) and a
DIFFERENT target (MicroBooNE on Ar) than the T2K-on-C of sec1.

ADoNIS side  : workflow banks output/adonis/{flux}_{mat}_{cc0pi,cc1pi}_batch*.npz -> the SAME
               signal.select_signal + observables the sec1 machinery uses, with the experiment's cuts.
ACHILLES side: the combined re-extracted oracle output/achilles/{tag}_{chan}_proc.npz (extract.py with
               the matching --experiment cuts, from the kept hepmc).
Both sides get the IDENTICAL numeric selection; the comparison is the validated chi2_ratio_panel.

Usage:  python -m analysis.paper.xval_flux_target [minerva|microboone|all]
"""
import glob
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from adonis.workflow.config import SignalDef                      # noqa: E402
from adonis.workflow import selection as SG, observables as OBS       # noqa: E402
from adonis.workflow.plotting import chi2_ratio_panel              # noqa: E402

ADO = str(ROOT / "output" / "adonis")
ACH = str(ROOT / "output" / "achilles")
OUT = ROOT / "output" / "paper"; OUT.mkdir(parents=True, exist_ok=True)
COS20 = float(np.cos(np.deg2rad(20.0)))

# per-experiment CC0pi signal cuts (mirror analysis/utils/extract.CC0PI_CUTS EXACTLY: muon cos vs
# proton cos are DIFFERENT for MINERvA -- cos_mu=cos20 (muon fwd), cth=cos70 (proton fwd)).
COS70 = float(np.cos(np.deg2rad(70.0)))
SD_CC0PI = {
    "minerva":    dict(mu_win=(1500.0, 10000.0), cos_mu=COS20, p_win=(450.0, 1200.0), cth=COS70),
    "microboone": dict(mu_win=(100.0, 1e9),      cos_mu=-1.0,  p_win=(300.0, 1200.0), cth=-1.0),
}


def _sd_cc0pi(exp):
    c = SD_CC0PI[exp]
    return SignalDef(mu_win=c["mu_win"], cos_mu=c["cos_mu"], p_win=c["p_win"], cth=c["cth"],
                     pi_win=None, pion_id="none", proton_lead="global", proton_count="ge1",
                     require_proton=True)


def _ado_concat(flux, mat, sd, muon_only=False):
    """ADoNIS: concat qe+res batches through select_signal (or muon-only for CC-inclusive).
    Each batch is an INDEPENDENT sigma estimate, so its weights are divided by the per-channel batch
    count before concatenation (mirror make_plots._merge_species: `w / nb`).  qe->cc0pi bank,
    res->cc1pi bank; both contribute to a CC0pi (res via pion absorption) or CC-inclusive sample."""
    out = {}
    for chan in ("cc0pi", "cc1pi"):                       # cc0pi = QE bank, cc1pi = RES bank
        batches = sorted(glob.glob(f"{ADO}/{flux}_{mat}_{chan}_batch*.npz"))
        nb = len(batches)
        for f in batches:
            if muon_only:
                b = SG._load(f); s = b["w"] > 0
                d = OBS.muon_obs(b["mu"][s], b["nu"][s]); d["w"] = b["w"][s]
            else:
                d = SG.select_signal(f, sd)
            d = {k: np.asarray(v) for k, v in d.items() if v is not None}
            d["w"] = d["w"] / nb                          # independent-estimate normalization
            for k, v in d.items():
                out.setdefault(k, []).append(v)
    if not out:
        raise SystemExit(f"no ADoNIS banks for {flux}_{mat}_* in {ADO}")
    return {k: np.concatenate(v) for k, v in out.items()}


def _ach(tag, chan):
    f = f"{ACH}/{tag}_{chan}_proc.npz"
    d = np.load(f, allow_pickle=True)
    w = np.asarray(d["w"]) * float(d["weight_to_nb"])
    return d, w


def _panel_grid(specs, ado, ach, achw, title, fname):
    """specs: list of (ado_key, ach_key, edges, label).  One chi2_ratio_panel column each."""
    n = len(specs)
    fig, ax = plt.subplots(2, n, figsize=(3.6 * n, 6.2), height_ratios=[3, 1],
                           squeeze=False, sharex="col")
    for c, (ak, hk, edges, label) in enumerate(specs):
        ref = {"values": np.asarray(ach[hk]), "w": achw}
        ad = {"values": np.asarray(ado[ak]), "w": np.asarray(ado["w"])}
        chi2_ratio_panel(ax[0, c], ax[1, c], np.asarray(edges), ref, ad, label=label,
                         ado_label="ADoNIS", ref_label="ACHILLES", ratio_ylim=(0.8, 1.2))
        if c == 0:
            ax[0, c].legend(fontsize=7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = OUT / fname; fig.savefig(p, dpi=150); print(f"[fig] {p}")


def do_minerva():
    ado = _ado_concat("minerva", "C", _sd_cc0pi("minerva"))
    d, w = _ach("MINERvA_C", "cc0pi")
    # STV
    _panel_grid([("dpt", "dpt", np.linspace(0, 800, 21), r"$\delta p_T$ [MeV/c]"),
                 ("dalphat", "dalphat", np.linspace(0, np.pi, 21), r"$\delta\alpha_T$ [rad]")],
                ado, d, w, "MINERvA CC0$\\pi$ on $^{12}$C — ADoNIS vs ACHILLES (STV)", "xval_minerva_stv.png")
    # muon kinematics
    _panel_grid([("p_mu", "pmu", np.linspace(1500, 10000, 21), r"$p_\mu$ [MeV/c]"),
                 ("cos_mu", "cos_mu", np.linspace(COS20, 1.0, 21), r"$\cos\theta_\mu$")],
                ado, d, w, "MINERvA CC0$\\pi$ on $^{12}$C — muon kinematics", "xval_minerva_muon.png")


def do_microboone():
    # CC-inclusive
    adoi = _ado_concat("microboone", "Ar", None, muon_only=True)
    di, wi = _ach("MicroBooNE_Ar", "cc_incl")
    _panel_grid([("p_mu", "pmu", np.linspace(0, 2000, 21), r"$p_\mu$ [MeV/c]"),
                 ("cos_mu", "cos_mu", np.linspace(-1, 1, 21), r"$\cos\theta_\mu$")],
                adoi, di, wi, "MicroBooNE CC-inclusive on $^{40}$Ar — ADoNIS vs ACHILLES", "xval_uboone_incl.png")
    # CC0pi-Np
    ado0 = _ado_concat("microboone", "Ar", _sd_cc0pi("microboone"))
    d0, w0 = _ach("MicroBooNE_Ar", "cc0pi")
    _panel_grid([("dpt", "dpt", np.linspace(0, 800, 21), r"$\delta p_T$ [MeV/c]"),
                 ("p_mu", "pmu", np.linspace(0, 2000, 21), r"$p_\mu$ [MeV/c]")],
                ado0, d0, w0, "MicroBooNE CC0$\\pi$Np on $^{40}$Ar — ADoNIS vs ACHILLES", "xval_uboone_cc0pi.png")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("minerva", "all"):
        do_minerva()
    if which in ("microboone", "all"):
        do_microboone()
