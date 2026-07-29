"""Paper Section 2 -- "gradient information for all knobs".

The exact autodiff Jacobian J[bin,knob] = d(dsigma/dx_bin)/dtheta_k of the measurable neutrino distributions
(the §1 data-overlay samples) w.r.t. all 28 typed PhysicsParams knobs, via workflow.gradients.bin_jacobian
(jax.jacfwd of the reweighted per-bin sum -- exact, not finite-difference).  Demonstrates that ADoNIS carries
dense gradient information across every knob -- the capability a non-differentiable generator lacks.

Each bank is heavy (~5M ragged events + jacfwd tangents), so the driver computes ONE BANK PER SUBPROCESS
(memory freed on exit), caches the fractional-response matrix to output/sec2/<bank>.npz, then assembles:
  fig_sec2_jacobian : signed-log heatmap of R[b,k] = (0.1*theta_k)*J[b,k]/nominal_b (stats-independent
                      % change per 10% knob step); rows = observable bins by sample, cols = 28 knobs by block.
  fig_sec2_reach    : per-knob max_b |R| -- every knob reachable by >=1 measurable sample.

  python -m analysis.paper.sec2_gradients.make_jacobian            # orchestrate (subprocess/bank) + plot
  python -m analysis.paper.sec2_gradients.make_jacobian --bank nu_T2K_C   # compute one bank -> npz
  python -m analysis.paper.sec2_gradients.make_jacobian --plot     # assemble cached npzs -> figures
"""
import sys
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from adonis.workflow.config import SignalDef                 # noqa: E402
from adonis.workflow import gradients as G                   # noqa: E402
from adonis.reweight import bank_plot as BP                  # noqa: E402
from analysis.paper import style                             # noqa: E402

BANKS = ROOT / "output/paper_banks_p4"
CACHE = ROOT / "output/sec2"
COS70, COS20 = float(np.cos(np.radians(70))), float(np.cos(np.radians(20)))

SD_T2K_CC0PI = SignalDef(mu_win=(250., 7e3), cos_mu=-0.6, p_win=(450., 1000.), cth=0.4,
                         pi_win=None, pion_id="none", proton_lead="global")
SD_T2K_CC1PI = SignalDef(mu_win=(250., 7e3), p_win=(450., 1200.), pi_win=(150., 1200.), cth=COS70,
                         cos_mu=None, pion_id="pip", proton_lead="in_window")
SD_MINERVA = SignalDef(mu_win=(1500., 1e4), cos_mu=COS20, p_win=(450., 1200.), cth=COS70,
                       pi_win=None, pion_id="none", proton_lead="global")
SD_UBOONE = SignalDef(mu_win=(100., 1200.), cos_mu=-1.0, p_win=(300., 1000.), cth=-1.0,
                      pi_win=None, pion_id="none", proton_lead="global", proton_count="eq1")

SAMPLES = [
    ("nu_T2K_C", "C", [
        (r"T2K CC0$\pi$ $\delta p_T$",  SD_T2K_CC0PI, "dpt",     np.linspace(0, 800, 21)),
        (r"T2K CC0$\pi$ $\delta\alpha_T$", SD_T2K_CC0PI, "dalphat", np.linspace(0, np.pi, 21)),
        (r"T2K CC1$\pi$ $p_N$",         SD_T2K_CC1PI, "pn",      np.linspace(0, 800, 21)),
        (r"T2K CC1$\pi$ $\delta p_{TT}$", SD_T2K_CC1PI, "dptt",  np.linspace(-700, 700, 29))]),
    ("nu_MINERvA_C", "C", [
        (r"MINERvA CC0$\pi$ $\delta\alpha_T$", SD_MINERVA, "dalphat", np.linspace(0, np.pi, 21)),
        (r"MINERvA CC0$\pi$ $p_n$",     SD_MINERVA, "pn",      np.linspace(0, 800, 21))]),
    ("nu_uBooNE_Ar", "Ar", [
        (r"$\mu$BooNE CC1p0$\pi$ $\delta p_T$", SD_UBOONE, "dpt", np.linspace(0, 800, 16))]),
]
# hadron beams -- clean FSI-only anchor (no hard vertex / SF): reacted sigma(p) via the kind-1 reweight
BEAM = [(r"$\pi^+$C reaction $\sigma(p)$", "beam_pip_C"),
        (r"$\pi^+$Ar reaction $\sigma(p)$", "beam_pip_Ar")]
NBEAM = 12
BLOCKS = [(0, "QE ax"), (2, "QE vec"), (7, "RES"), (11, "FSI $\\pi$"), (15, "FSI NN"), (22, "SF"), (26, "norm")]
STEP = 0.10


def _all_banks():
    return [s[0] for s in SAMPLES] + [b[1] for b in BEAM]


def compute_bank(bankname):
    """Compute the fractional-response matrix R (n_bins, 28) for every observable of ONE bank; cache to npz."""
    theta0_full = np.asarray(G.pack())
    if bankname in [b[1] for b in BEAM]:                    # ---- hadron beam: FSI-only anchor ----
        lab = next(b[0] for b in BEAM if b[1] == bankname)
        bdir = str(BANKS / bankname / "merged")
        Jfull, nom, _edges = G.fsi_jacobian_beam(bdir, nbins=NBEAM)
        with np.errstate(divide="ignore", invalid="ignore"):
            R = STEP * theta0_full[None, :] * Jfull / np.where(nom[:, None] > 0, nom[:, None], np.nan)
        R[nom < nom.max() * 0.05, :] = np.nan
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE / f"{bankname}.npz", labels=np.array([lab]), R0=R)
        print(f"  {lab:28s} nb={len(nom)} maxR={np.nanmax(np.abs(R)):.2f}  [FSI-only]", flush=True)
        print(f"[cached] {CACHE / (bankname + '.npz')}", flush=True)
        return
    bank, mat, obs_list = next(s for s in SAMPLES if s[0] == bankname)
    bdir = str(BANKS / bank / "merged")
    B = BP.load_bank(bdir)
    theta0 = np.asarray(G.pack())
    labels, mats = [], []
    for lab, sd, key, edges in obs_list:
        J, nom, stat = G.bin_jacobian(bdir, sd, key, edges, mat, B=B)
        with np.errstate(divide="ignore", invalid="ignore"):
            R = STEP * theta0[None, :] * J / np.where(nom[:, None] > 0, nom[:, None], np.nan)
        R[nom < nom.max() * 0.02, :] = np.nan          # drop <2%-of-peak (stat-noisy tail) bins
        labels.append(lab); mats.append(R)
        print(f"  {lab:28s} nb={len(nom)} sigma={nom.sum():.3e} maxR={np.nanmax(np.abs(R)):.2f}", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / f"{bank}.npz", labels=np.array(labels),
             **{f"R{i}": R for i, R in enumerate(mats)})
    print(f"[cached] {CACHE / (bank + '.npz')}", flush=True)


def _signed_log(R, eps=1e-3):
    return np.sign(R) * np.log10(1 + np.abs(R) / eps)


def assemble_and_plot():
    rows_R, rows_lab, seg = [], [], []
    for bank in _all_banks():
        d = np.load(CACHE / f"{bank}.npz", allow_pickle=True)
        for i, lab in enumerate(d["labels"]):
            R = d[f"R{i}"]; rows_R.append(R); rows_lab.append(str(lab)); seg.append(len(R))
    R = np.vstack(rows_R); S = _signed_log(R)
    style.use()
    # ---- heatmap ----
    fig, ax = plt.subplots(figsize=(13, 9))
    vmax = float(np.nanmax(np.abs(S))) or 1.0
    im = ax.imshow(S, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax), interpolation="nearest")
    ax.set_xticks(range(G.N_KNOBS)); ax.set_xticklabels(G.KNOB_LABELS, rotation=90, fontsize=6)
    ytick, y0 = [], 0
    for lab, n in zip(rows_lab, seg):
        ytick.append(y0 + n / 2 - 0.5); y0 += n
        if y0 < S.shape[0]:
            ax.axhline(y0 - 0.5, color="k", lw=0.8)
    ax.set_yticks(ytick); ax.set_yticklabels(rows_lab, fontsize=7)
    for c, _ in BLOCKS[1:]:
        ax.axvline(c - 0.5, color="k", lw=0.8)
    for c, name in BLOCKS:
        ax.text(c, -1.3, name, fontsize=7, color="0.3")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(r"signed $\log_{10}(1+|R|/10^{-3})$,  $R=0.1\,\theta_k\,\partial_k(d\sigma/dx_b)/(d\sigma/dx_b)$", fontsize=8)
    ax.set_title(r"ADoNIS gradient information: $\partial(d\sigma/dx_{\rm bin})/\partial\theta_k$ over the "
                 r"measurable distributions (all 28 knobs)", fontsize=11)
    fig.tight_layout(); style.save(fig, "fig_sec2_jacobian")
    # ---- reach ----
    reach = np.nanmax(np.abs(R), axis=0)
    fig2, ax2 = plt.subplots(figsize=(12, 4.2))
    ax2.bar(range(G.N_KNOBS), reach, color="tab:blue")
    for c, _ in BLOCKS[1:]:
        ax2.axvline(c - 0.5, color="0.6", lw=0.7, ls=":")
    ax2.set_xticks(range(G.N_KNOBS)); ax2.set_xticklabels(G.KNOB_LABELS, rotation=90, fontsize=6)
    ax2.set_yscale("log"); ax2.set_ylabel("max$_b$ |R| (10% step)")
    ax2.axhline(0.01, color="k", lw=0.6, ls="--"); ax2.text(0.2, 0.011, "1% response", fontsize=6)
    ax2.set_title("Gradient reach per knob -- exact autodiff gradient for ALL 28 knobs; "
                  "magnitude (response to a 10% step) set by physics", fontsize=10)
    fig2.tight_layout(); style.save(fig2, "fig_sec2_reach")
    nz = int((reach > 0).sum())
    print(f"  nonzero-gradient knobs: {nz}/{G.N_KNOBS};  weakest reach = {reach.min():.3e} "
          f"({G.KNOB_LABELS[int(np.argmin(reach))]}); strongest = {reach.max():.2f} "
          f"({G.KNOB_LABELS[int(np.argmax(reach))]})", flush=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--bank":
        compute_bank(argv[1]); return
    if "--plot" in argv:
        assemble_and_plot(); return
    for bank in _all_banks():                          # orchestrate: one subprocess per bank
        print(f"\n=== compute {bank} ===", flush=True)
        r = subprocess.run([sys.executable, "-m", "analysis.paper.sec2_gradients.make_jacobian",
                            "--bank", bank], cwd=str(ROOT))
        if r.returncode:
            print(f"[FAIL] {bank}: exit {r.returncode}", flush=True)
    assemble_and_plot()


if __name__ == "__main__":
    main()
