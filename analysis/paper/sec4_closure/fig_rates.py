"""FIGURE D -- the binned event rate in every sample, before and after the closure fit.

Three things per panel, all from the reference closure npz:

  data        the Asimov data, m(theta_true), with the fit's own per-bin sigma (5% syst; bins whose MC
              error exceeded 5% of the central value carry sigma=inf and are dropped from the fit -- they
              are drawn hollow here so the masking is visible rather than implied)
  pre-fit     m(theta_nominal), the prediction before any fitting
  post-fit    m(theta_hat), the prediction the fit arrives at

The point of the figure: the injected truth is a long way from nominal (37 sigma across 17 dials at the
P1 point), so the pre-fit curve misses the data visibly in the samples that constrain those dials, and
the post-fit curve lands on it.  chi2 per sample is quoted before and after, over LIVE bins only.

This is an Asimov closure, so "post-fit lands on the data" is exact by construction -- the residual is
~1e-27.  That is the statement being made: the model can reproduce its own truth from a blind start.  It
is NOT a goodness-of-fit test, which needs the toy ensemble (figure A, panel b).

Usage:  python -m analysis.paper.sec4_closure.fig_rates [label]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style

C_PRE, C_POST, C_DATA = "#c8842a", "#1f4b9c", "0.15"


_SAMP = {"t2k_cc0pi": "T2K CC0$\\pi$", "t2k_cc1pi_ch": "T2K CC1$\\pi$",
         "minerva_stv": "MINERvA CC0$\\pi$", "minerva_ptpz": "MINERvA incl.",
         "minerva_cc1pip_tpi": "MINERvA CC1$\\pi^+$", "minerva_cc1pip_q2": "MINERvA CC1$\\pi^+$",
         "ee_omega": "$(e,e')$C"}
_OBS = {"dpt": "$\\delta p_T$", "dalphat": "$\\delta\\alpha_T$", "pmu": "$p_\\mu$",
        "cos_mu": "$\\cos\\theta_\\mu$", "pn": "$p_N$", "dptt": "$\\delta p_{TT}$",
        "pt": "$p_T^\\mu$", "pz": "$p_\\parallel^\\mu$", "omega": "$\\omega$",
        "tpi": "$T_\\pi$", "q2": "$Q^2$"}
_BEAM = {"pip_react": "$\\pi^+$C $\\sigma_{\\rm reac}$", "pip_abs": "$\\pi^+$C $\\sigma_{\\rm abs}$",
         "prot_react": "$p$C $\\sigma_{\\rm reac}$", "prot_pipro": "$p$C $\\sigma_{\\pi\\rm prod}$",
         "neut_react": "$n$C $\\sigma_{\\rm reac}$", "neut_pipro": "$n$C $\\sigma_{\\pi\\rm prod}$"}


def _pretty(k):
    """Sample label in the same style the other sections use, instead of the raw dskey."""
    if k in _BEAM:
        return _BEAM[k]
    if ":" in k:
        smp, obs = k.split(":", 1)
        return f"{_SAMP.get(smp, smp)} {_OBS.get(obs, obs)}"
    return k


def main(label="sec4_P1"):
    style.use()
    z = np.load(style.ALTGEN / f"{label}.npz", allow_pickle=True)
    data = np.asarray(z["data"]); sig = np.asarray(z["sigma"])
    pre = np.asarray(z["model_nom"]); post = np.asarray(z["fit_model"])
    row0 = np.asarray(z["row0"]); dsk = [str(x) for x in z["dskeys"]]
    live = np.isfinite(sig) & (sig > 0)
    print(f"{len(dsk)} samples, {int(live.sum())}/{len(sig)} live bins")

    # 3 x 7 fits the 21 samples exactly; the old 4-wide grid left three empty slots.
    nd = len(dsk); nc = 3; nr = int(np.ceil(nd / nc))
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans"}):
        fig, ax = plt.subplots(nr, nc, figsize=(3.4 * nc, 2.05 * nr))
        for i, k in enumerate(dsk):
            A = ax.flat[i]; a, b = row0[i], row0[i + 1]
            m = live[a:b]
            # bin centres from the stored edges when present, else the bin index -- never silently
            # renumber, since several samples share an observable name across experiments
            ek = f"{k}_edges"
            if ek in z.files:
                e = np.asarray(z[ek], float); x = 0.5 * (e[1:] + e[:-1]); w = np.diff(e)
            else:
                x = np.arange(b - a) + 0.5; w = np.ones(b - a)
            d_, p_, q_, s_ = data[a:b], pre[a:b], post[a:b], sig[a:b]
            A.step(x, p_, where="mid", color=C_PRE, lw=1.5, label="pre-fit (nominal)")
            A.step(x, q_, where="mid", color=C_POST, lw=1.5, ls="--", label="post-fit")
            A.errorbar(x[m], d_[m], yerr=s_[m], fmt="o", ms=2.8, lw=0, elinewidth=0.9,
                       color=C_DATA, label="Asimov data", zorder=5)
            if (~m).any():          # masked bins drawn hollow: the cut is visible, not implied
                A.plot(x[~m], d_[~m], "o", ms=2.8, mfc="none", mec="0.6", mew=0.7, zorder=4)
            c_pre = float(np.sum(((p_[m] - d_[m]) / s_[m]) ** 2))
            c_post = float(np.sum(((q_[m] - d_[m]) / s_[m]) ** 2))
            # NO CLIPPING: `step(where="mid")` draws about bin CENTRES, so an axis auto-scaled to the
            # centres cuts the last bin in half whenever it is wider than its neighbours -- which is
            # exactly the overflow bin these releases end on.  Pin the limits to the outer EDGES.
            if ek in z.files:
                A.set_xlim(float(e[0]), float(e[-1]))
            A.tick_params(labelsize=6.5, top=False, right=False)
            # headroom for the in-panel text, without making the panel taller
            _top = max(np.nanmax(np.concatenate([p_, q_, d_])), 1e-30)
            A.set_ylim(0, 1.42 * _top)
            # LEFT or RIGHT, whichever half carries less: several of these peak at the left edge
            # (delta-p_T, p_n) and several at the right (the beam sigmas), so a fixed corner would
            # sit on the data in half the panels.
            _n = len(q_); _half = max(_n // 2, 1)
            _side = "right" if np.nansum(q_[:_half]) >= np.nansum(q_[_half:]) else "left"
            _xt, _ha = (0.97, "right") if _side == "right" else (0.03, "left")
            A.text(_xt, 0.965, _pretty(k), transform=A.transAxes, ha=_ha, va="top", fontsize=7.5)
            A.text(_xt, 0.855, f"$\\chi^2$ {c_pre:.0f} $\\to$ {c_post:.2g}  ({int(m.sum())} bins)",
                   transform=A.transAxes, ha=_ha, va="top", fontsize=6.5, color="0.3")
            print(f"  {k:>26} chi2 {c_pre:10.1f} -> {c_post:9.2e}   live {int(m.sum()):3d}/{b-a}")
        for j in range(nd, nr * nc):
            ax.flat[j].axis("off")
        _h, _l = ax.flat[0].get_legend_handles_labels()
        fig.legend(_h, _l, loc="upper center", ncol=3, fontsize=10, frameon=False,
                   bbox_to_anchor=(0.5, 1.0))
        fig.supylabel("rate per bin  (sample units)", fontsize=9)
        # explicit padding: the per-panel titles carry two lines (name + chi2), and at the
        # default pad they collided with the axis above.
        fig.tight_layout(pad=0.9, h_pad=1.0, w_pad=1.1, rect=(0, 0, 1, 0.982))
        style.save(fig, f"{label}_figD")


if __name__ == "__main__":
    p = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*(p[:1] or []))
