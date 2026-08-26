"""FIGURE D -- binned event rate in every sample, before and after the closure fit.

Three curves per panel, from the reference closure npz:

  data        Asimov data m(theta_true), with the fit's per-bin sigma; bins whose MC error exceeds the
              cut carry sigma=inf, are dropped from the fit, and are drawn HOLLOW here.
  pre-fit     m(theta_nominal), the prediction before fitting.
  post-fit    m(theta_hat), the prediction the fit arrives at.

chi2 per sample is quoted before/after, over live bins only.  This is an Asimov closure (post-fit lands
on the data by construction) -- not a goodness-of-fit test, which needs the toy ensemble
(fig_closure_summary panel b).

Usage:  python -m analysis.paper.inference.fig_rates [label]
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
_OVERFLOW = {"t2k_cc0pi:pmu", "ee_omega:omega"}
_XSCALE = {"pz": 1e-3, "pt": 1e-3}
_XUNIT = {"dpt": "[MeV/c]", "pn": "[MeV/c]", "dptt": "[MeV/c]", "pmu": "[MeV/c]",
          "pt": "[GeV/c]", "pz": "[GeV/c]", "tpi": "[MeV]", "omega": "[MeV]",
          "dalphat": "[rad]", "q2": "[GeV$^2$]", "cos_mu": ""}
_BEAMX = {"pip": "$p_{\\pi^+}$ [MeV/c]", "prot": "$p_p$ [MeV/c]", "neut": "$p_n$ [MeV/c]"}
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


def _xlab(k):
    """Axis title: the observable and its units.  The panel name carries the SAMPLE."""
    if ":" in k:
        obs = k.split(":", 1)[1]
        return f"{_OBS.get(obs, obs)} {_XUNIT.get(obs, '')}".strip()
    return _BEAMX.get(k.split("_", 1)[0], "")


def main(label="sec4_P1"):
    style.use()
    z = np.load(style.ALTGEN / f"{label}.npz", allow_pickle=True)
    data = np.asarray(z["data"]); sig = np.asarray(z["sigma"])
    pre = np.asarray(z["model_nom"]); post = np.asarray(z["fit_model"])
    row0 = np.asarray(z["row0"]); dsk = [str(x) for x in z["dskeys"]]
    live = np.isfinite(sig) & (sig > 0)
    print(f"{len(dsk)} samples, {int(live.sum())}/{len(sig)} live bins")

    nd = len(dsk); nc = 3; nr = int(np.ceil(nd / nc))
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans"}):
        fig, ax = plt.subplots(nr, nc, figsize=(3.4 * nc, 1.64 * nr))
        for i, k in enumerate(dsk):
            A = ax.flat[i]; a, b = row0[i], row0[i + 1]
            m = live[a:b]
            ek = f"{k}_edges"
            if ek in z.files:
                e = np.asarray(z[ek], float)
                _xs = _XSCALE.get(k.split(":", 1)[1], 1.0) if ":" in k else 1.0
                e = e * _xs
                x = 0.5 * (e[1:] + e[:-1]); w = np.diff(e)
            else:
                x = np.arange(b - a) + 0.5; w = np.ones(b - a)
            d_, p_, q_, s_ = data[a:b], pre[a:b], post[a:b], sig[a:b]
            if k in _OVERFLOW:
                d_, p_, q_, s_ = d_[:-1], p_[:-1], q_[:-1], s_[:-1]
                m = m[:-1]
                if ek in z.files:
                    e = e[:-1]; x = x[:-1]; w = w[:-1]
            _sc = float(np.nanmax(np.concatenate([p_, q_, d_])))
            _sc = _sc if _sc > 0 else 1.0
            p_, q_, d_, s_ = p_ / _sc, q_ / _sc, d_ / _sc, s_ / _sc
            if ek in z.files:
                A.stairs(p_, e, color=C_PRE, lw=1.5, label="Pre-fit (nominal)")
                A.stairs(q_, e, color=C_POST, lw=1.5, ls="--", label="Post-fit")
            else:
                A.step(x, p_, where="mid", color=C_PRE, lw=1.5, label="Pre-fit (nominal)")
                A.step(x, q_, where="mid", color=C_POST, lw=1.5, ls="--", label="Post-fit")
            A.errorbar(x[m], d_[m], yerr=s_[m], fmt="o", ms=2.8, lw=0, elinewidth=0.9,
                       color=C_DATA, label="Asimov data", zorder=5)
            if (~m).any():
                A.plot(x[~m], d_[~m], "o", ms=2.8, mfc="none", mec="0.45", mew=0.8, zorder=4)
            c_pre = float(np.sum(((p_[m] - d_[m]) / s_[m]) ** 2))
            c_post = float(np.sum(((q_[m] - d_[m]) / s_[m]) ** 2))
            if (~m).any():
                A.text(0.985, 0.62, f"{int(m.sum())}/{len(m)} bins",
                       transform=A.transAxes, ha="right", va="top", fontsize=7.0, color="0.35")
            if ek in z.files:
                A.set_xlim(float(e[0]), float(e[-1]))
            A.tick_params(labelsize=8.5, top=False, right=False)
            A.set_ylim(0, 1.50)
            A.set_yticks([0, 1]); A.set_yticklabels(["0", "1"])
            _n = len(q_); _half = max(_n // 2, 1)
            _side = "right" if np.nansum(q_[:_half]) >= np.nansum(q_[_half:]) else "left"
            _xt, _ha = (0.97, "right") if _side == "right" else (0.03, "left")
            A.text(_xt, 0.965, _pretty(k), transform=A.transAxes, ha=_ha, va="top", fontsize=10)
            A.text(_xt, 0.835, f"$\\chi^2$ {c_pre:.1f} $\\to$ {c_post:.0e}  ({int(m.sum())} bins)",
                   transform=A.transAxes, ha=_ha, va="top", fontsize=8.5, color="0.3")
            A.set_xlabel(_xlab(k), fontsize=8.5)
            print(f"  {k:>26} chi2 {c_pre:10.1f} -> {c_post:9.2e}   live {int(m.sum()):3d}/{b-a}")
        for j in range(nd, nr * nc):
            ax.flat[j].axis("off")
        _h, _l = ax.flat[0].get_legend_handles_labels()
        fig.legend(_h, _l, loc="upper center", ncol=3, fontsize=11.5, frameon=False,
                   bbox_to_anchor=(0.5, 1.012))
        fig.supylabel("differential cross-section  (a.u., peak normalised)", fontsize=11, x=0.005)
        _cut = float(z["mask_mcfrac"]) if "mask_mcfrac" in z.files else float("nan")
        _lv, _tt = int(live.sum()), int(live.size)
        print((f"  mask: MC-error cut {_cut:.0%} of central" if _cut == _cut else "  mask: nominal")
              + f" - {_lv}/{_tt} bins used, {_tt - _lv} masked (open circles)")
        fig.tight_layout(pad=0.9, h_pad=1.0, w_pad=1.1, rect=(0.012, 0, 1, 0.968))
        style.save(fig, "Asimov_xsec_samples")


if __name__ == "__main__":
    p = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*(p[:1] or []))
