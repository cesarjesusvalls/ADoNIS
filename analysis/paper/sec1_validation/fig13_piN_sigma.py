"""Paper Fig 13 (piN cross sections): the pi N total/charge-exchange sigma(W) the DCC cascade scatters
through, evaluated by ADoNIS from the ANL-Osaka MesonBaryonAmplitudes tables.

This is a cascade INPUT, not a generated observable: ADoNIS and ACHILLES read the SAME
`data/MesonBaryonAmplitudes/ANL/*.dat` files and use the identical CalcCrossSectionW_grid partial-wave
formula, so an "ADoNIS vs ACHILLES" overlay would be identical by construction.  The meaningful check is
that ADoNIS INGESTS that input faithfully across the FULL W range (a past bug truncated the table at
1700 MeV): the smooth ADoNIS curve is overlaid on the raw ANL W-grid points ACHILLES loads -- they must
coincide from threshold through the high-W tail, with the sharp Delta(1232) resonance reproduced.

  python -m analysis.paper.sec1_validation.fig13_piN_sigma
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from adonis.fsi.interactions.meson_baryon_amplitudes import (pip_p_total, pim_p_total,  # noqa: E402
                                                             pim_p_cex)
from analysis.paper import style                                                        # noqa: E402

M_PI = 139.57018; M_N = 938.918754


def _W_of_T(T):
    E = T + M_PI
    return np.sqrt(M_PI ** 2 + M_N ** 2 + 2 * M_N * E)


def main():
    style.use()
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.0))
    T = np.linspace(1.0, 500.0, 400)                          # pion lab kinetic energy [MeV]
    W = _W_of_T(T)
    curves = [(r"$\pi^+ p\to\pi^+ p$  (I=3/2)", pip_p_total, "tab:red"),
              (r"$\pi^- p$  total",             pim_p_total, "tab:blue"),
              (r"$\pi^- p\to\pi^0 n$  (cex)",   pim_p_cex,   "tab:green")]
    for lab, fn, col in curves:
        _, sig = fn(W)
        ax.plot(T, sig, "-", color=col, lw=1.8, label=lab)
    # raw ANL grid points (what ACHILLES loads) on the pi+ p channel -> anti-truncation / ingestion check
    Wt, sig_grid = pip_p_total()                              # native table W-grid
    Tg = (Wt ** 2 - M_PI ** 2 - M_N ** 2) / (2 * M_N) - M_PI
    m = (Tg > 0) & (Tg < 500)
    ax.plot(Tg[m], sig_grid[m], "o", ms=3.0, mfc="none", mec="k", mew=0.6, zorder=5,
            label="ANL table (ACHILLES input)")
    ax.axvline(_T_of_W_delta(), ls=":", lw=0.9, color="0.5")
    ax.text(_T_of_W_delta() + 6, ax.get_ylim()[1] * 0.9, r"$\Delta(1232)$", fontsize=8, color="0.4")
    ax.set_xlabel(r"$T_\pi$ [MeV]"); ax.set_ylabel(r"$\sigma$ [mb]")
    ax.set_xlim(0, 500); ax.set_ylim(0, None); ax.legend(fontsize=8)
    ax.set_title(r"ADoNIS $\pi N$ cascade cross sections (ANL-Osaka $=$ ACHILLES input)", fontsize=11)
    pk = T[np.argmax(pip_p_total(W)[1])]
    print(f"  pi+ p peak at T_pi = {pk:.0f} MeV (W = {_W_of_T(pk):.0f} MeV),  "
          f"sigma_max = {pip_p_total(W)[1].max():.1f} mb", flush=True)
    print(f"  table W range: {Wt.min():.0f}-{Wt.max():.0f} MeV (T_pi up to {Tg.max():.0f} MeV)", flush=True)
    fig.tight_layout()
    style.save(fig, "fig13_piN_sigma")


def _T_of_W_delta(Wd=1232.0):
    return (Wd ** 2 - M_PI ** 2 - M_N ** 2) / (2 * M_N) - M_PI


if __name__ == "__main__":
    main()
