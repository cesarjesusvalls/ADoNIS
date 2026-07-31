"""Paper Fig 2 (anl_bnl): free-nucleon RES single-pion sigma(E_nu) for the three CC channels
  nu_mu p -> mu- p pi+ ,  nu_mu n -> mu- n pi+ ,  nu_mu n -> mu- p pi0
ADoNIS vs ACHILLES (a pure cross-section curve -- no events run, no FSI).

BOTH sides are frozen scans on disk, read in ~0.1 s; this module only reduces and draws them.

ACHILLES: the free-nucleon monochromatic scan (output/oracle_freenucleon_scan/{H,N}_E{MeV}).  The
per-event weight is NOT constant (importance-sampled phase space, ~6k distinct values in 200k events),
so sigma_channel = sum of w * weight_to_nb over the channel's events and the statistical error is
sqrt(sum w^2) -- a binomial-on-counts error would be wrong (and vanishes for the single-channel proton
target).  H = proton target (all p pi+); N = neutron target (p pi0 + n pi+, split by pi_pid).

ADoNIS: the mirror-image scan, output/adonis_freenucleon_scan/<channel>_E<MeV>.npz, built by
`python -m analysis.paper.freenucleon_bank` (which owns ENERGIES + the channel table).  sigma(E) =
< a2 * flux_factor * SPIN_AVG * J_3body > over a monochromatic beam on a FREE nucleon at rest
(J_beam = 1), standard error std(w)/sqrt(n).  Units below: 10^-38 cm^2 (1 nb = 1e5).

  python -m analysis.paper.sec1_validation.fig2_anl_sigma
"""
import sys
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from analysis.paper import style                                  # noqa: E402
from analysis.paper.freenucleon_bank import (                     # noqa: E402
    ENERGIES, CHANNEL_SPECS, load_scan)

NB_TO_1E38 = 1.0e5                                                 # 1 nb = 1e-33 cm^2 = 1e5 x 10^-38 cm^2
SCAN = str(ROOT / "output" / "oracle_freenucleon_scan")


def achilles_sigma(E, species, pi_pid):
    """sigma(E) in nb + weighted MC error from the monochromatic scan npz.

    The per-event weight is NOT constant (importance-sampled phase space), so sigma_channel = sum of w
    over the channel's events and the statistical error is sqrt(sum w^2) -- a binomial-on-counts error
    would be wrong (and vanishes for the single-channel proton target)."""
    fs = sorted(glob.glob(f"{SCAN}/{species}_E{int(E)}/*.npz"))
    if not fs:
        return np.nan, np.nan
    d = np.load(fs[0], allow_pickle=True)
    w = np.asarray(d["w"], float) * float(d["weight_to_nb"])
    m = np.asarray(d["pi_pid"]) == pi_pid
    return float(w[m].sum()), float(np.sqrt((w[m] ** 2).sum()))


def main():
    style.use()
    scan = load_scan()                    # ADoNIS side: the frozen bank (raises with the build command)
    # ONE panel, all three channels (as in the ANL/BNL reference): colour = channel, and the usual
    # light-solid ADoNIS / dark-dashed ACHILLES pair within each colour.
    fig, ax = plt.subplots(2, 1, figsize=(4.6, 3.4), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0})
    top, bot = ax
    COLS = (style.C_TOTAL, style.C_QE, style.C_RES)
    for c, (tag, _ci, tex, sp, ach_pi) in enumerate(CHANNEL_SPECS):
        a_col, h_col = style.lighter(COLS[c]), style.darker(COLS[c])
        aS, aE = (v * NB_TO_1E38 for v in scan[tag])
        hS, hE = (np.array(v) * NB_TO_1E38 for v in
                  zip(*(achilles_sigma(E, sp, ach_pi) for E in ENERGIES)))
        # a genuine SCAN (sigma sampled at 10 energies), so points+line -- not steps
        top.fill_between(ENERGIES, aS - aE, aS + aE, color=a_col, alpha=0.25, lw=0)
        top.plot(ENERGIES, aS, "-", color=a_col, lw=1.6)
        top.errorbar(ENERGIES, hS, yerr=hE, fmt="s", ms=3.0, color=h_col, capsize=1.5,
                     lw=1.0, ls="--", zorder=3)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = aS / hS; re = np.abs(r) * np.sqrt((aE / aS) ** 2 + (hE / np.where(hS > 0, hS, np.nan)) ** 2)
        bot.errorbar(ENERGIES, r, yerr=re, fmt="o", ms=2.5, color=COLS[c], capsize=1.5, lw=1.0)
        m = np.isfinite(hS) & (hS > 0)                                  # points with an ACHILLES value
        chi2 = float(np.sum((aS[m] - hS[m]) ** 2 / (aE[m] ** 2 + hE[m] ** 2 + 1e-30)))
        print(f"  {tag}  chi2/ndf {chi2/max(m.sum(),1):7.2f} (ndf={int(m.sum())})", flush=True)
        for j, E in enumerate(ENERGIES):
            print(f"      E={E:5.0f}  ADO={aS[j]:.4e}  ACH={hS[j]:.4e}  ratio={aS[j]/hS[j]:.4f}"
                  f"  (SE_ado={aE[j]/aS[j]*100:.2f}% SE_ach={hE[j]/hS[j]*100:.2f}%)", flush=True)
    top.set_xlim(0, 4600); top.set_ylim(bottom=0)
    top.set_ylim(top.get_ylim()[0], top.get_ylim()[1] * 1.30)           # headroom for the legend
    top.set_ylabel(r"$\sigma$ [$10^{-38}$ cm$^2$]")
    handles, labels, hmap = style.swatches(
        [(tex, COLS[c]) for c, (_t, _ci, tex, _sp, _pid) in enumerate(CHANNEL_SPECS)])
    top.legend(handles, labels, handler_map=hmap, loc="upper left", fontsize=7,
               handlelength=3.0, labelspacing=0.3, borderpad=0.2)
    bot.axhline(1.0, ls="-", color="0.6", lw=0.8)
    for off in (0.1, 0.2):
        bot.axhline(1.0 - off, ls="--", color="0.7", lw=0.6)
        bot.axhline(1.0 + off, ls="--", color="0.7", lw=0.6)
    bot.set_ylim(0.6, 1.4); bot.set_yticks([0.8, 1.0, 1.2])
    bot.set_xlabel(r"$E_\nu$ [MeV]"); bot.set_ylabel("ratio")
    fig.suptitle(r"Free-nucleon RES single-pion $\sigma(E_\nu)$", fontsize=9, y=0.995, va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    style.save(fig, "fig02_anl_sigma")


def render(spec=None):
    """Figure-hook entry (analysis/paper/figures): render this figure. Compute is main()'s,
    unchanged; the paper style is set by style.use() inside main() and by the figures entry."""
    main()


if __name__ == "__main__":
    main()
