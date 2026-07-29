"""Paper Fig 2 (anl_bnl): free-nucleon RES single-pion sigma(E_nu) for the three CC channels
  nu_mu p -> mu- p pi+ ,  nu_mu n -> mu- n pi+ ,  nu_mu n -> mu- p pi0
ADoNIS vs ACHILLES (a pure cross-section curve -- no events run, no FSI).

ACHILLES: the free-nucleon monochromatic scan (output/oracle_freenucleon_scan/{H,N}_E{MeV}); each run
carries a per-event weight w (CONSTANT -> flat-phase-space MC) so sigma_channel = (#events in channel)
* w * weight_to_nb.  H = proton target (all p pi+); N = neutron target (p pi0 + n pi+, split by pi_pid).
Binomial count error sqrt(N_ch (1 - N_ch/N)) * w * weight_to_nb.

ADoNIS: sigma(E) = < a2 * flux_factor * SPIN_AVG * J_3body > over a monochromatic beam at rest on a FREE
nucleon (J_beam = 1) -- the exact free_proton.generate_H weight, generalized to the 3 channels via the
res.py channel table.  Standard error of the mean std(w)/sqrt(n).  Units: 10^-38 cm^2 (1 nb = 1e5).

  python -m analysis.paper.sec1_validation.fig2_anl_sigma [N_per_point]
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

import adonis.channels.dcc.current as _dcc                         # noqa: E402
_dcc.BATCH_INTERP = "spline"                                       # jitted spline (fast, 2e-12 vs bilinear)
from adonis.channels.res import sigma_free_nucleon, CHANNELS as RES_CHANNELS   # noqa: E402
from analysis.paper import style                                  # noqa: E402

NB_TO_1E38 = 1.0e5                                                 # 1 nb = 1e-33 cm^2 = 1e5 x 10^-38 cm^2
ENERGIES = np.array([400, 600, 800, 1000, 1300, 1600, 2000, 2600, 3400, 4400.0])   # MeV
SCAN = "/sdf/data/neutrino/cjesus/ADoNIS/output/oracle_freenucleon_scan"

# The ADoNIS-side physics comes from the CORE evaluator adonis.channels.res.sigma_free_nucleon over the
# shared res.CHANNELS table -- no cross-section code lives here.  Each row below pairs a res.CHANNELS entry
# (pdg_in, itiz, m_Nf, pi_pid, m_pi_phys, had_mass) with (tex label, ACHILLES scan species, ACHILLES pi_pid).
# res.CHANNELS order is [n->n pi+, n->p pi0, p->p pi+]; we plot p pi+, n pi+, n pi0.
_R = {(2212, 211): RES_CHANNELS[2], (2112, 211): RES_CHANNELS[0], (2112, 111): RES_CHANNELS[1]}
CHANNELS = [
    (r"$\nu_\mu\, p\to\mu^- p\,\pi^+$", _R[(2212, 211)], "H", 211),
    (r"$\nu_\mu\, n\to\mu^- n\,\pi^+$", _R[(2112, 211)], "N", 211),
    (r"$\nu_\mu\, n\to\mu^- p\,\pi^0$", _R[(2112, 111)], "N", 111),
]


def adonis_sigma(E, channel, n, seed):
    """sigma(E) in nb + SE, monochromatic free-nucleon (J_beam=1) -- thin call to the core evaluator."""
    return sigma_free_nucleon(E, channel, n=n, seed=seed)


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
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 80_000
    style.use()
    fig, ax = plt.subplots(2, 3, figsize=(12.5, 5.4), sharex="col",
                           gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0, "wspace": 0.28})
    for c, (tex, channel, sp, ach_pi) in enumerate(CHANNELS):
        aS, aE, hS, hE = [], [], [], []
        for j, E in enumerate(ENERGIES):
            s, e = adonis_sigma(E, channel, n, seed=1000 * c + j)
            aS.append(s); aE.append(e)
            s2, e2 = achilles_sigma(E, sp, ach_pi); hS.append(s2); hE.append(e2)
        aS = np.array(aS) * NB_TO_1E38; aE = np.array(aE) * NB_TO_1E38
        hS = np.array(hS) * NB_TO_1E38; hE = np.array(hE) * NB_TO_1E38
        top, bot = ax[0, c], ax[1, c]
        top.errorbar(ENERGIES, hS, yerr=hE, fmt="s", ms=4.5, color="k", capsize=2,
                     label="ACHILLES", zorder=3)
        top.plot(ENERGIES, aS, "-", color="tab:blue", lw=1.6, label="ADoNIS")
        top.fill_between(ENERGIES, aS - aE, aS + aE, color="tab:blue", alpha=0.22, lw=0)
        top.set_title(tex, fontsize=11); top.set_xlim(0, 4600)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = aS / hS; re = np.abs(r) * np.sqrt((aE / aS) ** 2 + (hE / np.where(hS > 0, hS, np.nan)) ** 2)
        bot.errorbar(ENERGIES, r, yerr=re, fmt="o", ms=3.5, color="navy", capsize=2)
        for y in (0.9, 1.0, 1.1):
            bot.axhline(y, ls="--" if y != 1 else "-", lw=0.8, color="0.6", zorder=0)
        bot.set_ylim(0.6, 1.4); bot.set_xlabel(r"$E_\nu$ [MeV]")
        # chi2/ndf (points with an ACHILLES value)
        m = np.isfinite(hS) & (hS > 0)
        chi2 = float(np.sum((aS[m] - hS[m]) ** 2 / (aE[m] ** 2 + hE[m] ** 2 + 1e-30)))
        print(f"  ch {c} {tex}  chi2/ndf {chi2/max(m.sum(),1):7.2f} (ndf={int(m.sum())})", flush=True)
        for j, E in enumerate(ENERGIES):
            print(f"      E={E:5.0f}  ADO={aS[j]:.4e}  ACH={hS[j]:.4e}  ratio={aS[j]/hS[j]:.4f}"
                  f"  (SE_ado={aE[j]/aS[j]*100:.2f}% SE_ach={hE[j]/hS[j]*100:.2f}%)", flush=True)
        if c == 0:
            top.legend(fontsize=8); top.set_ylabel(r"$\sigma$ [$10^{-38}$ cm$^2$]"); bot.set_ylabel("ratio")
    fig.suptitle(r"ADoNIS vs ACHILLES --- free-nucleon RES single-pion $\sigma(E_\nu)$", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    style.save(fig, "fig02_anl_sigma")


if __name__ == "__main__":
    main()
