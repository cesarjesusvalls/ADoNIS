"""Paper Fig 13 (fig:achilles-dcc-comparison): validation of the ANL-Osaka DCC meson-baryon model that
the Virtual-Resonances cascade scatters through, evaluated by ADoNIS from the same ANL tables ACHILLES
reads (data/MesonBaryonAmplitudes/ANL/*.dat + CalcCrossSectionW_grid).  Shared cascade INPUT, so the
ADoNIS analytic curve == the ACHILLES ANL-Osaka dashed line by construction.

Left  (mirrors FIG.13 left):  angle-integrated total sigma(W) [mb] vs W [GeV] off the PROTON for an
  incoming pi+ (red), pi0 (blue), pi- (green), eta (purple); each the sum over all open meson-baryon
  final states (piN + etaN + KLambda + KSigma).
Right (mirrors FIG.13 right): normalized angular distribution (1/sigma) dsigma/dOmega vs cos(theta_CM)
  for pi+ p (red) and eta p (purple) at meson lab momentum p = 300 MeV (pi0/pi- are identical to pi+).

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
from adonis.fsi.interactions.meson_baryon_amplitudes import (      # noqa: E402
    load_anl, _channel_sigma, pim_p_total, WAVES, wave_qn,
    conversion_sigma_grid, eta_elastic_sigma_grid, eta_backconv_sigma_grid)
from analysis.paper import style                                   # noqa: E402

_R2 = np.sqrt(2.0) / 3.0
M_PI = 139.57018; M_ETA = 547.86; M_N = 938.918754


def _interp(Wg, Wsrc, ysrc):
    return np.interp(Wg, Wsrc, ysrc, left=0.0, right=0.0)


def _W_of_p(p, mM):
    E = np.sqrt(p ** 2 + mM ** 2)
    return np.sqrt(mM ** 2 + M_N ** 2 + 2 * M_N * E)


# ------------------------------------------------------------------ left: total sigma(W) off the proton
def totals_off_proton():
    Wg, amps = load_anl(0, 0)                                      # piN grid (1080-2200 MeV) = master grid
    piN_pip = _channel_sigma(amps, Wg, {3: 1.0})                          # pi+ p -> pi+ p (I=3/2)
    piN_pi0 = (_channel_sigma(amps, Wg, {3: 2.0 / 3, 1: 1.0 / 3})         # pi0 p -> pi0 p
               + _channel_sigma(amps, Wg, {3: _R2, 1: -_R2}))            # pi0 p -> pi+ n (cex)
    piN_pim = pim_p_total(Wg)[1]                                          # pi- p -> pi- p + pi0 n
    Wc, conv = conversion_sigma_grid()                            # piN -> {etaN,KLam,KSig}, (3 pi,2 nuc,nW)
    conv_pip = _interp(Wg, Wc, conv[0, 0]); conv_pi0 = _interp(Wg, Wc, conv[1, 0])
    conv_pim = _interp(Wg, Wc, conv[2, 0])
    We, eel = eta_elastic_sigma_grid(); Wb, ebk = eta_backconv_sigma_grid()
    eta_tot = _interp(Wg, We, eel) + _interp(Wg, Wb, ebk[0].sum(axis=0))  # etaN->etaN + etaN->piN (proton)
    return Wg, {r"$\pi^+ p$": ("tab:red",    piN_pip + conv_pip),
                r"$\pi^0 p$": ("tab:blue",   piN_pi0 + conv_pi0),
                r"$\pi^- p$": ("tab:green",  piN_pim + conv_pim),
                r"$\eta p$":  ("tab:purple", eta_tot)}


# ------------------------------------------------------ right: normalized dsigma/dOmega (shape) at fixed W
def _dPL(L, c):
    return np.polynomial.legendre.legval(c, np.polynomial.legendre.legder([0] * L + [1]))


def dsig_dOmega(i, f, cg, W, cos_theta):
    """Unnormalized dsigma/dOmega(theta) shape for meson-baryon channel (i->f) at W, isospin weights cg.
    Standard spin-non-flip f + spin-flip g partial-wave sums (App. MB_amplitudes); channel-agnostic."""
    Wt, amps = load_anl(i, f)
    a = np.stack([np.interp(W, Wt, amps[:, k]) for k in range(amps.shape[1])])   # (20,) at this W
    aLp, aLm = {}, {}
    for k, name in enumerate(WAVES):
        L, twoI, twoJ = wave_qn(name)
        amp = cg.get(twoI, 0.0) * a[k]
        if twoJ == 2 * L + 1:
            aLp[L] = aLp.get(L, 0j) + amp
        elif twoJ == 2 * L - 1:
            aLm[L] = aLm.get(L, 0j) + amp
    c = np.asarray(cos_theta, float); s = np.sqrt(np.clip(1 - c ** 2, 0, 1)); Lmax = 5
    PL = [np.polynomial.legendre.legval(c, [0] * L + [1]) for L in range(Lmax + 1)]
    PL1 = [-s * _dPL(L, c) for L in range(Lmax + 1)]
    fa = np.zeros_like(c, complex); ga = np.zeros_like(c, complex)
    for L in range(Lmax + 1):
        fa += ((L + 1) * aLp.get(L, 0j) + L * aLm.get(L, 0j)) * PL[L]
        ga += (aLp.get(L, 0j) - aLm.get(L, 0j)) * PL1[L]
    return np.abs(fa) ** 2 + np.abs(ga) ** 2


def main():
    style.use()
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    # ---- left ----
    W, curves = totals_off_proton(); WG = W / 1000.0
    for lab, (col, sig) in curves.items():
        ax[0].plot(WG, sig, "-", color=col, lw=1.8, label=lab)
        print(f"  {lab:12s} peak {sig.max():6.1f} mb at W = {W[np.argmax(sig)]:.0f} MeV", flush=True)
    ax[0].set_xlim(1.08, 2.0); ax[0].set_ylim(0, None)
    ax[0].set_xlabel(r"$W$ [GeV]"); ax[0].set_ylabel(r"$\sigma$ [mb]")
    ax[0].legend(fontsize=9, title="off proton", title_fontsize=8)
    ax[0].set_title(r"total meson-baryon $\sigma(W)$", fontsize=10)
    # ---- right: angular at p = 300 MeV ----
    cth = np.linspace(-1, 1, 200)
    for lab, col, (i, f, cg, mM) in [(r"$\pi^+ p$", "tab:red", (0, 0, {3: 1.0}, M_PI)),
                                     (r"$\eta p$", "tab:purple", (1, 1, {1: 1.0}, M_ETA))]:
        W300 = _W_of_p(300.0, mM)
        d = dsig_dOmega(i, f, cg, W300, cth)
        _trap = getattr(np, "trapezoid", None) or np.trapz
        norm = _trap(d, cth) * 2 * np.pi                          # (1/sigma) dsigma/dOmega
        ax[1].plot(cth, d / norm, "-", color=col, lw=1.8, label=rf"{lab}  ($W$={W300/1000:.2f} GeV)")
    ax[1].set_xlim(-1, 1); ax[1].set_ylim(0, None)
    ax[1].set_xlabel(r"$\cos(\theta_{\rm CM})$"); ax[1].set_ylabel(r"$(1/\sigma)\, d\sigma/d\Omega$")
    ax[1].legend(fontsize=9); ax[1].set_title(r"angular distribution at $p=300$ MeV", fontsize=10)
    fig.suptitle(r"ADoNIS meson-baryon DCC (ANL-Osaka $=$ ACHILLES cascade input)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    style.save(fig, "fig13_piN_sigma")


if __name__ == "__main__":
    main()
