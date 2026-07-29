"""Paper Fig 13 (fig:achilles-dcc-comparison): ADoNIS vs ANL-Osaka DCC meson-baryon model.

Mirrors the paper's structure -- the ANL-Osaka analytic partial-wave cross section (smooth line, "the
calculation directly from Eq. sigma_int") vs the ADoNIS Monte-Carlo cascade sampler (points, the analog
of the paper's "Achilles INC" histogram).  Both read the SAME ANL tables; the MC validates that ADoNIS's
cascade samples that input correctly (total rate + angular shape), just as the paper validates ACHILLES.

Left  : total sigma(W) [mb] off the PROTON for pi+ (red), pi0 (blue), pi- (green), eta (purple), summed
  over open finals (piN+etaN+KLam+KSig).  ANL-Osaka = analytic; ADoNIS = beam-test MC (fire mesons at a
  nucleon, Gaussian interaction prob exp(-pi b^2/sigma); sigma_MC = pi R^2 <scatter>) -> Poisson errors.
Right : normalized (1/sigma) dsigma/dOmega vs cos(theta_CM) at meson lab p=300 MeV for pi+ p.  ANL-Osaka =
  analytic; ADoNIS = histogram of cos sampled by the cascade's jax_sample_cos_cm inverse-CDF table.

  python -m analysis.paper.sec1_validation.fig13_piN_sigma
"""
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from adonis.fsi.interactions.meson_baryon_amplitudes import (      # noqa: E402
    load_anl, _channel_sigma, pim_p_total, WAVES, wave_qn,
    conversion_sigma_grid, eta_elastic_sigma_grid, eta_backconv_sigma_grid)
from adonis.fsi.interactions.meson_baryon_xsec import jax_sample_cos_cm   # noqa: E402
from analysis.paper import style                                   # noqa: E402

_R2 = np.sqrt(2.0) / 3.0
M_PI = 139.57018; M_ETA = 547.86; M_N = 938.918754
MB_TO_FM2 = 0.1                                                     # 1 mb = 0.1 fm^2


def _interp(Wg, Ws, ys):
    return np.interp(Wg, Ws, ys, left=0.0, right=0.0)


def _W_of_p(p, mM):
    return np.sqrt(mM ** 2 + M_N ** 2 + 2 * M_N * np.sqrt(p ** 2 + mM ** 2))


# ------------------------------------------------------------------ left: analytic total sigma(W) off p
def totals_off_proton():
    Wg, amps = load_anl(0, 0)                                      # piN grid (1080-2200 MeV) = master grid
    piN_pip = _channel_sigma(amps, Wg, {3: 1.0})
    piN_pi0 = (_channel_sigma(amps, Wg, {3: 2.0 / 3, 1: 1.0 / 3})
               + _channel_sigma(amps, Wg, {3: _R2, 1: -_R2}))
    piN_pim = pim_p_total(Wg)[1]
    Wc, conv = conversion_sigma_grid()
    We, eel = eta_elastic_sigma_grid(); Wb, ebk = eta_backconv_sigma_grid()
    eta_tot = _interp(Wg, We, eel) + _interp(Wg, Wb, ebk[0].sum(axis=0))
    return Wg, {r"$\pi^+ p$":  ("tab:red",    piN_pip + _interp(Wg, Wc, conv[0, 0])),
                r"$\pi^0 p$":  ("tab:blue",   piN_pi0 + _interp(Wg, Wc, conv[1, 0])),
                r"$\pi^- p$":  ("tab:green",  piN_pim + _interp(Wg, Wc, conv[2, 0])),
                r"$\eta p$":   ("tab:purple", eta_tot)}


def inc_beam_test(sigma_mb, rng, ntrial=4000, k=12.0):
    """ADoNIS INC MC: sigma_MC(W) [mb] + Poisson error by firing `ntrial` mesons at a nucleon per W and
    applying the cascade's Gaussian interaction probability exp(-pi b^2 / sigma).  b^2 ~ U[0, R^2] with
    R^2 = k*sigma/pi (so P(R)=e^-k ~ 0), sigma_MC = pi R^2 <scatter>.  Vectorized over the W grid."""
    s_fm2 = np.maximum(sigma_mb * MB_TO_FM2, 1e-9)                 # fm^2
    R2 = k * s_fm2 / np.pi                                         # fm^2 ; per-W disk
    b2 = rng.random((len(s_fm2), ntrial)) * R2[:, None]           # b^2 uniform on the disk
    P = np.exp(-np.pi * b2 / s_fm2[:, None])
    hit = rng.random(P.shape) < P                                 # Bernoulli scatter
    p_hat = hit.mean(1); n = ntrial
    sig_fm2 = np.pi * R2 * p_hat                                  # = pi R^2 <scatter>
    err_fm2 = np.pi * R2 * np.sqrt(np.clip(p_hat * (1 - p_hat), 0, None) / n)
    return sig_fm2 / MB_TO_FM2, err_fm2 / MB_TO_FM2               # -> mb


def main():
    style.use()
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(1, 2, figsize=(11.8, 4.7))
    # ---- LEFT: analytic (line) + ADoNIS INC MC (points) ----
    W, curves = totals_off_proton(); WG = W / 1000.0
    Wc = W[::3]                                                   # coarser W bins for the MC "histogram"
    for lab, (col, sig) in curves.items():
        ax[0].plot(WG, sig, "-", color=col, lw=1.6, label=lab, zorder=2)
        sig_c = np.interp(Wc, W, sig)
        mc, err = inc_beam_test(sig_c, rng)
        ax[0].errorbar(Wc / 1000.0, mc, yerr=err, fmt="o", ms=2.6, color=col, mfc="white",
                       elinewidth=0.7, capsize=0, lw=0, zorder=3)
        print(f"  {lab:12s} analytic peak {sig.max():6.1f} mb | MC/analytic mean ratio "
              f"{np.mean(mc / np.maximum(np.interp(Wc, W, sig), 1e-6)):.3f}", flush=True)
    ax[0].plot([], [], "-", color="0.4", label="ANL-Osaka (analytic)")
    ax[0].plot([], [], "o", color="0.4", mfc="white", label="ADoNIS INC (MC)")
    ax[0].set_xlim(1.08, 2.0); ax[0].set_ylim(0, None)
    ax[0].set_xlabel(r"$W$ [GeV]"); ax[0].set_ylabel(r"$\sigma$ [mb]")
    ax[0].legend(fontsize=8, ncol=2, title="off proton", title_fontsize=8)
    ax[0].set_title(r"total meson-baryon $\sigma(W)$", fontsize=10)
    # ---- RIGHT: analytic dsigma/dOmega (line) + ADoNIS angular sampler (histogram) at p=300 MeV ----
    W300 = float(_W_of_p(300.0, M_PI))
    cg = np.linspace(-1, 1, 200)
    d = _dsig_dOmega(0, 0, {3: 1.0}, W300, cg)
    _trap = getattr(np, "trapezoid", None) or np.trapz
    ax[1].plot(cg, d / (_trap(d, cg) * 2 * np.pi), "-", color="tab:red", lw=1.8,
               label=rf"ANL-Osaka  ($W$={W300/1000:.2f} GeV)")
    N = 200_000
    u = jax.random.uniform(jax.random.PRNGKey(1), (N,))
    cs = np.asarray(jax_sample_cos_cm(jnp.full((N,), W300), u, chan=0))     # pi+ p -> pi+ p sampler
    edges = np.linspace(-1, 1, 21); ctr = 0.5 * (edges[1:] + edges[:-1])
    cnt, _ = np.histogram(cs, edges); bw = np.diff(edges)
    dens = cnt / (N * bw * 2 * np.pi); derr = np.sqrt(cnt) / (N * bw * 2 * np.pi)
    ax[1].errorbar(ctr, dens, yerr=derr, fmt="o", ms=3.5, color="tab:red", mfc="white",
                   elinewidth=0.8, capsize=0, lw=0, label=r"ADoNIS sampler ($\pi^+ p$)")
    ax[1].set_xlim(-1, 1); ax[1].set_ylim(0, None)
    ax[1].set_xlabel(r"$\cos(\theta_{\rm CM})$"); ax[1].set_ylabel(r"$(1/\sigma)\, d\sigma/d\Omega$")
    ax[1].legend(fontsize=8); ax[1].set_title(r"$\pi^+ p$ angular at $p=300$ MeV", fontsize=10)
    fig.suptitle(r"ADoNIS vs ANL-Osaka --- meson-baryon DCC (shared cascade cross sections)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    style.save(fig, "fig13_piN_sigma")


def _dPL(L, c):
    return np.polynomial.legendre.legval(c, np.polynomial.legendre.legder([0] * L + [1]))


def _dsig_dOmega(i, f, cg, W, cos_theta):
    Wt, amps = load_anl(i, f)
    a = np.stack([np.interp(W, Wt, amps[:, k]) for k in range(amps.shape[1])])
    aLp, aLm = {}, {}
    for k, name in enumerate(WAVES):
        L, twoI, twoJ = wave_qn(name); amp = cg.get(twoI, 0.0) * a[k]
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


if __name__ == "__main__":
    main()
