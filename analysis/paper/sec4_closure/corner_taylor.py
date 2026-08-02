"""S4D corner from the model derivatives (analytic, no bank grid) -- and its cross-check vs the exact grid.

Reads <label>_derivs.npz (J and d2m at the BFP).  Around the MLE minimum the residual is zero, so the
2nd-order-MODEL chi2 is
    chi2(D) = sum_b W_b ( (Jb . D)_b + 1/2 (D . Bb . D)_b )^2 ,
a >=0 polynomial in the dial shifts D.  For a pair (a,b) the NON-Gaussian marginal is this chi2 PROFILED
over the other 14 dials (a cheap Gauss-Newton on the polynomial); the GAUSSIAN is the D_pair^T V_pair^-1
D_pair ellipse.  `main()` builds every pair's profiled Delta-chi2 grid and caches it; `validate()` overlays
a few pairs on the exact numerical grid (<label>_corner.npz) to confirm the truncation is faithful to ~2sigma.

Usage:  python -m analysis.paper.sec4_closure.corner_taylor            # cache all pairs
        python -m analysis.paper.sec4_closure.corner_taylor validate  # cross-check vs the exact grid
"""
import sys
import itertools
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style

LABEL = "sec4_closure_r16_noprior"


def _resid(Jb, Bb, D, Cb=None):
    r = Jb @ D + 0.5 * np.einsum("bjk,j,k->b", Bb, D, D)          # model to 2nd order (- m0), residual
    if Cb is not None:
        r = r + (1.0 / 6.0) * np.einsum("bjkl,j,k,l->b", Cb, D, D, D)   # + 3rd-order model term
    return r


def profile_pair(Jb, Bb, W, a, b, ga, gb, nit=10, Cb=None):
    """Profiled Taylor-model Delta-chi2 on the (a,b) grid; others minimised by GN on the polynomial.
    Cb (the 3rd model derivative) is optional -> chi2 exact to 4th order instead of 3rd."""
    nsub = Jb.shape[1]; others = [i for i in range(nsub) if i not in (a, b)]
    out = np.full((len(ga), len(gb)), np.nan)
    for ia, da in enumerate(ga):
        for ib, db in enumerate(gb):
            D = np.zeros(nsub); D[a] = da; D[b] = db
            for _ in range(nit):
                r = _resid(Jb, Bb, D, Cb)
                Jr = Jb + np.einsum("bjk,k->bj", Bb, D)                  # d resid / d D
                if Cb is not None:
                    Jr = Jr + 0.5 * np.einsum("bjkl,k,l->bj", Cb, D, D)
                Jr = Jr[:, others]
                Ao = Jr.T @ (Jr * W[:, None]); go = Jr.T @ (W * r)
                D[others] += np.linalg.solve(Ao + 1e-9 * np.eye(len(others)), -go)
            out[ia, ib] = float(np.sum(W * _resid(Jb, Bb, D, Cb)**2))
    return out


def _load():
    z = np.load(style.ALTGEN / f"{LABEL}_derivs.npz", allow_pickle=True)
    Cb = np.asarray(z["Cb"]) if "Cb" in z.files else None
    return (np.asarray(z["Jb"]), np.asarray(z["Bb"]), np.asarray(z["W"]),
            np.asarray(z["V"]), [int(k) for k in z["subset"]], [str(x) for x in z["pnames"]], Cb)


def main():
    Jb, Bb, W, V, sub, pn, Cb = _load()
    nsub = len(sub); spost = np.sqrt(np.abs(np.diag(V)))
    ax = np.linspace(-3.5, 3.5, 19)
    pairs = list(itertools.combinations(range(nsub), 2))
    dchi2 = np.full((len(pairs), 19, 19), np.nan)
    for pi, (a, b) in enumerate(pairs):
        dchi2[pi] = profile_pair(Jb, Bb, W, a, b, ax * spost[a], ax * spost[b])
        if (pi + 1) % 20 == 0:
            print(f"  {pi+1}/{len(pairs)} pairs", flush=True)
    out = style.ALTGEN / f"{LABEL}_corner_taylor.npz"
    np.savez(out, subset=sub, pnames=pn, V=V, axis_sigma=ax, pairs=np.array(pairs), dchi2=dchi2)
    print(f"[out] {out}")


def validate():
    """Overlay the Taylor contours (3rd-order, and 4th-order if d3m is available) on the exact grid."""
    import matplotlib.pyplot as plt
    Jb, Bb, W, V, sub, pn, Cb = _load()
    spost = np.sqrt(np.abs(np.diag(V)))
    g = np.load(style.ALTGEN / f"{LABEL}_corner.npz", allow_pickle=True)      # exact grid
    gpairs = [tuple(p) for p in g["pairs"]]; gd = np.asarray(g["dchi2"]); gax = np.asarray(g["axis_sigma"])
    show = [(0, 1), (1, 3), (0, 13), (6, 9)]                                   # a few (indices into subset)
    style.use()
    fig, axes = plt.subplots(1, len(show), figsize=(4.2 * len(show), 4.2))
    lev = [2.3, 6.17]                                                          # 2-D 68% / 95%
    for ax_, (a, b) in zip(axes, show):
        exact = gd[gpairs.index((a, b))]
        t3 = profile_pair(Jb, Bb, W, a, b, gax * spost[a], gax * spost[b])                # 3rd-order chi2
        X, Y = np.meshgrid(gax, gax, indexing="ij")
        ax_.contour(X, Y, exact, levels=lev, colors="k", linewidths=1.4)
        ax_.contour(X, Y, t3, levels=lev, colors=style.FIT_EC, linewidths=1.2, linestyles="--")
        if Cb is not None:
            t4 = profile_pair(Jb, Bb, W, a, b, gax * spost[a], gax * spost[b], Cb=Cb)     # 4th-order chi2
            ax_.contour(X, Y, t4, levels=lev, colors="#2e7d32", linewidths=1.2, linestyles=":")
        Vp = V[np.ix_([a, b], [a, b])]; Pi = np.linalg.inv(Vp)
        GZ = (Pi[0, 0] * (X * spost[a])**2 + 2 * Pi[0, 1] * (X * spost[a]) * (Y * spost[b])
              + Pi[1, 1] * (Y * spost[b])**2)
        ax_.contour(X, Y, GZ, levels=lev, colors="0.55", linewidths=0.9, linestyles=(0, (1, 1)))
        ax_.set_title(f"{style.plab(pn[sub[a]])} $\\times$ {style.plab(pn[sub[b]])}", fontsize=9)
        ax_.set_xlabel(r"$\Delta_a/\sigma$"); ax_.set_ylabel(r"$\Delta_b/\sigma$")
    axes[0].plot([], [], "k", lw=1.4, label="exact")
    axes[0].plot([], [], color=style.FIT_EC, lw=1.2, ls="--", label="Taylor 3rd order")
    if Cb is not None:
        axes[0].plot([], [], color="#2e7d32", lw=1.2, ls=":", label="Taylor 4th order")
    axes[0].plot([], [], color="0.55", lw=0.9, ls=":", label="Gaussian")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Autodiff Taylor corner vs exact — 3rd vs 4th order (68/95%)", fontsize=11)
    fig.tight_layout()
    style.save(fig, f"{LABEL}_corner_validate")


if __name__ == "__main__":
    (validate if "validate" in sys.argv[1:] else main)()
