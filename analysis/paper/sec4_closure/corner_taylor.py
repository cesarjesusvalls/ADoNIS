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


def _resid(Jb, Bb, D):
    return Jb @ D + 0.5 * np.einsum("bjk,j,k->b", Bb, D, D)


def profile_pair(Jb, Bb, W, a, b, ga, gb, nit=10):
    """Profiled 2nd-order-model Delta-chi2 on the (a,b) grid; others minimised by GN on the polynomial."""
    nsub = Jb.shape[1]; others = [i for i in range(nsub) if i not in (a, b)]
    out = np.full((len(ga), len(gb)), np.nan)
    for ia, da in enumerate(ga):
        for ib, db in enumerate(gb):
            D = np.zeros(nsub); D[a] = da; D[b] = db
            for _ in range(nit):
                r = _resid(Jb, Bb, D)
                Jr = (Jb + np.einsum("bjk,k->bj", Bb, D))[:, others]     # d resid / d D_others
                Ao = Jr.T @ (Jr * W[:, None]); go = Jr.T @ (W * r)
                D[others] += np.linalg.solve(Ao + 1e-9 * np.eye(len(others)), -go)
            out[ia, ib] = float(np.sum(W * _resid(Jb, Bb, D)**2))
    return out


def _load():
    z = np.load(style.ALTGEN / f"{LABEL}_derivs.npz", allow_pickle=True)
    return (np.asarray(z["Jb"]), np.asarray(z["Bb"]), np.asarray(z["W"]),
            np.asarray(z["V"]), [int(k) for k in z["subset"]], [str(x) for x in z["pnames"]])


def main():
    Jb, Bb, W, V, sub, pn = _load()
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
    """Overlay the Taylor (analytic) contour on the exact numerical grid for a few representative pairs."""
    import matplotlib.pyplot as plt
    Jb, Bb, W, V, sub, pn = _load()
    spost = np.sqrt(np.abs(np.diag(V)))
    g = np.load(style.ALTGEN / f"{LABEL}_corner.npz", allow_pickle=True)      # exact grid
    gpairs = [tuple(p) for p in g["pairs"]]; gd = np.asarray(g["dchi2"]); gax = np.asarray(g["axis_sigma"])
    show = [(0, 1), (1, 3), (0, 13), (6, 9)]                                   # a few (indices into subset)
    style.use()
    fig, axes = plt.subplots(1, len(show), figsize=(4.2 * len(show), 4.0))
    for ax_, (a, b) in zip(axes, show):
        exact = gd[gpairs.index((a, b))]
        tay = profile_pair(Jb, Bb, W, a, b, gax * spost[a], gax * spost[b])
        X, Y = np.meshgrid(gax, gax, indexing="ij")
        lev = [1.0, 2.3, 6.17]                                                 # 1D-1sig, 2D-68, 2D-95
        ax_.contour(X, Y, exact, levels=lev, colors="k", linewidths=1.3)
        ax_.contour(X, Y, tay, levels=lev, colors=style.FIT_EC, linewidths=1.3, linestyles="--")
        # Gaussian ellipse (marginal V submatrix) in sigma_post units -> unit-ish
        Vp = V[np.ix_([a, b], [a, b])]; Pi = np.linalg.inv(Vp)
        GZ = (Pi[0, 0] * (X * spost[a])**2 + 2 * Pi[0, 1] * (X * spost[a]) * (Y * spost[b])
              + Pi[1, 1] * (Y * spost[b])**2)
        ax_.contour(X, Y, GZ, levels=[2.3], colors="#1f4b9c", linewidths=1.0, linestyles=":")
        ax_.set_title(f"{style.plab(pn[sub[a]])} x {style.plab(pn[sub[b]])}", fontsize=9)
        ax_.set_xlabel(r"$\Delta_a/\sigma$"); ax_.set_ylabel(r"$\Delta_b/\sigma$")
    fig.suptitle("exact grid (black) vs 2nd-order-model Taylor (orange --) vs Gaussian (blue :)", fontsize=11)
    fig.tight_layout()
    style.save(fig, f"{LABEL}_corner_validate")


if __name__ == "__main__":
    (validate if "validate" in sys.argv[1:] else main)()
