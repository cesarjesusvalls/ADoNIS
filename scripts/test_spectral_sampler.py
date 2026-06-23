"""STANDALONE accuracy test of the ADoNIS spectral-function importance sampler (audit 1.3/2.4).

SpectralImportanceSampler draws (|p|, E_rm) ~ |p|^2 S(p,E) via fine-grid trapezoid CDFs + linear
inverse-CDF.  The concern: does that reproduce the TRUE |p|^2 S(p,E) (what ACHILLES targets -- ACHILLES
draws flat + weights by the per-point Polint S, unbiased to |p|^2 S), or does the trapz/linear-inverse
bias the |p| / E_rm marginals (esp. the sharp removal-energy shell peak)?

Reference (ground truth) = the SF's OWN Polint table (the SAME table ACHILLES reads), integrated:
  P(|p|)  proportional to  |p|^2 * int S(p,E) dE
  P(E_rm) proportional to        int |p|^2 S(p,E) dp
Compared to the sampler's drawn marginals.  Pure ADoNIS-internal (no ACHILLES run needed -- the table
is the reference and ACHILLES reproduces it by construction).

Run: python -u scripts/test_spectral_sampler.py [C|Ar] [N]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from adonis.workflow.materials import resolve_targets
from adonis.xsec.spectral import SpectralFunction, SpectralImportanceSampler

MAT = sys.argv[1] if len(sys.argv) > 1 else "C"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4_000_000


def chi2_shape(hs, ht):
    """shape chi2/ndf of sampled hist hs vs true ht (both densities), Poisson-ish errors on the sample."""
    s = hs.sum(); t = ht.sum()
    if s <= 0 or t <= 0:
        return 0.0, 0
    ns = hs / s; nt = ht / t
    es = np.sqrt(np.clip(hs, 1, None)) / s
    use = (hs + ht) > 0
    d = (ns - nt) ** 2 / (es ** 2 + 1e-30)
    return float(d[use].sum()), int(use.sum())


def main():
    tg = resolve_targets(MAT)[0][0]
    sf = SpectralFunction(tg.spectral_n)
    samp = SpectralImportanceSampler(sf)
    rng = np.random.default_rng(0)
    # draw
    pvec, Erm = samp.sample(N, rng)
    pmag = np.linalg.norm(pvec, axis=1)
    # TRUE marginals from the SF's own Polint (fine grids)
    pf = np.linspace(sf.mom[0], sf.mom[-1], 400)
    Ef = np.linspace(sf.energy[0], sf.energy[-1], 400)
    PP, EE = np.meshgrid(pf, Ef, indexing="ij")
    S = np.clip(sf.batch(PP.ravel(), EE.ravel()).reshape(len(pf), len(Ef)), 0, None)
    dE = Ef[1] - Ef[0]; dp = pf[1] - pf[0]
    true_p = pf ** 2 * S.sum(axis=1) * dE            # P(|p|) ~ |p|^2 int S dE
    true_E = (pf[:, None] ** 2 * S).sum(axis=0) * dp  # P(E) ~ int |p|^2 S dp
    # sampled marginals on the SAME bin edges
    p_edges = np.linspace(sf.mom[0], sf.mom[-1], 81)
    E_edges = np.linspace(sf.energy[0], sf.energy[-1], 81)
    hp, _ = np.histogram(pmag, bins=p_edges)
    hE, _ = np.histogram(Erm, bins=E_edges)
    # true on the same (coarser) bin edges: integrate the fine true density into bins
    pc = 0.5 * (p_edges[:-1] + p_edges[1:]); Ec = 0.5 * (E_edges[:-1] + E_edges[1:])
    tp = np.interp(pc, pf, true_p); tE = np.interp(Ec, Ef, true_E)
    c2p, np_ = chi2_shape(hp, tp); c2E, nE = chi2_shape(hE, tE)
    rp = (hp / hp.sum()) / np.where(tp.sum() > 0, tp / tp.sum(), np.nan)
    rE = (hE / hE.sum()) / np.where(tE.sum() > 0, tE / tE.sum(), np.nan)
    print(f"spectral sampler accuracy  material={MAT}  N={N}")
    print(f"  |p| marginal : shape chi2/ndf = {c2p:.1f}/{np_} = {c2p/max(np_,1):.2f}   "
          f"max|ratio-1| = {np.nanmax(np.abs(rp-1))*100:.1f}%")
    print(f"  E_rm marginal: shape chi2/ndf = {c2E:.1f}/{nE} = {c2E/max(nE,1):.2f}   "
          f"max|ratio-1| = {np.nanmax(np.abs(rE-1))*100:.1f}%")
    # the sharp shell peak region (low E_rm) is the audit concern -- report it
    peakreg = Ec < (sf.energy[0] + 0.35 * (sf.energy[-1] - sf.energy[0]))
    if peakreg.any():
        print(f"  E_rm peak region max|ratio-1| = {np.nanmax(np.abs(rE[peakreg]-1))*100:.1f}%")
    # plot
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for col, (c, edges, h, tc, td, r, lab) in enumerate([
            (pc, p_edges, hp, pc, tp, rp, "|p| [MeV]"),
            (Ec, E_edges, hE, Ec, tE, rE, "E_rm [MeV]")]):
        a = ax[0, col]
        a.step(c, h / h.sum(), where="mid", label="sampled")
        a.plot(tc, td / td.sum(), "--", label="true |p|^2 S (Polint)")
        a.set_title(f"{lab} marginal"); a.set_xlabel(lab); a.legend(); a.grid(alpha=0.3)
        b = ax[1, col]; b.plot(c, r, "-", color="C3"); b.axhline(1, color="k", lw=0.8)
        b.axhline(1.01, color="grey", ls=":"); b.axhline(0.99, color="grey", ls=":")
        b.set_ylabel("sampled / true"); b.set_xlabel(lab); b.set_ylim(0.9, 1.1); b.grid(alpha=0.3)
    fig.suptitle(f"Spectral-function importance sampler vs true |p|^2 S  ({MAT})")
    fig.tight_layout(); fig.savefig(f"/tmp/spectral_sampler_test_{MAT}.png", dpi=120); plt.close(fig)
    print(f"\nwrote /tmp/spectral_sampler_test_{MAT}.png")


if __name__ == "__main__":
    main()
