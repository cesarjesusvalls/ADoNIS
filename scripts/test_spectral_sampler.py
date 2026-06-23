"""STANDALONE accuracy test of the ADoNIS spectral-function importance sampler (audit 1.3/2.4).

SpectralImportanceSampler draws (|p|, E_rm) ~ |p|^2 S(p,E) via fine-grid trapezoid CDFs (per-p
conditional in E) + linear inverse-CDF.  Tests whether it reproduces the TRUE |p|^2 S (the SF's OWN
Polint table = what ACHILLES targets) -- with binning that ACTUALLY resolves the narrow removal-energy
shell peak (FWHM ~6 MeV) and a 2D/CONDITIONAL check of the p-E correlation (the per-p e_cdf is exactly
what the trapz/linear-inverse could distort).

Reference (truth) = SF's own cubic-p/linear-E Polint:
  P(|p|)      ~ |p|^2 int S dE
  P(E_rm)     ~ int |p|^2 S dp
  P(E_rm|pa<|p|<pb) ~ int_pa^pb |p|^2 S dp   (the CORRELATION)
  <E_rm>(|p|) = int E |p|^2 S dE / int |p|^2 S dE

Chunked sampling (the per-event searchsorted loop OOMs at high N in one shot).
Run: python -u scripts/test_spectral_sampler.py [C|Ar] [N] [chunk]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from adonis.workflow.materials import resolve_targets
from adonis.xsec.spectral import SpectralFunction, SpectralImportanceSampler

MAT = sys.argv[1] if len(sys.argv) > 1 else "C"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4_000_000
CHUNK = int(sys.argv[3]) if len(sys.argv) > 3 else 80_000


def chi2_shape(hs, ht):
    s, t = hs.sum(), ht.sum()
    if s <= 0 or t <= 0:
        return 0.0, 0
    ns, nt = hs / s, ht / t
    es = np.sqrt(np.clip(hs, 1, None)) / s
    use = (hs + ht) > 0
    return float(((ns - nt) ** 2 / (es ** 2 + 1e-30))[use].sum()), int(use.sum())


def main():
    tg = resolve_targets(MAT)[0][0]
    sf = SpectralFunction(tg.spectral_n)
    samp = SpectralImportanceSampler(sf)
    rng = np.random.default_rng(0)
    pmin, pmax = float(sf.mom[0]), float(sf.mom[-1])
    Emin, Emax = float(sf.energy[0]), float(sf.energy[-1])

    # --- binning that RESOLVES the peak (E ~0.5 MeV over [Emin,80]) ---
    E_edges = np.arange(Emin, 80.0 + 0.5, 0.5)                    # 0.5 MeV bins through the peak+near tail
    p_edges = np.linspace(pmin, pmax, 41)                         # 40 |p| bins
    pslices = [(10, 80), (80, 160), (160, 260), (260, 400), (400, 790)]   # for the conditional E|p
    Ec_edges = np.arange(Emin, 80.0 + 1.0, 1.0)                   # 1 MeV bins for the conditional (fewer stats)

    hE = np.zeros(len(E_edges) - 1); hp = np.zeros(len(p_edges) - 1)
    hEp = [np.zeros(len(Ec_edges) - 1) for _ in pslices]         # E hist per |p| slice
    # <E>(p): running sum of E and count per p bin
    sumE = np.zeros(len(p_edges) - 1); cntE = np.zeros(len(p_edges) - 1)
    H2 = np.zeros((len(p_edges) - 1, len(Ec_edges) - 1))         # 2D (p, E) for the ratio map

    done = 0
    while done < N:
        n = min(CHUNK, N - done)
        pv, Erm = samp.sample(n, rng); pm = np.linalg.norm(pv, axis=1)
        hE += np.histogram(Erm, bins=E_edges)[0]
        hp += np.histogram(pm, bins=p_edges)[0]
        H2 += np.histogram2d(pm, Erm, bins=[p_edges, Ec_edges])[0]
        ip = np.clip(np.digitize(pm, p_edges) - 1, 0, len(p_edges) - 2)
        np.add.at(sumE, ip, Erm); np.add.at(cntE, ip, 1.0)
        for k, (pa, pb) in enumerate(pslices):
            sel = (pm >= pa) & (pm < pb)
            hEp[k] += np.histogram(Erm[sel], bins=Ec_edges)[0]
        done += n

    # --- TRUE from the SF Polint (fine grid) ---
    pf = np.linspace(pmin, pmax, 600); Ef = np.linspace(Emin, Emax, 3000)
    PP, EE = np.meshgrid(pf, Ef, indexing="ij")
    S = np.clip(sf.batch(PP.ravel(), EE.ravel()).reshape(len(pf), len(Ef)), 0, None)
    dEf = Ef[1] - Ef[0]; dpf = pf[1] - pf[0]
    w2 = pf[:, None] ** 2 * S                                     # |p|^2 S(p,E)

    def _bin_integral(x, dens, edges):
        # EXACT integral of dens(x) over each [edges[i],edges[i+1]] via cumulative-trapz + edge interp.
        # (Summing fine points with a >=/< mask aliases when dx does not divide the bin width -> ripple.)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(x))])
        return np.diff(np.interp(edges, x, cum))

    def true_in_E_bins(edges, pmask=None):                       # int |p|^2 S dp dE over E bins
        wp = w2 if pmask is None else w2[pmask]
        dens = wp.sum(axis=0) * dpf                              # int |p|^2 S dp  (E density on Ef)
        return _bin_integral(Ef, dens, edges)

    tE = true_in_E_bins(E_edges)
    tEp = [true_in_E_bins(Ec_edges, (pf >= pa) & (pf < pb)) for (pa, pb) in pslices]
    # true |p| marginal into p bins (same exact cumulative integration)
    densp = (pf ** 2 * S.sum(axis=1) * dEf)                       # |p|^2 int S dE on pf
    tp = _bin_integral(pf, densp, p_edges)
    # true <E>(p) on p bins
    Emean_t = (Ef[None, :] * w2).sum(axis=1) / np.clip(w2.sum(axis=1), 1e-30, None)   # <E>(pf)
    pc = 0.5 * (p_edges[:-1] + p_edges[1:])
    tEmean = np.interp(pc, pf, Emean_t)
    sEmean = sumE / np.clip(cntE, 1, None)

    # --- report ---
    c2E, nE = chi2_shape(hE, tE); c2p, npb = chi2_shape(hp, tp)
    print(f"spectral sampler accuracy  material={MAT}  N={done}  (E bins {E_edges[1]-E_edges[0]:.1f} MeV; peak FWHM ~6)")
    print(f"  E_rm marginal (PEAK-RESOLVED): shape chi2/ndf = {c2E:.1f}/{nE} = {c2E/max(nE,1):.2f}")
    Epk = 0.5 * (E_edges[np.argmax(tE)] + E_edges[np.argmax(tE) + 1])
    pkb = (0.5 * (E_edges[:-1] + E_edges[1:]) > Epk - 6) & (0.5 * (E_edges[:-1] + E_edges[1:]) < Epk + 6)
    rE = (hE / hE.sum()) / np.where(tE.sum() > 0, tE / tE.sum(), np.nan)
    print(f"    peak {Epk:.1f} MeV +-6: max|ratio-1| = {np.nanmax(np.abs(rE[pkb]-1))*100:.1f}%  "
          f"sampled/true peak-bin = {(hE/hE.sum())[np.argmax(tE)]/(tE/tE.sum())[np.argmax(tE)]:.3f}")
    print(f"  |p| marginal: shape chi2/ndf = {c2p:.1f}/{npb} = {c2p/max(npb,1):.2f}")
    print(f"  CORRELATION -- E_rm shape within |p| slices, and <E_rm>(|p|):")
    for k, (pa, pb) in enumerate(pslices):
        c2, nn = chi2_shape(hEp[k], tEp[k])
        ipk = np.argmax(tEp[k]); pkc = 0.5 * (Ec_edges[ipk] + Ec_edges[ipk + 1])
        print(f"    |p|=[{pa},{pb}): E|p shape chi2/ndf={c2/max(nn,1):5.2f}  peak@{pkc:5.1f}MeV  "
              f"N={int(hEp[k].sum())}")
    dEmean = sEmean - tEmean
    print(f"  <E_rm>(|p|) sampled-true: max |dMeV| = {np.nanmax(np.abs(dEmean)):.2f} MeV  "
          f"(true range {tEmean.min():.1f}-{tEmean.max():.1f})")

    # --- plots ---
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    Ecen = 0.5 * (E_edges[:-1] + E_edges[1:])
    ax[0, 0].step(Ecen, hE / hE.sum(), where="mid", label="sampled")
    ax[0, 0].plot(Ecen, tE / tE.sum(), "--", label="true |p|^2 S")
    ax[0, 0].set_title(f"E_rm marginal (0.5 MeV bins) -- peak FWHM~6"); ax[0, 0].set_xlim(0, 60)
    ax[0, 0].set_xlabel("E_rm [MeV]"); ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)
    ax[1, 0].plot(Ecen, rE, "-", color="C3"); ax[1, 0].axhline(1, color="k", lw=0.8)
    ax[1, 0].axhline(1.05, color="grey", ls=":"); ax[1, 0].axhline(0.95, color="grey", ls=":")
    ax[1, 0].set_xlim(0, 60); ax[1, 0].set_ylim(0.7, 1.3); ax[1, 0].set_xlabel("E_rm [MeV]")
    ax[1, 0].set_ylabel("sampled/true"); ax[1, 0].grid(alpha=0.3)
    # conditional slices
    Eccen = 0.5 * (Ec_edges[:-1] + Ec_edges[1:])
    for k, (pa, pb) in enumerate(pslices):
        ax[0, 1].plot(Eccen, tEp[k] / max(tEp[k].sum(), 1e-30), "--", color=f"C{k}")
        ax[0, 1].step(Eccen, hEp[k] / max(hEp[k].sum(), 1e-30), where="mid", color=f"C{k}", label=f"|p|[{pa},{pb})")
    ax[0, 1].set_title("E_rm | |p| (solid=sampled, dash=true) -- correlation"); ax[0, 1].set_xlim(0, 60)
    ax[0, 1].set_xlabel("E_rm [MeV]"); ax[0, 1].legend(fontsize=7); ax[0, 1].grid(alpha=0.3)
    ax[1, 1].plot(pc, tEmean, "--", label="true <E>(p)"); ax[1, 1].plot(pc, sEmean, "o-", ms=3, label="sampled")
    ax[1, 1].set_title("<E_rm>(|p|)"); ax[1, 1].set_xlabel("|p| [MeV]"); ax[1, 1].set_ylabel("<E_rm> [MeV]")
    ax[1, 1].legend(); ax[1, 1].grid(alpha=0.3)
    # 2D ratio map
    t2 = np.zeros_like(H2)
    for i in range(len(p_edges) - 1):
        t2[i] = true_in_E_bins(Ec_edges, (pf >= p_edges[i]) & (pf < p_edges[i + 1]))
    Hn = H2 / H2.sum(); Tn = t2 / t2.sum()
    ratio2 = np.where((Hn > 0) & (Tn > 1e-5), Hn / Tn, np.nan)
    im = ax[0, 2].imshow(ratio2.T, origin="lower", aspect="auto", vmin=0.8, vmax=1.2, cmap="RdBu_r",
                         extent=[pmin, pmax, Ec_edges[0], Ec_edges[-1]])
    ax[0, 2].set_title("2D sampled/true  (|p|,E_rm)"); ax[0, 2].set_xlabel("|p| [MeV]"); ax[0, 2].set_ylabel("E_rm [MeV]")
    ax[0, 2].set_ylim(0, 60); plt.colorbar(im, ax=ax[0, 2])
    ax[1, 2].axis("off")
    fig.suptitle(f"Spectral-function sampler vs true |p|^2 S  ({MAT}, N={done})")
    fig.tight_layout(); fig.savefig(f"/tmp/spectral_sampler_test_{MAT}.png", dpi=110); plt.close(fig)
    print(f"\nwrote /tmp/spectral_sampler_test_{MAT}.png")


if __name__ == "__main__":
    main()
