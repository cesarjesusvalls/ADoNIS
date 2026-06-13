"""DIRECT bilinear-vs-spline probe of dsigma/dW for the RES amplitude.
Mirrors res_xsec.generate_importance's inner loop, but computes the per-event amps2 TWICE on
the IDENTICAL sampled points -- once with BATCH_INTERP='bilinear', once 'spline'.  Everything
else (sampling, flux, spectral fn, phase space, channel weights) is bit-identical, so the dsigma/dW
ratio spline/bilinear ISOLATES the interpolation, directly as a function of W.

W = hadronic invariant mass sqrt((p_N+p_pi)^2) -- exactly the axis the DCC table is indexed on
(= wcm inside the amplitude).  ACHILLES uses the SPLINE, so spline is the faithful reference.

Usage: python scripts/interp_dsigdW.py [N_per_channel=200000] [seed=0]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import adonis.xsec.res_xsec as R
from adonis.xsec import dcc_current as DC
from adonis.xsec.backend import flux_factor

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0


def amps2_at(s, idx, itiz, ppid, mode):
    DC.BATCH_INTERP = mode
    a2 = np.zeros(len(s["valid"]))
    if len(idx):
        a2[idx] = DC.exclusive_amps2_batch(s["k_nu"][idx], s["k_mu"][idx], s["p_struck"][idx],
                                           s["p_N"][idx], s["p_pi"][idx], itiz, ppid)
    return a2


def main():
    rng = np.random.default_rng(SEED)
    flux = R.T2KFlux(); minE = flux.seed_min_GeV(); maxE = flux.max_energy
    Wl, wb_l, ws_l = [], [], []
    for (ipid, itiz, mNf, ppid, mpi, mstr) in R.CHANNELS:
        s = R._sample_channel(N, rng, flux, minE, maxE, R._pi_kin_mass(mpi), mNf)
        v = s["valid"]
        idx = np.where(v & (s["energy"] > 2.5) & (s["energy"] < 400) & (s["J"] > 0))[0]
        a2_b = amps2_at(s, idx, itiz, ppid, "bilinear")
        a2_s = amps2_at(s, idx, itiz, ppid, "spline")
        fl = np.asarray(flux_factor(s["k_nu"], s["p_struck"], had_mass=mstr))
        if ipid == 2212:
            sn = R._SF_N.batch(s["mom"], s["energy"]); sp = R._SF_P.batch(s["mom"], s["energy"])
            rw = np.where(sn > 0, sp / np.clip(sn, 1e-300, None), 0.0)
        else:
            rw = 1.0
        base = np.where(v, fl * R.N_NUC * R.SPIN_AVG * s["J"] * rw, 0.0)
        wb = np.where(np.isfinite(base) & (a2_b > 0), a2_b * base, 0.0)
        ws = np.where(np.isfinite(base) & (a2_s > 0), a2_s * base, 0.0)
        had = s["p_N"] + s["p_pi"]
        W = np.sqrt(np.clip(had[:, 0] ** 2 - np.sum(had[:, 1:] ** 2, axis=1), 0, None))
        keep = (wb > 0) | (ws > 0)
        Wl.append(W[keep]); wb_l.append(wb[keep] / N); ws_l.append(ws[keep] / N)
        print(f"  channel ({ipid},{ppid}): sigma_bilin={wb.mean():.4e} spline={ws.mean():.4e} "
              f"ratio s/b={ws.mean()/max(wb.mean(),1e-30):.4f}", flush=True)
    W = np.concatenate(Wl); wb = np.concatenate(wb_l); ws = np.concatenate(ws_l)
    print(f"\nTOTAL sigma_RES: bilinear={wb.sum():.4e}  spline={ws.sum():.4e}  "
          f"spline/bilinear={ws.sum()/wb.sum():.4f} nb")

    edges = np.linspace(1080, 1950, 30); ctr = 0.5 * (edges[1:] + edges[:-1]); bw = np.diff(edges)
    hb, _ = np.histogram(W, bins=edges, weights=wb); hs, _ = np.histogram(W, bins=edges, weights=ws)
    eb2, _ = np.histogram(W, bins=edges, weights=wb ** 2); es2, _ = np.histogram(W, bins=edges, weights=ws ** 2)
    hb, hs = hb / bw, hs / bw
    print(f"\n{'W[MeV]':>8} {'spline/bilin':>12}  (bilin dsig/dW   spline)")
    for i in range(len(ctr)):
        if hb[i] > 0:
            print(f"{ctr[i]:8.0f} {hs[i]/hb[i]:12.3f}  ({hb[i]:.3e} {hs[i]:.3e})")

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(8, 7), height_ratios=[3, 1], sharex=True)
    ax.step(edges, np.append(hb, hb[-1]), where="post", color="C1", lw=1.6, label="bilinear (current)")
    ax.step(edges, np.append(hs, hs[-1]), where="post", color="C0", lw=1.6, label="spline (faithful=ACHILLES)")
    ax.set_yscale("log"); ax.set_ylabel(r"d$\sigma$/dW [nb/MeV]"); ax.legend()
    ax.set_title("RES d$\\sigma$/dW: bilinear vs spline interpolation (identical events)")
    with np.errstate(divide="ignore", invalid="ignore"):
        r = hs / hb
    axr.axhspan(0.99, 1.01, color="green", alpha=0.15); axr.axhline(1.0, ls="--", color="green", lw=0.8)
    axr.plot(ctr[hb > 0], r[hb > 0], "o-", color="C3", ms=3)
    axr.set_ylim(0.8, 1.2); axr.set_ylabel("spline / bilinear"); axr.set_xlabel("hadronic W [MeV]")
    fig.tight_layout(); out = "paper_figures/interp_dsigdW.png"; fig.savefig(out, dpi=120)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
