"""IN-SITU decisive test: ADoNIS-spline dsigma/dW vs ACHILLES (no-FSI primary), absolute nb.
Generates ADoNIS RES events ONCE, computes per-event amps2 under BOTH bilinear and spline (same
events), bins absolute dsigma/dW by VERTEX W = sqrt((k_nu-k_mu+p_struck)^2) -- the same vertex-W
the ACHILLES oracle uses.  3-way overlay (ACHILLES / ADoNIS-bilinear / ADoNIS-spline) + ADO/ACH
ratio panels.  ACHILLES = no-FSI primary (matches ADoNIS res_xsec primary; no cascade either side).
All RES pions (pi+ and pi0), matching the ADoNIS channels and the oracle (pid in {111,211}).

Prediction under test: spline (=ACHILLES-faithful interp) sits ABOVE ACHILLES at high W -> the
high-W excess is non-amplitude (phase space / 3-body sampling), NOT the interpolation.

Usage: python scripts/insitu_spline_dsigdW.py [N_per_channel=200000] [seed=0]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import adonis.xsec.res_xsec as R
from adonis.xsec import dcc_current as DC
from adonis.xsec.backend import flux_factor
from adonis.data.oracle.normalization import hepmc_norm

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NOFSI_HEPMC = "_oracle_out/T2K_CH_virt_nofsi.hepmc"


def amps2_at(s, idx, itiz, ppid, mode):
    DC.BATCH_INTERP = mode
    a2 = np.zeros(len(s["valid"]))
    if len(idx):
        a2[idx] = DC.exclusive_amps2_batch(s["k_nu"][idx], s["k_mu"][idx], s["p_struck"][idx],
                                           s["p_N"][idx], s["p_pi"][idx], itiz, ppid)
    return a2


def gen_adonis():
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
        tot = s["k_nu"] - s["k_mu"] + s["p_struck"]                  # vertex hadronic 4-mom
        W = np.sqrt(np.clip(tot[:, 0] ** 2 - np.sum(tot[:, 1:] ** 2, axis=1), 0, None))
        keep = (wb > 0) | (ws > 0)
        Wl.append(W[keep]); wb_l.append(wb[keep] / N); ws_l.append(ws[keep] / N)
        print(f"  ({ipid},{ppid}): sig_bilin={wb.mean():.4e} spline={ws.mean():.4e}", flush=True)
    return np.concatenate(Wl), np.concatenate(wb_l), np.concatenate(ws_l)


def main():
    print("deriving no-FSI absolute norm from hepmc header ...", flush=True)
    w2nb = hepmc_norm(NOFSI_HEPMC)["weight_to_nb"]
    ach = np.load("data/oracle/t2k_res_w_achilles_nofsi.npz")
    aW = np.asarray(ach["W"]); aw = np.asarray(ach["w"]) * w2nb
    print(f"  ACH no-FSI weight_to_nb={w2nb:.4e}; sigma_RES(all-pi)={aw.sum():.4e} nb", flush=True)

    W, wb, ws = gen_adonis()
    print(f"\nADoNIS sigma_RES(all-pi): bilinear={wb.sum():.4e}  spline={ws.sum():.4e}  "
          f"ACH={aw.sum():.4e} nb\n  ADO/ACH: bilin={wb.sum()/aw.sum():.3f}  spline={ws.sum()/aw.sum():.3f}")

    edges = np.linspace(1080, 1950, 30); ctr = 0.5 * (edges[1:] + edges[:-1]); bw = np.diff(edges)
    def H(v, w): h, _ = np.histogram(v, bins=edges, weights=w); return h / bw
    ha, hb, hs = H(aW, aw), H(W, wb), H(W, ws)
    print(f"\n{'W[MeV]':>8} {'ADObilin/ACH':>12} {'ADOspline/ACH':>13}  (ACH dsig/dW)")
    for i in range(len(ctr)):
        if ha[i] > 0:
            print(f"{ctr[i]:8.0f} {hb[i]/ha[i]:12.3f} {hs[i]/ha[i]:13.3f}  ({ha[i]:.3e})")

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(8.5, 7.5), height_ratios=[3, 1.3], sharex=True)
    ax.step(edges, np.append(ha, ha[-1]), where="post", color="0.35", lw=1.8, label="ACHILLES (no-FSI)")
    ax.step(edges, np.append(hb, hb[-1]), where="post", color="C1", lw=1.5, label="ADoNIS bilinear")
    ax.step(edges, np.append(hs, hs[-1]), where="post", color="C0", lw=1.5, label="ADoNIS spline (faithful)")
    ax.set_yscale("log"); ax.set_ylabel(r"d$\sigma$/dW [nb/MeV]"); ax.legend()
    ax.set_title("IN-SITU: RES d$\\sigma$/dW absolute — ADoNIS (bilinear & spline) vs ACHILLES no-FSI")
    axr.axhspan(0.95, 1.05, color="green", alpha=0.12); axr.axhline(1.0, ls="--", color="green", lw=0.8)
    m = ha > 0
    axr.plot(ctr[m], (hb / ha)[m], "o-", color="C1", ms=3, label="bilinear/ACH")
    axr.plot(ctr[m], (hs / ha)[m], "s-", color="C0", ms=3, label="spline/ACH")
    axr.set_ylim(0.5, 2.0); axr.set_ylabel("ADoNIS / ACHILLES"); axr.set_xlabel("vertex W [MeV]")
    axr.legend(fontsize=8)
    fig.tight_layout(); out = "paper_figures/insitu_spline_dsigdW.png"; fig.savefig(out, dpi=120)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
