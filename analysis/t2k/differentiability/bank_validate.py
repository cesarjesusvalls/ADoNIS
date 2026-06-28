"""Validate the event bank reproduces the established results (within MC stats, since the bank is lower-N):
  (a) forward CC0pi dsigma/dx (dpt & dat) from the bank vs the 1M grad_all cache h0 -> integral & per-bin ratio
  (b) per-bin gradient d(dsigma/dx)/dtheta from the bank vs the 1M cache J -> ratio for the dominant knobs
  (c) sanity of the stored diagonal 2nd/3rd derivatives (finite, right order of magnitude)
Also re-renders the 1D arrow grids FROM THE BANK (grad_arrows._build_grid_fig) so they can be eyeballed
against grad_all's.  Pure plot-time: no cascade, no autodiff.

  python analysis/t2k/differentiability/bank_validate.py [bankdir]
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import analysis.t2k.differentiability.bank_plot as BP
from analysis.t2k.differentiability import grad_arrows as GA


def main():
    bankdir = sys.argv[1] if len(sys.argv) > 1 else "output/event_bank"
    sys.argv = [sys.argv[0], "dpt"]
    from analysis.t2k.differentiability import tune as T
    B = BP.load_bank(bankdir)
    n = len(B["w0"])
    print(f"bank: {n} events ({(B['channel']==0).sum()} QE + {(B['channel']==1).sum()} RES)", flush=True)
    mask, lead = BP.signal_cc0pi(B)                          # model_hist_full CC0pi definition
    print(f"CC0pi-selected events: {int(mask.sum())}", flush=True)
    OBS = {"dpt": BP.dpt, "dat": BP.dat}
    for obs, ofn in OBS.items():
        edges, conv, _ = GA._obs_binning(T, obs)
        vals = ofn(B, lead)
        h0 = BP.hist_forward(vals, B, mask, edges, conv)
        Jb = BP.hist_gradient(vals, B, mask, edges, conv).T          # (nk, nb)
        ref = np.load(f"output/jac_cache/jac_grid_{obs}.npz", allow_pickle=True)
        h0r, Jr, labels = ref["h0"], ref["J"], list(ref["labels"])
        ig_b, ig_r = float(h0.sum()), float(h0r.sum())
        print(f"\n=== {obs} ===", flush=True)
        print(f"forward integral: bank={ig_b:.4e}  1M={ig_r:.4e}  ratio={ig_b/ig_r:.4f}", flush=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            rb = np.where(h0r != 0, h0 / h0r, np.nan)
        print("  per-bin forward ratio bank/1M: " + " ".join(f"{x:.3f}" for x in rb), flush=True)
        # gradient: compare the dominant knobs (by 1M |J| sum)
        order = np.argsort(-np.abs(Jr).sum(1))[:6]
        print("  dominant-knob gradient ratio (bank sumJ / 1M sumJ):", flush=True)
        for k in order:
            sb, sr = float(Jb[k].sum()), float(Jr[k].sum())
            print(f"    {labels[k]:>16s}  bank={sb:+.3e} 1M={sr:+.3e} ratio={sb/sr:.3f}", flush=True)
        # diagonal 2nd/3rd sanity
        d2 = BP.hist_diag_curv(vals, B, mask, edges, 2, conv).T
        d3 = BP.hist_diag_curv(vals, B, mask, edges, 3, conv).T
        print(f"  diag 2nd-deriv max|.|={np.abs(d2).max():.2e}  3rd max|.|={np.abs(d3).max():.2e} "
              f"(finite: {np.isfinite(d2).all() and np.isfinite(d3).all()})", flush=True)
        # render the bank's arrow grid for this observable (distinct filename; don't clobber the 1M figs)
        fig = GA._build_grid_fig(obs, edges, h0, Jb, labels)
        F = f"output/figures/cc0pi_jacobian_arrows_grid_{obs}_BANK.png"
        fig.savefig(F, dpi=120); print(f"  wrote {F} (FROM BANK)", flush=True)


if __name__ == "__main__":
    main()
