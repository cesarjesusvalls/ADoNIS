"""Showcase plots from the 100k event bank (all plot-time, no re-run):
 (1) forward CC0pi dsigma/dx vs T2K data for dpt & dat  (the 'draw events like T2K plots' capability)
 (2) per-knob 2D Jacobian over (dpt,dat) from the bank  (reuses grad_2d.make_figure)
The 1D gradient arrow grids are produced by bank_validate.py (*_BANK.png).
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import uproot
import analysis.t2k.differentiability.bank_plot as BP
from analysis.t2k.differentiability import grad_arrows as GA
from analysis.t2k.differentiability import grad_2d as G2


def main():
    bankdir = sys.argv[1] if len(sys.argv) > 1 else "output/event_bank"
    sys.argv = [sys.argv[0], "dpt"]
    from analysis.t2k.differentiability import tune as T
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    B = BP.load_bank(bankdir)
    mask, lead = BP.signal_cc0pi(B)

    # (1) forward vs T2K data --------------------------------------------------------------------------- #
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for a, obs, ofn in ((ax[0], "dpt", BP.dpt), (ax[1], "dat", BP.dat)):
        edges, conv, _ = GA._obs_binning(T, obs)
        h = BP.hist_forward(ofn(B, lead), B, mask, edges, conv)
        r = uproot.open(f"../nuisance/data/T2K/CC0pi/STV/{'dpt' if obs=='dpt' else 'dat'}Results.root")
        data = np.asarray(r["Result"].values()) * 1e38; derr = np.asarray(r["Result"].errors()) * 1e38
        xsc = 1000.0 if obs == "dpt" else 1.0
        ctr = 0.5 * (edges[1:] + edges[:-1]) / xsc; xed = edges / xsc
        a.step(xed, np.append(h, h[-1]), where="post", color="C0", lw=2, label="ADoNIS bank (CC0$\\pi$, nominal)")
        a.errorbar(ctr, data, yerr=derr, fmt="o", color="k", capsize=3, label="T2K")
        a.set(xlabel=(r"$\delta p_T$ [GeV/c]" if obs == "dpt" else r"$\delta\alpha_T$ [rad]"),
              ylabel="d$\\sigma$/dx", title=f"forward CC0$\\pi$ {obs} (from 100k bank)")
        a.legend(fontsize=9); a.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig("output/figures/cc0pi_forward_BANK.png", dpi=120)
    print("wrote output/figures/cc0pi_forward_BANK.png", flush=True)

    # (2) 2D Jacobian over (dpt,dat) from the bank ----------------------------------------------------- #
    edpt, edat = G2._binning("dpt"), G2._binning("dat")
    n1, n2 = len(edpt) - 1, len(edat) - 1
    vdpt = BP.dpt(B, lead); vdat = BP.dat(B, lead)
    i = np.clip(np.searchsorted(edpt, vdpt) - 1, 0, n1 - 1); j = np.clip(np.searchsorted(edat, vdat) - 1, 0, n2 - 1)
    flat = (i * n2 + j)[mask]
    area = np.outer(np.diff(edpt), np.diff(edat)); CONV2D = 1e-33 / 12.0 * 1e38 * 1000.0
    labels = B["labels"]; nk = B["D1"].shape[1]
    h0 = np.bincount(flat, weights=B["w0"][mask].astype(np.float64), minlength=n1 * n2).reshape(n1, n2)
    J2d = np.zeros((nk, n1, n2))
    for k in range(nk):
        J2d[k] = np.bincount(flat, weights=B["D1"][mask, k].astype(np.float64), minlength=n1 * n2).reshape(n1, n2)
    norm2d = h0 / area * CONV2D; J2d = J2d / area[None] * CONV2D
    np.savez("/tmp/adonis_tune_runs/jac2d_bank.npz", edpt=edpt, edat=edat, norm2d=norm2d, J2d=J2d,
             labels=np.array(labels))
    G2.make_figure("/tmp/adonis_tune_runs/jac2d_bank.npz")
    for ext in ("png", "pdf"):
        os.replace(f"output/figures/cc0pi_jacobian_2d.{ext}", f"output/figures/cc0pi_jacobian_2d_BANK.{ext}")
    print("wrote output/figures/cc0pi_jacobian_2d_BANK.{png,pdf}", flush=True)


if __name__ == "__main__":
    main()
