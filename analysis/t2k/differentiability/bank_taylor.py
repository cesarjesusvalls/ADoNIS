"""Taylor-validity visualization from the event bank: for each bin, the KNOB RANGE over which the linear
and quadratic Taylor expansions of that bin's correction are good to 5%.

Per bin the correction vs knob shift Delta = theta - theta0 is C1=B1*D, C2=C1+B2*D^2/2, C3=C2+B3*D^3/6,
with B1,B2,B3 = bin-summed 1st/2nd/3rd derivatives (bank D1/D2/D3 for the chosen knob).  An order is
"good" while its deviation from the NEXT-order refinement is <=5% of that refined correction:
  linear good:    |C1-C2|/|C2| <= 0.05
  quadratic good: |C2-C3|/|C3| <= 0.05
Scanning Delta outward in each direction gives an ASYMMETRIC valid range per bin.  Drawn as color bands
around each bin (y = knob value): wide band = quadratic-good, narrow band = linear-good.

  python analysis/t2k/differentiability/bank_taylor.py [knob=M_A] [bankdir]
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import analysis.t2k.differentiability.bank_plot as BP
from analysis.t2k.differentiability import grad_arrows as GA

TOL = 0.05
DMAX = 0.8           # scan |theta - theta0| out to this (M_A nominal 1 -> 0.2 .. 1.8)
NDENSE = 2001


def _good_range(B1, B2, B3, order):
    """(lo, hi) signed Delta range where order is good to TOL; +/- scanned independently (asymmetric)."""
    out = []
    for sign in (-1.0, +1.0):
        d = sign * np.linspace(0.0, DMAX, NDENSE)
        C1 = B1 * d; C2 = C1 + 0.5 * B2 * d ** 2; C3 = C2 + B3 * d ** 3 / 6.0
        if order == 1:
            err = np.abs(C1 - C2) / (np.abs(C2) + 1e-300)
        else:
            err = np.abs(C2 - C3) / (np.abs(C3) + 1e-300)
        bad = np.where(err[1:] > TOL)[0]               # skip d=0 (err=0/0->0 there)
        out.append(d[bad[0] + 1] if bad.size else sign * DMAX)
    return out[0], out[1]                               # (lo<=0, hi>=0)


def main():
    knob = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "M_A"
    bankdir = sys.argv[2] if len(sys.argv) > 2 else "output/event_bank"
    sys.argv = [sys.argv[0], "dpt"]
    from analysis.t2k.differentiability import tune as T
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    B = BP.load_bank(bankdir)
    ki = list(B["labels"]).index(knob)
    nom = 1.0 if knob not in ("Eb_shift", "f_NN_cex") else (0.0 if knob == "Eb_shift" else 0.5)
    mask, lead = BP.signal_cc0pi(B)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, obs, ofn in ((axes[0], "dpt", BP.dpt), (axes[1], "dat", BP.dat)):
        edges, _conv, _ = GA._obs_binning(T, obs)
        xsc = 1000.0 if obs == "dpt" else 1.0
        vals = ofn(B, lead); nb = len(edges) - 1
        idx = np.clip(np.searchsorted(edges, vals) - 1, 0, nb - 1)
        B1 = np.bincount(idx[mask], weights=B["D1"][mask, ki], minlength=nb)
        B2 = np.bincount(idx[mask], weights=B["D2"][mask, ki], minlength=nb)
        B3 = np.bincount(idx[mask], weights=B["D3"][mask, ki], minlength=nb)
        xed = edges / xsc
        for b in range(nb):
            x0, x1 = xed[b], xed[b + 1]
            lo2, hi2 = _good_range(B1[b], B2[b], B3[b], order=2)
            lo1, hi1 = _good_range(B1[b], B2[b], B3[b], order=1)
            ax.fill_between([x0, x1], nom + lo2, nom + hi2, color="#5dade2", alpha=0.85,
                            label="quadratic good (5%)" if b == 0 else None, lw=0)
            ax.fill_between([x0, x1], nom + lo1, nom + hi1, color="#27ae60", alpha=0.95,
                            label="linear good (5%)" if b == 0 else None, lw=0)
        ax.axhline(nom, color="k", lw=1, ls="--", label=f"nominal {knob}={nom:g}")
        ax.set(xlabel=(r"$\delta p_T$ [GeV/c]" if obs == "dpt" else r"$\delta\alpha_T$ [rad]"),
               ylabel=knob, title=f"{obs}: Taylor validity range in {knob}",
               ylim=(nom - DMAX * 1.05, nom + DMAX * 1.05))
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Per-bin Taylor-expansion validity in {knob} (5% deviation-vs-next-order; from 100k bank)",
                 fontsize=12)
    os.makedirs("output/figures", exist_ok=True)
    F = f"output/figures/cc0pi_taylor_validity_{knob}.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(F, dpi=130); print(f"wrote {F}", flush=True)


if __name__ == "__main__":
    main()
