"""Per-bin Jacobian ARROW grids for additional CC0pi differential cross sections, from the event bank:
    dsigma/dQ^2, dsigma/dW, dsigma/dcos(theta_mu), dsigma/dp_mu, dsigma/dp_lead
All observables are built from the stored per-event kinematics (k_nu, p_struck, k_mu) + leading proton, so
this is pure plot-time (no re-run).  Each page = one observable's 27-knob arrow grid (forward histogram +
per-knob gradient arrows, per-subplot 0.2*y-span scaling), same style as grad_arrows.

Signal = CC0pi (primary pion absorbed, prim_pi_pid==0); NO detector acceptance applied (these are the
physical differential cross sections, not the STV-acceptance distributions).

  python analysis/t2k/differentiability/bank_arrows_obs.py [bankdir]
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import analysis.t2k.differentiability.bank_plot as BP
from analysis.t2k.differentiability import grad_arrows as GA

CONV = 1e-33 / 12.0 * 1e38            # nb/12C -> 1e-38 cm^2/nucleon (per raw-unit after /binwidth)


def _Q2(B, lead):
    q = B["k_nu"].astype(np.float64) - B["k_mu"].astype(np.float64)
    return np.sum(q[:, 1:] ** 2, 1) - q[:, 0] ** 2                      # MeV^2
def _W(B, lead):
    q = B["k_nu"].astype(np.float64) - B["k_mu"].astype(np.float64)
    had = B["p_struck"].astype(np.float64) + q
    return np.sqrt(np.clip(had[:, 0] ** 2 - np.sum(had[:, 1:] ** 2, 1), 0, None))   # MeV
def _cosmu(B, lead):
    p = B["k_mu"].astype(np.float64)[:, 1:]; return p[:, 2] / np.linalg.norm(p, axis=1)
def _pmu(B, lead):  return np.linalg.norm(B["k_mu"].astype(np.float64)[:, 1:], axis=1)     # MeV
def _plead(B, lead):  return np.linalg.norm(lead[:, 1:], axis=1)                            # MeV

# name -> (fn, edges (raw units), xsc, xlabel, needs_proton)
OBSDEF = {
    "Q2":     (_Q2,    np.linspace(0.0, 1.5e6, 16), 1e6,  r"$Q^2$ [GeV$^2$]", False),
    "W":      (_W,     np.linspace(900.0, 1700.0, 17), 1e3, r"$W$ [GeV]", False),
    "cosmu":  (_cosmu, np.linspace(-0.2, 1.0, 13), 1.0,   r"$\cos\theta_\mu$", False),
    "pmu":    (_pmu,   np.linspace(0.0, 1600.0, 17), 1e3, r"$p_\mu$ [GeV/c]", False),
    "plead":  (_plead, np.linspace(0.0, 1200.0, 13), 1e3, r"$p_{\mathrm{lead}}$ [GeV/c]", True),
}


def main():
    bankdir = sys.argv[1] if len(sys.argv) > 1 else "output/event_bank"
    B = BP.load_bank(bankdir)
    labels = B["labels"]; nk = B["D1"].shape[1]
    lead, has_p = BP.leading_proton(B)
    base = (B["prim_pi_pid"] == 0)                                      # CC0pi (no detector acceptance)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    os.makedirs("output/figures", exist_ok=True)
    out = "output/figures/cc0pi_arrows_observables.pdf"
    with PdfPages(out) as pdf:
        for name, (fn, edges, xsc, xlabel, needp) in OBSDEF.items():
            vals = fn(B, lead); mask = base & has_p if needp else base
            nb = len(edges) - 1; idx = np.clip(np.searchsorted(edges, vals) - 1, 0, nb - 1)
            bw = np.diff(edges)
            h0 = np.bincount(idx[mask], weights=B["w0"][mask].astype(np.float64), minlength=nb) / bw * CONV
            Js = np.zeros((nk, nb))
            for k in range(nk):
                Js[k] = np.bincount(idx[mask], weights=B["D1"][mask, k].astype(np.float64), minlength=nb) / bw * CONV
            fig = GA._build_grid_fig(name, edges, h0, Js, labels, xlabel=xlabel, xsc=xsc)
            pdf.savefig(fig); plt.close(fig)
            print(f"  page: {name}  ({int(mask.sum())} CC0pi events)", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
