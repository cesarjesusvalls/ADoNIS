"""Paper Fig. 1 (inclusive (e,e')) as ADoNIS-vs-ACHILLES: dsigma/domega on 12C (and 40Ar) at
E=2.222 GeV, theta_e=15.541 deg, with the ADoNIS/ACHILLES ratio panel -- the validation analogue of the
paper's ACHILLES-vs-data inclusive figure.

ACHILLES side: parse the inclusive (e,e') hepmc produced by the run cards
  configs/achilles/run_inclusive_ee_{C,Ar}_{qe,res}.yml  (via analysis.utils.run_achilles ->
  output/achilles/inclusive_ee_{C,Ar}_{qe,res}.hepmc) into the omega = E_beam - E_e' spectrum.
ADoNIS side: the inclusive response model (adonis.nuclear.qe_inclusive / inclusive_1pi).
The ADoNIS model is a SHAPE (arb. units), so the comparison is PEAK-NORMALISED dsigma/domega per component
(QE, 1pi) -- exactly the old scripts/make_inclusive_ee_validation comparison, now ACHILLES-from-hepmc.

Data overlay is OPTIONAL and OFF by default: the JLab file in data/experiment/jlab_ee is a DIFFERENT
kinematic (E=2.020 GeV, theta=15.022) than the paper's 2.222 GeV (Murphy:2019wed) -- fetch the matching
release for an apples-to-apples overlay.

Usage (after running the ACHILLES cards):  python -m analysis.paper.fig1_inclusive_ee.make [C|Ar]
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.utils.hepmc import parse_events
from adonis.nuclear.qe_inclusive import qe_dsigma_domega
from adonis.nuclear.inclusive_1pi import onepi_dsigma_domega

E0, TH = 2222.0, 15.541
ACH_DIR = ROOT / "output" / "achilles"
OUT = ROOT / "output" / "figures"
# (component, ACHILLES hepmc stem, ADoNIS model, omega window, color)
COMPONENTS = {
    "qe":  dict(model=lambda w, sf, n: qe_dsigma_domega(E0, TH, w, sf, n, n),
                w=(50.0, 500.0), bins=30, color="tab:orange", label="QE"),
    "res": dict(model=lambda w, sf, n: onepi_dsigma_domega(E0, TH, w, sf, 2 * n),
                w=(250.0, 900.0), bins=22, color="tab:green", label=r"1$\pi$ (Δ)"),
}
_SF = {"C": ("pke12p_tot.data", 6), "Ar": ("pke40p_tot.data", 18)}     # spectral-fn file, n_p(=n_n approx)


def ach_omega(hepmc, bins, w_lo, w_hi):
    """Peak-normalised ACHILLES dsigma/domega shape + error from the inclusive (e,e') hepmc."""
    oms, ws = [], []
    for ev in parse_events(Path(hepmc)):
        els = [(p4, st) for pid, st, p4 in ev["parts"] if pid == 11]
        sc = [p4 for p4, st in els if st == 1] or [p4 for p4, st in els if p4[0] < E0 - 1]
        if not sc:
            continue
        oms.append(E0 - sc[0][0]); ws.append(ev["w"] or 1.0)
    oms, ws = np.array(oms), np.array(ws)
    h, edges = np.histogram(oms, bins=bins, range=(w_lo, w_hi), weights=ws)
    h2, _ = np.histogram(oms, bins=bins, range=(w_lo, w_hi), weights=ws ** 2)
    cen = 0.5 * (edges[1:] + edges[:-1]); norm = h.max() or 1.0
    return cen, h / norm, np.sqrt(h2) / norm


def main(argv=None):
    mat = (argv or sys.argv[1:] or ["C"])[0]
    sf, n = _SF[mat]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), height_ratios=[3, 1], sharex="col")
    any_ach = False
    for c, (key, spec) in enumerate(COMPONENTS.items()):
        a0, a1 = axes[0, c], axes[1, c]
        cen = np.linspace(*spec["w"], spec["bins"])
        ado = np.asarray(spec["model"](cen, sf, n)); ado = ado / (ado.max() or 1.0)   # peak-norm shape
        a0.plot(cen, ado, "s-", color="C0", ms=3, lw=1.0, label="ADoNIS")
        hp = ACH_DIR / f"inclusive_ee_{mat}_{key}.hepmc"
        if hp.exists():
            any_ach = True
            xc, da, ea = ach_omega(hp, spec["bins"], *spec["w"])
            a0.fill_between(xc, da - ea, da + ea, color="0.6", alpha=0.4, lw=0, label="ACH stat")
            a0.step(xc, da, where="mid", color="0.3", lw=1.3, label="ACHILLES")
            with np.errstate(divide="ignore", invalid="ignore"):
                r = ado / np.interp(cen, xc, da)
            a1.axhspan(0.9, 1.1, color="green", alpha=0.12); a1.axhline(1.0, ls="--", color="green", lw=0.7)
            a1.plot(cen, r, "o", color="C3", ms=3)
        else:
            a1.text(0.5, 0.5, f"run {hp.name}\n(analysis.utils.run_achilles)", transform=a1.transAxes,
                    ha="center", va="center", fontsize=8, color="r")
        a0.set_title(f"{mat}: {spec['label']}", fontsize=10); a0.set_ylabel(r"d$\sigma$/d$\omega$ (peak-norm)")
        a0.set_ylim(bottom=0); a0.legend(fontsize=8)
        a1.set_ylim(0.5, 1.5); a1.set_xlabel(r"$\omega$ [MeV]"); a1.set_ylabel("ADO/ACH")
    fig.suptitle(rf"Inclusive (e,e') on $^{{}}${mat}, E=2.222 GeV, $\theta$=15.541° — ADoNIS vs ACHILLES "
                 f"(paper Fig. 1)", fontsize=12)
    os.makedirs(OUT, exist_ok=True); out = OUT / f"fig1_inclusive_ee_{mat}.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); print("wrote", out, flush=True)
    if not any_ach:
        print("  (ADoNIS-only: no ACHILLES hepmc yet -- run the inclusive cards via analysis.utils.run_achilles)", flush=True)


if __name__ == "__main__":
    main()
