"""Phase B2/B3 figure: inclusive (e,e') dsigma/domega, QE + 1pi, on 12C and 40Ar (~Fig 1).

Two panels: the quasi-elastic peak (PWIA fold of the single-nucleon elastic response over
the target spectral function S(p,E)) plus the Delta/1pi bump (EM W_T/W_L folded over S),
for 12C (pke12p) and 40Ar (pke40p). The heavier, more loosely-bound Ar broadens and shifts
the QE peak relative to C -- the target-dependence the paper's Fig 1 shows.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from adonis.nuclear.qe_inclusive import qe_dsigma_domega
from adonis.nuclear.inclusive_1pi import onepi_dsigma_domega
ROOT = Path(__file__).resolve().parents[1]

E, TH = 2222.0, 15.541
w = np.linspace(20, 900, 130)      # extend past the Delta so the 1pi resonance tail is shown
TARGETS = [
    dict(name=r"$^{12}$C", sf="pke12p_tot.data", n_p=6, n_n=6, n_nuc=12),
    dict(name=r"$^{40}$Ar", sf="pke40p_tot.data", n_p=18, n_n=22, n_nuc=40),
]

# optional ACHILLES (e,e') oracle overlays (12C panel): QE_Spectral_Func + RES_Spectral_Func
def _load(name):
    f = ROOT / "data" / "oracle" / name
    if f.exists():
        d = np.loadtxt(f); return d[:, 0], d[:, 1]
    return None
oracle_qe = _load("inclusive_ee_12C_qe.csv")
oracle_res = _load("inclusive_ee_12C_res.csv")

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
for ax, tg in zip(axes, TARGETS):
    qe = qe_dsigma_domega(E, TH, w, tg["sf"], tg["n_p"], tg["n_n"])
    pi = onepi_dsigma_domega(E, TH, w, tg["sf"], tg["n_nuc"])
    pi_scale = qe.max() * 0.45
    pi = pi / pi.max() * pi_scale               # relative scaling (Delta bump ~half QE peak)
    ax.plot(w, qe, label="QE", color="tab:orange")
    ax.plot(w, pi, label=r"1$\pi$ (Δ)", color="tab:green")
    ax.plot(w, qe + pi, label="total", color="tab:blue", lw=2)
    if tg["sf"].startswith("pke12"):
        if oracle_qe is not None:
            ow, osh = oracle_qe
            ax.plot(ow, osh * qe.max(), "o", ms=4, color="saddlebrown", label="ACHILLES QE")
        if oracle_res is not None:
            ow, osh = oracle_res
            ax.plot(ow, osh * pi.max(), "s", ms=4, color="darkgreen", label="ACHILLES RES")
    # QE-peak FWHM annotation
    half = qe.max() / 2
    above = np.where(qe >= half)[0]
    fwhm = w[above[-1]] - w[above[0]] if above.size else float("nan")
    ax.set_title(f"{tg['name']}  (QE peak {w[qe.argmax()]:.0f} MeV, FWHM {fwhm:.0f})")
    ax.set_xlabel(r"$\omega$ [MeV]"); ax.legend()
axes[0].set_ylabel(r"$d\sigma/d\omega$ (arb.)")
fig.suptitle(r"Inclusive (e,e'): QE + 1$\pi$, $E$=2.222 GeV, $\theta$=15.5$^\circ$ (~Fig 1)")
fig.tight_layout()
out = ROOT / "figures"; out.mkdir(exist_ok=True)
fig.savefig(out / "inclusive_ee_c12_ar40.png", dpi=130)
# keep the single-target 12C figure too (back-compat)
figc, axc = plt.subplots(figsize=(6, 4))
qe = qe_dsigma_domega(E, TH, w, "pke12p_tot.data", 6, 6)
pi = onepi_dsigma_domega(E, TH, w, "pke12p_tot.data", 12)
pi = pi / pi.max() * qe.max() * 0.45
axc.plot(w, qe, label="QE", color="tab:orange")
axc.plot(w, pi, label=r"1$\pi$ (Δ)", color="tab:green")
axc.plot(w, qe + pi, label="total", color="tab:blue", lw=2)
axc.set_xlabel(r"$\omega$ [MeV]"); axc.set_ylabel(r"$d\sigma/d\omega$ (arb.)")
axc.set_title(r"Inclusive (e,e') on $^{12}$C: QE + 1$\pi$ (~Fig 1)")
axc.legend(); figc.tight_layout(); figc.savefig(out / "inclusive_ee_c12.png", dpi=130)
print("wrote", out / "inclusive_ee_c12_ar40.png", "and", out / "inclusive_ee_c12.png")
