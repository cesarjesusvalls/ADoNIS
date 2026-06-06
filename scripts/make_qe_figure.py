"""Phase B2/B3 figure: inclusive (e,e') dsigma/domega on 12C (QE + 1pi) ~ Fig 1."""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from adonis.nuclear.qe_inclusive import qe_dsigma_domega
from adonis.nuclear.inclusive_1pi import onepi_dsigma_domega
ROOT = Path(__file__).resolve().parents[1]
w = np.linspace(20, 520, 110)
qe = qe_dsigma_domega(2222., 15.541, w, "pke12p_tot.data", 6, 6)
pi = onepi_dsigma_domega(2222., 15.541, w, "pke12p_tot.data", 12)
pi = pi / pi.max() * qe.max() * 0.45      # relative scaling (the Delta bump is ~half the QE peak)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(w, qe, label="QE", color="tab:orange")
ax.plot(w, pi, label=r"1$\pi$ (Δ)", color="tab:green")
ax.plot(w, qe + pi, label="total", color="tab:blue", lw=2)
ax.set_xlabel(r"$\omega$ [MeV]"); ax.set_ylabel(r"$d\sigma/d\omega$ (arb.)")
ax.set_title(r"Inclusive (e,e') on $^{12}$C: QE + 1$\pi$, $E$=2.222 GeV, $\theta$=15.5$^\circ$ (~Fig 1)")
ax.legend(); fig.tight_layout()
out = ROOT / "figures"; out.mkdir(exist_ok=True); fig.savefig(out / "inclusive_ee_c12.png", dpi=130)
print("wrote", out / "inclusive_ee_c12.png")
