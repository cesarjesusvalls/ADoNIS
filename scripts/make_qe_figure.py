"""Phase B2/B3 figure: QE inclusive (e,e') dsigma/domega on 12C and 40Ar (~ Fig 1)."""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from adonis.nuclear.qe_inclusive import qe_dsigma_domega
ROOT = Path(__file__).resolve().parents[1]
w = np.linspace(20, 400, 100)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(w, qe_dsigma_domega(2222., 15.541, w, "pke12p_tot.data", 6, 6), label=r"$^{12}$C (QE)", color="tab:blue")
try:
    ax.plot(w, qe_dsigma_domega(2222., 15.541, w, "pke40p_tot.data", 18, 22), label=r"$^{40}$Ar (QE)", color="tab:green")
except Exception:
    pass
ax.set_xlabel(r"$\omega$ [MeV]"); ax.set_ylabel(r"$d\sigma/d\omega$ (arb.)")
ax.set_title(r"PWIA QE inclusive (e,e'): $E$=2.222 GeV, $\theta$=15.5$^\circ$ (~Fig 1)")
ax.legend(); fig.tight_layout()
out = ROOT / "figures"; out.mkdir(exist_ok=True); fig.savefig(out / "qe_inclusive_c12_ar40.png", dpi=130)
print("wrote", out / "qe_inclusive_c12_ar40.png")
