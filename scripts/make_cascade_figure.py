"""Phase G/H figure: ACHILLES cascade pi+ on 12C reaction cross section, both modes.

Virtual Resonances (DCC + Oset) and Propagating Resonances (GiBUU Delta), from the
cascade-enabled image we built (docker/Dockerfile.cascade -> achilles-cascade). The
beam is uniform in p_pi, so the hit-momentum histogram is proportional to the reaction
cross section sigma(p_pi); both modes peak at the Delta(1232).  ~ paper Fig c12_ar40.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
fig, ax = plt.subplots(figsize=(6, 4))
for csv, label, col in [("cascade_pip_c12_reaction.csv", "Virtual Resonances", "tab:blue"),
                        ("cascade_pip_c12_reaction_prop.csv", "Propagating Resonances", "tab:red")]:
    path = ROOT / "data" / "oracle" / csv
    if not path.exists():
        continue
    ref = np.loadtxt(path)
    ax.plot(ref[:, 0], ref[:, 1], "o-", color=col, label=label, ms=4)
ax.axvspan(240, 320, color="gray", alpha=0.15, label="Delta(1232) region")
ax.set_xlabel(r"$p_\pi$ [MeV]")
ax.set_ylabel(r"reaction $\sigma$ (shape, peak-normalised)")
ax.set_title(r"ACHILLES cascade: $\pi^+$ on $^{12}$C reaction $\sigma$ (both modes)")
ax.legend(fontsize=8)
fig.tight_layout()
out = ROOT / "figures"; out.mkdir(exist_ok=True)
fig.savefig(out / "cascade_pip_c12.png", dpi=130)
print(f"wrote {out/'cascade_pip_c12.png'}")
