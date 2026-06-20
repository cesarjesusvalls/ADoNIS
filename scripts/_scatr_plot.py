"""Both-generators NN-scatter radial distribution: ADoNIS (/tmp/ado_nscatr_C.npz) vs ACHILLES
(NSCATR lines in the fate log).  Left: all NN scatters; right: DEGRADING scatters (an outgoing
nucleon below 137 MeV = KE<10, the capture-feeding ones)."""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
np.seterr(all="ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ach_log = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ach_fate_C.log"
d = dict(np.load("/tmp/ado_nscatr_C.npz"))
R_d, PL_d, PR_d, W_d, radius = d["r"], d["pl"], d["pr"], d["w"], float(d["radius"])
pmin_d = np.minimum(PL_d, PR_d)                                  # softer outgoing nucleon

# ACHILLES NSCATR r + outgoing momenta
rx = re.compile(r"NSCATR r=([\d.eE+-]+)((?: p=[\d.eE+-]+)+)")
R_a, pmin_a = [], []
for ln in open(ach_log):
    m = rx.search(ln)
    if m:
        r = float(m.group(1)); ps = [float(x) for x in re.findall(r"p=([\d.eE+-]+)", m.group(2))]
        R_a.append(r); pmin_a.append(min(ps) if ps else np.nan)
R_a = np.array(R_a); pmin_a = np.array(pmin_a)
print(f"ADoNIS NN scatters={len(R_d)}  ACHILLES NN scatters={len(R_a)}")

edges = np.linspace(0, radius * 1.05, 31)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
# all scatters
ax[0].hist(R_d, bins=edges, weights=W_d, density=True, histtype="step", lw=2, color="crimson", label="ADoNIS")
ax[0].hist(R_a, bins=edges, density=True, histtype="step", lw=2, color="navy", label="ACHILLES")
ax[0].axvline(3.14, ls=":", c="green", label="$k_F(r)=137$ (r=3.14 fm)"); ax[0].axvline(radius, ls="--", c="k", lw=1)
ax[0].set_xlabel("NN-scatter radius [fm]"); ax[0].set_ylabel("normalized"); ax[0].legend()
ax[0].set_title("All NN-scatter vertices (C)")
# degrading scatters (outgoing < 137 MeV)
deg_d = pmin_d < 137; deg_a = pmin_a < 137
ax[1].hist(R_d[deg_d], bins=edges, weights=W_d[deg_d], density=True, histtype="step", lw=2, color="crimson",
           label=f"ADoNIS ({np.average(deg_d,weights=W_d)*100:.1f}% of scat)")
ax[1].hist(R_a[deg_a], bins=edges, density=True, histtype="step", lw=2, color="navy",
           label=f"ACHILLES ({deg_a.mean()*100:.1f}% of scat)")
ax[1].axvline(3.14, ls=":", c="green"); ax[1].axvline(radius, ls="--", c="k", lw=1)
ax[1].set_xlabel("NN-scatter radius [fm]"); ax[1].set_ylabel("normalized"); ax[1].legend()
ax[1].set_title("DEGRADING scatters (outgoing $|p|$<137 MeV)")
plt.tight_layout(); out = "paper_figures/nscatr_adovsach_C.png"; plt.savefig(out, dpi=110)
print(f"ADoNIS degrading-scatter frac={np.average(deg_d,weights=W_d):.4f}  ACHILLES={deg_a.mean():.4f}")
print(f"ADoNIS scatters beyond r=3.14: {np.average(R_d>3.14,weights=W_d):.3f}  ACHILLES: {(R_a>3.14).mean():.3f}")
print(f"wrote {out}")
