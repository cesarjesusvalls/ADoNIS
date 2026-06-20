"""MATCHED single-pass: FIRST-scatter radius of a PRIMARY proton (where it's consumed), ADoNIS vs
ACHILLES.  ACHILLES: FATE 'posr' for status==29 (scattered) protons.  ADoNIS: /tmp/ado_firstscat_C.npz
first_r for scattered primaries.  Both = the primary's first-interaction radius, no daughters."""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
np.seterr(all="ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ach_log = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ach_fate_C2.log"
# ACHILLES: posr for scattered (status 29) protons
rx = re.compile(r"FATE pidin=(-?\d+) pin=[\d.eE+-]+ pidout=-?\d+ pout=[\d.eE+-]+ status=(\d+) posr=([\d.eE+-]+)")
posr_a = []
for ln in open(ach_log):
    mm = rx.search(ln)
    if mm and int(mm.group(1)) == 2212 and int(mm.group(2)) == 29:
        posr_a.append(float(mm.group(3)))
posr_a = np.array(posr_a)
# ADoNIS
d = dict(np.load("/tmp/ado_firstscat_C.npz")); fr = d["first_r"]; w = d["w"]; radius = float(d["radius"])
scat = fr >= 0; fr_s = fr[scat]; w_s = w[scat]
print(f"ADoNIS scattered N(eff)~{(w_s.sum()**2/(w_s**2).sum()):.0f}  ACHILLES scattered N={len(posr_a)}")
fig, ax = plt.subplots(figsize=(7.5, 5))
edges = np.linspace(0, radius * 1.05, 31)
ax.hist(fr_s, bins=edges, weights=w_s, density=True, histtype="step", lw=2.2, color="crimson", label="ADoNIS")
ax.hist(posr_a, bins=edges, density=True, histtype="step", lw=2.2, color="navy", label="ACHILLES")
ax.axvline(3.14, ls=":", c="green", lw=1.8, label="$k_F(r)=137$ MeV (r=3.14 fm)")
ax.axvline(radius, ls="--", c="k", lw=1, label=f"nuclear radius {radius:.2f} fm")
ax.set_xlabel("radius of the primary's FIRST scatter [fm]"); ax.set_ylabel("normalized")
ax.set_title("Single-pass primary: first-scatter radius (C proton) — matched, no daughters")
ax.legend()
plt.tight_layout(); out = "paper_figures/firstscat_adovsach_C.png"; plt.savefig(out, dpi=120)
print(f"ADoNIS <first-scat r>={np.average(fr_s,weights=w_s):.2f}  ACHILLES <r>={posr_a.mean():.2f}")
print(f"ADoNIS frac first-scat in core (<3.14): {np.average(fr_s<3.14,weights=w_s):.3f}  ACHILLES: {(posr_a<3.14).mean():.3f}")
print(f"wrote {out}")
