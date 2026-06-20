"""ADoNIS-vs-ACHILLES on the axes the FATE dump provides: (1) capture fraction vs initial |p|,
(2) the normalized initial-|p| spectrum (to check the primaries match).  Carbon proton."""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
np.seterr(all="ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ach_log = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ach_fate_C.log"
ado_npz = sys.argv[2] if len(sys.argv) > 2 else "/tmp/fate_C_p_ms600.npz"
# ACHILLES
pin, st = [], []
rx = re.compile(r"FATE pidin=(-?\d+) pin=([\d.eE+-]+) pidout=(-?\d+) pout=([\d.eE+-]+) status=(\d+)")
for ln in open(ach_log):
    m = rx.search(ln)
    if m and int(m.group(1)) == 2212:
        pin.append(float(m.group(2))); st.append(int(m.group(5)))
pin = np.array(pin); st = np.array(st); cap_a = (st == 26)
# ADoNIS
d = dict(np.load(ado_npz)); w = d["w"]; pi = d["p_init"]; cap_d = d["fate"] == 3

edges = np.array([0, 50, 100, 150, 200, 250, 300, 400, 500, 700, 1000])
ctr = 0.5 * (edges[:-1] + edges[1:])
def capfrac(p, mask, wt=None):
    fa, ea = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (p >= lo) & (p < hi)
        if wt is None:
            n = b.sum(); f = mask[b].mean() if n else np.nan; e = np.sqrt(max(f*(1-f),0)/max(n,1))
        else:
            wb = wt[b]; f = (wb*mask[b]).sum()/wb.sum() if wb.sum() else np.nan
            neff = wb.sum()**2/(wb**2).sum() if wb.sum() else 1; e = np.sqrt(max(f*(1-f),0)/max(neff,1))
        fa.append(f); ea.append(e)
    return np.array(fa), np.array(ea)
fa, ea = capfrac(pin, cap_a); fd, ed = capfrac(pi, cap_d, w)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].errorbar(ctr, fd, yerr=ed, fmt="o-", color="crimson", label="ADoNIS", capsize=2)
ax[0].errorbar(ctr, fa, yerr=ea, fmt="s-", color="navy", label="ACHILLES", capsize=2)
ax[0].axvline(137, ls=":", c="gray", label="$|p|$=137 (KE=10)")
ax[0].set_xlabel("initial proton $|p|$ [MeV]"); ax[0].set_ylabel("capture fraction")
ax[0].set_title("Capture vs initial $|p|$: ADoNIS vs ACHILLES (C)"); ax[0].legend()
# spectrum
sb = np.linspace(0, 1000, 41)
ax[1].hist(pi, bins=sb, weights=w, density=True, histtype="step", lw=2, color="crimson", label="ADoNIS")
ax[1].hist(pin, bins=sb, density=True, histtype="step", lw=2, color="navy", label="ACHILLES")
ax[1].axvline(137, ls=":", c="gray")
ax[1].set_xlabel("initial proton $|p|$ [MeV]"); ax[1].set_ylabel("normalized")
ax[1].set_title("Primary proton spectrum (normalized)"); ax[1].legend()
plt.tight_layout(); out = "paper_figures/capture_adovsach_C.png"; plt.savefig(out, dpi=110)
# integrated capture + low-|p| fractions
def wf(m, p, lo, hi, wt=None):
    b = (p >= lo) & (p < hi)
    return (m[b]).mean() if wt is None else (wt[b]*m[b]).sum()/wt[b].sum()
print(f"integrated capture: ADoNIS={np.average(cap_d,weights=w):.4f}  ACHILLES={cap_a.mean():.4f}")
print(f"frac primaries <137 MeV: ADoNIS={np.average(pi<137,weights=w):.4f}  ACHILLES={(pin<137).mean():.4f}")
print(f"frac primaries <200 MeV: ADoNIS={np.average(pi<200,weights=w):.4f}  ACHILLES={(pin<200).mean():.4f}")
print(f"wrote {out}")
