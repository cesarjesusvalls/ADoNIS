"""High-W ANGULAR audit: per-event amps2_ACH/amps2_ADO ratio from a free-proton mono RESDUMP,
binned in (W, cos theta*_pi) -- tests whether the matrix elements agree DIFFERENTIALLY in the
second-resonance region (the Delta-region audit was flat; W>1400 was never gated).  A flat
ratio in cos theta* per W slice exonerates the amplitude assembly; structure implicates it.
Also compares the ACHILLES phase-space weight (psw) shape vs cos theta* per W slice.

Usage: python scripts/resdump_angular_audit.py <reslog> [nmax]
"""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import adonis.xsec.dcc_current as dcc
dcc.BATCH_INTERP = "spline"
from adonis.xsec.dcc_current import exclusive_amps2_batch

LOG = Path(sys.argv[1])
NMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 400000

def v3(s): return [float(x) for x in s.split(",")]
num = r"([-+0-9.eE]+)"
pat = re.compile(
    r"liE=" + num + r" li=([-+0-9.eE,]+) loE=" + num + r" lo=([-+0-9.eE,]+) "
    r"hiE=" + num + r" hi=([-+0-9.eE,]+) hiID=(\d+) hNID=(\d+) hNE=" + num + r" hN=([-+0-9.eE,]+) "
    r"hPID=(-?\d+) hPE=" + num + r" hP=([-+0-9.eE,]+) amps2=" + num +
    r" flux=" + num + r" initwgt=" + num + r" spinavg=" + num + r" psw=" + num)

knu, kmu, pst, pN, pPi, a2_ach, psw = [], [], [], [], [], [], []
with open(LOG) as fh:
    for line in fh:
        if "RESDUMP" not in line: continue
        m = pat.search(line)
        if not m: continue
        g = m.groups()
        liE, li, loE, lo, hiE, hi, hiID, hNID, hNE, hN, hPID, hPE, hP, amps2, flux, initwgt, spinavg, ps = g
        if int(hiID) != 2212 or int(hNID) != 2212 or int(hPID) != 211: continue
        if float(initwgt) != 1.0 or float(amps2) <= 0.0: continue
        knu.append([float(liE)] + v3(li)); kmu.append([float(loE)] + v3(lo))
        pst.append([float(hiE)] + v3(hi)); pN.append([float(hNE)] + v3(hN)); pPi.append([float(hPE)] + v3(hP))
        a2_ach.append(float(amps2)); psw.append(float(ps))
        if len(a2_ach) >= NMAX: break
knu, kmu, pst, pN, pPi = (np.array(x) for x in (knu, kmu, pst, pN, pPi))
a2_ach = np.array(a2_ach); psw = np.array(psw)
print(f"parsed {len(a2_ach)} physical p->p pi+ RESDUMP events", flush=True)

q = knu - kmu
P = q + pst
W = np.sqrt(np.clip(P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, axis=1), 0, None))
# cos theta*: boost pion to the hadronic CM; q is already || z in the dump frame
beta = P[:, 1:] / P[:, [0]]
b2 = np.sum(beta ** 2, axis=1); g = 1 / np.sqrt(1 - b2)
bp = np.sum(beta * pPi[:, 1:], axis=1)
pi_cm3 = pPi[:, 1:] + ((g - 1) * bp / np.clip(b2, 1e-30, None) + g * pPi[:, 0])[:, None] * (-beta) * (-1)
# (standard boost to CM: p' = p + [ (g-1) (b.p)/b2 - g E ] b )
pi_cm3 = pPi[:, 1:] + (((g - 1) * bp / np.clip(b2, 1e-30, None)) - g * pPi[:, 0])[:, None] * beta
cth = pi_cm3[:, 2] / np.clip(np.linalg.norm(pi_cm3, axis=1), 1e-12, None)

a2_ado = np.asarray(exclusive_amps2_batch(knu, kmu, pst, pN, pPi, +1, 211))
good = (a2_ado > 0) & np.isfinite(a2_ado)
r = a2_ach / np.clip(a2_ado, 1e-300, None)
print(f"usable {int(good.sum())};  global ratio mean {r[good].mean():.4f}  spread {r[good].std()/r[good].mean():.2e}")
WB = [(1100, 1300), (1300, 1450), (1450, 1600), (1600, 1750), (1750, 1950)]
CB = np.linspace(-1, 1, 7)
for lo_, hi_ in WB:
    m = good & (W >= lo_) & (W < hi_)
    if m.sum() < 50:
        print(f"W [{lo_},{hi_}): <50 events"); continue
    row = []
    for i in range(len(CB) - 1):
        mm = m & (cth >= CB[i]) & (cth < CB[i + 1])
        row.append(f"{np.median(r[mm]):.3f}({mm.sum()})" if mm.sum() >= 10 else "--")
    print(f"W [{lo_},{hi_}): amps2 ACH/ADO median per cth* bin:", " ".join(row), flush=True)
