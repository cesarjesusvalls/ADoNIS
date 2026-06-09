"""Decisive amps2 normalization audit: parse ACHILLES free-proton mono RESDUMP (per-event
kinematics + ACHILLES amps2), recompute ADoNIS exclusive_amps2_batch on the IDENTICAL momenta,
and form the per-event ratio amps2_ACH/amps2_ADO.  A FLAT constant ratio => the only disagreement
is the overall _NORM; its mean is the exact correction.  Momenta in the RESDUMP are already q||z."""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import adonis.xsec.dcc_current as dcc
dcc.BATCH_INTERP = "spline"
from adonis.xsec.dcc_current import exclusive_amps2_batch

LOG = Path("/Users/homelab/Lab/Playground/projects/DIFFGEN/Achilles/_resrun_out/reslog_H_mono_gen.txt")
NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 20000

def v3(s): return [float(x) for x in s.split(",")]
num = r"([-+0-9.eE]+)"
pat = re.compile(
    r"liE=" + num + r" li=([-+0-9.eE,]+) loE=" + num + r" lo=([-+0-9.eE,]+) "
    r"hiE=" + num + r" hi=([-+0-9.eE,]+) hiID=(\d+) hNID=(\d+) hNE=" + num + r" hN=([-+0-9.eE,]+) "
    r"hPID=(-?\d+) hPE=" + num + r" hP=([-+0-9.eE,]+) amps2=" + num +
    r" flux=" + num + r" initwgt=" + num + r" spinavg=" + num + r" psw=" + num)

knu, kmu, pst, pN, pPi, a2_ach = [], [], [], [], [], []
n_read = 0
with open(LOG) as fh:
    for line in fh:
        if "RESDUMP" not in line: continue
        m = pat.search(line)
        if not m: continue
        g = m.groups()
        liE, li, loE, lo, hiE, hi, hiID, hNID, hNE, hN, hPID, hPE, hP, amps2, flux, initwgt, spinavg, psw = g
        # physical free-proton CC channel: struck proton -> proton + pi+, selected (initwgt=1, amps2>0)
        if int(hiID) != 2212 or int(hNID) != 2212 or int(hPID) != 211: continue
        if float(initwgt) != 1.0 or float(amps2) <= 0.0: continue
        knu.append([float(liE)] + v3(li)); kmu.append([float(loE)] + v3(lo))
        pst.append([float(hiE)] + v3(hi)); pN.append([float(hNE)] + v3(hN)); pPi.append([float(hPE)] + v3(hP))
        a2_ach.append(float(amps2))
        n_read += 1
        if n_read >= NMAX: break

knu = np.array(knu); kmu = np.array(kmu); pst = np.array(pst); pN = np.array(pN); pPi = np.array(pPi)
a2_ach = np.array(a2_ach)
print(f"parsed {n_read} physical free-proton RESDUMP events", flush=True)

# q||z sanity: q spatial should already be along z
q = knu - kmu
qperp = np.hypot(q[:, 1], q[:, 2])  # x,y components
print(f"  q||z check: max |q_x,q_y|/|q_z| = {np.max(qperp/np.abs(q[:,3])):.2e} (should be ~0)")

a2_ado = np.asarray(exclusive_amps2_batch(knu, kmu, pst, pN, pPi, +1, 211))
good = (a2_ado > 0) & np.isfinite(a2_ado)
r = a2_ach[good] / a2_ado[good]
print(f"  usable {good.sum()} events")
print(f"  ratio amps2_ACH/amps2_ADO:  mean={r.mean():.6f}  median={np.median(r):.6f}  std={r.std():.2e}  "
      f"spread(std/mean)={r.std()/r.mean():.2e}")
print(f"  -> if flat, multiply ADoNIS amps2 by {r.mean():.5f}  (equiv: divide _NORM by {r.mean():.5f})")
print(f"  current _NORM={dcc._NORM:.6e}  ->  corrected _NORM={dcc._NORM/r.mean():.6e}")
print(f"  percentiles r: 1%={np.percentile(r,1):.5f} 25%={np.percentile(r,25):.5f} "
      f"75%={np.percentile(r,75):.5f} 99%={np.percentile(r,99):.5f}")
