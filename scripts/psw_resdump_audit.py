"""Phase-space weight audit: for each ACHILLES free-proton RESDUMP event, reconstruct s23 from the
dumped momenta and compute ADoNIS's 3-body phase-space Jacobian J_3body(s23) with the SAME formula
free_proton_gen uses, then compare to ACHILLES's dumped psw.  J_3body depends only on s23 (the
angles factor out), so this is a clean per-event comparison.  A flat ratio != 1 => the phase-space
normalization (or a mass convention inside it) is the ~0.85% culprit."""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
from adonis.xsec.res_xsec import _sqlam, M_MU, M_PIP, M_P

LOG = Path("/Users/homelab/Lab/Playground/projects/DIFFGEN/Achilles/_resrun_out/reslog_H_mono_gen.txt")
NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
num = r"([-+0-9.eE]+)"
pat = re.compile(
    r"liE=" + num + r" li=([-+0-9.eE,]+) loE=" + num + r" lo=([-+0-9.eE,]+) "
    r"hiE=" + num + r" hi=([-+0-9.eE,]+) hiID=(\d+) hNID=(\d+) hNE=" + num + r" hN=([-+0-9.eE,]+) "
    r"hPID=(-?\d+) hPE=" + num + r" hP=([-+0-9.eE,]+) amps2=" + num +
    r" flux=" + num + r" initwgt=" + num + r" spinavg=" + num + r" psw=" + num)
def v3(s): return [float(x) for x in s.split(",")]

knu, kmu, pN, pPi, psw_ach = [], [], [], [], []
n = 0
with open(LOG) as fh:
    for line in fh:
        if "RESDUMP" not in line: continue
        m = pat.search(line)
        if not m: continue
        (liE, li, loE, lo, hiE, hi, hiID, hNID, hNE, hN, hPID, hPE, hP,
         amps2, flux, initwgt, spinavg, psw) = m.groups()
        if int(hiID) != 2212 or int(hNID) != 2212 or int(hPID) != 211: continue
        if float(initwgt) != 1.0 or float(amps2) <= 0.0: continue
        knu.append([float(liE)] + v3(li)); kmu.append([float(loE)] + v3(lo))
        pN.append([float(hNE)] + v3(hN)); pPi.append([float(hPE)] + v3(hP))
        psw_ach.append(float(psw)); n += 1
        if n >= NMAX: break

knu = np.array(knu); kmu = np.array(kmu); pN = np.array(pN); pPi = np.array(pPi)
psw_ach = np.array(psw_ach)
print(f"parsed {n} physical free-proton events", flush=True)

# total invariant: P = knu + p_struck.  struck at rest with the dumped energy (939.57)
M_STRUCK = 939.57
P = knu.copy(); P[:, 0] += M_STRUCK
s = P[:, 0]**2 - np.sum(P[:, 1:]**2, axis=1)
# s23 = (mu + N)^2  -- the (muN) intermediate of split A (total -> (muN) + pi)
muN = kmu + pN
s23 = muN[:, 0]**2 - np.sum(muN[:, 1:]**2, axis=1)
m_pi, m_Nf = M_PIP, M_P
s23max = (np.sqrt(s) - m_pi)**2
s23min = max((M_MU + m_Nf)**2, 1e-8)
# ADoNIS density (free_proton_gen): (2pi)^5 * I2W_A * I2W_B / (s23max - s23min)
I2W_A = 2.0 / np.pi / np.clip(_sqlam(s, s23, m_pi**2), 1e-12, None)
I2W_B = 2.0 / np.pi / np.clip(_sqlam(s23, M_MU**2, m_Nf**2), 1e-12, None)
density = (2*np.pi)**5 * I2W_A * I2W_B / (s23max - s23min)
J_3body = np.where(density > 0, 1.0 / density, 0.0)

good = (J_3body > 0) & np.isfinite(J_3body) & (psw_ach > 0)
r = J_3body[good] / psw_ach[good]
print(f"  usable {good.sum()}")
print(f"  ratio J_3body_ADO / psw_ACH:  mean={r.mean():.6f}  median={np.median(r):.6f}  "
      f"std={r.std():.2e}  spread={r.std()/r.mean():.2e}")
print(f"  percentiles: 1%={np.percentile(r,1):.5f} 25%={np.percentile(r,25):.5f} "
      f"50%={np.median(r):.5f} 75%={np.percentile(r,75):.5f} 99%={np.percentile(r,99):.5f}")
print(f"  s23 range sampled: [{np.sqrt(s23[good].min()):.1f}, {np.sqrt(s23[good].max()):.1f}] MeV "
      f"(sqrt s23 = M(muN))")
