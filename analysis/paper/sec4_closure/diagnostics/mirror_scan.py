"""Is the BINNED prediction two-to-one in delta_strength, and if so where is the partner?

The RES hard-vertex weight is g^T M g / g0^T M g0 with g = (u, dl*u), u = (1, r, r*pp), so it is exactly
quadratic in dl = delta_strength with vertex dl* = -(u^T M12 u)/(u^T M22 u).  u depends on Q^2, so dl* is
EVENT-dependent and there is no guaranteed global mirror.  What decides whether a mirror basin exists in
the FIT is whether the degeneracy survives binning and the sum over events -- so test that directly:

  scan dl at the best fit, and for every pair (dl_i, dl_j) compute the chi2-metric distance between the
  two binned predictions,  D_ij = sum_b W_b (m_b(dl_i) - m_b(dl_j))^2 .

D_ij = 0 off the diagonal is an exact mirror; a shallow valley is an approximate one.  Also reports, for
the TRUTH value of dl, the partner that minimises D and the chi2 the fit would see there.
"""
import os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
from adonis.fit.stages.multisample import build_multisample_engine
from adonis.fit.stages.physical_fit_run import parse_inject
from adonis.reweight.reweight_model import nominal_knobs

from adonis.fit.config import FitConfig
cfg = FitConfig.load(os.environ.get("ADONIS_FIT_CONFIG", "configs/fits/sec4_P1.yaml"))
eng = build_multisample_engine(log, cfg)
truth, _ = parse_inject(os.environ["PHYSFIT_INJECT"], nominal_knobs())
eng.set_closure_data(truth)
data, sigma = eng.data_sigma()
ok = np.isfinite(sigma) & (sigma > 0)
W = np.where(ok, 1.0 / np.where(ok, sigma, 1.0) ** 2, 0.0)
pn = list(eng.pnames); kd = pn.index("delta_strength")
log(f"truth delta_strength = {truth[kd]:.4f}")

N = int(os.environ.get("MIRROR_N", "61"))
lo, hi = 0.05, 3.0
grid = np.linspace(lo, hi, N)
M = np.empty((N, len(data)))
for i, v in enumerate(grid):
    th = truth.copy(); th[kd] = v
    M[i] = eng.model(th)
    if (i + 1) % 10 == 0: log(f"  {i+1}/{N}")

D = np.zeros((N, N))
for i in range(N):
    d = (M - M[i]) * np.sqrt(W)
    D[i] = np.sum(d * d, axis=1)
np.savez("output/altgen/sec4_P1_mirror.npz", grid=grid, D=D, chi2=np.sum(((M - data) ** 2) * W, axis=1),
         truth_dl=truth[kd])

it = int(np.argmin(np.abs(grid - truth[kd])))
row = D[it].copy(); row[max(it-3,0):it+4] = np.inf      # ignore the trivial self-neighbourhood
jm = int(np.argmin(row))
c2 = np.sum(((M - data) ** 2) * W, axis=1)
log(f"TRUTH dl={grid[it]:.3f}  chi2={c2[it]:.3e}")
log(f"best non-local partner dl={grid[jm]:.3f}  ||dm||^2_W={row[jm]:.3e}  chi2 there={c2[jm]:.3e}")
log(f"  (for scale: chi2 at dl={grid[min(it+6,N-1)]:.2f} is {c2[min(it+6,N-1)]:.3e})")
print("\n dl      chi2(dl)     ||m(dl)-m(truth)||^2_W")
for i in range(0, N, max(N // 30, 1)):
    print(f"{grid[i]:6.3f} {c2[i]:12.4e} {D[it, i]:14.4e}")
