"""Prototype Pareto-smoothed importance sampling (PSIS) on the RES Vegas weights.
Fit a generalized Pareto to the top-M weights, replace them by the fitted order statistics (capped),
report k-hat (tail-heaviness diagnostic), the tail before/after, N_eff, and the bias on sigma."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import genpareto
from adonis.xsec import res_xsec as R
from adonis.xsec.vegas_grid import VegasGrid

grid = VegasGrid.load("data/oracle/res_vegasgrid_C12.npz")
def neffN(w): p = w[w > 0]; return p.sum() ** 2 / np.sum(p ** 2) / len(p)

# generate one RES Vegas weight set
ws = []
for sd in range(3):
    ws.append(R.generate(120000, seed=sd, return_events=True, grid=grid)["events"]["w"])
w = np.concatenate(ws); w = w[w > 0]
N = len(w); med = np.median(w)

def psis(w):
    w = w.copy(); N = len(w)
    M = int(min(0.2 * N, 3.0 * np.sqrt(N)))           # tail size (Vehtari et al.)
    order = np.argsort(w); tail_idx = order[-M:]
    u = w[order[-M - 1]]                               # cutoff = largest non-tail weight
    exc = w[tail_idx] - u                              # exceedances over the cutoff
    k, loc, sigma = genpareto.fit(exc, floc=0.0)      # GPD fit (loc fixed at 0)
    z = (np.arange(M) + 0.5) / M
    smoothed = u + genpareto.ppf(z, k, loc=0.0, scale=sigma)
    smoothed = np.minimum(smoothed, w[tail_idx].max())  # cap at the largest raw weight
    w[tail_idx] = smoothed
    return w, k

wp, khat = psis(w)
print(f"N={N}  M_tail={int(min(0.2*N,3*np.sqrt(N)))}  k_hat={khat:.3f}  "
      f"(<0.5 good, 0.5-0.7 ok, >0.7 problematic)")
print(f"{'':8s} {'sum_w (sigma proxy)':>20s} {'N_eff/N':>8s} {'max/med':>8s} {'p99.9/med':>9s}")
print(f"{'raw':8s} {w.sum():20.6e} {neffN(w):8.3f} {w.max()/med:8.0f} {np.percentile(w,99.9)/med:9.0f}")
print(f"{'PSIS':8s} {wp.sum():20.6e} {neffN(wp):8.3f} {wp.max()/med:8.0f} {np.percentile(wp,99.9)/med:9.0f}")
print(f"sigma bias (PSIS/raw - 1) = {wp.sum()/w.sum()-1:+.4%}")
