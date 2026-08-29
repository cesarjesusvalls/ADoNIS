"""Aggregate the MH-vs-NUTS grid: convergence gate FIRST, efficiency only if it passes.

  GATE 1  every parameter, every method: rank-normalised split-Rhat < RHAT_MAX.
  GATE 2  MH and NUTS agree on each 1-D marginal: |mean_MH - mean_NUTS| within a few combined MCSE, and
          the posterior sd's agree to a few percent (the "same stationary distribution" check).

Only then the efficiency table, which separates:

  ESS/s                      hardware- and implementation-specific
  ESS per model evaluation   portable across implementations of the same model
  ESS per EVENT-PASS         portable across hardware AND model implementations

    python -m analysis.benchmarks.bench_mcmc_report [--glob '<results>/mcmc_*.npz']
"""
from __future__ import annotations

from analysis._cli import results_dir

import argparse
import glob as globmod
import sys
from collections import defaultdict

import numpy as np

from adonis.fit.mcmc_diag import ess_bulk, ess_tail, split_rhat, tau_int

RHAT_MAX = 1.01
MCSE_SIGMA = 4.0


def load(paths):
    by = defaultdict(list)
    for p in sorted(paths):
        z = np.load(p, allow_pickle=True)
        by[(int(z["ndials"]), str(z["method"]))].append(z)
    return by


def stack(zs):
    """(nchain, ndraw, npar) truncated to the SHORTEST chain, so no chain is over-weighted."""
    n = min(int(z["draws"].shape[0]) for z in zs)
    return np.stack([np.asarray(z["draws"])[:n] for z in zs]), n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default=str(results_dir() / "mcmc_*.npz"))
    ap.add_argument("--rhat-max", type=float, default=RHAT_MAX)
    a = ap.parse_args(argv)
    by = load(globmod.glob(a.glob))
    if not by:
        raise SystemExit(f"no files matching {a.glob}")
    ns = sorted({k[0] for k in by})
    print(f"loaded {sum(len(v) for v in by.values())} chains over n = {ns}\n")

    print("==== GATE 1: rank-normalised split-Rhat (max over parameters) ====")
    print(f"{'n':>3} {'method':>6} {'chains':>7} {'draws/chain':>12} {'max Rhat':>9} {'min ESS_bulk':>13} "
          f"{'min ESS_tail':>13} {'verdict':>9}")
    ok_conv = {}
    S = {}
    for n in ns:
        for m in ("mh", "nuts"):
            zs = by.get((n, m))
            if not zs:
                continue
            d, nd = stack(zs)
            names = [str(x) for x in zs[0]["names"]]
            rh = [split_rhat(d[:, :, k]) for k in range(d.shape[2])]
            eb = [ess_bulk(d[:, :, k]) for k in range(d.shape[2])]
            et = [ess_tail(d[:, :, k]) for k in range(d.shape[2])]
            good = np.nanmax(rh) < a.rhat_max
            ok_conv[(n, m)] = good
            S[(n, m)] = dict(d=d, names=names, rhat=rh, eb=eb, et=et, nd=nd, zs=zs)
            print(f"{n:>3} {m:>6} {len(zs):>7} {nd:>12,} {np.nanmax(rh):9.4f} {np.nanmin(eb):13.1f} "
                  f"{np.nanmin(et):13.1f} {'PASS' if good else 'FAIL':>9}")

    from scipy.stats import kstwo
    print(f"\n==== GATE 2: MH and NUTS are sampling the SAME distribution ====")
    print(f"{'n':>3} {'worst dial':>20} {'|dmean|/mcse':>13} {'KS D':>8} {'KS p':>8} "
          f"{'|dq05|/mcse':>12} {'|dq95|/mcse':>12} {'verdict':>9}")
    ok_same = {}
    for n in ns:
        if (n, "mh") not in S or (n, "nuts") not in S:
            continue
        A, B = S[(n, "mh")], S[(n, "nuts")]
        worst, wr, wD, wp, wq5, wq95 = None, 0.0, 0.0, 1.0, 0.0, 0.0
        for k, nm in enumerate(A["names"]):
            xa, xb = A["d"][:, :, k].ravel(), B["d"][:, :, k].ravel()
            ea, eb_ = max(A["eb"][k], 1.0), max(B["eb"][k], 1.0)
            sa, sb = xa.std(ddof=1), xb.std(ddof=1)
            mcse = np.sqrt(sa ** 2 / ea + sb ** 2 / eb_)
            r = abs(xa.mean() - xb.mean()) / mcse if mcse > 0 else np.inf
            grid = np.union1d(xa, xb)
            D = float(np.max(np.abs(np.searchsorted(np.sort(xa), grid, "right") / xa.size
                                    - np.searchsorted(np.sort(xb), grid, "right") / xb.size)))
            neff = ea * eb_ / (ea + eb_)
            pv = float(kstwo.sf(D, max(int(round(neff)), 2)))
            dq = []
            for qq in (0.05, 0.95):
                qa, qb = np.quantile(xa, qq), np.quantile(xb, qq)
                w = 0.1 * (np.quantile(xa, 0.75) - np.quantile(xa, 0.25)) + 1e-300
                fa = max(np.mean(np.abs(xa - qa) < w) / (2 * w), 1e-300)
                se = np.sqrt(qq * (1 - qq)) / fa * np.sqrt(1 / ea + 1 / eb_)
                dq.append(abs(qa - qb) / se if se > 0 else np.inf)
            if r > wr:
                worst, wr, wD, wp, wq5, wq95 = nm, r, D, pv, dq[0], dq[1]
        good = (wr < MCSE_SIGMA) and (wp > 0.01) and (wq5 < MCSE_SIGMA) and (wq95 < MCSE_SIGMA)
        ok_same[n] = good
        print(f"{n:>3} {worst:>20} {wr:13.2f} {wD:8.4f} {wp:8.3f} {wq5:12.2f} {wq95:12.2f} "
              f"{'PASS' if good else 'FAIL':>9}")

    print("\n==== EFFICIENCY (median over parameters of the per-parameter ESS) ====")
    print("hardware-specific ->            | portable ->")
    print(f"{'n':>3} {'method':>6} {'ESSb/s':>9} {'ESSt/s':>9} {'ESSb/1e3 visit':>15} "
          f"{'ESSb/1e3 pass':>14} {'tau_int':>8} {'accept':>7} {'grad/draw':>10} "
          f"{'MINessb/s':>9} {'worst par':>18} {'ndiv':>6}")
    eff = {}
    for n in ns:
        for m in ("mh", "nuts"):
            if (n, m) not in S:
                continue
            s = S[(n, m)]
            zs = s["zs"]
            fr = [s["nd"] / float(np.asarray(z["draws"]).shape[0]) for z in zs]
            t = float(np.sum([f * float(z["t_sample"]) for f, z in zip(fr, zs)]))
            vis = float(np.sum([f * float(z["visits"] if "visits" in z.files else z["n_logp"])
                                for f, z in zip(fr, zs)]))
            ngr = float(np.sum([f * float(z["n_grad"]) for f, z in zip(fr, zs)]))
            pas = float(np.sum([f * float(z["passes"]) for f, z in zip(fr, zs)]))
            acc = float(np.mean([float(z["accept"]) for z in zs]))
            ndv = int(np.sum([int(z["ndiv"]) for z in zs])) if "ndiv" in zs[0].files else -1
            ndraw = float(s["d"].shape[0] * s["d"].shape[1])
            eb, et = float(np.median(s["eb"])), float(np.median(s["et"]))
            ebmin = float(np.min(s["eb"]))
            worstpar = s["names"][int(np.argmin(s["eb"]))]
            tau = float(np.median([tau_int(s["d"][:, :, k]) for k in range(s["d"].shape[2])]))
            eff[(n, m)] = dict(eb=eb, et=et, t=t, vis=vis, pas=pas, ebmin=ebmin, worst=worstpar,
                               ndiv=ndv, spread=eb / max(ebmin, 1e-300))
            print(f"{n:>3} {m:>6} {eb/t:9.2f} {et/t:9.2f} {1e3*eb/max(vis,1):14.2f} "
                  f"{1e3*eb/max(pas,1):14.2f} {tau:8.1f} {acc:7.3f} {ngr/max(ndraw,1):10.2f} "
                  f"{ebmin/t:9.2f} {worstpar:>18} {ndv:>6}")

    print("\n==== NUTS / MH advantage vs parameter count ====")
    print(f"{'n':>3} {'ESSb/s':>9} {'ESSt/s':>9} {'ESSb/visit':>11} {'ESSb/pass':>11} {'gate':>6}")
    for n in ns:
        if (n, "mh") not in eff or (n, "nuts") not in eff:
            continue
        A, B = eff[(n, "mh")], eff[(n, "nuts")]
        g = "ok" if (ok_conv.get((n, "mh")) and ok_conv.get((n, "nuts")) and ok_same.get(n)) else "CHECK"
        print(f"{n:>3} {(B['eb']/B['t'])/(A['eb']/A['t']):9.2f} "
              f"{(B['et']/B['t'])/(A['et']/A['t']):9.2f} "
              f"{(B['eb']/B['vis'])/(A['eb']/A['vis']):11.2f} "
              f"{(B['eb']/B['pas'])/(A['eb']/A['pas']):11.2f} {g:>6}")
    print("\n(>1 means NUTS is more efficient.  ESSb/pass is the portable one: it removes this GPU and "
          "this harness and asks how much independent information each FULL PASS OVER THE EVENTS buys.)")


if __name__ == "__main__":
    sys.exit(main())
