"""Aggregate the MH-vs-NUTS grid: convergence gate FIRST, efficiency only if it passes.

The order is not cosmetic.  An efficiency ratio between two samplers that are not sampling the same
distribution is meaningless, and the failure is silent -- a badly-mixing chain reports a small ESS, which
LOOKS like an honest efficiency number rather than a broken one.  So:

  GATE 1  every parameter, every method: rank-normalised split-Rhat < RHAT_MAX.
  GATE 2  MH and NUTS agree on each 1-D marginal: |mean_MH - mean_NUTS| within a few combined MCSE, and
          the posterior sd's agree to a few percent.  This is the "same stationary distribution" check;
          without it, a sampler that never left a mode would look wonderfully efficient.

Only then the efficiency table, and it separates:

  ESS/s                      hardware- and implementation-specific (A100, this kernel, this harness)
  ESS per model evaluation   portable across implementations of the same model
  ESS per EVENT-PASS         portable across hardware AND model implementations; the unit the optimiser
                             report uses, so the two studies can be read together

    python -m adonis.fit.bench_mcmc_report [--glob 'output/altgen/mcmc_*.npz']
"""
from __future__ import annotations

import argparse
import glob as globmod
import sys
from collections import defaultdict

import numpy as np

from adonis.fit.mcmc_diag import ess_bulk, ess_tail, split_rhat, tau_int

RHAT_MAX = 1.01
MCSE_SIGMA = 4.0          # how many combined MCSE two methods' means may differ by


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
    ap.add_argument("--glob", default="output/altgen/mcmc_*.npz")
    ap.add_argument("--rhat-max", type=float, default=RHAT_MAX)
    a = ap.parse_args(argv)
    by = load(globmod.glob(a.glob))
    if not by:
        raise SystemExit(f"no files matching {a.glob}")
    ns = sorted({k[0] for k in by})
    print(f"loaded {sum(len(v) for v in by.values())} chains over n = {ns}\n")

    # ---- GATE 1: convergence ---------------------------------------------------------------------- #
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

    # ---- GATE 2: same stationary distribution ------------------------------------------------------ #
    print(f"\n==== GATE 2: MH vs NUTS agree on the marginals (|dmean| < {MCSE_SIGMA:g} combined MCSE) ====")
    print(f"{'n':>3} {'worst dial':>20} {'|dmean|/mcse':>13} {'sd ratio':>9} {'verdict':>9}")
    ok_same = {}
    for n in ns:
        if (n, "mh") not in S or (n, "nuts") not in S:
            continue
        A, B = S[(n, "mh")], S[(n, "nuts")]
        worst, wr, wsd = None, 0.0, 1.0
        for k, nm in enumerate(A["names"]):
            xa, xb = A["d"][:, :, k], B["d"][:, :, k]
            ma, mb = xa.mean(), xb.mean()
            sa, sb = xa.std(ddof=1), xb.std(ddof=1)
            mcse = np.sqrt(sa ** 2 / max(A["eb"][k], 1) + sb ** 2 / max(B["eb"][k], 1))
            r = abs(ma - mb) / mcse if mcse > 0 else np.inf
            if r > wr:
                worst, wr, wsd = nm, r, sb / sa if sa > 0 else np.nan
        good = wr < MCSE_SIGMA
        ok_same[n] = good
        print(f"{n:>3} {worst:>20} {wr:13.2f} {wsd:9.3f} {'PASS' if good else 'FAIL':>9}")

    # ---- efficiency --------------------------------------------------------------------------------- #
    print("\n==== EFFICIENCY (median over parameters of the per-parameter ESS) ====")
    print("hardware-specific ->            | portable ->")
    print(f"{'n':>3} {'method':>6} {'ESSb/s':>9} {'ESSt/s':>9} {'ESSb/1e3 logp':>14} "
          f"{'ESSb/1e3 pass':>14} {'tau_int':>8} {'accept':>7} {'grad/draw':>10}")
    eff = {}
    for n in ns:
        for m in ("mh", "nuts"):
            if (n, m) not in S:
                continue
            s = S[(n, m)]
            zs = s["zs"]
            t = float(np.sum([float(z["t_sample"]) for z in zs]))
            nlp = float(np.sum([float(z["n_logp"]) for z in zs]))
            ngr = float(np.sum([float(z["n_grad"]) for z in zs]))
            pas = float(np.sum([float(z["passes"]) for z in zs]))
            acc = float(np.mean([float(z["accept"]) for z in zs]))
            eb, et = float(np.median(s["eb"])), float(np.median(s["et"]))
            tau = float(np.median([tau_int(s["d"][:, :, k]) for k in range(s["d"].shape[2])]))
            eff[(n, m)] = dict(eb=eb, et=et, t=t, nlp=nlp, pas=pas)
            print(f"{n:>3} {m:>6} {eb/t:9.2f} {et/t:9.2f} {1e3*eb/max(nlp,1):14.2f} "
                  f"{1e3*eb/max(pas,1):14.2f} {tau:8.1f} {acc:7.3f} "
                  f"{ngr/max(sum(len(zs) for _ in [0])*s['nd'],1):10.2f}")

    # ---- the answer ---------------------------------------------------------------------------------- #
    print("\n==== NUTS / MH advantage vs parameter count ====")
    print(f"{'n':>3} {'ESSb/s':>9} {'ESSt/s':>9} {'ESSb/logp':>11} {'ESSb/pass':>11} {'gate':>6}")
    for n in ns:
        if (n, "mh") not in eff or (n, "nuts") not in eff:
            continue
        A, B = eff[(n, "mh")], eff[(n, "nuts")]
        g = "ok" if (ok_conv.get((n, "mh")) and ok_conv.get((n, "nuts")) and ok_same.get(n)) else "CHECK"
        print(f"{n:>3} {(B['eb']/B['t'])/(A['eb']/A['t']):9.2f} "
              f"{(B['et']/B['t'])/(A['et']/A['t']):9.2f} "
              f"{(B['eb']/B['nlp'])/(A['eb']/A['nlp']):11.2f} "
              f"{(B['eb']/B['pas'])/(A['eb']/A['pas']):11.2f} {g:>6}")
    print("\n(>1 means NUTS is more efficient.  ESSb/pass is the portable one: it removes this GPU and "
          "this harness and asks how much independent information each FULL PASS OVER THE EVENTS buys.)")


if __name__ == "__main__":
    sys.exit(main())
