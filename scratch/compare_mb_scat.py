"""Compare ACHILLES MesonBaryonInteraction::CrossSection (MBSCAT stderr) vs the ADoNIS DCC
scatter sigma (cascade_mb) at identical (pidm, pidb, W).  Decides whether sigma_scatter is the
source of the cascade transport over-prediction."""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from adonis.fsi.mb.anl_xsec import load_anl, _channel_sigma
from adonis.fsi.mb.cascade_mb import _CHANNELS

Wt, amps = load_anl(0, 0)
PIDX = {211: 0, 111: 1, -211: 2}
NUC = {2212: "p", 2112: "n"}


def ado_scat(pidm, pidb, W):
    pin = PIDX.get(int(pidm)); nuc = NUC.get(int(pidb))
    if pin is None or nuc is None or (pin, nuc) not in _CHANNELS:
        return None
    tot = 0.0
    for po, no, cg in _CHANNELS[(pin, nuc)]:
        tot += max(0.0, float(np.interp(W, Wt, _channel_sigma(amps, Wt, cg), left=0, right=0)))
    return tot


pat = re.compile(r"MBSCAT pidm=(-?\d+) pidb=(-?\d+) W=([\d.eE+-]+) tot=([\d.eE+-]+)")


def main(path):
    rows = []
    for line in open(path, errors="ignore"):
        m = pat.search(line)
        if m:
            pm, pb, W, tot = int(m[1]), int(m[2]), float(m[3]), float(m[4])
            a = ado_scat(pm, pb, W)
            if a is not None and tot > 1e-6:
                rows.append((pm, pb, W, tot, a))
    rows = np.array([(W, t, a) for _, _, W, t, a in rows])
    if not len(rows):
        print("no MBSCAT lines"); return
    W, ta, ad = rows.T
    r = ad / ta
    print(f"MBSCAT samples: {len(rows)}")
    print(f"ADoNIS/ACHILLES sigma_scatter ratio: median={np.median(r):.4f} mean={np.mean(r):.4f} "
          f"[{np.percentile(r,5):.3f},{np.percentile(r,95):.3f}]")
    print(f"\n{'W':>8} {'ACH_scat':>9} {'ADO_scat':>9} {'ratio':>6}")
    for i in np.argsort(W)[:: max(1, len(W) // 12)]:
        print(f"{W[i]:8.1f} {ta[i]:9.3f} {ad[i]:9.3f} {ad[i]/ta[i]:6.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "_oracle_out/osetabs.log")
