"""Assert the corner used the SAME per-dial spans as the merged 1-D profile.

This is the check that was missing.  `profile2d` reads `<label>_profile.npz` for its axes, but the stage
only warns when that file is ABSENT -- a STALE one from another campaign is accepted silently, and the
provenance machinery validates shards against each other, never a stage against the stage it consumed.
Result: a corner built on M_A_res [-17.50,+5.40] while the profile said [-5.40,+3.00], which moved every
wall test and every crop.  Read-only.
"""
import sys, glob, numpy as np
from analysis.paper.sec4_closure.fig_corner_prof import load_views
lab = sys.argv[1] if len(sys.argv)>1 else 'sec4_P2'
z = np.load(f'output/altgen/{lab}_profile.npz', allow_pickle=True)
pn=[str(x) for x in z['pnames']]; sub=[int(k) for k in z['subset']]; g=np.asarray(z['grids_sigma'])
prof={pn[k]:(float(np.nanmin(g[c])), float(np.nanmax(g[c]))) for c,k in enumerate(sub)}
views, meta = load_views(lab, True)
seen, bad = {}, []
for (ni,nj),w in views.items():
    for nm,ax in ((ni,w['axi']),(nj,w['axj'])):
        seen.setdefault(nm,(float(ax[0]),float(ax[-1])))
print(f"{'dial':>20} {'corner span':>20} {'profile span':>20}   match")
for nm,(a0,a1) in seen.items():
    p0,p1 = prof.get(nm,(np.nan,np.nan))
    ok = abs(a0-p0)<1e-3 and abs(a1-p1)<1e-3
    if not ok: bad.append(nm)
    print(f'{nm:>20} [{a0:+7.2f},{a1:+7.2f}] [{p0:+7.2f},{p1:+7.2f}]   {"OK" if ok else "MISMATCH"}')
print("\nRESULT:", "all dials consistent" if not bad else f"MISMATCH on {bad} -- the corner used a different profile")
sys.exit(1 if bad else 0)
