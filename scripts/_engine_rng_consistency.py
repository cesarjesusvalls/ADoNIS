"""Statistical-consistency check for the per-event RNG re-key (Stage 3a): the new per-event keying
draws DIFFERENT randoms per event than the old shared-key scheme, so individual events differ -- but the
DISTRIBUTIONS must be identical (same physics).  Compares aggregate observables between the old golden
(/tmp/refeng_C_oldRNG.npz) and the new one (/tmp/refeng_C.npz): terminal-pid multiplicities, escaped
counts, overflow.  A faithful re-key => all within Poisson noise.

Run: python -u scripts/_engine_rng_consistency.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

old = dict(np.load("/tmp/refeng_C_oldRNG.npz"))
new = dict(np.load("/tmp/refeng_C.npz"))


def pid_counts(d, tag):
    """Counts of escaped final-state nucleon/pion pids over all events (nterms alive + pterm)."""
    al = d[f"{tag}_nt_alive"]; pid = d[f"{tag}_nt_pid"]
    vals, cnts = np.unique(pid[al], return_counts=True)
    out = {int(v): int(c) for v, c in zip(vals, cnts)}
    pt = d[f"{tag}_pterm_pid"]
    pv, pc = np.unique(pt, return_counts=True)
    for v, c in zip(pv, pc):
        out[f"pterm{int(v)}"] = int(c)
    return out


print(f"{'observable':<22} {'old':>10} {'new':>10} {'d':>8} {'d/sqrtN':>9}")
for tag in ("res", "qe"):
    co, cn = pid_counts(old, tag), pid_counts(new, tag)
    keys = sorted(set(co) | set(cn), key=str)
    print(f"--- {tag} ---")
    for k in keys:
        o = co.get(k, 0); m = cn.get(k, 0); d = m - o
        sig = d / np.sqrt(max(o, 1))
        print(f"{str(k):<22} {o:>10} {m:>10} {d:>8} {sig:>9.2f}")
    print(f"{'ofl':<22} {int(old[tag+'_ofl']):>10} {int(new[tag+'_ofl']):>10}")
    print(f"{'logofl':<22} {int(old[tag+'_logofl']):>10} {int(new[tag+'_logofl']):>10}")
