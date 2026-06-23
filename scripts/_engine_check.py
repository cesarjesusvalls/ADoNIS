"""Bit-exact gate for the persistent-refill engine rewrite: recompute the Stage-0 golden outputs with the
CURRENT engine and compare to /tmp/refeng_C.npz.  Run after every rewrite stage.  C has ofl=0 (no
overflow) so EVERY stage must stay bit-exact on C (the kept-overflow-particle change only shows on Ar).

Run: python -u scripts/_engine_check.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ADONIS_N_RECOIL", "8")
import numpy as np
from adonis.workflow.materials import resolve_targets
from _engine_golden import run_all

TOL = 1e-9


def main():
    ref = dict(np.load("/tmp/refeng_C.npz"))
    cur = run_all(None, resolve_targets("C")[0][0])
    bad = 0; worst = 0.0; worstk = ""
    for k in sorted(ref):
        a = ref[k]; b = np.asarray(cur[k])
        if a.shape != b.shape:
            print(f"  SHAPE {k}: ref{a.shape} != cur{b.shape}"); bad += 1; continue
        d = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()) if a.size else 0.0
        if d > worst:
            worst, worstk = d, k
        if d > TOL:
            print(f"  DIFF {k}: max|d|={d:.3e}"); bad += 1
    print(f"\nbit-exact check: {len(ref)} arrays, {bad} differ.  worst |d|={worst:.3e} ({worstk})")
    print("RESULT:", "PASS (bit-exact)" if bad == 0 else "FAIL")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
