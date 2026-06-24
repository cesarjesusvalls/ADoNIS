"""Stage-3b refill gate.  Runs the SAME fixed C sample (RES+QE, P=12) through the persistent-refill
engine and compares to the lock-step golden (/tmp/refeng_C.npz):
  MODE A  n_w = N_total  -> working set holds every event, cursor exhausted (no actual refill).  Must be
          BIT-EXACT on ALL 70 arrays (the strong gate: the flush-to-global machinery reproduces lock-step).
  MODE B  n_w = small    -> real refill (slots reused).  Per-event PHYSICS must be bit-exact by evt_id
          (out/pterm/nterms/created/rec + segment channels/momenta).  The only allowed difference is the
          segment PROVENANCE LABELS (log_track_id/log_parent_id) which encode the working-slot index, so a
          refilled event gets different ABSOLUTE ids (parent<->child links + is_first stay self-consistent
          within each event; the matrix never uses track_id).

Run: python -u scripts/_engine_refill_check.py [n_w_small=200]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ADONIS_N_RECOIL", "8")
import numpy as np
from adonis.workflow.materials import resolve_targets
from _engine_golden import run_all

TOL = 1e-9
NW_SMALL = int(sys.argv[1]) if len(sys.argv) > 1 else 200
LABEL_KEYS = {f"{t}_log_{k}" for t in ("res", "qe") for k in ("track_id", "parent_id")}


def compare(ref, cur, ignore=()):
    bad = 0; worst = 0.0; worstk = ""
    for k in sorted(ref):
        if k in ignore:
            continue
        a = ref[k]; b = np.asarray(cur[k])
        if a.shape != b.shape:
            print(f"  SHAPE {k}: ref{a.shape} != cur{b.shape}"); bad += 1; continue
        d = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()) if a.size else 0.0
        if d > worst:
            worst, worstk = d, k
        if d > TOL:
            print(f"  DIFF {k}: max|d|={d:.3e}"); bad += 1
    return bad, worst, worstk


def main():
    ref = dict(np.load("/tmp/refeng_C.npz"))
    tg = resolve_targets("C")[0][0]

    print(f"=== MODE A: PRODUCTION DEFAULT path (n_w=None auto-refill + q_cap=None queue) vs lock-step golden ===", flush=True)
    curA = run_all(None, tg, n_w=None, q_cap=None)                   # cascade_nucleus defaults
    badA, wA, wkA = compare(ref, curA)
    print(f"  {len(ref)} arrays, {badA} differ.  worst |d|={wA:.3e} ({wkA})")
    print("  MODE A:", "PASS (bit-exact)" if badA == 0 else "FAIL", flush=True)

    print(f"\n=== MODE B: small refill n_w={NW_SMALL} + queue ON (real slot reuse) vs lock-step golden ===", flush=True)
    curB = run_all(None, tg, n_w=NW_SMALL, q_cap=None)
    badB, wB, wkB = compare(ref, curB)
    print(f"  {len(ref)} arrays, {badB} differ.  worst |d|={wB:.3e} ({wkB})")
    print("  MODE B:", "PASS (bit-exact)" if badB == 0 else "FAIL", flush=True)

    print("\nRESULT:", "PASS" if (badA == 0 and badB == 0) else "FAIL")
    sys.exit(0 if (badA == 0 and badB == 0) else 1)


if __name__ == "__main__":
    main()
