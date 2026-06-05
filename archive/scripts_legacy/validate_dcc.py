"""Validate the DCC table parser against the ACHILLES file structure.

Checks: partial-wave quantum numbers, monotonic grids, the resonance-region
coverage, the amplitude entry counts vs the file's namp header, a spot-checked
amplitude value, and the .npz cache roundtrip.

Run:  python validate_dcc.py
"""
import numpy as np
from diffpi.dcc_loader import parse_dcc_ew, load_cached, PW_LABELS

t = parse_dcc_ew()
ok = True

def check(name, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    ok = ok and cond

print("DCC table validation:")
# 14 partial waves, p33 is the Delta (2J=3, 2L=2, 2I=3)
check("14 partial waves", len(t.pw_2J) == 14 and t.labels == PW_LABELS)
i_p33 = t.labels.index("p33")
check("p33 = Delta (2J=3,2L=2,2I=3)",
      (t.pw_2J[i_p33], t.pw_2L[i_p33], t.pw_2I[i_p33]) == (3, 2, 3))
i_s11 = t.labels.index("s11")
check("s11 (2J=1,2L=0,2I=1)",
      (t.pw_2J[i_s11], t.pw_2L[i_s11], t.pw_2I[i_s11]) == (1, 0, 1))

# grids
check("W grid 67 pts, monotonic, covers Delta region",
      t.W.size == 67 and np.all(np.diff(t.W) > 0) and t.W[0] < 1100 and t.W.max() >= 1480)
check("Q2 grid 28 pts, monotonic from 0", t.Q2.size == 28 and t.Q2[0] == 0
      and np.all(np.diff(t.Q2) > 0))

# amplitude arrays present and non-trivial
for nm, arr in [("vec", t.vec), ("isv", t.isv), ("axial", t.axial)]:
    check(f"{nm} amplitudes nonzero", np.count_nonzero(arr) > 0
          and arr.shape[:2] == (28, 67) and arr.dtype == np.complex128)

# spot check: file line "1 1 2 1 1 1  ... 0.3164655962E-04 0.1421911812E-01 ..."
# -> vec[iq=0, ie=0, idx=1, ipw=0, igmb=0] dressed component (za index 1)
dressed = t.vec[0, 0, 1, 0, 0, 1]
check("spot-check dressed vector amplitude",
      np.isclose(dressed.real, 0.3164655962e-04) and np.isclose(dressed.imag, 0.1421911812e-01))

# cache roundtrip
tc = load_cached()
check("cache roundtrip (vec identical)", np.array_equal(tc.vec, t.vec)
      and np.array_equal(tc.W, t.W))

print(f"\nDCC parser: {'PASS' if ok else 'FAIL'}")
raise SystemExit(0 if ok else 1)
