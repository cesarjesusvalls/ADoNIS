"""Bit-exact carbon refactor gate: regenerate the 2-seed carbon baseline and np-compare to the
golden snapshot.  Used after each generalization stage to prove carbon is byte-for-byte unchanged.
Run: python -u scripts/_gate_carbon_bitexact.py   (regenerate gen_basecarb.yaml first)."""
import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def cmp(tag):
    new = np.load(ROOT / f"data/oracle/t2k_{tag}_engine_rich_basecarb.npz")
    gold = np.load(ROOT / f"data/oracle/t2k_{tag}_engine_rich_basegold.npz")
    ok = True
    keys = sorted(set(new.files) | set(gold.files))
    for k in keys:
        if k not in new.files or k not in gold.files:
            print(f"  [{tag}] {k}: MISSING ({'new' if k not in new.files else 'gold'})"); ok = False; continue
        a, b = new[k], gold[k]
        if a.shape != b.shape:
            print(f"  [{tag}] {k}: SHAPE {a.shape} vs {b.shape}"); ok = False; continue
        if not np.array_equal(a, b, equal_nan=True):
            d = np.abs(a.astype(float) - b.astype(float)); m = np.nanmax(d)
            print(f"  [{tag}] {k}: DIFF max|Δ|={m:.3e} ({np.sum(d>0)} elems)"); ok = False
    print(f"[{tag}] {'BIT-EXACT' if ok else 'DIFFERS'}")
    return ok

if __name__ == "__main__":
    all_ok = all([cmp("cc0pi"), cmp("cc1pi")])
    print("GATE:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)
