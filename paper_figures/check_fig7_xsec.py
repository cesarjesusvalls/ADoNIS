"""Per-bin ADoNIS/ACHILLES ratio for Fig 7 (T2K CC0pi STV) using the BIT-EXACT primary sample
(t2k_cc0pi_tki_adonis_xsec.npz = spectral-fn QE + RES-pion-absorbed + cascade).  THE criterion:
|ADoNIS/ACHILLES - 1| < 3% in every bin."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import uproot

ROOT = Path(__file__).resolve().parents[1]
DDIR = ROOT.parent / "nuisance" / "data" / "T2K" / "CC0pi" / "STV"
ach = np.load(ROOT / "data" / "oracle" / "t2k_cc0pi_tki_achilles.npz")
ado = np.load(ROOT / "data" / "oracle" / "t2k_cc0pi_tki_adonis_xsec.npz")


def edges(f):
    return uproot.open(DDIR / f)["Result"].to_numpy()[1]


def shape(x, w, ed):
    h, _ = np.histogram(x, bins=ed, weights=w); wd = np.diff(ed); d = h / wd
    s = np.sum(d * wd); return d / s if s > 0 else d


print(f"ADoNIS CC0pi events: {len(ado['w'])}")
for key, f, conv in [("dpt", "dptResults.root", 1e-3), ("dalphat", "datResults.root", 1.0)]:
    ed = edges(f)
    sa = shape(ach[key] * conv, ach["w"], ed); sd = shape(ado[key] * conv, ado["w"], ed)
    r = sd / np.clip(sa, 1e-12, None)
    mx = np.max(np.abs(r - 1)) * 100
    print(f"--- {key} ---  ratio {np.array2string(r, precision=3)}")
    print(f"    max|r-1| = {mx:.1f}%   {'PASS <3%' if mx < 3 else 'FAIL'}")
