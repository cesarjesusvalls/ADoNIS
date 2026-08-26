"""Dump the fit's live-bin mask -- which bins survive the sparse-bin cut AT THE FIT'S OWN STATISTICS.

The engine caps selected events per sample (banks.sig_cap) while the Gate-I Jacobian streams the whole
bank, so the same cut drops a different set of bins in each; recomputing it downstream from the Jacobian
would produce a figure that looks masked to the fit's statistics and is not.

    python -m analysis.benchmarks.dump_live_mask [config]   ->  output/altgen/sec4_live_mask.npz

Read by analysis/paper/grad_info/make.py.  Re-run whenever the sample list, the bank caps or
data.sigma.mask_mcfrac change.
"""
import numpy as np
from adonis.fit.config import FitConfig
from analysis.campaign.stages.multisample import build_multisample_engine
import sys
cfg = FitConfig.load(sys.argv[1] if len(sys.argv) > 1 else "configs/fits/sec4_P1.yaml")
eng = build_multisample_engine(lambda m: None, cfg)
sig = cfg.data.sigma
keys, keep, ntot, nkeep = [], [], 0, 0
for s in eng.samples:
    for d in s.ds:
        c0 = np.abs(np.asarray(d["data"], float))
        mc = np.asarray(d["mcerr"], float)
        good = (c0 > 0) & (mc <= sig.mask_mcfrac * np.where(c0 > 0, c0, 1.0))
        keys.append(d["key"]); keep.append(good); ntot += len(good); nkeep += int(good.sum())
        print(f"  {d['key']:28s} {int(good.sum()):3d}/{len(good):3d} kept", flush=True)
print(f"TOTAL {nkeep}/{ntot} bins kept at the fit's statistics (nu_chunks={cfg.banks.nu_chunks})")
np.savez("output/altgen/sec4_live_mask.npz",
         keys=np.array(keys, dtype=object),
         **{f"{k}_keep": m for k, m in zip(keys, keep)})
print("[out] output/altgen/sec4_live_mask.npz")
