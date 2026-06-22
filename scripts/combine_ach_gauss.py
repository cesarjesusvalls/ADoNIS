"""Combine the per-seed Gaussian ACHILLES batches (data/oracle/_ach_gauss_batches/s*.npz) into one
reference with the correct POOLED normalization (weights events ~equally, so a tiny crashed batch
contributes proportionally to its stats, not 1/B):
  sigma_b = sum_w_b * weight_to_nb_b   (per batch absolute nb);  mean_sigma = mean_b sigma_b
  combined per-event weight [nb] = w_i * mean_sigma / (sum over ALL batches of w_j) ;  weight_to_nb = 1
For B=1 this reduces to w_i*weight_to_nb_1 (the single-file ref).  Usage: python scripts/combine_ach_gauss.py [out.npz]"""
import sys, glob
import numpy as np
OUT = sys.argv[1] if len(sys.argv) > 1 else "data/oracle/t2k_cc1pi_rich_ach_FSI_C_gauss.npz"
files = sorted(glob.glob("data/oracle/_ach_gauss_batches/s*.npz"))
assert files, "no batch npzs found"
META = {"weight_to_nb", "gen_xs_pb", "sum_w_all"}
sig_b, sumw_b, ds = [], [], []
for f in files:
    d = np.load(f, allow_pickle=True); ds.append(d)
    sw = float(d["w"].sum()); sig_b.append(sw * float(d["weight_to_nb"])); sumw_b.append(sw)
mean_sigma = float(np.mean(sig_b)); total_raw_w = float(np.sum(sumw_b))
scale = mean_sigma / total_raw_w
print(f"{len(files)} batches  events={sum(len(d['w']) for d in ds)}  "
      f"sigma_b mean={mean_sigma:.4e} spread={np.std(sig_b)/mean_sigma*100:.1f}%  scale={scale:.4e}", flush=True)
out = {}
for k in ds[0].files:
    if k in META:
        continue
    out[k] = np.concatenate([(d["w"].astype(float) * scale) if k == "w" else d[k] for d in ds])
out["weight_to_nb"] = 1.0
np.savez(OUT, **out)
print(f"DONE -> {OUT}  ({len(out['w'])} events, pooled, weight_to_nb=1)", flush=True)
