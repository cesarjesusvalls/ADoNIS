"""Were the toy best fits actually the global minimum?  Refit each toy from a SECOND starting point.

Every toy in the ensemble was fitted BLIND: started at nominal, which for the P1 truth is 37 sigma away
across 17 dials.  On a surface with a two-to-one RES direction that is ample opportunity to settle in the
reflected basin.  This re-runs the SAME toys -- the noise is reproducible from the seed -- from a second
start and keeps both objectives:

  chi2_nominal : what the ensemble recorded (start = nominal)
  chi2_truth   : start = the toy's own injected truth   (closure-only diagnostic)
  chi2_mirror  : start = theta_hat with delta_strength reflected about MIRROR_VERTEX, if set
                 (this one needs no knowledge of the truth, so it is available on REAL data)

A toy whose restart reaches a LOWER chi2 was a minimiser failure, not a statistical fluctuation.
Env: S4_TOY_BASE, S4_NTOYS, MIRROR_VERTEX (optional), ADONIS_LABEL.
"""
import os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
from adonis.fit.stages.multisample import build_multisample_engine, MULTISAMPLE_NPZ, fit_subset
from adonis.fit.stages.physical_fit_run import trf_fit, parse_inject
from adonis.reweight.reweight_model import nominal_knobs

BASE = int(os.environ.get("S4_TOY_BASE", "0")); NT = int(os.environ.get("S4_NTOYS", "125"))
NIT = int(os.environ.get("ALTGEN_NIT", "200"))
VERT = os.environ.get("MIRROR_VERTEX", "").strip()
from adonis.fit.config import FitConfig
cfg = FitConfig.load(os.environ.get("ADONIS_FIT_CONFIG", "configs/fits/sec4_P1.yaml"))
eng = build_multisample_engine(log, cfg)
g = np.load(MULTISAMPLE_NPZ, allow_pickle=True)
subset = fit_subset(g, eng.pnames, cfg, log)
eng.prior = eng.prior * 1e6                       # MLE, same as the ensemble (S4_PRIOR_SCALE=0)
star, _ = parse_inject(os.environ["S4_FIXED_TRUTH"], nominal_knobs())
kd = list(eng.pnames).index("delta_strength")

# the ensemble's recorded fits, to compare against
E = {}
import glob
for f in sorted(glob.glob(f"output/altgen/{os.environ.get('ENS_LABEL','sec4_P1_ens')}_*.npz")):
    z = np.load(f, allow_pickle=True)
    b = int(f.rsplit("_", 1)[1].split(".")[0])
    for t in range(len(z["th_fit"])):
        E[b + t] = (np.asarray(z["th_fit"])[t], float(np.asarray(z["chi2_data"])[t]))
log(f"{len(E)} recorded toys; this shard does seeds {BASE}..{BASE+NT-1}")

rows = []
out = f"output/altgen/{os.environ.get('ADONIS_LABEL','sec4_P1_restart')}_{BASE:04d}.npz"
for t in range(NT):
    seed = BASE + t
    if seed not in E: continue
    rng = np.random.default_rng(1_000_000 + seed)
    eng.set_closure_data(star)                             # identical to the ensemble's construction
    for d in eng.ds:
        sd = np.where(np.isfinite(d["sigma"]), d["sigma"], 0.0)
        d["data"] = d["data"] + rng.normal(0.0, sd)
    th_rec, c_rec = E[seed]
    th_t, _, _, _, _, c_truth = trf_fit(eng, subset, f"t{seed}", nit=NIT, th_init=star.copy())
    c_mirr = np.nan; dl_m = np.nan
    if VERT:
        th0 = th_rec.copy(); th0[kd] = max(2.0 * float(VERT) - th_rec[kd], 0.05)
        th_m, _, _, _, _, c_mirr = trf_fit(eng, subset, f"m{seed}", nit=NIT, th_init=th0)
        dl_m = th_m[kd]
    rows.append((seed, c_rec, c_truth, c_mirr, th_rec[kd], th_t[kd], dl_m))
    log(f"toy {seed}: chi2 rec={c_rec:.2f} truth-start={c_truth:.2f} mirror={c_mirr:.2f}  "
        f"dS rec={th_rec[kd]:.3f} truth-start={th_t[kd]:.3f}")
    if (t + 1) % 5 == 0 or t == NT - 1:
        a = np.array(rows)
        np.savez(out, seed=a[:, 0], chi2_rec=a[:, 1], chi2_truth=a[:, 2], chi2_mirror=a[:, 3],
                 dS_rec=a[:, 4], dS_truth=a[:, 5], dS_mirror=a[:, 6], truth_dS=star[kd])
log(f"[out] {out}")
