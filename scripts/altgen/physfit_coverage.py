"""COVERAGE TOY: one pseudo-experiment per invocation (shard over a SLURM array).

Throw truth from the PRIOR (the model's ~20% multiplicative knob priors), generate pseudo-data with a
per-bin statistical+systematic throw, fit it, and persist the postfit summary. The ensemble over N
toys tests coverage: postfit chi2 should follow a chi2 distribution and the pulls
(theta_fit - theta*)/sigma_fit should be unit normal (physfit_coverage_fig.py renders both).

Per toy (TOY_SEED = array task id):
  1. theta*_i  = theta0 + prior * N(0,1)   on the Gate-I fit subset      [SYST throw, the 20% priors]
  2. data_i    = model(theta*_i) + N(0, sigma_bin)  per bin              [STAT+SYST bin throw;
     sigma_bin is the fit's total per-bin error: 5% syst (+) MC stat -- the SAME sigma the fit uses]
  3. LM/GN fit (prior penalty centered at theta0, exactly like the closure fits)
  4. save output/altgen/coverage/toy_<seed>.npz : th_star, th_fit, sig_fit, chi2 (total/data), nbins

Env: ADONIS_EVENT_BANK  GATE1_NPZ  TOY_SEED  NIT (default 12)  PHYSFIT_OBS (optional)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax
jax.config.update("jax_enable_x64", True)

from analysis.t2k.differentiability import bank_plot as BP, bank_reweight as BR       # noqa: E402
from analysis.t2k.differentiability.full_knobs import nominal_knobs                   # noqa: E402
from physical_fit import build_physfit_datasets, OBS_SUBSETS                          # noqa: E402
from physical_fit_run import Engine                                                    # noqa: E402

BANKDIR = os.environ.get("ADONIS_EVENT_BANK", "output/event_bank")
GATE1_NPZ = os.environ.get("GATE1_NPZ", "output/altgen/physfit_gate1.npz")
SEED = int(os.environ.get("TOY_SEED", "0"))
NIT = int(os.environ.get("NIT", "12"))
STEP_SCALE = float(os.environ.get("STEP_SCALE", "1.0"))   # <1 = shorter damped steps (robust, no multi-start)

_t0 = time.time()
def log(m):
    print(f"[{time.time()-_t0:7.1f}s] {m}", flush=True)


def main():
    B = BP.load_bank(BANKDIR)
    JB = BR.to_jax(B); grids = BR.default_grids(); nom = nominal_knobs()
    w0 = np.asarray(BR.weight_jit(JB, nom, grids))
    _obs = os.environ.get("PHYSFIT_OBS", "")
    ds = build_physfit_datasets(B, w0, log, obs=OBS_SUBSETS[_obs] if _obs else None)
    eng = Engine(ds, JB, grids, nom, B=B)
    g1 = np.load(GATE1_NPZ, allow_pickle=True)
    subset = [int(i) for i in np.where(g1["shrink"] < 0.5)[0]]
    nsub = len(subset)
    _, sigma = eng.data_sigma()
    row0 = eng.row0

    # ---- the toy: prior throw of truth + per-bin stat/syst throw of the data ---------------------- #
    rng = np.random.default_rng(90_000 + SEED)
    th_star = eng.th0.copy()
    th_star[subset] = eng.th0[subset] + eng.prior[subset] * rng.standard_normal(nsub)
    # Eb_shift is a ONE-SIDED knob (ACHILLES's clamped linear-in-E SF makes the Eb<0 branch
    # non-smooth; see commit 67b3d9c) -- throw its truth from the HALF-NORMAL |N(0,prior)|, the
    # consistent truncated prior on the physical domain (no pileup exactly at the boundary).
    for c, k in enumerate(subset):
        if eng.pnames[k] == "Eb_shift":
            th_star[k] = 1e-2 + eng.prior[k] * abs(rng.standard_normal())
    m_star = eng.model(th_star)
    data = m_star + sigma * rng.standard_normal(len(sigma))
    for j, d in enumerate(ds):
        d["data"] = data[row0[j]:row0[j + 1]]
    log(f"toy {SEED}: theta* thrown ({', '.join(f'{eng.pnames[k]}={th_star[k]:.3f}' for k in subset)})")

    # ---- LM/GN fit (same machinery as the closure fits) ------------------------------------------- #
    prior_w = 1.0 / eng.prior[subset] ** 2
    th = eng.th0.copy(); lam = 1e-3

    def chi2_of(thv):
        m = eng.model(thv); u = (m - data) / sigma
        c_data = float(np.sum(u ** 2))
        return c_data + float(np.sum(prior_w * (thv[subset] - eng.th0[subset]) ** 2)), c_data, m

    c_cur, c_data, m = chi2_of(th)
    for it in range(NIT):
        c_before = c_cur
        J = eng.jac(th, subset)
        W = 1.0 / sigma ** 2
        A = J.T @ (J * W[:, None]) + np.diag(prior_w)
        g = J.T @ (W * (m - data)) + prior_w * (th[subset] - eng.th0[subset])
        for _ in range(12):
            dth = STEP_SCALE * np.linalg.solve(A + lam * np.diag(np.maximum(np.diag(A), 1e-12)), -g)
            th_try = th.copy(); th_try[subset] = th[subset] + dth
            # box constraint: Eb_shift >= eps.  Below 0 the clamped model is EXACTLY flat (zero
            # gradient) -- an absorbing trap for LM.  Projecting onto the boundary keeps the
            # one-sided gradient alive so the fit can climb back off it if the data demands.
            for _c, _k in enumerate(subset):
                if eng.pnames[_k] == "Eb_shift":
                    th_try[_k] = max(th_try[_k], 1e-2)
            c_try, cd_try, m_try = chi2_of(th_try)
            if c_try < c_cur:
                th, c_cur, c_data, m = th_try, c_try, cd_try, m_try
                lam = max(lam / 3, 1e-8); break
            lam *= 5
        log(f"it {it:2d} chi2={c_cur:9.3f} (data {c_data:9.3f})")
        if np.linalg.norm(dth) < 1e-6 or (c_before - c_cur) / max(c_before, 1e-9) < 1e-4:
            break
    J = eng.jac(th, subset)
    A = J.T @ (J * (1.0 / sigma ** 2)[:, None]) + np.diag(prior_w)
    V = np.linalg.pinv(A, rcond=1e-12)

    outdir = Path("output/altgen/coverage"); outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"toy_{SEED:04d}.npz"
    np.savez(out, seed=SEED, pnames_sub=[eng.pnames[k] for k in subset],
             th_star=th_star[subset], th_fit=th[subset], sig_fit=np.sqrt(np.abs(np.diag(V))),
             th_nom=eng.th0[subset], prior_sub=eng.prior[subset],
             chi2_total=c_cur, chi2_data=c_data, nbins=int(row0[-1]), nsub=nsub)
    log(f"[out] {out}  chi2_total={c_cur:.2f} chi2_data={c_data:.2f} nbins={int(row0[-1])}")


if __name__ == "__main__":
    main()
