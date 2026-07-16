"""COMPUTE the closure-fit diagnostics: the LM/Gauss-Newton parameter TRAJECTORY (theta vs iteration)
plus the quadratic (Gauss-Newton) expansion of the chi2 at the best-fit point -- everything the corner /
landscape / streamline figures need, persisted to ONE npz so the plotting can iterate on style without
ever re-running the fit (physfit_traj_fig.py renders it).

Same setup as the closure5 run (physical_fit_run.py MODE=closure): 10M bank, Gate-I subset from
GATE1_NPZ (shrink < 0.5), closure fake data at the injected knobs. The LM loop is a trajectory-recording
copy of physical_fit_run.lm_fit (same steps, same damping) -- the recorded path IS the real fit.

Saved npz (output/altgen/<ADONIS_LABEL>.npz):
  pnames_sub, subset       : fitted knob names / indices
  traj                     : (n_it+1, n_sub) accepted theta after each iteration (row 0 = start)
  chi2_traj, chi2_data_traj: total / data-only chi2 after each iteration (row 0 = start)
  lam_traj                 : LM damping after each iteration
  th_bfp, th_truth, th_nom : best fit / injected truth / nominal (subset values)
  prior_sub                : prior sigmas
  A, V                     : GN normal matrix (J^T W J + prior) and covariance (pinv A) at the BFP
  chi2_min, chi2_truth     : chi2 at BFP and at the injected truth (exact model, not quadratic)

Env:  ADONIS_EVENT_BANK  GATE1_NPZ  PHYSFIT_INJECT (default = the closure5 injection)
      ADONIS_LABEL (default physfit_traj_closure5)   NIT (default 20)   PHYSFIT_OBS (optional subset)

Run (batch; ~15-20 min on the 10M bank):
  ADONIS_EVENT_BANK=$ADONIS_OUT/event_bank_10M/merged \
  GATE1_NPZ=$ADONIS_OUT/altgen/physfit_gate1_full_10M.npz \
  python -u scripts/altgen/physfit_traj.py

EXACT-GRID mode (one knob PAIR per invocation -> shard pairs over a SLURM array): set EXACT_PAIR=<p>
(flat pair index 0..n_pairs-1) and optionally GRID_N (default 25). Reads the trajectory npz (must run
first) for the BFP/ranges, then evaluates the EXACT chi2 (full bank reweight per point, others clamped
at the BFP) on a GRID_N x GRID_N grid -> output/altgen/<label>_exact_p<p>.npz. ~1-3 s/point.
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
from physical_fit import build_physfit_datasets, OBS_SUBSETS, NPAR                    # noqa: E402
from physical_fit_run import Engine, apply_mode, parse_inject                          # noqa: E402

BANKDIR = os.environ.get("ADONIS_EVENT_BANK", "output/event_bank")
GATE1_NPZ = os.environ.get("GATE1_NPZ", "output/altgen/physfit_gate1.npz")
INJECT = os.environ.get("PHYSFIT_INJECT",
                        "M_A_res=0.85,kF_sf=1.10,Eb_shift=3.0,s_NN_elastic[pn]=1.25,f_NN_cex=0.40")
LABEL = os.environ.get("ADONIS_LABEL", "physfit_traj_closure5")
NIT = int(os.environ.get("NIT", "20"))
STEP_SCALE = float(os.environ.get("STEP_SCALE", "1.0"))   # <1 = shorter (smoother) LM/GN steps

_t0 = time.time()
def log(m):
    print(f"[{time.time()-_t0:7.1f}s] {m}", flush=True)


def lm_fit_traj(eng, subset, nit=NIT):
    """physical_fit_run.lm_fit with per-iteration recording (no Huber, no mask -- the closure case)."""
    data, sigma = eng.data_sigma()
    prior_w = 1.0 / eng.prior[subset] ** 2
    th = eng.th0.copy()
    lam = 1e-3

    def chi2_of(thv):
        m = eng.model(thv)
        u = (m - data) / sigma
        c_data = float(np.sum(u ** 2))
        c_pri = float(np.sum(prior_w * (thv[subset] - eng.th0[subset]) ** 2))
        return c_data + c_pri, c_data, m

    c_cur, c_data, m = chi2_of(th)
    traj = [th[subset].copy()]; c_traj = [c_cur]; cd_traj = [c_data]; lam_traj = [lam]
    log(f"start chi2 {c_cur:.2f} (data {c_data:.2f})")
    for it in range(nit):
        c_before = c_cur
        J = eng.jac(th, subset)
        W = 1.0 / sigma ** 2
        A = J.T @ (J * W[:, None]) + np.diag(prior_w)
        g = J.T @ (W * (m - data)) + prior_w * (th[subset] - eng.th0[subset])
        for _ in range(12):
            dth = STEP_SCALE * np.linalg.solve(A + lam * np.diag(np.maximum(np.diag(A), 1e-12)), -g)
            th_try = th.copy(); th_try[subset] = th[subset] + dth
            c_try, cd_try, m_try = chi2_of(th_try)
            if c_try < c_cur:
                th, c_cur, c_data, m = th_try, c_try, cd_try, m_try
                lam = max(lam / 3, 1e-8)
                break
            lam *= 5
        traj.append(th[subset].copy()); c_traj.append(c_cur); cd_traj.append(c_data); lam_traj.append(lam)
        log(f"it {it:2d} chi2={c_cur:9.3f} (data {c_data:9.3f}) lam={lam:.1e} " +
            " ".join(f"{eng.pnames[k]}={th[k]:.4f}" for k in subset))
        if np.linalg.norm(dth) < 1e-6 or (c_before - c_cur) / max(c_before, 1e-9) < 1e-3:
            log(f"converged/plateau at it {it}")
            break
    # GN quadratic at the BFP (same as lm_fit's final block)
    J = eng.jac(th, subset)
    A = J.T @ (J * (1.0 / sigma ** 2)[:, None]) + np.diag(prior_w)
    V = np.linalg.pinv(A, rcond=1e-12)
    return th, np.array(traj), np.array(c_traj), np.array(cd_traj), np.array(lam_traj), A, V, chi2_of


def exact_grid(eng, subset, chi2_of, pair_idx, grid_n):
    """Exact chi2 (full-model) on a 2D grid for the pair_idx-th (j,i) pair, others at the BFP.
    Range matches physfit_traj_fig._pair_range so quadratic and exact panels align."""
    zt = np.load(f"output/altgen/{LABEL}.npz", allow_pickle=True)
    bfp_s = np.asarray(zt["th_bfp"]); sig = np.sqrt(np.abs(np.diag(np.asarray(zt["V"]))))
    traj = np.asarray(zt["traj"]); truth_s = np.asarray(zt["th_truth"])
    pairs = [(a, b) for b in range(len(subset)) for a in range(b)]        # (j, i) j<i, row-major lower
    j, i = pairs[pair_idx]

    def rng(k):
        # tight +-4 sigma window around the BFP -- matches physfit_traj_fig._pair_range so the exact
        # grid resolves the ellipse (the old traj-extent range left thin valleys with <8 points).
        half = max(4.0 * sig[k], 1.25 * abs(truth_s[k] - bfp_s[k]))
        return bfp_s[k] - half, bfp_s[k] + half

    th = eng.th0.copy(); th[subset] = bfp_s                                # clamp all at BFP
    xr = np.linspace(*rng(j), grid_n); yr = np.linspace(*rng(i), grid_n)
    CH = np.zeros((grid_n, grid_n))
    for b, yv in enumerate(yr):
        for a, xv in enumerate(xr):
            t = th.copy(); t[subset[j]] = xv; t[subset[i]] = yv
            CH[b, a] = chi2_of(t)[0]
        log(f"row {b+1}/{grid_n} done")
    out = f"output/altgen/{LABEL}_exact_p{pair_idx}.npz"
    np.savez(out, pair=(j, i), xr=xr, yr=yr, chi2=CH, chi2_min_traj=float(zt["chi2_min"]))
    log(f"[out] {out}")


def main():
    B = BP.load_bank(BANKDIR)
    JB = BR.to_jax(B); grids = BR.default_grids(); nom = nominal_knobs()
    w0 = np.asarray(BR.weight_jit(JB, nom, grids))
    log(f"bank {len(w0):,} events")
    _obs = os.environ.get("PHYSFIT_OBS", "")
    ds = build_physfit_datasets(B, w0, log, obs=OBS_SUBSETS[_obs] if _obs else None)
    eng = Engine(ds, JB, grids, nom, B=B)

    g1 = np.load(GATE1_NPZ, allow_pickle=True)
    subset = [int(i) for i in np.where(g1["shrink"] < 0.5)[0]]
    log(f"fit subset ({len(subset)}): " + " ".join(eng.pnames[k] for k in subset))

    # PHYSFIT_FLUCT=<seed>: statistically fluctuate the Asimov data by its per-bin sigma (same
    # convention as physical_fit_run) -- the BFP then scatters around truth within the fit errors.
    truth, desc = apply_mode(ds, eng, "closure", inject=INJECT,
                             fluct=int(os.environ.get("PHYSFIT_FLUCT", "0")))
    log(f"fake data: {desc}")

    ep = os.environ.get("EXACT_PAIR", "")
    if ep:                                                                 # exact-grid mode
        data, sigma = eng.data_sigma()
        prior_w = 1.0 / eng.prior[subset] ** 2

        def chi2_of(thv):
            m = eng.model(thv); u = (m - data) / sigma
            return (float(np.sum(u ** 2)) +
                    float(np.sum(prior_w * (thv[subset] - eng.th0[subset]) ** 2)),)
        exact_grid(eng, subset, chi2_of, int(ep), int(os.environ.get("GRID_N", "25")))
        return

    th, traj, c_traj, cd_traj, lam_traj, A, V, chi2_of = lm_fit_traj(eng, subset)
    c_truth = chi2_of(np.asarray(truth))[0]
    log(f"chi2: bfp {c_traj[-1]:.3f}  truth {c_truth:.3f}")

    out = f"output/altgen/{LABEL}.npz"
    np.savez(out,
             pnames_sub=[eng.pnames[k] for k in subset], subset=subset,
             traj=traj, chi2_traj=c_traj, chi2_data_traj=cd_traj, lam_traj=lam_traj,
             th_bfp=th[subset], th_truth=np.asarray(truth)[subset], th_nom=eng.th0[subset],
             prior_sub=eng.prior[subset], A=A, V=V,
             chi2_min=c_traj[-1], chi2_truth=c_truth, inject=desc)
    log(f"[out] {out}")


if __name__ == "__main__":
    main()
