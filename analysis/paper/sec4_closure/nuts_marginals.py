"""Sec 4.1 companion -- TRUE marginal posteriors by NUTS, vs the profile, on the SAME model.

The 4.1 curves use the profile likelihood (maximize over the 15 nuisance dials).  The Bayesian marginal
INTEGRATES over them instead; the two differ by the nuisance-volume (Occam) factor.  To isolate that effect
cleanly we work on ONE model -- the autodiff Taylor posterior built from the cached derivative tensors
(Jb, Bb, and Cb = d3m at the BFP), whose chi2 is
    chi2(D) = sum_b W_b ( Jb.D + 1/2 D.Bb.D + 1/6 D.Cb.D.D )^2 ,   D = theta - BFP .
On this model we compute, per dial:
  * the NUTS MARGINAL   -- blackjax NUTS samples the full 16-D posterior; the 1-D histogram is the marginal,
  * the surrogate PROFILE -- min over the other 15 dials on a grid (the same object 4.1 plots, same model),
  * the GAUSSIAN        -- N(0, sigma_post).
Overlaying the three answers "why not marginalize?": they coincide where the posterior is Gaussian and split
(marginal vs profile) exactly where the nuisance volume changes with the dial.

Env: ADONIS_LABEL (default sec4_closure_r16_noprior), NUTS_ORDER (2 or 3, default 3),
NUTS_WARMUP (1000), NUTS_SAMPLES (4000), NUTS_CHAINS (4).  Writes output/altgen/<label>_nuts.npz.
"""
import os
import sys
import time
import itertools
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    import jax
    import jax.numpy as jnp
    import blackjax
    from analysis.paper import style

    LABEL = os.environ.get("ADONIS_LABEL", "sec4_closure_r16_noprior")
    ORDER = int(os.environ.get("NUTS_ORDER", "3"))
    NWARM = int(os.environ.get("NUTS_WARMUP", "1000"))
    NSAMP = int(os.environ.get("NUTS_SAMPLES", "4000"))
    NCH = int(os.environ.get("NUTS_CHAINS", "4"))
    log(f"jax backend: {jax.default_backend()}  devices={jax.devices()}")

    z = np.load(style.ALTGEN / f"{LABEL}_derivs.npz", allow_pickle=True)
    Jb = jnp.asarray(z["Jb"]); Bb = jnp.asarray(z["Bb"]); W = jnp.asarray(z["W"])
    V = np.asarray(z["V"]); sub = [int(k) for k in z["subset"]]; pn = [str(x) for x in z["pnames"]]
    Cb = jnp.asarray(z["Cb"]) if ("Cb" in z.files and ORDER >= 3) else None
    nsub = len(sub); spost = np.sqrt(np.abs(np.diag(V)))
    sp = jnp.asarray(spost)
    log(f"model: {nsub} dials, order {ORDER}, Cb={'yes' if Cb is not None else 'no'}")

    # ---- Taylor-surrogate chi2, sampled in scaled coords u (D = sigma_post * u) ---------------------- #
    def resid(D):
        r = Jb @ D + 0.5 * jnp.einsum("bjk,j,k->b", Bb, D, D)
        if Cb is not None:
            r = r + (1.0 / 6.0) * jnp.einsum("bjkl,j,k,l->b", Cb, D, D, D)
        return r

    def logdensity(u):
        r = resid(sp * u)
        return -0.5 * jnp.sum(W * r * r)

    # ---- NUTS (blackjax) with window adaptation, NCH chains ----------------------------------------- #
    def run_chain(seed):
        key = jax.random.PRNGKey(seed)
        wk, sk = jax.random.split(key)
        warmup = blackjax.window_adaptation(blackjax.nuts, logdensity, progress_bar=False)
        (state, params), _ = warmup.run(wk, jnp.zeros(nsub), num_steps=NWARM)
        step = jax.jit(blackjax.nuts(logdensity, **params).step)

        def one(st, k):
            st, info = step(k, st)
            return st, (st.position, info.is_divergent)
        _, (pos, div) = jax.lax.scan(one, state, jax.random.split(sk, NSAMP))
        return pos, div

    log(f"NUTS: {NCH} chains x {NSAMP} samples ({NWARM} warmup)")
    U = []; ndiv = 0
    for c in range(NCH):
        pos, div = run_chain(c)
        pos.block_until_ready()
        U.append(np.asarray(pos)); ndiv += int(np.asarray(div).sum())
        log(f"  chain {c}: {NSAMP} samples, {int(np.asarray(div).sum())} divergences")
    U = np.concatenate(U, axis=0)                         # (NCH*NSAMP, nsub) in sigma_post units
    Dsamp = U * spost[None, :]                            # dial shift from BFP (native theta units)
    log(f"NUTS done: {U.shape[0]} samples, {ndiv} divergences total")

    # ---- surrogate 1-D PROFILE per dial (same model, min over the other 15) ------------------------- #
    Jn = np.asarray(z["Jb"]); Bn = np.asarray(z["Bb"]); Wn = np.asarray(z["W"])
    Cn = np.asarray(z["Cb"]) if Cb is not None else None

    def resid_np(D):
        r = Jn @ D + 0.5 * np.einsum("bjk,j,k->b", Bn, D, D)
        if Cn is not None:
            r = r + (1.0 / 6.0) * np.einsum("bjkl,j,k,l->b", Cn, D, D, D, optimize=True)
        return r

    grid = np.linspace(-3.5, 3.5, 71)                     # sigma_post units
    prof = np.full((nsub, len(grid)), np.nan)
    for a in range(nsub):
        others = [i for i in range(nsub) if i != a]
        for gi, gv in enumerate(grid):
            D = np.zeros(nsub); D[a] = gv * spost[a]
            for _ in range(12):                           # GN on the polynomial over the other dials
                r = resid_np(D)
                Jr = Jn + np.einsum("bjk,k->bj", Bn, D)
                if Cn is not None:
                    Jr = Jr + 0.5 * np.einsum("bjkl,k,l->bj", Cn, D, D, optimize=True)
                Jo = Jr[:, others]
                D[others] += np.linalg.solve(Jo.T @ (Jo * Wn[:, None]) + 1e-9 * np.eye(len(others)),
                                             -Jo.T @ (Wn * r))
            prof[a, gi] = float(np.sum(Wn * resid_np(D)**2))
        prof[a] -= np.nanmin(prof[a])
        if (a + 1) % 4 == 0:
            log(f"  profiled {a+1}/{nsub} dials")

    out = style.ALTGEN / f"{LABEL}_nuts.npz"
    np.savez(out, subset=sub, pnames=pn, spost=spost, V=V, order=ORDER,
             samples_D=Dsamp, grid_sigma=grid, prof_dchi2=prof, n_div=ndiv,
             n_samples=U.shape[0])
    log(f"[out] {out}  (samples {Dsamp.shape}, prof {prof.shape})")


if __name__ == "__main__":
    main()
