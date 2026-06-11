"""Concurrent 3-parameter closure on the CC0pi-Np observables (differentiable ADoNIS, option B).

Recover (M_A, sigma_abs, sigma_scatter) jointly by gradient descent from
  - dsigma/dQ^2            -> M_A         (axial dipole)
  - CC0pi absorbed fraction -> sigma_abs   (pion absorption rate)
  - dsigma/d(delta_pT)     -> sigma_scatter (leading-proton NN rescattering)

Kind-1 reweighting throughout: a detached proposal is sampled ONCE; every stochastic decision reads
the FROZEN reference knobs, and the fitted knobs enter ONLY through a per-event likelihood-ratio
weight -> d/dtheta E[obs] is exact, autodiff == finite difference at a fixed key.  The proton/pion
trajectories use the smooth GAUSSIAN interaction probability (reweightable), per the design choice.

Loss: two-replica unbiased chi^2  mean((m1-d)(m2-d)/(d+1))  (the 2nd replica cancels the model-MC
variance bias).  Truth is injected; the SAME differentiable model generates the pseudo-data, so this
is a genuine closure (recovers truth up to MC noise).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# ---- reference knobs (truth/init straddle them) and fixed kinematic constants ------------------ #
MA0, SABS0, SSC0 = 1.03, 1.0, 1.0          # reference (proposal) knobs: M_A [GeV], abs & scat scales
MN, MMU = 939.0, 105.7
NSTEP, DX, RHO, MB_FM2 = 14, 0.45, 0.16, 0.1   # cascade: steps, fm/step, rho[fm^-3], mb->fm^2
SIG_NN0, SIG_ABS0 = 40.0, 30.0             # reference NN-scatter & pion-abs cross sections [mb]
F_RES = 0.18                               # RES fraction of the produced sample (rest QE)
Q2_EDGES = np.linspace(0.0, 1.4, 16)
DPT_EDGES = np.linspace(0.0, 700.0, 16)    # MeV


def _dipole(Q2_GeV2, MA):
    return 1.0 / (1.0 + Q2_GeV2 / MA ** 2) ** 2


def proposal(key, n):
    """Detached CC0pi-Np proposal sampled at the reference knobs.  Returns the (frozen) kinematics +
    the per-event stochastic decisions needed to reweight sigma_abs and sigma_scatter."""
    sg = jax.lax.stop_gradient
    kq, kr, ka, ksc, kdef, kfermi, kphi = jax.random.split(key, 7)
    # Q^2 ~ proposal at MA0 (importance: dipole^2), drawn on [0, 1.4] GeV^2
    Q2 = jax.random.uniform(kq, (n,), minval=0.0, maxval=1.4)
    is_res = jax.random.uniform(kr, (n,)) < F_RES
    # pion absorption (RES only): Bernoulli at reference p_abs0; non-absorbed RES are NOT CC0pi
    p_abs0 = SIG_ABS0 / (SIG_ABS0 + SIG_NN0)                # reference absorb-vs-scatter branch
    absorbed = jax.random.uniform(ka, (n,)) < p_abs0
    # leading proton: starts with the Fermi imbalance (delta_pT ~ |p_fermi_T|), then NN-rescatters.
    pf = sg(jnp.abs(jax.random.normal(kfermi, (n,)) * 90.0))    # initial transverse imbalance [MeV]
    phi = jax.random.uniform(kphi, (n,)) * 2 * jnp.pi
    dpt0 = pf
    dvec = jnp.stack([dpt0 * jnp.cos(phi), dpt0 * jnp.sin(phi)], axis=1)
    # NN cascade (continuum, Gaussian/stochastic): per step interact ~ Bernoulli(lam0*DX); on
    # interact add a transverse kick (detached).  Record the per-step rolls for kind-1 reweighting.
    lam0 = SIG_NN0 * MB_FM2 * RHO                            # reference interaction rate [1/fm]
    rolls = jax.random.uniform(ksc, (n, NSTEP))             # interaction Bernoulli rolls
    kicks = sg(jax.random.normal(kdef, (n, NSTEP, 2)) * 110.0)   # transverse kick per scatter [MeV]
    p_int0 = float(-np.expm1(-lam0 * DX))                    # reference per-step interaction prob (const)
    interacted = rolls < p_int0                             # (n, NSTEP) sampled at reference
    dvec = dvec + jnp.sum(jnp.where(interacted[:, :, None], kicks, 0.0), axis=1)
    dpt = jnp.linalg.norm(dvec, axis=1)                     # final delta_pT [MeV] (detached)
    return dict(Q2=sg(Q2), is_res=is_res, absorbed=absorbed, dpt=sg(dpt),
                rolls=sg(rolls), p_int0=float(p_int0), p_abs0=float(p_abs0))


def _soft_hist(x, w, edges):
    """Differentiable-in-w histogram (hard bin assignment is detached; only w carries gradient)."""
    idx = jnp.clip(jnp.searchsorted(jnp.asarray(edges), x) - 1, 0, len(edges) - 2)
    return jax.ops.segment_sum(w, jax.lax.stop_gradient(idx), num_segments=len(edges) - 1)


def model(theta, S):
    """Weighted (Q2 hist, absorbed-fraction-vs-Q2, dpt hist) for knobs theta=(M_A, s_abs, s_scat).
    Kind-1 reweighting of the frozen proposal S."""
    MA, sabs, ssc = theta[0], theta[1], theta[2]
    # production reweight (axial dipole, |F_A|^2)
    w_prod = (_dipole(S["Q2"], MA) / _dipole(S["Q2"], MA0)) ** 2
    # CC0pi membership: QE protons always; RES only if the pion absorbed
    is_cc0pi = (~S["is_res"]) | (S["is_res"] & S["absorbed"])
    # sigma_abs reweight (kind-1): absorbed RES weighted by p_abs(s_abs)/p_abs0
    p_abs = (sabs * SIG_ABS0) / (sabs * SIG_ABS0 + SIG_NN0)
    w_abs = jnp.where(S["is_res"], jnp.where(S["absorbed"], p_abs / S["p_abs0"], 1.0), 1.0)
    # sigma_scatter reweight (kind-1): per-step interact/not likelihood ratio at the frozen rolls
    p_int = -jnp.expm1(-(ssc * SIG_NN0) * MB_FM2 * RHO * DX)
    inter = S["rolls"] < S["p_int0"]
    ratio = jnp.where(inter, p_int / S["p_int0"], (1.0 - p_int) / (1.0 - S["p_int0"]))
    w_scat = jnp.prod(ratio, axis=1)
    w = w_prod * w_abs * w_scat * is_cc0pi
    h_q2 = _soft_hist(S["Q2"], w, Q2_EDGES)                                       # total CC0pi vs Q2 -> M_A
    h_abs = _soft_hist(S["Q2"], w * (S["is_res"] & S["absorbed"]), Q2_EDGES)      # RES-absorbed vs Q2 -> sigma_abs
    h_dpt = _soft_hist(S["dpt"], w, DPT_EDGES)                                    # CC0pi vs delta_pT -> sigma_scatter
    return jnp.concatenate([h_q2, h_abs, h_dpt])    # all COUNTS (linear in w -> two-replica unbiased)


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    N = 300_000
    MA_T, SABS_T, SSC_T = 1.15, 1.30, 0.80                  # injected TRUTH
    MA_I, SABS_I, SSC_I = 1.00, 1.00, 1.00                  # init
    NAMES = ["M_A", "sigma_abs", "sigma_scatter"]
    true_p = jnp.array([MA_T, SABS_T, SSC_T]); init_p = jnp.array([MA_I, SABS_I, SSC_I])

    S_data = proposal(jax.random.PRNGKey(1), N)
    data = jax.lax.stop_gradient(model(true_p, S_data))
    log(f"data generated at truth {np.asarray(true_p)}")

    def loss(theta, key):
        S1 = proposal(key, N); S2 = proposal(jax.random.fold_in(key, 1), N)
        m1, m2 = model(theta, S1), model(theta, S2)
        inv = 1.0 / (data + 1.0)
        return jnp.mean((m1 - data) * (m2 - data) * inv)

    vg = jax.jit(jax.value_and_grad(loss))
    log("compiling value+grad ..."); tc = time.time()
    l0, g0 = vg(init_p, jax.random.PRNGKey(2)); log(f"  compiled {time.time()-tc:.1f}s  loss0={float(l0):.3e}  grad0={np.asarray(g0)}")
    # gradient check (autodiff vs finite-difference) on the first parameter set
    fd = []
    for i in range(3):
        e = jnp.zeros(3).at[i].set(2e-3)
        fd.append(float((loss(init_p + e, jax.random.PRNGKey(2)) - loss(init_p - e, jax.random.PRNGKey(2))) / (2 * 2e-3)))
    log(f"  grad check  AD={np.asarray(g0)}  FD={np.array(fd)}")

    theta = init_p; lr = 0.03; m = jnp.zeros(3); v = jnp.zeros(3); b1, b2 = 0.9, 0.999
    traj = [np.asarray(theta)]
    for it in range(400):
        l, g = vg(theta, jax.random.PRNGKey(100 + it))
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g ** 2
        mh = m / (1 - b1 ** (it + 1)); vh = v / (1 - b2 ** (it + 1))
        theta = theta - lr * mh / (jnp.sqrt(vh) + 1e-8)
        theta = jnp.clip(theta, 0.3, 3.0)
        traj.append(np.asarray(theta))
        if it % 40 == 0 or it == 399:
            log(f"  it {it:3d} loss={float(l):.3e}  M_A={float(theta[0]):.3f} s_abs={float(theta[1]):.3f} s_sc={float(theta[2]):.3f}")
    traj = np.array(traj); fit = np.asarray(theta)
    log("RECOVERED:")
    for i, nm in enumerate(NAMES):
        log(f"  {nm:>14s}: init {float(init_p[i]):.3f} -> fit {fit[i]:.3f}  (truth {float(true_p[i]):.3f}, |err| {abs(fit[i]-float(true_p[i])):.3f})")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # ---- binned pseudo-data vs initial vs final prediction (same proposal S_data) ---------------- #
    nQ = len(Q2_EDGES) - 1; nD = len(DPT_EDGES) - 1
    d_all = np.asarray(data); i_all = np.asarray(model(init_p, S_data)); f_all = np.asarray(model(jnp.asarray(fit), S_data))
    Q2c = 0.5 * (Q2_EDGES[1:] + Q2_EDGES[:-1]); Dc = 0.5 * (DPT_EDGES[1:] + DPT_EDGES[:-1])
    PAN = [(Q2c, slice(0, nQ), r"$Q^2$ [GeV$^2$]", "CC0$\\pi$ counts", "Q$^2$ (sets M_A)"),
           (Q2c, slice(nQ, 2 * nQ), r"$Q^2$ [GeV$^2$]", "RES-absorbed counts", "absorbed vs Q$^2$ (sets $\\sigma_{abs}$)"),
           (Dc, slice(2 * nQ, 2 * nQ + nD), r"$\delta p_T$ [MeV]", "CC0$\\pi$ counts", "$\\delta p_T$ (sets $\\sigma_{scat}$)")]
    figs, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax2, (xc, sl, xl, yl, ti) in zip(axs, PAN):
        ax2.step(xc, d_all[sl], where="mid", color="k", lw=2, label="pseudo-data (truth)")
        ax2.step(xc, i_all[sl], where="mid", color="C3", ls="--", lw=1.5, label="initial")
        ax2.step(xc, f_all[sl], where="mid", color="C0", lw=1.5, label="final (fit)")
        ax2.set(xlabel=xl, ylabel=yl, title=ti); ax2.legend(fontsize=8); ax2.set_ylim(bottom=0)
    figs.suptitle("CC0$\\pi$-Np closure — binned pseudo-data, initial and final predictions (same proposal)")
    figs.tight_layout(); figs.savefig("paper_figures/cc0pi_closure_spectra.png", dpi=120)
    print("wrote paper_figures/cc0pi_closure_spectra.png")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for i, nm in enumerate(NAMES):
        ax[i].plot(traj[:, i], color="C0", lw=1.8)
        ax[i].axhline(float(true_p[i]), color="k", ls="--", lw=1.2, label=f"truth {float(true_p[i]):.3f}")
        ax[i].axhline(float(init_p[i]), color="C3", ls=":", lw=1.0, alpha=0.6, label=f"init {float(init_p[i]):.2f}")
        ax[i].set(xlabel="iteration", title=f"{nm}\nfit {fit[i]:.3f}"); ax[i].legend(fontsize=8)
    fig.suptitle("CC0$\\pi$-Np concurrent 3-parameter closure (M_A, $\\sigma_{abs}$, $\\sigma_{scat}$) — "
                 "kind-1 reweighting, two-replica $\\chi^2$")
    fig.tight_layout(); fig.savefig("paper_figures/cc0pi_closure.png", dpi=120)
    print("wrote paper_figures/cc0pi_closure.png")


if __name__ == "__main__":
    main()
