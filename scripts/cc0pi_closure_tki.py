"""Closure on the T2K-MEASURED CC0pi-Np observables: recover the FSI knobs (sigma_abs, sigma_scatter)
by gradient descent from dsigma/d(delta_alphaT) and dsigma/d(delta_pT), binned on the ACTUAL T2K STV
bins (t2k_cc0pi_stv_data.npz) -- i.e. set up exactly as a fit to the real T2K data would be.

The TKI shapes constrain the two FSI parameters: sigma_scatter broadens delta_pT / pushes delta_alphaT
toward pi (proton rescattering); sigma_abs sets the RES-absorbed contribution, a distinctly HARD
delta_pT proton (the pi-absorption kick).  M_A's handle is Q^2 (not in the STV data), so it is held.

Kind-1 reweighting (Gaussian/stochastic interaction, frozen proposal); two-replica unbiased chi^2.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

SABS0, SSC0 = 1.0, 1.0
NSTEP, DX, RHO, MB_FM2 = 14, 0.45, 0.16, 0.1
SIG_NN0, SIG_ABS0 = 40.0, 30.0
F_RES = 0.18
_d = np.load("data/oracle/t2k_cc0pi_stv_data.npz")
DAT_EDGES = _d["dalphat_edges"]                  # rad
DPT_EDGES = _d["dpt_edges"] * 1000.0             # GeV -> MeV (same bins as the T2K data)


def proposal(key, n):
    sg = jax.lax.stop_gradient
    kmu, kr, ka, kf, kph, ksc, kkick, kabs = jax.random.split(key, 8)
    pmu = sg(250.0 + 450.0 * jax.random.uniform(kmu, (n,)))      # muon |pT| [MeV], along +x
    is_res = jax.random.uniform(kr, (n,)) < F_RES
    p_abs0 = SIG_ABS0 / (SIG_ABS0 + SIG_NN0)
    absorbed = jax.random.uniform(ka, (n,)) < p_abs0
    # transverse imbalance dvec = pT_mu + pT_p ; pre-FSI ~ Fermi (small, random direction)
    pf = sg(jnp.abs(jax.random.normal(kf, (n,)) * 90.0)); phi = jax.random.uniform(kph, (n,)) * 2 * jnp.pi
    dvec = jnp.stack([pf * jnp.cos(phi), pf * jnp.sin(phi)], axis=1)
    # RES-absorbed proton: a HARD extra kick (pi-absorption energy) -> distinct high-delta_pT bump
    kdir = jax.random.uniform(kabs, (n,)) * 2 * jnp.pi
    abs_kick = sg(jnp.where((is_res & absorbed)[:, None],
                            260.0 * jnp.stack([jnp.cos(kdir), jnp.sin(kdir)], axis=1), 0.0))
    dvec = dvec + abs_kick
    # NN cascade (continuum, Gaussian): per-step interact ~ Bernoulli(lam0 DX); kick on interact
    lam0 = SIG_NN0 * MB_FM2 * RHO; p_int0 = float(-np.expm1(-lam0 * DX))
    rolls = jax.random.uniform(ksc, (n, NSTEP)); kicks = sg(jax.random.normal(kkick, (n, NSTEP, 2)) * 110.0)
    inter = rolls < p_int0
    dvec = dvec + jnp.sum(jnp.where(inter[:, :, None], kicks, 0.0), axis=1)
    dpt = jnp.linalg.norm(dvec, axis=1)
    dat = jnp.arccos(jnp.clip(-dvec[:, 0] / (dpt + 1e-9), -1.0, 1.0))   # angle to -pT_mu (+x)
    return dict(is_res=is_res, absorbed=absorbed, dpt=sg(dpt), dat=sg(dat),
                rolls=sg(rolls), p_int0=p_int0, p_abs0=float(p_abs0))


def _hist(x, w, edges):
    idx = jnp.clip(jnp.searchsorted(jnp.asarray(edges), x) - 1, 0, len(edges) - 2)
    return jax.ops.segment_sum(w, jax.lax.stop_gradient(idx), num_segments=len(edges) - 1)


def model(theta, S):
    sabs, ssc = theta[0], theta[1]
    is_cc0pi = (~S["is_res"]) | (S["is_res"] & S["absorbed"])
    p_abs = (sabs * SIG_ABS0) / (sabs * SIG_ABS0 + SIG_NN0)
    w_abs = jnp.where(S["is_res"] & S["absorbed"], p_abs / S["p_abs0"], 1.0)
    p_int = -jnp.expm1(-(ssc * SIG_NN0) * MB_FM2 * RHO * DX)
    inter = S["rolls"] < S["p_int0"]
    w_scat = jnp.prod(jnp.where(inter, p_int / S["p_int0"], (1.0 - p_int) / (1.0 - S["p_int0"])), axis=1)
    w = w_abs * w_scat * is_cc0pi
    return jnp.concatenate([_hist(S["dat"], w, DAT_EDGES), _hist(S["dpt"], w, DPT_EDGES)])


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    N = 400_000
    true_p = jnp.array([1.30, 0.80]); init_p = jnp.array([1.00, 1.00]); NAMES = ["sigma_abs", "sigma_scatter"]
    S_data = proposal(jax.random.PRNGKey(1), N)
    data = jax.lax.stop_gradient(model(true_p, S_data)); log(f"data at truth {np.asarray(true_p)}")

    def loss(theta, key):
        S1 = proposal(key, N); S2 = proposal(jax.random.fold_in(key, 1), N)
        inv = 1.0 / (data + 1.0)
        return jnp.mean((model(theta, S1) - data) * (model(theta, S2) - data) * inv)

    vg = jax.jit(jax.value_and_grad(loss)); tc = time.time()
    l0, g0 = vg(init_p, jax.random.PRNGKey(2)); log(f"compiled {time.time()-tc:.1f}s loss0={float(l0):.3e} grad0={np.asarray(g0)}")
    fd = [float((loss(init_p + jnp.zeros(2).at[i].set(2e-3), jax.random.PRNGKey(2))
                 - loss(init_p - jnp.zeros(2).at[i].set(2e-3), jax.random.PRNGKey(2))) / 4e-3) for i in range(2)]
    log(f"grad check AD={np.asarray(g0)} FD={np.array(fd)}")

    theta = init_p; lr = 0.03; m = jnp.zeros(2); v = jnp.zeros(2); traj = [np.asarray(theta)]
    for it in range(400):
        l, g = vg(theta, jax.random.PRNGKey(100 + it))
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g ** 2
        mh = m / (1 - 0.9 ** (it + 1)); vh = v / (1 - 0.999 ** (it + 1))
        theta = jnp.clip(theta - lr * mh / (jnp.sqrt(vh) + 1e-8), 0.3, 3.0); traj.append(np.asarray(theta))
        if it % 50 == 0 or it == 399:
            log(f"  it {it:3d} loss={float(l):.3e} s_abs={float(theta[0]):.3f} s_sc={float(theta[1]):.3f}")
    traj = np.array(traj); fit = np.asarray(theta)
    for i, nm in enumerate(NAMES):
        log(f"  {nm:>14s}: init {float(init_p[i]):.3f} -> fit {fit[i]:.3f} (truth {float(true_p[i]):.3f}, |err| {abs(fit[i]-float(true_p[i])):.3f})")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    nA = len(DAT_EDGES) - 1
    d = np.asarray(data); ii = np.asarray(model(init_p, S_data)); ff = np.asarray(model(jnp.asarray(fit), S_data))
    Ac = 0.5 * (DAT_EDGES[1:] + DAT_EDGES[:-1]); Pc = 0.5 * (DPT_EDGES[1:] + DPT_EDGES[:-1])
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.9))
    for a, xc, sl, xl, ti in [(ax[0], Ac, slice(0, nA), r"$\delta\alpha_T$ [rad]", "$\\delta\\alpha_T$"),
                              (ax[1], Pc, slice(nA, None), r"$\delta p_T$ [MeV]", "$\\delta p_T$")]:
        a.step(xc, d[sl], where="mid", color="k", lw=2, label="pseudo-data (truth)")
        a.step(xc, ii[sl], where="mid", color="C3", ls="--", lw=1.5, label="initial")
        a.step(xc, ff[sl], where="mid", color="C0", lw=1.5, label="final (fit)")
        a.set(xlabel=xl, ylabel="CC0$\\pi$ counts", title=ti); a.legend(fontsize=8); a.set_ylim(bottom=0)
    for i, nm in enumerate(NAMES):
        ax[2].plot(traj[:, i], lw=1.8, label=f"{nm} -> {fit[i]:.3f}")
        ax[2].axhline(float(true_p[i]), ls="--", color="k", lw=1, alpha=0.6)
    ax[2].set(xlabel="iteration", ylabel="parameter", title="recovery (truth dashed)"); ax[2].legend(fontsize=8)
    fig.suptitle("CC0$\\pi$-Np closure on the T2K STV observables ($\\delta\\alpha_T$, $\\delta p_T$ on T2K bins): "
                 "recover ($\\sigma_{abs}$, $\\sigma_{scat}$)")
    fig.tight_layout(); fig.savefig("paper_figures/cc0pi_closure_tki.png", dpi=120); print("wrote paper_figures/cc0pi_closure_tki.png")


if __name__ == "__main__":
    main()
