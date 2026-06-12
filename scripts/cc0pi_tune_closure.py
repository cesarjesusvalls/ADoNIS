"""CLOSURE of the real-chain T2K tune: pseudo-data = ADoNIS at MODIFIED knobs
theta* = (sigma_abs*, sigma_scatter*, MA*), generated from INDEPENDENT cascade walks; then
the exact tune machinery (cc0pi_tune_adonis: replica bank, two-replica chi2 with the T2K
covariance, Adam, Hessian errors) runs from nominal and must (a) recover theta* within
errors and (b) move the prediction onto the pseudo-data.

The optimisation HISTORY (trajectory, histograms, BFP, Hessian, pseudo-data) is saved to
/tmp/adonis_tune_runs/<name>.npz so the figure can be re-rendered without re-running:

  python scripts/cc0pi_tune_closure.py [dpt|dat] [sabs*] [sscat*] [MA*_GeV] [--noA]
  python scripts/cc0pi_tune_closure.py --plot-only /tmp/adonis_tune_runs/<name>.npz

(default dpt 1.6 0.7 1.0; MA* != 1.0 engages the exact quadratic M_A reweight as a 3rd
knob; --noA = absolute normalization, no profiled A.)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

RUN_DIR = "/tmp/adonis_tune_runs"
NAMES = ("sigma_abs", "sigma_scatter", "M_A")
NREP, NPSE, NITERS = 8, 4, 400


def run():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import scripts.cc0pi_tune_adonis as T        # observable picked by argv[1] there too

    th_true = np.array([float(sys.argv[2]) if len(sys.argv) > 2 else 1.6,
                        float(sys.argv[3]) if len(sys.argv) > 3 else 0.7,
                        float(sys.argv[4]) if len(sys.argv) > 4 else 1.0])
    profile_a = "--noA" not in sys.argv
    clip_lo = jnp.array([0.3, 0.3, 0.7]); clip_hi = jnp.array([3.0, 3.0, 1.5])

    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    log(f"closure on {T.OBS}: theta* = {th_true}  norm={'profiled' if profile_a else 'ABSOLUTE'}")
    qe, qw, res, rw = T.build_proposal(); log("proposal sampled")
    M = T.build_ma_records(qe, res); log("M_A records built (quadratic amps2 decomposition)")

    # pseudo-data: ADoNIS at theta*, from cascade walks INDEPENDENT of the fit bank
    pse = []
    for i in range(NPSE):
        R = T.build_replica(jax.random.PRNGKey(200 + i), qe, qw, res, rw)
        pse.append(np.asarray(T.model_hist(jnp.asarray(th_true), R, M)))
        log(f"  pseudo-data replica {i+1}/{NPSE}")
    dstar = np.mean(pse, axis=0); DSTAR = jnp.asarray(dstar)

    bank = []
    for r_i in range(NREP):
        bank.append(T.build_replica(jax.random.PRNGKey(50 + r_i), qe, qw, res, rw))
        log(f"  fit replica {r_i+1}/{NREP}")

    nom = np.mean([np.asarray(T.model_hist(jnp.array([1.0, 1.0, 1.0]), R, M)) for R in bank], axis=0)
    A = (float((jnp.asarray(nom) @ T.COVINV @ DSTAR) / (jnp.asarray(nom) @ T.COVINV @ jnp.asarray(nom)))
         if profile_a else 1.0)
    r0 = A * nom - dstar; chi2_nom = float(jnp.asarray(r0) @ T.COVINV @ jnp.asarray(r0))
    log(f"NOMINAL vs pseudo-data: chi2/ndf = {chi2_nom/(8-2):.2f}  (A_nom={A:.3f})")

    def loss(theta, R1, R2):
        r1 = A * T.model_hist(theta, R1, M) - DSTAR; r2 = A * T.model_hist(theta, R2, M) - DSTAR
        return r1 @ T.COVINV @ r2
    vg = jax.jit(jax.value_and_grad(loss))
    theta = jnp.array([1.0, 1.0, 1.0]); m = jnp.zeros(3); v = jnp.zeros(3); lr = 0.02
    traj = [np.asarray(theta)]; losses = []
    for it in range(NITERS):
        off = 1 + (it // NREP) % (NREP - 1)
        l, g = vg(theta, bank[it % NREP], bank[(it + off) % NREP])
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g ** 2
        mh = m / (1 - 0.9 ** (it + 1)); vh = v / (1 - 0.999 ** (it + 1))
        theta = jnp.clip(theta - lr * mh / (jnp.sqrt(vh) + 1e-8), clip_lo, clip_hi)
        traj.append(np.asarray(theta)); losses.append(float(l))
        if it % 50 == 0 or it == NITERS - 1:
            log(f"  it {it:3d}/{NITERS}  chi2/ndf={float(l)/(8-2):.2f}  s_abs={float(theta[0]):.3f} "
                f"s_sc={float(theta[1]):.3f} MA={float(theta[2]):.3f}")
    bfp = np.asarray(theta); traj = np.array(traj)

    def chi2_1(theta, R):
        r = A * T.model_hist(theta, R, M) - DSTAR; return r @ T.COVINV @ r
    chi2_bf = float(np.mean([float(chi2_1(theta, R)) for R in bank]))
    H = np.mean([np.asarray(jax.hessian(chi2_1)(theta, R)) for R in bank], axis=0)
    V = 2.0 * np.linalg.inv(H); sig = np.sqrt(np.clip(np.diag(V), 0.0, None))
    pull = (bfp - th_true) / np.where(sig > 0, sig, np.inf)
    for i, nm in enumerate(NAMES):
        log(f"  {nm:>14s} = {bfp[i]:.4f} +/- {sig[i]:.4f}   (true {th_true[i]:.3f}, pull {pull[i]:+.2f}sig)")
    log(f"chi2/ndf {chi2_nom/(8-2):.2f} -> {chi2_bf/(8-2):.2f}")
    mb = A * np.mean([np.asarray(T.model_hist(jnp.asarray(bfp), R, M)) for R in bank], axis=0)

    os.makedirs(RUN_DIR, exist_ok=True)
    tag = ("" if th_true[2] == 1.0 else "_ma") + ("" if profile_a else "_abs")
    npz = f"{RUN_DIR}/closure_{T.OBS}{tag}.npz"
    np.savez(npz, obs=T.OBS, th_true=th_true, profile_a=profile_a, A=A, dstar=dstar, nom=nom,
             mb=mb, traj=traj, losses=np.array(losses), bfp=bfp, sig=sig, V=V, chi2_nom=chi2_nom,
             chi2_bf=chi2_bf, edges=np.asarray(T.EDGES), derr=np.sqrt(np.diag(np.asarray(T.COV))))
    log(f"history saved -> {npz}")
    make_figure(npz)


def make_figure(npz_path):
    """Render the closure figure from a saved run history (cosmetics-only re-runs)."""
    d = np.load(npz_path, allow_pickle=True)
    obs = str(d["obs"]); th_true = d["th_true"]; profile_a = bool(d["profile_a"]); A = float(d["A"])
    dstar, nom, mb, traj, bfp, sig, V = d["dstar"], d["nom"], d["mb"], d["traj"], d["bfp"], d["sig"], d["V"]
    chi2_nom, chi2_bf, edges, derr = float(d["chi2_nom"]), float(d["chi2_bf"]), d["edges"], d["derr"]

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    xsc = 1000.0 if obs == "dpt" else 1.0
    ctr = 0.5 * (edges[1:] + edges[:-1]) / xsc; xed = edges / xsc
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    ax[0].errorbar(ctr, dstar, yerr=derr, fmt="o", color="k", capsize=3,
                   label=f"pseudo-data: ADoNIS @ $\\theta^*$=({th_true[0]}, {th_true[1]}, {th_true[2]})")
    if profile_a:
        ax[0].step(xed, np.append(nom, nom[-1]), where="post", color="0.45", lw=1.4, ls=":",
                   label="pre-tune ADoNIS (absolute, A=1)")
    norm_lbl = f" $\\times$ profiled A={A:.2f}" if profile_a else " (absolute)"
    ax[0].step(xed, np.append(A * nom, (A * nom)[-1]), where="post", color="C2", lw=2,
               label=f"pre-tune{norm_lbl}  $\\chi^2$/ndf {chi2_nom/(8-2):.1f}")
    ax[0].step(xed, np.append(mb, mb[-1]), where="post", color="C0", lw=1.6, ls="--",
               label=f"post-tune  $\\chi^2$/ndf {chi2_bf/(8-2):.1f}")
    XL = r"$\delta p_T$ [GeV/c]" if obs == "dpt" else r"$\delta\alpha_T$ [rad]"
    ax[0].set(xlabel=XL, ylabel="d$\\sigma$/dx [data units]",
              title=f"closure on {obs} (T2K covariance, norm {'profiled' if profile_a else 'absolute'})")
    ax[0].legend(fontsize=8); ax[0].set_ylim(bottom=0)
    ax[1].plot(traj[:, 0], traj[:, 1], "-", color="0.6", lw=1, label="Adam trajectory")
    ax[1].plot(1, 1, "s", color="0.5", ms=9, label="start (nominal)")
    ax[1].plot(th_true[0], th_true[1], "*", color="C3", ms=16, label="truth $\\theta^*$")
    ax[1].plot(bfp[0], bfp[1], "X", color="k", ms=10, label="best fit")
    V2 = V[:2, :2]
    if np.all(np.diag(V2) > 0):
        ew, evec = np.linalg.eigh(V2); th_ = np.linspace(0, 2 * np.pi, 100)
        for k, c in ((1, "C3"), (2, "C0")):
            xy = (evec @ (np.sqrt(np.clip(ew, 0, None))[:, None] * np.array([np.cos(th_), np.sin(th_)])) * k)
            ax[1].plot(bfp[0] + xy[0], bfp[1] + xy[1], color=c, lw=1.4)
    ax[1].set(xlabel=r"$\sigma_{abs}$", ylabel=r"$\sigma_{scatter}$", title="parameter recovery")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle(f"REAL-chain tune CLOSURE on {obs}: recover $\\theta^*$ from nominal")
    tag = ("" if th_true[2] == 1.0 else "_ma") + ("" if profile_a else "_abs")
    FIG = f"paper_figures/cc0pi_tune_closure_{obs}{tag}.png"
    fig.tight_layout(); fig.savefig(FIG, dpi=120); print(f"wrote {FIG}", flush=True)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        make_figure(sys.argv[sys.argv.index("--plot-only") + 1])
    else:
        run()
