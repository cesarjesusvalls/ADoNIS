"""MULTI-MECHANISM CLOSURE of the differentiable T2K CC0pi tune via model_hist_full.

Where closure.py recovers theta* = (sigma_abs, sigma_scatter, M_A) through the legacy FSI+M_A
model_hist, THIS closure exercises the full-knob path (full_knobs.model_hist_full): it perturbs and
recovers knobs that span ALL THREE reweight mechanisms at once --
  * hard vertex (amps2 quadratic) : M_A
  * FSI (kind-1 record)           : sabs
  * spectral function (density)   : kF_sf
Pseudo-data = ADoNIS at theta* from INDEPENDENT cascade walks; the exact tune machinery (replica bank,
two-replica chi2 with the T2K covariance, Adam, Hessian errors) runs from nominal and must (a) recover
theta* within errors and (b) move the prediction onto the pseudo-data -- proving model_hist_full is a
usable fit predictor, not just gradient-correct.

  python analysis/t2k/differentiability/closure_full.py [dpt|dat] [--noA]
  python analysis/t2k/differentiability/closure_full.py --plot-only /tmp/adonis_tune_runs/<name>.npz

The fit-knob set + truth + clips live in FIT (edit there to extend).  History -> /tmp/adonis_tune_runs.
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np

RUN_DIR = "/tmp/adonis_tune_runs"
NREP, NPSE, NITERS = 8, 4, 400

# fit-knob registry: name -> (truth value theta*, clip_lo, clip_hi, Adam lr).  One per mechanism.
FIT = [
    ("M_A",   1.10, 0.70, 1.50, 0.010),   # hard vertex (QE+RES amps2 quadratic)
    ("sabs",  1.30, 0.30, 3.00, 0.020),   # FSI pion absorption (kind-1)
    ("kF_sf", 1.10, 0.80, 1.30, 0.010),   # spectral function (density ratio)
]
NAMES = [f[0] for f in FIT]


def run():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from analysis.t2k.differentiability import tune as T
    from analysis.t2k.differentiability.full_knobs import nominal_knobs, build_hv_sf, model_hist_full
    from adonis.workflow.materials import resolve_targets
    from adonis.xsec.spectral import SpectralFunction

    T.NQE = T.NRES = int(os.environ.get("CC0PI_N", "40000"))
    profile_a = "--noA" not in sys.argv
    NOM = nominal_knobs()
    th_true = np.array([f[1] for f in FIT])
    clip_lo = jnp.array([f[2] for f in FIT]); clip_hi = jnp.array([f[3] for f in FIT])
    lrv = jnp.array([f[4] for f in FIT])
    nb = len(np.asarray(T.EDGES)) - 1; ndf = nb - len(FIT)
    EDGES = np.asarray(T.EDGES); CONV = T.CONV; COVINV = T.COVINV

    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    log(f"full-knob closure on {T.OBS}: fit {NAMES}  theta*={th_true}  norm={'profiled' if profile_a else 'ABSOLUTE'}")

    qe, qw, res, rw = T.build_proposal(); log("proposal sampled")
    sf = SpectralFunction(resolve_targets("C")[0][0].spectral_n)
    HV, SF = build_hv_sf(qe, res, sf); log("hard-vertex amps2 records + SF grids built")

    def knobs_of(theta):
        k = dict(NOM)
        for i, nm in enumerate(NAMES):
            k[nm] = theta[i]
        return k

    def hist(theta, R):
        return model_hist_full(knobs_of(theta), R, HV, SF, EDGES, CONV)

    # pseudo-data: ADoNIS at theta*, INDEPENDENT walks from the fit bank
    th_true_j = jnp.asarray(th_true)
    pse = []
    for i in range(NPSE):
        R = T.build_replica(jax.random.PRNGKey(200 + i), qe, qw, res, rw)
        pse.append(np.asarray(hist(th_true_j, R)))
        log(f"  pseudo-data replica {i+1}/{NPSE}")
    dstar = np.mean(pse, axis=0); DSTAR = jnp.asarray(dstar)

    bank = []
    for r_i in range(NREP):
        bank.append(T.build_replica(jax.random.PRNGKey(50 + r_i), qe, qw, res, rw))
        log(f"  fit replica {r_i+1}/{NREP}")

    one = jnp.ones(len(FIT))
    nom = np.mean([np.asarray(hist(one, R)) for R in bank], axis=0)
    A = (float((jnp.asarray(nom) @ COVINV @ DSTAR) / (jnp.asarray(nom) @ COVINV @ jnp.asarray(nom)))
         if profile_a else 1.0)
    r0 = A * nom - dstar; chi2_nom = float(jnp.asarray(r0) @ COVINV @ jnp.asarray(r0))
    log(f"NOMINAL vs pseudo-data: chi2/ndf = {chi2_nom/ndf:.2f}  (A_nom={A:.3f})")

    def loss(theta, R1, R2):
        r1 = A * hist(theta, R1) - DSTAR; r2 = A * hist(theta, R2) - DSTAR
        return r1 @ COVINV @ r2
    vg = jax.jit(jax.value_and_grad(loss))
    theta = jnp.ones(len(FIT)); m = jnp.zeros(len(FIT)); v = jnp.zeros(len(FIT))
    traj = [np.asarray(theta)]; losses = []
    for it in range(NITERS):
        off = 1 + (it // NREP) % (NREP - 1)
        l, g = vg(theta, bank[it % NREP], bank[(it + off) % NREP])
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g ** 2
        mh = m / (1 - 0.9 ** (it + 1)); vh = v / (1 - 0.999 ** (it + 1))
        theta = jnp.clip(theta - lrv * mh / (jnp.sqrt(vh) + 1e-8), clip_lo, clip_hi)
        traj.append(np.asarray(theta)); losses.append(float(l))
        if it % 50 == 0 or it == NITERS - 1:
            log(f"  it {it:3d}/{NITERS}  chi2/ndf={float(l)/ndf:.2f}  " +
                " ".join(f"{nm}={float(theta[i]):.3f}" for i, nm in enumerate(NAMES)))
    bfp = np.asarray(theta); traj = np.array(traj)

    def chi2_1(theta, R):
        r = A * hist(theta, R) - DSTAR; return r @ COVINV @ r
    chi2_bf = float(np.mean([float(chi2_1(theta, R)) for R in bank]))
    H = np.mean([np.asarray(jax.hessian(chi2_1)(theta, R)) for R in bank], axis=0)
    V = 2.0 * np.linalg.inv(H); sig = np.sqrt(np.clip(np.diag(V), 0.0, None))
    pull = (bfp - th_true) / np.where(sig > 0, sig, np.inf)
    for i, nm in enumerate(NAMES):
        log(f"  {nm:>8s} = {bfp[i]:.4f} +/- {sig[i]:.4f}   (true {th_true[i]:.3f}, pull {pull[i]:+.2f}sig)")
    log(f"chi2/ndf {chi2_nom/ndf:.2f} -> {chi2_bf/ndf:.2f}")
    mb = A * np.mean([np.asarray(hist(jnp.asarray(bfp), R)) for R in bank], axis=0)

    os.makedirs(RUN_DIR, exist_ok=True)
    tag = "" if profile_a else "_abs"
    npz = f"{RUN_DIR}/closure_full_{T.OBS}{tag}.npz"
    np.savez(npz, obs=T.OBS, names=np.array(NAMES), th_true=th_true, profile_a=profile_a, A=A,
             dstar=dstar, nom=nom, mb=mb, traj=traj, losses=np.array(losses), bfp=bfp, sig=sig, V=V,
             chi2_nom=chi2_nom, chi2_bf=chi2_bf, ndf=ndf, edges=EDGES, derr=np.sqrt(np.diag(np.asarray(T.COV))))
    log(f"history saved -> {npz}")
    make_figure(npz)


def make_figure(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    obs = str(d["obs"]); names = list(d["names"]); th_true = d["th_true"]; profile_a = bool(d["profile_a"])
    A = float(d["A"]); dstar, nom, mb, traj, bfp, sig, V = d["dstar"], d["nom"], d["mb"], d["traj"], d["bfp"], d["sig"], d["V"]
    chi2_nom, chi2_bf, ndf, edges, derr = float(d["chi2_nom"]), float(d["chi2_bf"]), int(d["ndf"]), d["edges"], d["derr"]

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    xsc = 1000.0 if obs == "dpt" else 1.0
    ctr = 0.5 * (edges[1:] + edges[:-1]) / xsc; xed = edges / xsc
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    tlbl = ", ".join(f"{n}={th_true[i]:.2f}" for i, n in enumerate(names))
    ax[0].errorbar(ctr, dstar, yerr=derr, fmt="o", color="k", capsize=3, label=f"pseudo-data: ADoNIS @ ({tlbl})")
    if profile_a:
        ax[0].step(xed, np.append(nom, nom[-1]), where="post", color="0.45", lw=1.4, ls=":", label="pre-tune (absolute, A=1)")
    norm_lbl = f" $\\times$ A={A:.2f}" if profile_a else " (absolute)"
    ax[0].step(xed, np.append(A * nom, (A * nom)[-1]), where="post", color="C2", lw=2,
               label=f"pre-tune{norm_lbl}  $\\chi^2$/ndf {chi2_nom/ndf:.1f}")
    ax[0].step(xed, np.append(mb, mb[-1]), where="post", color="C0", lw=1.6, ls="--",
               label=f"post-tune  $\\chi^2$/ndf {chi2_bf/ndf:.1f}")
    XL = r"$\delta p_T$ [GeV/c]" if obs == "dpt" else r"$\delta\alpha_T$ [rad]"
    ax[0].set(xlabel=XL, ylabel="d$\\sigma$/dx [data units]",
              title=f"full-knob closure on {obs} (norm {'profiled' if profile_a else 'absolute'})")
    ax[0].legend(fontsize=8); ax[0].set_ylim(bottom=0)

    nb = len(bfp); x = np.arange(nb)
    ax[1].axhline(1.0, color="0.7", lw=1)
    ax[1].errorbar(x, bfp, yerr=sig, fmt="X", color="k", ms=10, capsize=4, label="best fit $\\pm 1\\sigma$")
    ax[1].plot(x, th_true, "*", color="C3", ms=16, label="truth $\\theta^*$")
    ax[1].plot(x, np.ones(nb), "s", color="0.5", ms=8, label="start (nominal)")
    ax[1].set_xticks(x); ax[1].set_xticklabels(names)
    ax[1].set(ylabel="knob value", title="multi-mechanism parameter recovery")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle(f"model_hist_full CLOSURE on {obs}: recover $\\theta^*$ across hard-vertex + FSI + SF")
    tag = "" if profile_a else "_abs"
    os.makedirs("output/figures", exist_ok=True)
    FIG = f"output/figures/cc0pi_tune_closure_full_{obs}{tag}.png"
    fig.tight_layout(); fig.savefig(FIG, dpi=120); print(f"wrote {FIG}", flush=True)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        make_figure(sys.argv[sys.argv.index("--plot-only") + 1])
    else:
        run()
