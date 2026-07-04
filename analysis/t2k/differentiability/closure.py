"""PSEUDO-DATA CLOSURES of the differentiable T2K CC0pi tune: pseudo-data = ADoNIS at MODIFIED knobs
theta* from INDEPENDENT cascade walks; the exact tune machinery (replica bank, two-replica chi2 with the
T2K covariance, Adam, Hessian errors) runs from nominal and must (a) recover theta* within errors and
(b) move the prediction onto the pseudo-data.

Two modes (merged from the former closure.py / closure_full.py):

  legacy 3-param (default) -- theta* = (sigma_abs, sigma_scatter, M_A) through tune.model_hist:
     CC0PI_N=40000 python analysis/t2k/differentiability/closure.py [dpt|dat] [sabs*] [sscat*] [MA*] [--noA]
     (defaults dpt 1.6 0.7 1.0; MA* != 1.0 engages the exact quadratic M_A reweight; --noA = absolute norm)

  full-knob (--full) -- knobs spanning ALL reweight mechanisms via full_knobs.model_hist_full; the knob
  set + truths + clips live in the FIT registry below (edit there to extend):
     CC0PI_N=40000 python analysis/t2k/differentiability/closure.py --full [dpt|dat] [--noA]

  re-render a saved history:  python analysis/t2k/differentiability/closure.py --plot-only <npz>
  (closure_full npz's carry a 'names' key and are dispatched to the full-mode figure automatically.)

Histories -> /tmp/adonis_tune_runs/closure*.npz; figures -> output/figures/cc0pi_tune_closure*.png.
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root
import numpy as np

RUN_DIR = "/tmp/adonis_tune_runs"
NAMES3 = ("sigma_abs", "sigma_scatter", "M_A")
NREP, NPSE, NITERS = 8, 4, 400

# --full fit-knob registry: name -> (truth value theta*, clip_lo, clip_hi, Adam lr).  One per mechanism.
FIT = [
    ("M_A_qe",  1.10, 0.70, 1.50, 0.010),   # hard vertex QE (amps2 quadratic; M_A split per channel)
    ("M_A_res", 0.90, 0.70, 1.50, 0.010),   # hard vertex RES (independent axial dipole mass)
    ("sabs",  1.30, 0.30, 3.00, 0.020),   # FSI pion absorption (kind-1)
    ("kF_sf", 1.10, 0.80, 1.30, 0.010),   # spectral function (density ratio)
]
NAMES_FULL = [f[0] for f in FIT]


# ================================================================ legacy 3-param mode =================== #
def run_legacy():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from analysis.t2k.differentiability import tune as T   # observable picked by argv[1] there too
    T.NQE = T.NRES = int(os.environ.get("CC0PI_N", "40000"))
    _build, _hist = T.build_replica, T.model_hist

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
        R = _build(jax.random.PRNGKey(200 + i), qe, qw, res, rw)
        pse.append(np.asarray(_hist(jnp.asarray(th_true), R, M)))
        log(f"  pseudo-data replica {i+1}/{NPSE}")
    dstar = np.mean(pse, axis=0); DSTAR = jnp.asarray(dstar)

    bank = []
    for r_i in range(NREP):
        bank.append(_build(jax.random.PRNGKey(50 + r_i), qe, qw, res, rw))
        log(f"  fit replica {r_i+1}/{NREP}")

    nom = np.mean([np.asarray(_hist(jnp.array([1.0, 1.0, 1.0]), R, M)) for R in bank], axis=0)
    A = (float((jnp.asarray(nom) @ T.COVINV @ DSTAR) / (jnp.asarray(nom) @ T.COVINV @ jnp.asarray(nom)))
         if profile_a else 1.0)
    r0 = A * nom - dstar; chi2_nom = float(jnp.asarray(r0) @ T.COVINV @ jnp.asarray(r0))
    log(f"NOMINAL vs pseudo-data: chi2/ndf = {chi2_nom/(8-2):.2f}  (A_nom={A:.3f})")

    def loss(theta, R1, R2):
        r1 = A * _hist(theta, R1, M) - DSTAR; r2 = A * _hist(theta, R2, M) - DSTAR
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
        r = A * _hist(theta, R, M) - DSTAR; return r @ T.COVINV @ r
    chi2_bf = float(np.mean([float(chi2_1(theta, R)) for R in bank]))
    H = np.mean([np.asarray(jax.hessian(chi2_1)(theta, R)) for R in bank], axis=0)
    V = 2.0 * np.linalg.inv(H); sig = np.sqrt(np.clip(np.diag(V), 0.0, None))
    pull = (bfp - th_true) / np.where(sig > 0, sig, np.inf)
    for i, nm in enumerate(NAMES3):
        log(f"  {nm:>14s} = {bfp[i]:.4f} +/- {sig[i]:.4f}   (true {th_true[i]:.3f}, pull {pull[i]:+.2f}sig)")
    log(f"chi2/ndf {chi2_nom/(8-2):.2f} -> {chi2_bf/(8-2):.2f}")
    mb = A * np.mean([np.asarray(_hist(jnp.asarray(bfp), R, M)) for R in bank], axis=0)

    os.makedirs(RUN_DIR, exist_ok=True)
    tag = ("" if th_true[2] == 1.0 else "_ma") + ("" if profile_a else "_abs")
    npz = f"{RUN_DIR}/closure_{T.OBS}{tag}.npz"
    np.savez(npz, obs=T.OBS, th_true=th_true, profile_a=profile_a, A=A, dstar=dstar, nom=nom,
             mb=mb, traj=traj, losses=np.array(losses), bfp=bfp, sig=sig, V=V, chi2_nom=chi2_nom,
             chi2_bf=chi2_bf, edges=np.asarray(T.EDGES), derr=np.sqrt(np.diag(np.asarray(T.COV))))
    log(f"history saved -> {npz}")
    make_figure_legacy(npz)


def make_figure_legacy(npz_path):
    """Render the 3-param closure figure from a saved run history (cosmetics-only re-runs)."""
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
    os.makedirs("output/figures", exist_ok=True)
    FIG = f"output/figures/cc0pi_tune_closure_{obs}{tag}.png"
    fig.tight_layout(); fig.savefig(FIG, dpi=120); print(f"wrote {FIG}", flush=True)


# ================================================================ full-knob mode (--full) =============== #
def run_full():
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
    log(f"full-knob closure on {T.OBS}: fit {NAMES_FULL}  theta*={th_true}  norm={'profiled' if profile_a else 'ABSOLUTE'}")

    qe, qw, res, rw = T.build_proposal(); log("proposal sampled")
    sf = SpectralFunction(resolve_targets("C")[0][0].spectral_n)
    HV, SF = build_hv_sf(qe, res, sf); log("hard-vertex amps2 records + SF grids built")

    def knobs_of(theta):
        k = dict(NOM)
        for i, nm in enumerate(NAMES_FULL):
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
                " ".join(f"{nm}={float(theta[i]):.3f}" for i, nm in enumerate(NAMES_FULL)))
    bfp = np.asarray(theta); traj = np.array(traj)

    def chi2_1(theta, R):
        r = A * hist(theta, R) - DSTAR; return r @ COVINV @ r
    chi2_bf = float(np.mean([float(chi2_1(theta, R)) for R in bank]))
    H = np.mean([np.asarray(jax.hessian(chi2_1)(theta, R)) for R in bank], axis=0)
    V = 2.0 * np.linalg.inv(H); sig = np.sqrt(np.clip(np.diag(V), 0.0, None))
    pull = (bfp - th_true) / np.where(sig > 0, sig, np.inf)
    for i, nm in enumerate(NAMES_FULL):
        log(f"  {nm:>8s} = {bfp[i]:.4f} +/- {sig[i]:.4f}   (true {th_true[i]:.3f}, pull {pull[i]:+.2f}sig)")
    log(f"chi2/ndf {chi2_nom/ndf:.2f} -> {chi2_bf/ndf:.2f}")
    mb = A * np.mean([np.asarray(hist(jnp.asarray(bfp), R)) for R in bank], axis=0)

    os.makedirs(RUN_DIR, exist_ok=True)
    tag = "" if profile_a else "_abs"
    npz = f"{RUN_DIR}/closure_full_{T.OBS}{tag}.npz"
    np.savez(npz, obs=T.OBS, names=np.array(NAMES_FULL), th_true=th_true, profile_a=profile_a, A=A,
             dstar=dstar, nom=nom, mb=mb, traj=traj, losses=np.array(losses), bfp=bfp, sig=sig, V=V,
             chi2_nom=chi2_nom, chi2_bf=chi2_bf, ndf=ndf, edges=EDGES, derr=np.sqrt(np.diag(np.asarray(T.COV))))
    log(f"history saved -> {npz}")
    make_figure_full(npz)


def make_figure_full(npz_path):
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
        p = sys.argv[sys.argv.index("--plot-only") + 1]
        d = np.load(p, allow_pickle=True)
        (make_figure_full if "names" in d.files else make_figure_legacy)(p)
    elif "--full" in sys.argv:
        sys.argv.remove("--full")
        run_full()
    else:
        run_legacy()
