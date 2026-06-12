"""Tune the REAL ADoNIS CC0pi-Np prediction to the T2K delta_pT data, starting from its nominal.

The differentiable ADoNIS predictor: QE-C + RES-C CC0pi events are sampled ONCE (frozen proposal);
the pion cascade (sigma_abs, sigma_scatter) and nucleon cascade (sigma_scatter) carry kind-1 weights,
so dsigma/d(delta_pT)(theta) is differentiable in (sigma_abs, sigma_scatter) and at theta=(1,1)
reproduces the nominal forward prediction (the chi^2/ndf 1.44 curve) bit-for-bit -- NO toy, NO free
normalization.  Covariance-weighted two-replica chi^2; Hessian parameter covariance at the BFP.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, uproot
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import adonis.xsec.dcc_current as dcc; dcc.BATCH_INTERP = "spline"
from adonis.xsec import qe_xsec, res_xsec
from adonis.core.event import EventRecord
from adonis.fsi.cascade_discrete import DiscreteCascadeFSI, DiscreteNucleonFSI, DiscreteCascadeConfig

CFG = lambda **k: DiscreteCascadeConfig(cylinder=False, step=0.04, max_steps=260, **k)   # Gaussian (diff'able)
# max_steps 260 (10.4 fm) is saturation-verified == 325; fast_xsec=True (default) slab-restricts the
# Oset/DCC cross sections (bit-exact, ~1.15x).  Together ~1.4x vs the original 325/dense.
MU_LO, COSMU, P_LO, P_HI, COSP = 250.0, -0.6, 450.0, 1000.0, 0.4
NQE, NRES = 120000, 120000
CONV = 1e-33 / 12.0 * 1000.0 * 1e38                       # nb/MeV per-12C -> 1e-38 cm^2/(GeV/c)/nucleon

# ---- T2K delta_pT data + covariance (1e-38 units) -------------------------------------------- #
_r = uproot.open("../nuisance/data/T2K/CC0pi/STV/dptResults.root")
EDGES = np.asarray(_r["Result"].axis().edges()) * 1000.0    # MeV
DATA = jnp.asarray(np.asarray(_r["Result"].values()) * 1e38)
D_ERR = np.asarray(_r["Result"].errors()) * 1e38
COV = np.asarray(_r["Covariance_Matrix"].values())
COVINV = jnp.asarray(np.linalg.inv(COV + 1e-12 * np.eye(8)))


def _mk_ev(d, pid_Ni, p_N, p_pi, ppid):
    m = len(d["w"])
    return EventRecord(k=jnp.asarray(d["k_nu"]), kp=jnp.asarray(d["k_mu"]), p_struck=jnp.asarray(d["p_struck"]),
                       p_pi=jnp.asarray(p_pi), p_N=jnp.asarray(p_N), w=jnp.ones(m),
                       channel=jnp.zeros(m, jnp.int32), pid_pi=jnp.asarray(ppid, jnp.int32),
                       pid_N=jnp.full((m,), 2212, jnp.int32), pid_Ni=jnp.asarray(pid_Ni, jnp.int32),
                       W=jnp.zeros(m), Q2_adj=jnp.zeros(m))


def _dpt(kmu, lead):
    lt = kmu[:, 1:3]; dv = lt + lead[:, 1:3]
    return jnp.linalg.norm(dv, axis=1)


def _sel(kmu, lead):
    pmu = jnp.linalg.norm(kmu[:, 1:], axis=1); cmu = kmu[:, 3] / jnp.clip(pmu, 1e-9, None)
    pl = jnp.linalg.norm(lead[:, 1:], axis=1); cl = lead[:, 3] / jnp.clip(pl, 1e-9, None)
    return (pmu > MU_LO) & (cmu > COSMU) & (pl > P_LO) & (pl < P_HI) & (cl > COSP)


def build_proposal():
    """Sample QE + RES CC0pi events ONCE (frozen)."""
    qe = qe_xsec.sample_importance(NQE, seed=0)
    qw = np.asarray(qe["w"]) / NQE
    res = res_xsec.generate(NRES, seed=0, return_events=True)["events"]
    rw = np.asarray(res["w"])
    return qe, qw, res, rw


def hist_nb(theta, kcasc, qe, qw, res, rw):
    """Differentiable CC0pi dsigma/ddpt [1e-38 units] for theta=(sabs, sscat); kcasc fixes the
    sampled cascade (frozen proposal) so only the knobs move."""
    sabs, sscat = theta[0], theta[1]
    k1, k2, k3 = jax.random.split(kcasc, 3)
    # --- QE: proton through the nucleon cascade (sigma_scatter) ---
    qev = _mk_ev(qe, np.full(NQE, 2112), qe["p_out"], np.zeros((NQE, 4)), np.zeros(NQE))
    nf = DiscreteNucleonFSI(CFG(seed=2)); qev2 = nf.apply(None, qev, key=k1, sscat=sscat)
    q_lead = qev2.p_N; q_w = jnp.asarray(qw) * nf.last_w_scat
    q_dpt = _dpt(jnp.asarray(qe["k_mu"]), q_lead); q_keep = _sel(jnp.asarray(qe["k_mu"]), q_lead)
    # --- RES: pion cascade (absorb) then nucleon cascade ---
    rev = _mk_ev(res, np.asarray(res["ipid"]), res["p_N"], res["p_pi"], np.asarray(res["ppid"]))
    pion = DiscreteCascadeFSI(CFG(seed=1)); rev2 = pion.apply(None, rev, key=k2, sabs=sabs, sscat=sscat)
    absb = pion.last_absorbed.astype(float); abs_p = pion.last_abs_proton; w_fsi = pion.last_w_fsi
    nf2 = DiscreteNucleonFSI(CFG(seed=2)); rev3 = nf2.apply(None, rev2, key=k3, sscat=sscat)
    pNf = rev3.p_N; mom_p = jnp.linalg.norm(pNf[:, 1:], axis=1) * (rev3.pid_N == 2212)
    mom_a = jnp.linalg.norm(abs_p[:, 1:], axis=1)
    r_lead = jnp.where((mom_a > mom_p)[:, None], abs_p, pNf)
    has_p = (mom_a > 1) | (rev3.pid_N == 2212)
    r_w = jnp.asarray(rw) * w_fsi * nf2.last_w_scat * absb * has_p.astype(float)
    r_dpt = _dpt(jnp.asarray(res["k_mu"]), r_lead); r_keep = _sel(jnp.asarray(res["k_mu"]), r_lead)
    # --- histogram (absolute nb -> data units) ---
    def H(x, w, keep):
        idx = jnp.clip(jnp.searchsorted(jnp.asarray(EDGES), x) - 1, 0, len(EDGES) - 2)
        return jax.ops.segment_sum(w * keep, jax.lax.stop_gradient(idx), num_segments=len(EDGES) - 1)
    h_nb_per_MeV = (H(q_dpt, q_w, q_keep) + H(r_dpt, r_w, r_keep)) / jnp.diff(jnp.asarray(EDGES))
    return h_nb_per_MeV * CONV


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    qe, qw, res, rw = build_proposal(); log("proposal sampled")
    nom = np.asarray(hist_nb(jnp.array([1.0, 1.0]), jax.random.PRNGKey(0), qe, qw, res, rw))
    A = float((jnp.asarray(nom) @ COVINV @ DATA) / (jnp.asarray(nom) @ COVINV @ jnp.asarray(nom)))
    r = A * nom - np.asarray(DATA); chi2_nom = float(jnp.asarray(r) @ COVINV @ jnp.asarray(r))
    log(f"NOMINAL chi2/ndf = {chi2_nom/(8-2):.2f}  (A_nom={A:.3f})")

    def loss(theta, key):
        k1, k2 = jax.random.split(key)
        m1 = A * hist_nb(theta, k1, qe, qw, res, rw); m2 = A * hist_nb(theta, k2, qe, qw, res, rw)
        r1 = m1 - DATA; r2 = m2 - DATA
        return r1 @ COVINV @ r2
    vg = jax.jit(jax.value_and_grad(loss))
    l0, g0 = vg(jnp.array([1.0, 1.0]), jax.random.PRNGKey(1)); log(f"loss(nominal)={float(l0):.2f} grad={np.asarray(g0)}")

    NITERS = 150
    theta = jnp.array([1.0, 1.0]); m = jnp.zeros(2); v = jnp.zeros(2); lr = 0.02; traj = [np.asarray(theta)]
    t_loop = time.time(); per = []
    for it in range(NITERS):
        ti = time.time()
        l, g = vg(theta, jax.random.PRNGKey(50 + it))
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g ** 2
        mh = m / (1 - 0.9 ** (it + 1)); vh = v / (1 - 0.999 ** (it + 1))
        theta = jnp.clip(theta - lr * mh / (jnp.sqrt(vh) + 1e-8), 0.3, 3.0); traj.append(np.asarray(theta))
        dt = time.time() - ti; per.append(dt)
        # the first iter pays jit-compile; forecast from the median of the steady-state iters
        med = float(np.median(per[1:])) if len(per) > 1 else dt
        eta = med * (NITERS - it - 1)
        log(f"  it {it:3d}/{NITERS}  chi2/ndf={float(l)/(8-2):.2f}  s_abs={float(theta[0]):.3f} s_sc={float(theta[1]):.3f}  "
            f"[{dt:5.1f}s/it  median {med:5.1f}s  ETA {eta/60:4.1f}min]")
    bfp = np.asarray(theta); traj = np.array(traj)
    log(f"loop done in {(time.time()-t_loop)/60:.1f}min (median {float(np.median(per[1:])):.1f}s/it)")

    def chi2_1(theta, key):
        r = A * hist_nb(theta, key, qe, qw, res, rw) - DATA; return r @ COVINV @ r
    NBF, NHES = 8, 12
    tb = time.time()
    cbf = []
    for i in range(NBF):
        cbf.append(float(chi2_1(theta, jax.random.PRNGKey(900 + i))))
        log(f"  chi2_bf eval {i+1}/{NBF}  ({(time.time()-tb)/max(i+1,1):.1f}s/eval, ETA {(time.time()-tb)/(i+1)*(NBF-i-1)/60:.1f}min)")
    chi2_bf = float(np.mean(cbf))
    th = time.time(); Hs = []
    for i in range(NHES):
        Hs.append(np.asarray(jax.hessian(chi2_1)(theta, jax.random.PRNGKey(700 + i))))
        per_h = (time.time() - th) / (i + 1)
        log(f"  hessian eval {i+1}/{NHES}  ({per_h:.1f}s/eval, ETA {per_h*(NHES-i-1)/60:.1f}min)")
    H = np.mean(Hs, axis=0)
    V = 2.0 * np.linalg.inv(H); sig = np.sqrt(np.diag(V)); corr = V[0, 1] / (sig[0] * sig[1])
    log(f"BFP: s_abs={bfp[0]:.4f}+/-{sig[0]:.4f}  s_scat={bfp[1]:.4f}+/-{sig[1]:.4f}  corr={corr:+.3f}  chi2/ndf={chi2_bf/(8-2):.2f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ctr = 0.5 * (EDGES[1:] + EDGES[:-1]) / 1000.0
    mb = A * np.asarray(hist_nb(jnp.asarray(bfp), jax.random.PRNGKey(0), qe, qw, res, rw))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    ax[0].errorbar(ctr, np.asarray(DATA), yerr=D_ERR, fmt="o", color="k", capsize=3, label="T2K data")
    ax[0].step(EDGES / 1000, np.append(A * nom, (A * nom)[-1]), where="post", color="C2", lw=2,
               label=f"ADoNIS nominal ($\\chi^2$/ndf {chi2_nom/(8-2):.1f})")
    ax[0].step(EDGES / 1000, np.append(mb, mb[-1]), where="post", color="C0", lw=1.6, ls="--",
              label=f"ADoNIS best fit ($\\chi^2$/ndf {chi2_bf/(8-2):.1f})")
    ax[0].set(xlabel=r"$\delta p_T$ [GeV/c]", ylabel=r"d$\sigma$/d$\delta p_T$ [$10^{-38}$cm$^2$/(GeV/c)/nuc]",
              title="ADoNIS tune to T2K $\\delta p_T$"); ax[0].legend(fontsize=8); ax[0].set_ylim(bottom=0)
    ew, evec = np.linalg.eigh(V); th = np.linspace(0, 2 * np.pi, 100)
    for k, c in ((1, "C3"), (2, "C0")):
        xy = (evec @ (np.sqrt(ew)[:, None] * np.array([np.cos(th), np.sin(th)])) * k)
        ax[1].plot(bfp[0] + xy[0], bfp[1] + xy[1], color=c, lw=1.6, label=f"{k}$\\sigma$")
    ax[1].plot(1, 1, "s", color="0.5", ms=8, label="nominal"); ax[1].plot(bfp[0], bfp[1], "X", color="k", ms=10, label="best fit")
    ax[1].set(xlabel=r"$\sigma_{abs}$", ylabel=r"$\sigma_{scatter}$", title=f"Hessian param covariance (corr={corr:+.2f})")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle("REAL ADoNIS tune to T2K CC0$\\pi$-Np $\\delta p_T$ (full covariance, from nominal)")
    fig.tight_layout(); fig.savefig("paper_figures/cc0pi_tune_adonis.png", dpi=120); print("wrote paper_figures/cc0pi_tune_adonis.png")


if __name__ == "__main__":
    main()
