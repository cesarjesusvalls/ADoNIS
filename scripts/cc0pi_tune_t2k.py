"""Tune to the REAL T2K CC0pi-Np data: fit the FSI knobs (sigma_abs, sigma_scatter) + an overall
normalization A to the measured dsigma/d(delta_pT) with the FULL 8x8 covariance, and report the
parameter covariance at the best-fit point from the Hessian (V_theta = 2 H^-1, H = d^2 chi^2/dtheta^2).

Loss: covariance-weighted, TWO-REPLICA unbiased chi^2  mean over keys of (m1-d)^T Cinv (m2-d) -- the
independent 2nd replica cancels the model-MC-variance bias in the gradient.  Model = the structural
differentiable CC0pi predictor (cc0pi_closure_tki) reweighted in (sigma_abs, sigma_scatter), scaled
by A; the Hessian uses a single high-stat evaluation.  Units: 1e-38 cm^2 (data & cov consistent).

NOTE: this is the TUNE MACHINERY on real data with the structural model + a free normalization; the
absolute differentiable pipeline would replace the free A.  delta_pT only (delta_alphaT is correlated
-> needs the joint covariance).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, uproot
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scripts.cc0pi_closure_tki import proposal, _hist, DPT_EDGES, SIG_NN0, SIG_ABS0, MB_FM2, RHO, DX

# ---- T2K delta_pT data + covariance (work in 1e-38 cm^2 units) -------------------------------- #
_r = uproot.open("../nuisance/data/T2K/CC0pi/STV/dptResults.root")
_edges = _r["Result"].axis().edges()                     # GeV/c
d_val = np.asarray(_r["Result"].values()) * 1e38         # dsigma/ddpt [1e-38 cm^2 /(GeV/c)/nucleon]
d_err = np.asarray(_r["Result"].errors()) * 1e38
COV = np.asarray(_r["Covariance_Matrix"].values())       # 8x8, (1e-38 units)^2
assert np.allclose(np.sqrt(np.diag(COV)), d_err, rtol=0.05), "covariance/error unit mismatch"
BW = np.diff(_edges)                                      # GeV bin widths (matches DPT_EDGES/1000)
COVINV = jnp.asarray(np.linalg.inv(COV + 1e-12 * np.eye(8)))
DATA = jnp.asarray(d_val)


def shape_dpt(theta, S):
    """Model dsigma/ddpt SHAPE (per GeV/c, arbitrary norm) for (s_abs, s_scat); kind-1 reweight."""
    sabs, ssc = theta[0], theta[1]
    is_cc0pi = (~S["is_res"]) | (S["is_res"] & S["absorbed"])
    p_abs = (sabs * SIG_ABS0) / (sabs * SIG_ABS0 + SIG_NN0)
    w_abs = jnp.where(S["is_res"] & S["absorbed"], p_abs / S["p_abs0"], 1.0)
    p_int = -jnp.expm1(-(ssc * SIG_NN0) * MB_FM2 * RHO * DX)
    inter = S["rolls"] < S["p_int0"]
    w_scat = jnp.prod(jnp.where(inter, p_int / S["p_int0"], (1.0 - p_int) / (1.0 - S["p_int0"])), axis=1)
    h = _hist(S["dpt"], w_abs * w_scat * is_cc0pi, DPT_EDGES)
    return h / jnp.asarray(BW) / S["dpt"].shape[0]         # per (GeV/c), per-event normalized


def model(theta, S):                                      # theta = (s_abs, s_scat, A)
    return jnp.exp(theta[2]) * shape_dpt(theta, S)        # A = exp(theta2) > 0


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    N = 400_000
    # init A near the data integral so the fit starts in range
    S0 = proposal(jax.random.PRNGKey(0), N)
    A0 = float(jnp.sum(DATA * jnp.asarray(BW)) / jnp.sum(shape_dpt(jnp.array([1.0, 1.0]), S0) * jnp.asarray(BW)))
    theta = jnp.array([1.0, 1.0, np.log(A0)])
    log(f"init A0={A0:.3f}")

    def chi2_1rep(theta, key):                            # single-replica chi^2 (for the Hessian)
        m = model(theta, proposal(key, N)); r = m - DATA
        return r @ COVINV @ r

    def loss(theta, key):                                 # two-replica unbiased chi^2
        k1, k2 = jax.random.split(key)
        r1 = model(theta, proposal(k1, N)) - DATA; r2 = model(theta, proposal(k2, N)) - DATA
        return r1 @ COVINV @ r2

    vg = jax.jit(jax.value_and_grad(loss))
    m = jnp.zeros(3); v = jnp.zeros(3); lr = 0.02
    for it in range(500):
        l, g = vg(theta, jax.random.PRNGKey(10 + it))
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g ** 2
        mh = m / (1 - 0.9 ** (it + 1)); vh = v / (1 - 0.999 ** (it + 1))
        theta = theta - lr * mh / (jnp.sqrt(vh) + 1e-8)
        if it % 100 == 0 or it == 499:
            log(f"  it {it:3d} chi2={float(l):.2f}  s_abs={float(theta[0]):.3f} s_sc={float(theta[1]):.3f} A={float(jnp.exp(theta[2])):.3f}")
    bfp = np.asarray(theta)
    # chi2/ndf at BFP (high-stat single eval) ; ndf = 8 bins - 3 params
    chi2_bf = float(np.mean([float(chi2_1rep(theta, jax.random.PRNGKey(900 + i))) for i in range(12)]))
    log(f"BFP: s_abs={bfp[0]:.4f}  s_scat={bfp[1]:.4f}  A={np.exp(bfp[2]):.4f}   chi2/ndf={chi2_bf/(8-3):.2f}")

    # ---- parameter covariance from the Hessian: V = 2 H^-1, H = d^2 chi^2/dtheta^2 -------------- #
    Hs = np.mean([np.asarray(jax.hessian(chi2_1rep)(theta, jax.random.PRNGKey(700 + i))) for i in range(16)], axis=0)
    V = 2.0 * np.linalg.inv(Hs)
    sig = np.sqrt(np.diag(V)); corr = V / np.outer(sig, sig)
    NAMES = ["sigma_abs", "sigma_scatter", "lnA"]
    log("parameter covariance (Hessian, V=2H^-1):")
    for i, nm in enumerate(NAMES):
        log(f"  {nm:>14s} = {bfp[i]:.4f} +/- {sig[i]:.4f}")
    log(f"  corr(s_abs, s_scat) = {corr[0,1]:+.3f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ctr = 0.5 * (_edges[1:] + _edges[:-1])
    mb = np.asarray(model(theta, S0))
    ax[0].errorbar(ctr, d_val, yerr=d_err, fmt="o", color="k", capsize=3, label="T2K data")
    ax[0].step(_edges, np.append(mb, mb[-1]), where="post", color="C0", lw=1.8, label="best fit")
    ax[0].set(xlabel=r"$\delta p_T$ [GeV/c]", ylabel=r"d$\sigma$/d$\delta p_T$ [$10^{-38}$cm$^2$/(GeV/c)/nucleon]",
              title=f"T2K CC0$\\pi$-Np tune  $\\chi^2$/ndf={chi2_bf/(8-3):.2f}"); ax[0].legend(); ax[0].set_ylim(bottom=0)
    # 2-sigma error ellipse in (s_abs, s_scat)
    Vab = V[:2, :2]; ew, evec = np.linalg.eigh(Vab); th = np.linspace(0, 2 * np.pi, 100)
    for k, c in ((1, "C3"), (2, "C0")):
        xy = (evec @ (np.sqrt(ew)[:, None] * np.array([np.cos(th), np.sin(th)])) * k)
        ax[1].plot(bfp[0] + xy[0], bfp[1] + xy[1], color=c, lw=1.6, label=f"{k}$\\sigma$")
    ax[1].plot(bfp[0], bfp[1], "X", color="k", ms=10, label="best fit")
    ax[1].set(xlabel=r"$\sigma_{abs}$", ylabel=r"$\sigma_{scatter}$",
              title=f"parameter covariance (corr={corr[0,1]:+.2f})"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle("Tune to T2K CC0$\\pi$-Np d$\\sigma$/d$\\delta p_T$ (full covariance) + Hessian parameter errors")
    fig.tight_layout(); fig.savefig("paper_figures/cc0pi_tune_t2k.png", dpi=120); print("wrote paper_figures/cc0pi_tune_t2k.png")


if __name__ == "__main__":
    main()
