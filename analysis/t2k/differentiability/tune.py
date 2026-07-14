"""Tune the REAL ADoNIS CC0pi-Np prediction to T2K STV data (delta_pT default; argv[1]="dat"
for delta_alphaT), starting from its nominal.

The differentiable ADoNIS predictor: QE-C + RES-C CC0pi events are sampled ONCE (frozen proposal);
the pion cascade (sigma_abs, sigma_scatter) and nucleon cascade (sigma_scatter) carry kind-1 weights,
so dsigma/dx(theta) is differentiable in (sigma_abs, sigma_scatter) and at theta=(1,1) reproduces
the nominal forward prediction bit-for-bit -- NO toy, NO free normalization (A is profiled once at
nominal).  Covariance-weighted two-replica chi^2; Hessian parameter covariance at the BFP.

WALK/WEIGHT SPLIT: the cascade walk is theta-independent (kind-1), so a bank of NREP walk replicas
is precomputed once (the sampling step) and each fit iteration is a pure reweight via pool_fsi_reweight.
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root (analysis/t2k/differentiability/ -> .)
import numpy as np, uproot
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import adonis.xsec.dcc_current as dcc; dcc.BATCH_INTERP = "spline"
from adonis.xsec import qe_xsec, res_xsec
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig

MU_LO, COSMU, P_LO, P_HI, COSP = 250.0, -0.6, 450.0, 1000.0, 0.4
NQE, NRES = 120000, 120000
CONV = 1e-33 / 12.0 * 1000.0 * 1e38                       # nb/MeV per-12C -> 1e-38 cm^2/(GeV/c)/nucleon

# ---- observable: delta_pT (default) or delta_alphaT (argv[1] = "dat") ------------------------- #
OBS = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "dpt"
assert OBS in ("dpt", "dat"), OBS

# ---- T2K data + covariance (1e-38 units) ------------------------------------------------------ #
_r = uproot.open(f"../nuisance/data/T2K/CC0pi/STV/{'dpt' if OBS == 'dpt' else 'dat'}Results.root")
if OBS == "dpt":
    EDGES = np.asarray(_r["Result"].axis().edges()) * 1000.0    # MeV
else:
    EDGES = np.asarray(_r["Result"].axis().edges())             # rad
    CONV = 1e-33 / 12.0 * 1e38                                  # nb/rad per-12C -> 1e-38 cm^2/rad/nucleon
DATA = jnp.asarray(np.asarray(_r["Result"].values()) * 1e38)
D_ERR = np.asarray(_r["Result"].errors()) * 1e38
COV = np.asarray(_r["Covariance_Matrix"].values())
COVINV = jnp.asarray(np.linalg.inv(COV + 1e-12 * np.eye(8)))


def _dpt(kmu, lead):
    lt = kmu[:, 1:3]; dv = lt + lead[:, 1:3]
    return jnp.linalg.norm(dv, axis=1)


def _dat(kmu, lead):
    """delta_alphaT = acos(-p_T^mu . delta_pT_vec / (|p_T^mu| |delta_pT_vec|))  [rad]."""
    lt = kmu[:, 1:3]; dv = lt + lead[:, 1:3]
    num = -jnp.sum(lt * dv, axis=1)
    den = jnp.linalg.norm(lt, axis=1) * jnp.clip(jnp.linalg.norm(dv, axis=1), 1e-9, None)
    return jnp.arccos(jnp.clip(num / den, -1.0, 1.0))


_obs = _dpt if OBS == "dpt" else _dat


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


from adonis.analysis.ma_records import build_qe_ma_records, build_res_ma_records, ma_reweight


def build_ma_records(qe, res):
    """ONE-TIME (per proposal) exact M_A-reweight records (see adonis.analysis.ma_records)."""
    qa, qb, qc, qq2 = build_qe_ma_records(qe["k_nu"], qe["k_mu"], qe["p_struck"], qe["p_out"])
    ra, rb, rc, rq2 = build_res_ma_records(res["k_nu"], res["k_mu"], res["p_struck"],
                                           res["p_N"], res["p_pi"], res["ipid"], res["ppid"])
    j = jnp.asarray
    return dict(qe=(j(qa), j(qb), j(qc), j(qq2)), res=(j(ra), j(rb), j(rc), j(rq2)))


# ============================================================================================== #
# POOL-BACKED blueprint (the differentiable core): ONE joint cascade per channel (cascade_nucleus,
# engine="pool"), emitting the joint kind-1 FSI record reweighted by pool_fsi_reweight.  The walk is
# theta-independent and the knobs enter only through the kind-1 reweight; the single validated+fixed
# pool engine underneath (re-cascades both absorption nucleons, charge-resolved sigma, etc.).
# ============================================================================================== #
import adonis.fsi.cascade_full as _CF
# Gaussian interaction probability everywhere -> ADoNIS forward + differentiable tuning share ONE model and the
# kind-1 sigma-reweight is exact (see all-gaussian decision).  max_steps=100000 (= production / the runaway
# ceiling): with the M=1 serial pool the per-event nstep is the SUM of all particles' steps, so the physics
# bound is path_budget_R*radius, not a small step cap (a 600 cap wrongly trips the runaway guard at M=1).
POOLCFG = lambda **k: DiscreteCascadeConfig(step=0.04, max_steps=100000, path_budget_R=3.0, engine="pool", **k)
REC_CAPS = (96, 64)            # pion IN-SLAB CANDIDATE steps<=96 (was hits<=32: the pion record now logs
                               # every candidate step, hit or not, to carry the sigma_tot/mean-free-path
                               # response -- measured mean 4.8, max 31 over 5.3k events, so 96 leaves tail
                               # headroom for a 1M-event bank), nucleon candidate steps<=64 (ns_max~43).
                               # build_walk ASSERTS on overflow, it never truncates -- keep the headroom.
_P_BUF = 12


def _lead_proton_pool(nt):
    """Leading (global max-momentum) escaped proton from a pool nucleon-terminal batch."""
    p4 = nt["p4"]; isp = (nt["pid"] == 2212) & nt["alive"]
    mom = jnp.linalg.norm(p4[:, :, 1:], axis=2) * isp
    n = p4.shape[0]; ar = jnp.arange(n); j = jnp.argmax(mom, axis=1)
    return jnp.where((mom[ar, j] > 0)[:, None], p4[ar, j], 0.0)


def build_walk(kcasc, qe, qw, res, rw):
    """Run one OBSERVABLE-INDEPENDENT cascade replica: one cascade_nucleus pool run per channel ->
    leading proton + the joint kind-1 FSI record (theta enters via pool_fsi_reweight).  Returns the raw
    walk (leading protons, k_mu, w0, records); bin it into an R with bin_walk (the only obs-dependent step).
    CC0pi-RES = primary pion absorbed (pterm pid==0)."""
    kq, kr = jax.random.split(kcasc, 2); j = jnp.asarray
    nq = len(qe["w"]); nr = len(res["w"])                       # actual event counts (RES generate != NRES)
    # QE: struck neutron -> proton through the pool (no pion); record carries only nucleon scatters.
    _pt, ntq, oflq, _c, recq = _CF.cascade_nucleus(
        jnp.zeros((nq, 4)), j(qe["p_out"]), jnp.zeros(nq, jnp.int32), jnp.full(nq, 2112, jnp.int32),
        jnp.full(nq, 2212, jnp.int32), POOLCFG(seed=2), kq, channel="qe", rec_caps=REC_CAPS)
    q_lead = _lead_proton_pool(ntq[0])
    # RES: primary pion + recoil through the pool (joint pion+nucleon record).
    ptr, ntr, oflr, _c2, recr = _CF.cascade_nucleus(
        j(res["p_pi"]), j(res["p_N"]), j(res["ppid"]).astype(jnp.int32), j(res["ipid"]).astype(jnp.int32),
        j(res["Npid"]).astype(jnp.int32), POOLCFG(seed=1), kr, channel="res", rec_caps=REC_CAPS)
    r_lead = _lead_proton_pool(ntr[0])
    absb = (ptr["pid"] == 0).astype(float)                      # primary pion absorbed -> CC0pi
    W = dict(q_kmu=j(qe["k_mu"]), q_lead=q_lead, q_w0=j(qw), q_rec=recq,
             r_kmu=j(res["k_mu"]), r_lead=r_lead, r_w0=j(rw) * absb, r_rec=recr)
    nhmax = max(int(jnp.max(recq["nh"])), int(jnp.max(recr["nh"])))
    nsmax = max(int(jnp.max(recq["ns"])), int(jnp.max(recr["ns"])))
    assert nhmax <= REC_CAPS[0] and nsmax <= REC_CAPS[1], ("rec overflow", nhmax, nsmax)
    # pool buffer overflow (P stack / M_out finals): production tolerates a handful per 1e4-1e5 events
    # (logged, not fatal -- those events drop a low-rank particle).  LOG it, don't crash.
    if int(oflq) or int(oflr):
        print(f"  [build_walk] pool buffer overflow: QE={int(oflq)} RES={int(oflr)} "
              f"of (nq={nq}, nr={nr}) events", flush=True)
    return jax.block_until_ready(W)


def bin_walk(W, edges=None, obs=None):
    """Bin a build_walk output into an R (idx/keep/w0/rec) for a given observable.  Defaults to this
    module's OBS (EDGES, _obs); pass edges/obs to bin the SAME walk for a different observable."""
    edges = EDGES if edges is None else np.asarray(edges)
    obs = _obs if obs is None else obs
    ej = jnp.asarray(edges); nbm = len(edges) - 2
    q_x = obs(W["q_kmu"], W["q_lead"]); q_keep = _sel(W["q_kmu"], W["q_lead"])
    r_x = obs(W["r_kmu"], W["r_lead"]); r_keep = _sel(W["r_kmu"], W["r_lead"])
    return dict(q_idx=jnp.clip(jnp.searchsorted(ej, q_x) - 1, 0, nbm), q_keep=q_keep,
                q_w0=W["q_w0"], q_rec=W["q_rec"],
                r_idx=jnp.clip(jnp.searchsorted(ej, r_x) - 1, 0, nbm), r_keep=r_keep,
                r_w0=W["r_w0"], r_rec=W["r_rec"])


def build_replica(kcasc, qe, qw, res, rw):
    """Walk one replica and bin it for this module's observable (build_walk -> bin_walk).  Behaviour is
    identical to before the walk/bin split."""
    return jax.block_until_ready(bin_walk(build_walk(kcasc, qe, qw, res, rw)))


@jax.jit
def model_hist(theta, R, M=None):
    """Differentiable CC0pi dsigma/dx [1e-38 units]: theta=(sabs,sscat[,MA]); reweight a precomputed pool walk."""
    sabs, sscat = theta[0], theta[1]
    w_ma_q = ma_reweight(M["qe"], theta[2]) if M is not None else 1.0
    w_ma_r = ma_reweight(M["res"], theta[2]) if M is not None else 1.0
    q_w = R["q_w0"] * w_ma_q * _CF.pool_fsi_reweight(R["q_rec"], sabs, sscat)
    r_w = R["r_w0"] * w_ma_r * _CF.pool_fsi_reweight(R["r_rec"], sabs, sscat)
    nb = len(EDGES) - 1
    h = (jax.ops.segment_sum(q_w * R["q_keep"], R["q_idx"], num_segments=nb)
         + jax.ops.segment_sum(r_w * R["r_keep"], R["r_idx"], num_segments=nb))
    return h / jnp.diff(jnp.asarray(EDGES)) * CONV


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    # ENGINE: the pool is the single faithful + differentiable cascade core.  N and NREP are env-tunable;
    # size them to the statistics the fit needs.
    global NQE, NRES
    NQE = NRES = int(os.environ.get("CC0PI_N", "40000"))
    NREP = int(os.environ.get("CC0PI_NREP", "4"))
    _build, _hist = build_replica, model_hist
    log(f"ENGINE=pool (differentiable core)  NQE=NRES={NQE}  NREP={NREP}")
    qe, qw, res, rw = build_proposal(); log("proposal sampled")

    # ---- precompute the cascade-walk replica bank (the sampling step) ------------------------- #
    bank = []
    for r_i in range(NREP):
        bank.append(_build(jax.random.PRNGKey(50 + r_i), qe, qw, res, rw))
        log(f"  replica {r_i+1}/{NREP} walked  ({(time.time()-t0)/(r_i+1):.1f}s/replica)")

    nom = np.mean([np.asarray(_hist(jnp.array([1.0, 1.0]), R)) for R in bank], axis=0)
    A = float((jnp.asarray(nom) @ COVINV @ DATA) / (jnp.asarray(nom) @ COVINV @ jnp.asarray(nom)))
    r = A * nom - np.asarray(DATA); chi2_nom = float(jnp.asarray(r) @ COVINV @ jnp.asarray(r))
    log(f"NOMINAL chi2/ndf = {chi2_nom/(8-2):.2f}  (A_nom={A:.3f})")

    def loss(theta, R1, R2):                       # two-replica unbiased chi^2 (independent walks)
        r1 = A * _hist(theta, R1) - DATA; r2 = A * _hist(theta, R2) - DATA
        return r1 @ COVINV @ r2
    vg = jax.jit(jax.value_and_grad(loss))
    l0, g0 = vg(jnp.array([1.0, 1.0]), bank[0], bank[1]); log(f"loss(nominal)={float(l0):.2f} grad={np.asarray(g0)}")

    NITERS = 400
    theta = jnp.array([1.0, 1.0]); m = jnp.zeros(2); v = jnp.zeros(2); lr = 0.02; traj = [np.asarray(theta)]
    t_loop = time.time()
    for it in range(NITERS):
        off = 1 + (it // NREP) % (NREP - 1)                  # in [1, NREP-1]: R2 is never R1
        R1 = bank[it % NREP]; R2 = bank[(it + off) % NREP]
        l, g = vg(theta, R1, R2)
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g ** 2
        mh = m / (1 - 0.9 ** (it + 1)); vh = v / (1 - 0.999 ** (it + 1))
        theta = jnp.clip(theta - lr * mh / (jnp.sqrt(vh) + 1e-8), 0.3, 3.0); traj.append(np.asarray(theta))
        if it % 50 == 0 or it == NITERS - 1:
            log(f"  it {it:3d}/{NITERS}  chi2/ndf={float(l)/(8-2):.2f}  s_abs={float(theta[0]):.3f} s_sc={float(theta[1]):.3f}")
    bfp = np.asarray(theta); traj = np.array(traj)
    log(f"loop done in {time.time()-t_loop:.1f}s ({(time.time()-t_loop)/NITERS*1e3:.0f} ms/it)")

    def chi2_1(theta, R):
        r = A * _hist(theta, R) - DATA; return r @ COVINV @ r
    chi2_bf = float(np.mean([float(chi2_1(theta, R)) for R in bank]))
    H = np.mean([np.asarray(jax.hessian(chi2_1)(theta, R)) for R in bank], axis=0)
    V = 2.0 * np.linalg.inv(H); sig = np.sqrt(np.diag(V)); corr = V[0, 1] / (sig[0] * sig[1])
    log(f"BFP: s_abs={bfp[0]:.4f}+/-{sig[0]:.4f}  s_scat={bfp[1]:.4f}+/-{sig[1]:.4f}  corr={corr:+.3f}  chi2/ndf={chi2_bf/(8-2):.2f}")

    os.makedirs("/tmp/adonis_tune_runs", exist_ok=True)
    npz = f"/tmp/adonis_tune_runs/datafit_{OBS}.npz"
    np.savez(npz, obs=OBS, A=A, data=np.asarray(DATA), derr=D_ERR, nom=nom, mb_traj=traj,
             bfp=bfp, sig=sig, V=V, chi2_nom=chi2_nom, chi2_bf=chi2_bf, edges=EDGES)
    log(f"history saved -> {npz}")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ctr = 0.5 * (EDGES[1:] + EDGES[:-1]) / (1000.0 if OBS == "dpt" else 1.0)
    XL = r"$\delta p_T$ [GeV/c]" if OBS == "dpt" else r"$\delta\alpha_T$ [rad]"
    YL = (r"d$\sigma$/d$\delta p_T$ [$10^{-38}$cm$^2$/(GeV/c)/nuc]" if OBS == "dpt"
          else r"d$\sigma$/d$\delta\alpha_T$ [$10^{-38}$cm$^2$/rad/nuc]")
    os.makedirs("output/figures", exist_ok=True)
    FIG = f"output/figures/cc0pi_tune_adonis{'' if OBS == 'dpt' else '_dat'}.png"
    mb = A * np.mean([np.asarray(_hist(jnp.asarray(bfp), R)) for R in bank], axis=0)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    ax[0].errorbar(ctr, np.asarray(DATA), yerr=D_ERR, fmt="o", color="k", capsize=3, label="T2K data")
    xed = EDGES / (1000.0 if OBS == "dpt" else 1.0)
    ax[0].step(xed, np.append(nom, nom[-1]), where="post", color="0.45", lw=1.4, ls=":",
               label="pre-tune ADoNIS (absolute, A=1)")
    ax[0].step(xed, np.append(A * nom, (A * nom)[-1]), where="post", color="C2", lw=2,
               label=f"pre-tune $\\times$ profiled A={A:.2f} ($\\chi^2$/ndf {chi2_nom/(8-2):.1f})")
    ax[0].step(xed, np.append(mb, mb[-1]), where="post", color="C0", lw=1.6, ls="--",
              label=f"ADoNIS best fit ($\\chi^2$/ndf {chi2_bf/(8-2):.1f})")
    ax[0].set(xlabel=XL, ylabel=YL,
              title=f"ADoNIS tune to T2K {OBS}"); ax[0].legend(fontsize=8); ax[0].set_ylim(bottom=0)
    ew, evec = np.linalg.eigh(V); th = np.linspace(0, 2 * np.pi, 100)
    for k, c in ((1, "C3"), (2, "C0")):
        xy = (evec @ (np.sqrt(ew)[:, None] * np.array([np.cos(th), np.sin(th)])) * k)
        ax[1].plot(bfp[0] + xy[0], bfp[1] + xy[1], color=c, lw=1.6, label=f"{k}$\\sigma$")
    ax[1].plot(1, 1, "s", color="0.5", ms=8, label="nominal"); ax[1].plot(bfp[0], bfp[1], "X", color="k", ms=10, label="best fit")
    ax[1].set(xlabel=r"$\sigma_{abs}$", ylabel=r"$\sigma_{scatter}$", title=f"Hessian param covariance (corr={corr:+.2f})")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle(f"REAL ADoNIS tune to T2K CC0$\\pi$-Np {OBS} (full covariance, from nominal)")
    fig.tight_layout(); fig.savefig(FIG, dpi=120); print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
