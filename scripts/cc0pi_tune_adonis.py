"""Tune the REAL ADoNIS CC0pi-Np prediction to T2K STV data (delta_pT default; argv[1]="dat"
for delta_alphaT), starting from its nominal.

The differentiable ADoNIS predictor: QE-C + RES-C CC0pi events are sampled ONCE (frozen proposal);
the pion cascade (sigma_abs, sigma_scatter) and nucleon cascade (sigma_scatter) carry kind-1 weights,
so dsigma/dx(theta) is differentiable in (sigma_abs, sigma_scatter) and at theta=(1,1) reproduces
the nominal forward prediction bit-for-bit -- NO toy, NO free normalization (A is profiled once at
nominal).  Covariance-weighted two-replica chi^2; Hessian parameter covariance at the BFP.

WALK/WEIGHT SPLIT: the cascade walk is theta-independent (kind-1), so a bank of NREP walk replicas
is precomputed once (the only expensive step) and each fit iteration is a pure reweight via
pion_branch_reweight / nucleon_scat_reweight (~ms, vs ~minutes for a re-walk at this N).
Verified: model_hist(theta, replica) == hist_nb(theta, key) to <5e-16 and identical gradients.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, uproot
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import adonis.xsec.dcc_current as dcc; dcc.BATCH_INTERP = "spline"
from adonis.xsec import qe_xsec, res_xsec
from adonis.xsec.backend import me_cross_section
from adonis.core.event import EventRecord
from adonis.fsi.cascade_discrete import (DiscreteCascadeFSI, DiscreteNucleonFSI, DiscreteCascadeConfig,
                                         pion_branch_reweight, nucleon_scat_reweight)
from adonis.primary.dcc.form_factors import axial_reweight_dipole

CFG = lambda **k: DiscreteCascadeConfig(cylinder=False, step=0.04, max_steps=260, **k)   # Gaussian (diff'able)
# max_steps 260 (10.4 fm) is saturation-verified == 325; fast_xsec=True (default) slab-restricts the
# Oset/DCC cross sections (bit-exact, ~1.15x).  Together ~1.4x vs the original 325/dense.
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


def hist_nb(theta, kcasc, qe, qw, res, rw):
    """Differentiable CC0pi dsigma/ddpt [1e-38 units] for theta=(sabs, sscat); kcasc fixes the
    sampled cascade (frozen proposal) so only the knobs move."""
    sabs, sscat = theta[0], theta[1]
    k1, k2, k3 = jax.random.split(kcasc, 3)
    # --- QE: proton through the nucleon cascade (sigma_scatter) ---
    qev = _mk_ev(qe, np.full(NQE, 2112), qe["p_out"], np.zeros((NQE, 4)), np.zeros(NQE))
    nf = DiscreteNucleonFSI(CFG(seed=2)); qev2 = nf.apply(None, qev, key=k1, sscat=sscat)
    q_lead = qev2.p_N; q_w = jnp.asarray(qw) * nf.last_w_scat
    q_dpt = _obs(jnp.asarray(qe["k_mu"]), q_lead); q_keep = _sel(jnp.asarray(qe["k_mu"]), q_lead)
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
    r_dpt = _obs(jnp.asarray(res["k_mu"]), r_lead); r_keep = _sel(jnp.asarray(res["k_mu"]), r_lead)
    # --- histogram (absolute nb -> data units) ---
    def H(x, w, keep):
        idx = jnp.clip(jnp.searchsorted(jnp.asarray(EDGES), x) - 1, 0, len(EDGES) - 2)
        return jax.ops.segment_sum(w * keep, jax.lax.stop_gradient(idx), num_segments=len(EDGES) - 1)
    h_nb_per_MeV = (H(q_dpt, q_w, q_keep) + H(r_dpt, r_w, r_keep)) / jnp.diff(jnp.asarray(EDGES))
    return h_nb_per_MeV * CONV


# ---- walk/weight split: precompute theta-INDEPENDENT walks once, reweight per theta ----------- #
# The cascade trajectory is sampled at NOMINAL cross sections (kind-1); (sabs, sscat) enter only
# through pion_branch_reweight / nucleon_scat_reweight, pure functions of the compressed walk
# records.  So each cascade replica is run ONCE; a fit iteration is then an O(n*K) elementwise
# reweight + histogram (~ms), not a 260-step re-walk (~minutes at production N).

def build_replica(kcasc, qe, qw, res, rw):
    """Run the full FSI chain ONCE at nominal for cascade key `kcasc`; return everything the
    theta-reweight needs.  Mirrors hist_nb's chain exactly (same keys, same selections)."""
    k1, k2, k3 = jax.random.split(kcasc, 3)
    # --- QE: proton through the nucleon cascade ---
    qev = _mk_ev(qe, np.full(NQE, 2112), qe["p_out"], np.zeros((NQE, 4)), np.zeros(NQE))
    nf = DiscreteNucleonFSI(CFG(seed=2)); qev2 = nf.apply(None, qev, key=k1, sscat=1.0)
    q_lead = qev2.p_N
    q_dpt = _obs(jnp.asarray(qe["k_mu"]), q_lead); q_keep = _sel(jnp.asarray(qe["k_mu"]), q_lead)
    # --- RES: pion cascade (absorb) then nucleon cascade ---
    rev = _mk_ev(res, np.asarray(res["ipid"]), res["p_N"], res["p_pi"], np.asarray(res["ppid"]))
    pion = DiscreteCascadeFSI(CFG(seed=1)); rev2 = pion.apply(None, rev, key=k2, sabs=1.0, sscat=1.0)
    absb = pion.last_absorbed.astype(float); abs_p = pion.last_abs_proton
    nf2 = DiscreteNucleonFSI(CFG(seed=2)); rev3 = nf2.apply(None, rev2, key=k3, sscat=1.0)
    pNf = rev3.p_N; mom_p = jnp.linalg.norm(pNf[:, 1:], axis=1) * (rev3.pid_N == 2212)
    mom_a = jnp.linalg.norm(abs_p[:, 1:], axis=1)
    r_lead = jnp.where((mom_a > mom_p)[:, None], abs_p, pNf)
    has_p = (mom_a > 1) | (rev3.pid_N == 2212)
    r_dpt = _obs(jnp.asarray(res["k_mu"]), r_lead); r_keep = _sel(jnp.asarray(res["k_mu"]), r_lead)
    edges = jnp.asarray(EDGES)
    R = dict(
        q_idx=jnp.clip(jnp.searchsorted(edges, q_dpt) - 1, 0, len(EDGES) - 2),
        q_keep=q_keep, q_w0=jnp.asarray(qw), q_srec=nf.last_srec,
        r_idx=jnp.clip(jnp.searchsorted(edges, r_dpt) - 1, 0, len(EDGES) - 2),
        r_keep=r_keep, r_w0=jnp.asarray(rw) * absb * has_p.astype(float),
        r_brec=pion.last_brec, r_srec=nf2.last_srec)
    # overflow guards on the compressed records (capacities _K_BR / _K_SLAB_REC)
    nh = int(jnp.max(R["r_brec"][4]))
    nsmax = max(int(jnp.max(s[2])) for s in (R["q_srec"][0], R["q_srec"][1], R["r_srec"][0], R["r_srec"][1]))
    assert nh <= R["r_brec"][1].shape[1] and nsmax <= R["q_srec"][0][1].shape[1], (nh, nsmax)
    return jax.block_until_ready(R)


def _w_nuc(srec, sscat):
    s1, s2, has_ko = srec
    return nucleon_scat_reweight(s1, sscat) * jnp.where(has_ko, nucleon_scat_reweight(s2, sscat), 1.0)


@jax.jit
def model_hist(theta, R, M=None):
    """Differentiable CC0pi dsigma/dx [1e-38 units] from a precomputed walk replica.
    theta = (sabs, sscat) or (sabs, sscat, MA_GeV); MA needs M = build_ma_records(...)."""
    sabs, sscat = theta[0], theta[1]
    w_ma_q = ma_reweight(M["qe"], theta[2]) if M is not None else 1.0
    w_ma_r = ma_reweight(M["res"], theta[2]) if M is not None else 1.0
    q_w = R["q_w0"] * w_ma_q * _w_nuc(R["q_srec"], sscat)
    r_w = R["r_w0"] * w_ma_r * pion_branch_reweight(R["r_brec"], sabs, sscat) * _w_nuc(R["r_srec"], sscat)
    nb = len(EDGES) - 1
    h = (jax.ops.segment_sum(q_w * R["q_keep"], R["q_idx"], num_segments=nb)
         + jax.ops.segment_sum(r_w * R["r_keep"], R["r_idx"], num_segments=nb))
    return h / jnp.diff(jnp.asarray(EDGES)) * CONV


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    qe, qw, res, rw = build_proposal(); log("proposal sampled")

    # ---- precompute the cascade-walk replica bank (the ONLY expensive step) ------------------- #
    NREP = 8
    bank = []
    for r_i in range(NREP):
        bank.append(build_replica(jax.random.PRNGKey(50 + r_i), qe, qw, res, rw))
        log(f"  replica {r_i+1}/{NREP} walked  ({(time.time()-t0)/(r_i+1):.1f}s/replica)")

    nom = np.mean([np.asarray(model_hist(jnp.array([1.0, 1.0]), R)) for R in bank], axis=0)
    A = float((jnp.asarray(nom) @ COVINV @ DATA) / (jnp.asarray(nom) @ COVINV @ jnp.asarray(nom)))
    r = A * nom - np.asarray(DATA); chi2_nom = float(jnp.asarray(r) @ COVINV @ jnp.asarray(r))
    log(f"NOMINAL chi2/ndf = {chi2_nom/(8-2):.2f}  (A_nom={A:.3f})")

    def loss(theta, R1, R2):                       # two-replica unbiased chi^2 (independent walks)
        r1 = A * model_hist(theta, R1) - DATA; r2 = A * model_hist(theta, R2) - DATA
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
        r = A * model_hist(theta, R) - DATA; return r @ COVINV @ r
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
    FIG = f"paper_figures/cc0pi_tune_adonis{'' if OBS == 'dpt' else '_dat'}.png"
    mb = A * np.mean([np.asarray(model_hist(jnp.asarray(bfp), R)) for R in bank], axis=0)
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
