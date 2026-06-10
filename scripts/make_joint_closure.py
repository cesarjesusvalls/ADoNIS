"""JOINT 2-parameter closure test on 12C (differentiable ADoNIS).

Generate ADoNIS "data" on a 12C spectral function at KNOWN (axial_MA, pw_norm[Delta]) and recover
BOTH simultaneously by gradient descent.  The two knobs have near-orthogonal observable signatures
-- axial_MA shapes dsigma/dQ2 (axial dipole), the Delta(1232) P33 partial-wave norm shapes
dsigma/dW (resonance peak) -- so the joint fit is well-identified.

Kind-1 reweighting: the proposal is sampled ONCE (detached), only the weight carries the knobs, so
weight_from_sample is pure JAX and value_and_grad over (M_A, pw5) is exact.  Two-replica unbiased
chi2 (split the model sample) removes the model-variance bias.  Loss on the W and Q2 histograms,
built differentiably by segment_sum of per-event weights into detached-observable bins.

Run:  PYTHONPATH=. python scripts/make_joint_closure.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from adonis.primary.dcc.amplitudes import DCCKnobs
from adonis.primary.dcc.structure import HadronStructure
from adonis.primary.dcc.channel import sample_final_state, weight_from_sample
from adonis.core.event import EventRecord
from adonis import observables as obs
from adonis.core.autodiff import Adam

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

hs = HadronStructure(spline=False)                       # 12C spectral function (default sf=pke12p)
MA_TRUE, MA_INIT = 1.150, 1.000                          # axial mass [GeV]
PW_TRUE, PW_INIT = 0.150, 0.000                          # Delta P33 partial-wave norm (1+pw)
PW_IDX = 5
NDATA  = int(os.environ.get("ADONIS_NDATA", 600_000))
NMODEL = int(os.environ.get("ADONIS_NMODEL", 120_000))
DATA_CHUNK = min(100_000, NDATA)
ITERS  = int(os.environ.get("ADONIS_ITERS", 160))
EDGES = {"W": np.linspace(1080, 1640, 25), "Q2": np.linspace(0.0, 1.6e6, 22)}


def knobs_of(theta):
    pw = tuple(jnp.where(jnp.arange(14) == PW_IDX, theta[1], 0.0))   # differentiable Delta-wave norm
    return DCCKnobs(axial_MA=theta[0], pw_norm=pw)


def ev_of(S):
    z = jnp.zeros(S["n"])
    return EventRecord(k=S["k_lab"], kp=S["kp_lab"], p_struck=S["p_struck"], p_pi=S["p_pi"],
                       p_N=S["p_N"], w=z, channel=z, pid_pi=z, pid_N=z, pid_Ni=z,
                       W=S["W"], Q2_adj=S["Q2_adj"])


def bin_idx(S, name):
    e = jnp.asarray(EDGES[name]); nb = e.shape[0] - 1
    v = obs.OBSERVABLES[name](ev_of(S))
    return jax.lax.stop_gradient(jnp.clip(jnp.searchsorted(e, v) - 1, 0, nb - 1)), nb


# ---- DATA target at the TRUE params (high stat, chunked) -------------------------------------- #
log(f"generating DATA on 12C at (M_A={MA_TRUE}, pw_Delta={PW_TRUE}): {NDATA:,} events ...")
nb = {n: len(EDGES[n]) - 1 for n in EDGES}
accw = {n: np.zeros(nb[n]) for n in EDGES}; accw2 = {n: np.zeros(nb[n]) for n in EDGES}
for c in range(NDATA // DATA_CHUNK):
    Sd = sample_final_state(jax.random.PRNGKey(101 + c), n=DATA_CHUNK, hs=hs)
    wd = np.asarray(weight_from_sample(knobs_of(jnp.array([MA_TRUE, PW_TRUE])), Sd, use_spline=False)[0])
    for n in EDGES:
        idx = np.asarray(bin_idx(Sd, n)[0])
        accw[n] += np.bincount(idx, weights=wd, minlength=nb[n])
        accw2[n] += np.bincount(idx, weights=wd ** 2, minlength=nb[n])
    log(f"  data chunk {c+1}/{NDATA // DATA_CHUNK}")
DATA = {n: (accw[n] / NDATA, np.sqrt(accw2[n]) / NDATA + 1e-15) for n in EDGES}

# ---- MODEL proposal (sampled ONCE) ----------------------------------------------------------- #
log(f"sampling MODEL proposal (N={NMODEL:,}) ...")
Sm = sample_final_state(jax.random.PRNGKey(202), n=NMODEL, hs=hs)
H = NMODEL // 2
MIDX = {n: bin_idx(Sm, n)[0] for n in EDGES}; NB = {n: bin_idx(Sm, n)[1] for n in EDGES}


def disp_hist(theta, name):
    w, _ = weight_from_sample(knobs_of(theta), Sm, use_spline=False)
    return np.asarray(jax.ops.segment_sum(w, MIDX[name], num_segments=NB[name])) / NMODEL


def loss(theta):
    w, _ = weight_from_sample(knobs_of(theta), Sm, use_spline=False)   # compute the weight ONCE
    tot, ndf = 0.0, 0
    for n in EDGES:
        d, e = jnp.asarray(DATA[n][0]), jnp.asarray(DATA[n][1])
        mA = jax.ops.segment_sum(w[:H], MIDX[n][:H], num_segments=NB[n]) / H
        mB = jax.ops.segment_sum(w[H:], MIDX[n][H:], num_segments=NB[n]) / H
        tot = tot + jnp.sum((mA - d) * (mB - d) / (e ** 2)); ndf += d.shape[0]
    return tot / ndf


# ---- joint fit: reverse-mode value_and_grad over (M_A, pw_Delta) ------------------------------ #
vg = jax.jit(jax.value_and_grad(loss))
log("compiling value+grad (first call) ...")
tc = time.time(); l0, g0 = vg(jnp.array([MA_INIT, PW_INIT]))
log(f"  compiled in {time.time()-tc:.1f}s; loss0={float(l0):.3e}  grad0={np.asarray(g0)}")
# AD vs finite-diff gate on both components
eps = 5e-3
fd = []
for i in range(2):
    d = jnp.array([eps if j == i else 0.0 for j in range(2)])
    fd.append(float((vg(jnp.array([MA_INIT, PW_INIT]) + d)[0] - vg(jnp.array([MA_INIT, PW_INIT]) - d)[0]) / (2 * eps)))
log(f"  grad AD={np.asarray(g0)}  FD={np.array(fd)}")

theta = jnp.array([MA_INIT, PW_INIT]); opt = Adam(0.04); st = opt.init(theta)
traj = [np.asarray(theta)]; losses = []; titer = time.time()
for it in range(ITERS):
    opt.lr = 0.04 * (0.03 ** (it / max(ITERS - 1, 1)))
    l, g = vg(theta); theta, st = opt.update(theta, jnp.nan_to_num(g), st)
    traj.append(np.asarray(theta)); losses.append(float(l))
    if it % 20 == 0 or it == ITERS - 1:
        r = (it + 1) / (time.time() - titer)
        log(f"  iter {it:3d}/{ITERS} loss={float(l):.3e} M_A={float(theta[0]):.4f} pw_Delta={float(theta[1]):.4f} ({r:.1f} it/s)")
traj = np.array(traj); MAf, PWf = float(theta[0]), float(theta[1])
log(f"RECOVERED: M_A {MA_INIT}->{MAf:.4f} (true {MA_TRUE});  pw_Delta {PW_INIT}->{PWf:.4f} (true {PW_TRUE})")

# ---- figure: concurrent convergence + 2D trajectory + observables ---------------------------- #
fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(2, 3)
# (a) M_A vs iter
ax = fig.add_subplot(gs[0, 0])
ax.axhline(MA_TRUE, color="darkorange", ls="--", lw=1.5, label=f"true {MA_TRUE}")
ax.plot(traj[:, 0], color="C0", lw=2, label=f"fit -> {MAf:.3f}")
ax.set_xlabel("iteration"); ax.set_ylabel("$M_A$ [GeV]"); ax.set_title("$M_A$ convergence"); ax.legend(fontsize=8)
# (b) pw_Delta vs iter
ax = fig.add_subplot(gs[1, 0])
ax.axhline(PW_TRUE, color="darkorange", ls="--", lw=1.5, label=f"true {PW_TRUE}")
ax.plot(traj[:, 1], color="C2", lw=2, label=f"fit -> {PWf:.3f}")
ax.set_xlabel("iteration"); ax.set_ylabel(r"$pw_{\Delta}$ (P33 norm)"); ax.set_title(r"$pw_\Delta$ convergence"); ax.legend(fontsize=8)
# (c) 2D trajectory in param space
ax = fig.add_subplot(gs[:, 1])
ax.plot(traj[:, 0], traj[:, 1], "-o", color="0.4", ms=2.5, lw=1, label="optimizer path")
ax.plot(MA_INIT, PW_INIT, "s", color="k", ms=9, label="init")
ax.plot(MA_TRUE, PW_TRUE, "*", color="darkorange", ms=20, label="truth", zorder=5)
ax.plot(MAf, PWf, "X", color="C3", ms=12, label="recovered", zorder=5)
ax.set_xlabel("$M_A$ [GeV]"); ax.set_ylabel(r"$pw_\Delta$ (P33 norm)")
ax.set_title("concurrent 2-parameter optimisation (12C)"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
# (d,e) data / init / final observables
for r, name in enumerate(("W", "Q2")):
    ax = fig.add_subplot(gs[r, 2])
    e = np.asarray(EDGES[name]); x = 0.5 * (e[:-1] + e[1:]); dx = np.diff(e)
    sc = 1e-6 if name == "Q2" else 1.0
    ax.errorbar(x * sc, DATA[name][0] / dx, yerr=DATA[name][1] / dx, fmt="o", ms=3, color="darkorange",
                capsize=2, lw=1, label="data (truth)", zorder=3)
    ax.plot(x * sc, disp_hist(jnp.array([MA_INIT, PW_INIT]), name) / dx, ":", color="0.5", lw=1.5, label="init")
    ax.plot(x * sc, disp_hist(jnp.array([MAf, PWf]), name) / dx, "-", color="C3", lw=1.5, label="fit")
    ax.set_xlabel("W [MeV]" if name == "W" else "$Q^2$ [GeV$^2$]")
    ax.set_ylabel(r"d$\sigma$/dx [a.u.]"); ax.set_title(f"d$\\sigma$/d{name}"); ax.legend(fontsize=7)
fig.suptitle(f"Joint closure on 12C: recover ($M_A$, $pw_\\Delta$) by gradient descent  "
             f"[{MA_TRUE},{PW_TRUE}] -> [{MAf:.3f},{PWf:.3f}]", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = "paper_figures/joint_closure_12C.png"
fig.savefig(out, dpi=120, bbox_inches="tight"); log(f"saved -> {out}")
