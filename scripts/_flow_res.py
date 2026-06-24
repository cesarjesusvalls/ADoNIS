"""The ACTUAL flow we need: a 6-D coupling flow over the RES final-state hypercube dims
[beam, 3-body] trained by the SCORE-FUNCTION estimator (amps2 is black-box numpy) to importance-sample
the summed 3-channel RES integrand, ON TOP of the resonance(BW) mapper.  Drop-in via the u_active hook.

Reports N_eff/N (vs resonance+Vegas 0.32 baseline) and the sigma estimate vs sigma_RES (UNBIASEDNESS).
Live status -> /tmp/flow_res.log + /tmp/flow_res_loss.npz (plot with _flow_lossplot.py)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
print("jax devices:", jax.devices(), flush=True)
import adonis.xsec.res_xsec as R
from adonis.xsec.flux import T2KFlux
R.SAMPLER_3BODY = "resonance"

# ---------- 6-D floored PL coupling flow ----------
NDIM, K, HID, NLAY, FLOOR = 6, 16, 64, 8, 0.10
LAYERS = [([d for d in range(NDIM) if d % 2 == L % 2], [d for d in range(NDIM) if d % 2 != L % 2]) for L in range(NLAY)]
def init_mlp(kin, kout, key):
    s = []; dims = [kin, HID, HID, kout]
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        key, k1 = jax.random.split(key)
        W = jnp.zeros((a, b)) if i == len(dims) - 2 else jax.random.normal(k1, (a, b)) / np.sqrt(a)
        s.append((W, jnp.zeros(b)))
    return s
def mlp(p, x):
    h = x
    for i, (W, b) in enumerate(p):
        h = h @ W + b; h = jnp.tanh(h) if i < len(p) - 1 else h
    return h
def init_flow(key):
    out = []
    for (td, pd) in LAYERS:
        key, k = jax.random.split(key); out.append(init_mlp(len(pd), len(td) * K, k))
    return out
def pl_block(xt, logits):                                # xt (n,m), logits (n,m,K)
    q = jax.nn.softmax(logits, -1); q = (1 - FLOOR) * q + FLOOR / K
    cum = jnp.cumsum(q, -1); cum0 = jnp.concatenate([jnp.zeros(q.shape[:-1] + (1,)), cum[..., :-1]], -1)
    s = jnp.clip(xt * K, 0.0, K - 1e-7); b = jnp.floor(s).astype(jnp.int32); fr = s - b
    qb = jnp.take_along_axis(q, b[..., None], -1)[..., 0]; cb = jnp.take_along_axis(cum0, b[..., None], -1)[..., 0]
    return cb + fr * qb, jnp.sum(jnp.log(qb * K), axis=1)
def flow_forward(flow, z):
    x = z; lj = jnp.zeros(z.shape[0])
    for lay, (td, pd) in zip(flow, LAYERS):
        logits = mlp(lay, x[:, pd]).reshape(x.shape[0], len(td), K)
        yt, l = pl_block(x[:, jnp.array(td)], logits)
        x = x.at[:, jnp.array(td)].set(yt); lj = lj + l
    return x, lj

# ---------- RES integrand as a function of the 6 flow dims (struck sampled internally per channel) ----------
flux = T2KFlux(); minE = flux.seed_min_GeV(); maxE = flux.max_energy; imp = R._imp_for(R._SF_N)
def res_weight(x6, rng):                                  # x6:(n,6) numpy -> summed 3-channel weight (n,)
    n = len(x6); tot = np.zeros(n)
    for (ipid, itiz, mNf, ppid, mpi, mstr) in R.CHANNELS:
        s = R._sample_channel(n, rng, flux, minE, maxE, R._pi_kin_mass(mpi), mNf, imp=imp, u_active=x6)
        tot = tot + R._channel_weight(s, ipid, itiz, ppid, mstr, R._SF_N, R._SF_P, 6, 6)
    return tot
def neffN(w): w = w[w > 0]; return float(w.sum() ** 2 / np.sum(w ** 2) / w.size)

NSTEP = int(sys.argv[1]) if len(sys.argv) > 1 else 600
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
rng = np.random.default_rng(0)
# baselines (no flow): flat-6dim and the known resonance+Vegas ~0.32
xb = rng.random((120000, 6)); wb = res_weight(xb, rng)
SIG_RES = float(np.mean(wb))                              # this is the channel-summed sigma (uniform-6dim est)
print(f"flat-6dim(resonance map, no flow): N_eff/N={neffN(wb):.3f}  sigma~{SIG_RES:.4e}  (Vegas baseline 0.32)", flush=True)

flow = init_flow(jax.random.PRNGKey(0))
def surr(flow, z, coef):                                  # score-function surrogate: grad = E[(w^2-b) d_theta lj]
    _, lj = flow_forward(flow, z); return jnp.mean(coef * lj)
vg = jax.jit(jax.value_and_grad(surr))
fwd = jax.jit(flow_forward)
m = jax.tree_util.tree_map(jnp.zeros_like, flow); v = jax.tree_util.tree_map(jnp.zeros_like, flow)
lr0, lrT, b1, b2, eps, CLIP = 3e-4, 1e-5, 0.9, 0.999, 1e-8, 1.0
LOG = "/tmp/flow_res.log"; open(LOG, "w").close(); hist = {k: [] for k in ("step", "neff", "ratio")}
t0 = time.time()
for step in range(1, NSTEP + 1):
    lr = lrT + 0.5 * (lr0 - lrT) * (1 + np.cos(np.pi * step / NSTEP))
    z = jnp.asarray(rng.random((BATCH, NDIM)))
    x6, lj = fwd(flow, z); G = res_weight(np.asarray(x6), rng); w = G * np.exp(np.asarray(lj))
    coef = jnp.asarray(w ** 2 - np.mean(w ** 2))         # detached baseline-corrected weights^2
    _, g = vg(flow, z, coef)
    gn = jnp.sqrt(sum(jnp.sum(a ** 2) for a in jax.tree_util.tree_leaves(g)))
    g = jax.tree_util.tree_map(lambda a: a * jnp.minimum(1.0, CLIP / (gn + 1e-12)), g)
    m = jax.tree_util.tree_map(lambda a, c: b1 * a + (1 - b1) * c, m, g)
    v = jax.tree_util.tree_map(lambda a, c: b2 * a + (1 - b2) * c * c, v, g)
    flow = jax.tree_util.tree_map(lambda p, a, c: p - lr * (a / (1 - b1 ** step)) / (jnp.sqrt(c / (1 - b2 ** step)) + eps), flow, m, v)
    if step % 5 == 0 or step == 1:
        ze = jnp.asarray(rng.random((60000, NDIM))); xe, lje = fwd(flow, ze)
        we = res_weight(np.asarray(xe), rng) * np.exp(np.asarray(lje))
        nf = neffN(we); ratio = float(np.mean(we)) / SIG_RES
        for k, val in zip(hist, (step, nf, ratio)): hist[k].append(val)
        with open(LOG, "a") as fh:
            fh.write(f"step {step:4d}  N_eff/N {nf:.3f}  sigma/ref {ratio:.4f}  ({time.time()-t0:.0f}s)\n"); fh.flush()
        np.savez("/tmp/flow_res_loss.npz", step=np.array(hist["step"]), neff=np.array(hist["neff"]), ratio=np.array(hist["ratio"]))
# converged eval
ze = jnp.asarray(rng.random((400000, NDIM))); xe, lje = fwd(flow, ze)
we = res_weight(np.asarray(xe), rng) * np.exp(np.asarray(lje))
nn = np.array(hist["neff"]); tail = nn[int(0.85 * len(nn)):]
print(f"FLOW RES: N_eff/N = {tail.mean():.3f} +/- {tail.std():.3f} (converged; final {nn[-1]:.3f}; max {nn.max():.3f})", flush=True)
print(f"  sigma_est/ref = {float(np.mean(we))/SIG_RES:.4f}  (unbiasedness; 1.0 = correct sigma_RES)", flush=True)
print("FLOW RES DONE", flush=True)
