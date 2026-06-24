"""PoC: hand-rolled piecewise-linear COUPLING FLOW (pure JAX) trained to importance-sample a 2-D
integrand with a DIAGONAL ridge -- the correlated structure separable Vegas cannot capture.  A PL
coupling layer is "Vegas whose output bins depend (via a small MLP) on the OTHER coordinate", so it
can tilt with the ridge.  Trains on the Apple-Silicon CPU (JAX CpuDevice; no Metal for jax 0.10).

Live status: appends `step loss neffN` to /tmp/flow_toy.log (flushed) and saves the history to
/tmp/flow_toy_loss.npz every few steps, so a loss-evolution plot can be made ON REQUEST while training.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
print("jax devices:", jax.devices(), flush=True)

K = 16          # PL bins per transformed dim
HID = 32        # conditioner MLP hidden width
NLAY = 6        # coupling layers (alternating which dim is transformed)
rng = np.random.default_rng(0)

def init_mlp(kin, kout, key):
    s = []
    dims = [kin, HID, HID, kout]
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        key, k1 = jax.random.split(key)
        W = jax.random.normal(k1, (a, b)) * (1.0 / np.sqrt(a))
        if i == len(dims) - 2:                        # ZERO last layer -> logits=0 -> identity flow start
            W = jnp.zeros((a, b))
        s.append((W, jnp.zeros(b)))
    return s

def mlp(params, x):                                   # x:(n,kin) -> (n,kout)
    h = x
    for i, (W, bts) in enumerate(params):
        h = h @ W + bts
        if i < len(params) - 1:
            h = jnp.tanh(h)
    return h

TRANS = [L % 2 for L in range(NLAY)]                  # static: which dim each layer transforms

def init_flow(key):
    lays = []
    for L in range(NLAY):
        key, k = jax.random.split(key)
        lays.append(init_mlp(1, K, k))                # differentiable params only (MLP per layer)
    return lays

FLOOR = 0.10                                          # defensive floor: every bin keeps >= FLOOR/K of
                                                      #   the mass -> bounded Jacobian, no empty regions
def pl_layer(xt, logits):                             # xt:(n,) in [0,1], logits:(n,K)
    q = jax.nn.softmax(logits, axis=1)                # output bin widths (sum 1)
    q = (1.0 - FLOOR) * q + FLOOR / K                 # floored mixture (still sums to 1; q >= FLOOR/K)
    cum = jnp.cumsum(q, axis=1)
    cum0 = jnp.concatenate([jnp.zeros((q.shape[0], 1)), cum[:, :-1]], axis=1)
    s = jnp.clip(xt * K, 0.0, K - 1e-7); b = jnp.floor(s).astype(jnp.int32); frac = s - b
    qb = jnp.take_along_axis(q, b[:, None], 1)[:, 0]
    cb = jnp.take_along_axis(cum0, b[:, None], 1)[:, 0]
    y = cb + frac * qb
    return y, jnp.log(qb * K)                         # y, log|dy/dxt|

def flow_forward(flow, z):                            # z:(n,2) uniform -> x:(n,2), logjac:(n,)
    x = z; lj = jnp.zeros(z.shape[0])
    for lay, t in zip(flow, TRANS):
        p = 1 - t
        logits = mlp(lay, x[:, p:p+1])
        yt, l = pl_layer(x[:, t], logits)
        x = x.at[:, t].set(yt); lj = lj + l
    return x, lj

# ---- toy integrand: sharp DIAGONAL ridge in [0,1]^2 (correlated; separable Vegas fails) ----
def f_toy(x):
    d = (x[:, 1] - x[:, 0])                           # along the anti-diagonal
    a = (x[:, 0] + x[:, 1]) / 2
    return jnp.exp(-0.5 * (d / 0.03) ** 2) * (1.0 + 0.5 * jnp.sin(6 * a))

def neffN(w): w = w[w > 0]; return float(w.sum() ** 2 / jnp.sum(w ** 2) / w.size)

def loss_fn(flow, z):
    x, lj = flow_forward(flow, z)
    w = f_toy(x) * jnp.exp(lj)                        # w = f(x)*|dx/dz| = f/p
    return jnp.mean(w ** 2) / jnp.mean(w) ** 2        # true variance loss (large batch smooths spikes)

def neffN_flow(flow, z):                              # held-out efficiency on RAW (unclipped) weights
    x, lj = flow_forward(flow, z); w = f_toy(x) * jnp.exp(lj)
    return neffN(w)

# ---- flat-MC baseline + separable-Vegas-ish baseline for context ----
zf = jnp.asarray(rng.random((200000, 2)))
print(f"flat MC  N_eff/N = {neffN(f_toy(zf)):.3f}", flush=True)

NSTEP = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 32000
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
TAG = f"s{SEED}"; rng = np.random.default_rng(1000 + SEED)
flow = init_flow(jax.random.PRNGKey(SEED))
gloss = jax.jit(jax.value_and_grad(loss_fn))
m = jax.tree_util.tree_map(jnp.zeros_like, flow); v = jax.tree_util.tree_map(jnp.zeros_like, flow)
lr0, lrT, b1, b2, eps, CLIP = 3e-4, 1e-5, 0.9, 0.999, 1e-8, 1.0   # cosine 3e-4 -> 1e-5 (gentler)
LOG = f"/tmp/flow_toy_{TAG}.log"; open(LOG, "w").close()          # UNIQUE per-seed log (no contamination)
hist = {k: [] for k in ("step", "lr", "gnorm", "loss", "neff")}
t0 = time.time()
for step in range(1, NSTEP + 1):
    lr = lrT + 0.5 * (lr0 - lrT) * (1 + np.cos(np.pi * step / NSTEP))
    z = jnp.asarray(rng.random((BATCH, 2)))                       # FRESH samples (can't memorize)
    L, g = gloss(flow, z)
    gnorm = jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g)))
    scale = jnp.minimum(1.0, CLIP / (gnorm + 1e-12))
    g = jax.tree_util.tree_map(lambda x: x * scale, g)
    m = jax.tree_util.tree_map(lambda mm, gg: b1 * mm + (1 - b1) * gg, m, g)
    v = jax.tree_util.tree_map(lambda vv, gg: b2 * vv + (1 - b2) * gg * gg, v, g)
    mh = jax.tree_util.tree_map(lambda mm: mm / (1 - b1 ** step), m)
    vh = jax.tree_util.tree_map(lambda vv: vv / (1 - b2 ** step), v)
    flow = jax.tree_util.tree_map(lambda p, mm, vv: p - lr * mm / (jnp.sqrt(vv) + eps), flow, mh, vh)
    if step % 5 == 0 or step == 1:
        nf = neffN_flow(flow, jnp.asarray(rng.random((100000, 2))))   # FRESH eval each time (honest)
        for k, val in zip(hist, (step, lr, float(gnorm), float(L), nf)): hist[k].append(val)
        with open(LOG, "a") as fh:
            fh.write(f"step {step:5d}  lr {lr:.1e}  gnorm {float(gnorm):8.2f}  loss {float(L):9.3f}  "
                     f"N_eff/N {nf:.3f}  ({time.time()-t0:.0f}s)\n"); fh.flush()
        np.savez(f"/tmp/flow_toy_loss_{TAG}.npz", **{k: np.array(v) for k, v in hist.items()})
nn = np.array(hist["neff"]); tail = nn[int(0.85 * len(nn)):]     # CONVERGED regime (last 15% of evals)
print(f"FLOW[seed {SEED}] converged N_eff/N = {tail.mean():.3f} +/- {tail.std():.3f}  "
      f"(mean of last {len(tail)} evals; final {nn[-1]:.3f}; max {nn.max():.3f})", flush=True)
print("FLOW TOY DONE", flush=True)
