"""RIGOROUS validation of the flow IS system on a KNOWN target: a correlated 2-D Gaussian on [0,1]^2.
Checks BOTH:
  (1) UNBIASEDNESS  -- the flow IS estimate of integral f  ==  a fine-grid reference (to MC error).
  (2) EFFICIENCY    -- N_eff/N climbs toward 1 (optimal proposal = the Gaussian itself).
Run for rho=0 (axis-aligned: separable Vegas can do it) and rho=0.85 (TILTED: only a non-separable
flow can) so the correlation point is explicit.  Floored PL coupling flow, fresh-sample eval.
Usage: python scripts/_flow_gauss.py [rho] [nstep] [batch]"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

K, HID, NLAY, FLOOR = 16, 32, 6, 0.10
TRANS = [L % 2 for L in range(NLAY)]
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
    for _ in range(NLAY):
        key, k = jax.random.split(key); out.append(init_mlp(1, K, k))
    return out
def pl_layer(xt, logits):
    q = jax.nn.softmax(logits, 1); q = (1 - FLOOR) * q + FLOOR / K
    cum = jnp.cumsum(q, 1); cum0 = jnp.concatenate([jnp.zeros((q.shape[0], 1)), cum[:, :-1]], 1)
    s = jnp.clip(xt * K, 0.0, K - 1e-7); b = jnp.floor(s).astype(jnp.int32); fr = s - b
    qb = jnp.take_along_axis(q, b[:, None], 1)[:, 0]; cb = jnp.take_along_axis(cum0, b[:, None], 1)[:, 0]
    return cb + fr * qb, jnp.log(qb * K)
def flow_forward(flow, z):
    x = z; lj = jnp.zeros(z.shape[0])
    for lay, t in zip(flow, TRANS):
        logits = mlp(lay, x[:, 1 - t:2 - t]); yt, l = pl_layer(x[:, t], logits)
        x = x.at[:, t].set(yt); lj = lj + l
    return x, lj

RHO = float(sys.argv[1]) if len(sys.argv) > 1 else 0.85
NSTEP = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 64000
SIG = 0.10; MU = jnp.array([0.5, 0.5])
COV = jnp.array([[SIG**2, RHO*SIG**2], [RHO*SIG**2, SIG**2]]); PREC = jnp.linalg.inv(COV)
def f_gauss(x):
    d = x - MU; return jnp.exp(-0.5 * jnp.einsum('ni,ij,nj->n', d, PREC, d))
# fine-grid REFERENCE integral over [0,1]^2
g = np.linspace(0, 1, 3000); XX, YY = np.meshgrid(g, g)
I_ref = float(jnp.mean(f_gauss(jnp.stack([jnp.asarray(XX).ravel(), jnp.asarray(YY).ravel()], 1))))
print(f"rho={RHO}  fine-grid reference  integral = {I_ref:.6e}", flush=True)
def neffN(w): w = w[w > 0]; return float(w.sum()**2 / jnp.sum(w**2) / w.size)

rng = np.random.default_rng(0)
# flat-MC baseline: estimate + unbiasedness + N_eff
uf = jnp.asarray(rng.random((400000, 2))); wf = f_gauss(uf)
print(f"flat MC : integral = {float(jnp.mean(wf)):.6e}  ( /ref = {float(jnp.mean(wf))/I_ref:.4f})  N_eff/N = {neffN(wf):.3f}", flush=True)

flow = init_flow(jax.random.PRNGKey(0))
def loss_fn(flow, z):
    # SCORE-FUNCTION estimator (the path RES needs: f is a BLACK BOX, no df/dx).
    # dV/dtheta = E[w^2 d_theta log p]; for forward-sampled x, log p = -lj, so the surrogate
    # mean((w^2 - baseline)*lj) has gradient E[(w^2-b) d_theta lj] = dV/dtheta (baseline E[d lj]=0).
    x, lj = flow_forward(flow, z)
    w = jax.lax.stop_gradient(f_gauss(x)) * jnp.exp(jax.lax.stop_gradient(lj))   # DETACHED values
    w2 = w ** 2; b = jnp.mean(w2)
    return jnp.mean((w2 - b) * lj)
gloss = jax.jit(jax.value_and_grad(loss_fn))
m = jax.tree_util.tree_map(jnp.zeros_like, flow); v = jax.tree_util.tree_map(jnp.zeros_like, flow)
lr0, lrT, b1, b2, eps, CLIP = 3e-4, 1e-5, 0.9, 0.999, 1e-8, 1.0
LOG = f"/tmp/flow_gauss_rho{RHO}.log"; open(LOG, "w").close()
hist = {k: [] for k in ("step", "neff", "ratio")}
t0 = time.time()
for step in range(1, NSTEP + 1):
    lr = lrT + 0.5 * (lr0 - lrT) * (1 + np.cos(np.pi * step / NSTEP))
    z = jnp.asarray(rng.random((BATCH, 2))); L, gr = gloss(flow, z)
    gn = jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(gr)))
    gr = jax.tree_util.tree_map(lambda x: x * jnp.minimum(1.0, CLIP / (gn + 1e-12)), gr)
    m = jax.tree_util.tree_map(lambda a, c: b1*a + (1-b1)*c, m, gr)
    v = jax.tree_util.tree_map(lambda a, c: b2*a + (1-b2)*c*c, v, gr)
    flow = jax.tree_util.tree_map(lambda p, a, c: p - lr * (a/(1-b1**step)) / (jnp.sqrt(c/(1-b2**step)) + eps), flow, m, v)
    if step % 10 == 0 or step == 1:
        zt = jnp.asarray(rng.random((200000, 2))); xt, ljt = flow_forward(flow, zt)
        wt = f_gauss(xt) * jnp.exp(ljt); ratio = float(jnp.mean(wt)) / I_ref
        for k, val in zip(hist, (step, neffN(wt), ratio)): hist[k].append(val)
        with open(LOG, "a") as fh:
            fh.write(f"step {step:5d}  N_eff/N {neffN(wt):.3f}  est/ref {ratio:.4f}  ({time.time()-t0:.0f}s)\n"); fh.flush()
        np.savez(f"/tmp/flow_gauss_loss_rho{RHO}.npz", **{k: np.array(vv) for k, vv in hist.items()})
# final converged eval on a big fresh sample
zt = jnp.asarray(rng.random((1000000, 2))); xt, ljt = flow_forward(flow, zt); wt = f_gauss(xt) * jnp.exp(ljt)
est = float(jnp.mean(wt)); err = float(jnp.std(wt) / np.sqrt(wt.size))
print(f"FLOW    : integral = {est:.6e} +/- {err:.1e}  ( /ref = {est/I_ref:.4f}, pull = {(est-I_ref)/err:+.2f} sigma)  "
      f"N_eff/N = {neffN(wt):.3f}", flush=True)
print(f"  => UNBIASED: |est-ref| = {abs(est-I_ref):.2e} vs MC err {err:.2e}  ({'PASS' if abs(est-I_ref) < 4*err else 'FAIL'})", flush=True)
print("GAUSS DONE", flush=True)
