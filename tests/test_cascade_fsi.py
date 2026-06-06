"""Phase D1 gates for the concrete cascade FSIModel (`adonis.fsi.cascade.ToyCascadeFSI`).

Two gates:
  (closure)  the module's own differentiability -- d/d(sigma_sc, sigma_abs) of a final-state
             observable, autodiff vs FD with a frozen proposal (exact kind-1 reweighting).
  (joint)    the Phase-D deliverable: synthesise a post-FSI observable on REAL produced
             EventRecords at a known (M_A, sigma_sc, sigma_abs), then recover all three
             jointly by Adam on the chi^2 of the (Q^2, |p_pi|)+CC0pi histogram. Production
             (M_A, lepton side) and FSI (sigma, hadron side) knobs are simultaneously
             identifiable from one observable, with the gradient flowing through the whole
             differentiable chain (production reweight x FSI likelihood-ratio weight).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from adonis import GenConfig, DCCSinglePion, PhysicsParams
from adonis.core.autodiff import Adam
from adonis.fsi.cascade import ToyCascadeFSI, CascadeConfig
from adonis.fsi.toy.kernel import to_positive, from_positive


# --- (closure) module differentiability gate -------------------------------- #
def test_cascade_closure():
    m = ToyCascadeFSI()
    r = m.closure_test()
    assert r.passed, r.detail


def test_oset_shape_closure_and_delta_peak():
    """The momentum-dependent Oset-shaped absorption mode (a) stays exactly differentiable
    (closure), and (b) absorbs Delta-region pions (T_pi ~ 180-210 MeV) preferentially over
    lower/higher momenta -- the Oset Delta-peaked absorption signature fed into the cascade."""
    from adonis.core.event import EventRecord
    M_PI = 139.57
    m = ToyCascadeFSI(CascadeConfig(oset_shape=True, seed=2))
    assert m.closure_test().passed                          # differentiable in the Oset mode

    def mono(p_mev, n=6000):
        key = jax.random.PRNGKey(int(p_mev))
        k1, k2 = jax.random.split(key)
        ct = jax.random.uniform(k1, (n,), minval=-1, maxval=1); ph = jax.random.uniform(k2, (n,)) * 2 * np.pi
        st = jnp.sqrt(1 - ct ** 2)
        d = jnp.stack([st * jnp.cos(ph), st * jnp.sin(ph), ct], 1) * p_mev
        E = jnp.sqrt(p_mev ** 2 + M_PI ** 2); z = jnp.zeros((n, 4)); o = jnp.ones(n)
        p_pi = jnp.concatenate([jnp.full((n, 1), E), d], 1)
        return EventRecord(k=z, kp=z, p_struck=z, p_pi=p_pi, p_N=z, w=o,
                           channel=jnp.zeros(n, jnp.int32), pid_pi=jnp.full(n, 211, jnp.int32),
                           pid_N=jnp.full(n, 2212, jnp.int32), pid_Ni=jnp.full(n, 2112, jnp.int32),
                           W=jnp.full(n, 1232.0), Q2_adj=jnp.full(n, 1.0e5))

    p = PhysicsParams(fsi_sigma_scatter=0.05, fsi_sigma_abs=0.30)  # low scatter -> isolate absorption
    af = {pp: float(np.mean(np.asarray(m.apply(p, mono(pp), key=jax.random.PRNGKey(7)).pid_pi == 0)))
          for pp in (140, 320, 400)}                              # below / at / above the Delta
    assert af[320] > af[140] and af[320] > af[400], af            # absorption peaks at the Delta


# --- (joint) production + FSI joint recovery on real EventRecords ------------ #
FAST = bool(_os.environ.get("ADONIS_CI_FAST"))
N = 8_000 if FAST else 24_000
ITERS = 100 if FAST else 200

cfg = GenConfig(spline=False)
ch = DCCSinglePion(cfg)
fsi = ToyCascadeFSI(CascadeConfig(seed=3))

# truth / init for the three jointly-fit knobs
MA_T, SC_T, AB_T = 1.15, 0.35, 0.22
MA_0, SC_0, AB_0 = 0.95, 0.18, 0.40

# binning: a coarse Q^2 x |p_pi| grid + a CC0pi (absorbed) overflow bin
Q2_EDGES = np.linspace(0.0, 1.4e6, 7)            # MeV^2
PPI_EDGES = np.linspace(0.0, 450.0, 7)           # MeV
NQ, NP = len(Q2_EDGES) - 1, len(PPI_EDGES) - 1
N_BINS = NQ * NP + 1                              # + CC0pi overflow


def _params(theta):
    return PhysicsParams(axial_MA=theta[0], fsi_sigma_scatter=theta[1], fsi_sigma_abs=theta[2])


def _histogram(theta, key, proposal):
    """Differentiable post-FSI (Q^2, |p_pi|)+CC0pi histogram on the fixed sample S."""
    p = _params(theta)
    w_prod, LWc = ch.weight(p, S)
    from adonis.primary.dcc.channel import assemble_event
    ev = assemble_event(S, w_prod, LWc)
    ev = fsi.apply(p, ev, key=key, proposal=proposal)

    sg = jax.lax.stop_gradient
    Q2 = sg(jnp.clip(jnp.searchsorted(jnp.asarray(Q2_EDGES), ev.Q2_adj) - 1, 0, NQ - 1))
    ppi = jnp.linalg.norm(ev.p_pi[:, 1:], axis=1)
    pb = sg(jnp.clip(jnp.searchsorted(jnp.asarray(PPI_EDGES), ppi) - 1, 0, NP - 1))
    absorbed = ev.pid_pi == 0
    flat = jnp.where(absorbed, N_BINS - 1, Q2 * NP + pb)
    return jax.ops.segment_sum(ev.w, flat, num_segments=N_BINS) / theta.shape[0] / N


# fixed sample (kinematics detached) + frozen FSI proposal & key, shared by data and model.
# Common random numbers: with one frozen proposal the cascade path never moves as theta
# varies (pure kind-1 reweighting), so the loss is smooth and data==model exactly at truth
# -- the closure then probes identifiability + the optimiser, not MC noise.
S = ch.sample(jax.random.PRNGKey(11), N)
_FSI_KEY = jax.random.PRNGKey(7)
_Q_FIXED = PhysicsParams(fsi_sigma_scatter=0.30, fsi_sigma_abs=0.30)   # frozen proposal

theta_true = jnp.array([MA_T, SC_T, AB_T])
_data = np.asarray(_histogram(theta_true, _FSI_KEY, proposal=_Q_FIXED))
_derr = np.sqrt(np.maximum(_data, 0) / N) + 1e-9


def _loss(u):
    theta = jnp.stack([to_positive(u[0]), to_positive(u[1]), to_positive(u[2])])
    model = _histogram(theta, _FSI_KEY, proposal=_Q_FIXED)
    return jnp.sum((model - jnp.asarray(_data)) ** 2 / jnp.asarray(_derr) ** 2) / N_BINS


def _run_fit():
    u = jnp.array([from_positive(MA_0), from_positive(SC_0), from_positive(AB_0)])
    vg = jax.jit(jax.value_and_grad(_loss))
    opt = Adam(0.05); st = opt.init(u)
    for it in range(ITERS):
        opt.lr = 0.05 * (0.02 ** (it / max(ITERS - 1, 1)))
        L, g = vg(u)
        g = jax.tree_util.tree_map(jnp.nan_to_num, g)
        u, st = opt.update(u, g, st)
    return np.array([float(to_positive(u[0])), float(to_positive(u[1])), float(to_positive(u[2]))])


_fit = _run_fit()
print(f"joint closure  truth=(MA {MA_T}, sc {SC_T}, ab {AB_T})  "
      f"init=({MA_0}, {SC_0}, {AB_0})  ->  fit=({_fit[0]:.3f}, {_fit[1]:.3f}, {_fit[2]:.3f})")


def test_joint_recovery():
    assert abs(_fit[0] - MA_T) < 0.03, f"M_A {_fit[0]}"
    assert abs(_fit[1] - SC_T) < 0.04, f"sigma_scatter {_fit[1]}"
    assert abs(_fit[2] - AB_T) < 0.03, f"sigma_abs {_fit[2]}"
