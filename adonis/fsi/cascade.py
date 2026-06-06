"""Concrete FSIModel: the toy intranuclear cascade applied to the real EventRecord (Phase D1).

The toy cascade (`adonis.fsi.toy`) was validated as a stand-alone histogram estimator.
This module ports it to the production contract -- `FSIModel.apply(params, EventRecord) ->
EventRecord` -- so the produced pion is actually propagated through the nucleus and the FSI
cross sections (`fsi_sigma_scatter`, `fsi_sigma_abs`) are fit jointly with the production
knobs from one final-state observable.

The pion (`event.p_pi`) starts at the nucleus centre moving along its production direction
and propagates through a uniform sphere of radius R [fm].  Per step it either escapes,
scatters (deflect + lose a fixed momentum fraction -- in-medium pi N energy loss), or is
absorbed (the pion is removed: a CC0pi event).

Contract / differentiability.  Like ACHILLES' cascade, the per-event outcome is a definite
sampled final state, so the EventRecord stays a per-event record (not an expected-value
histogram).  Every stochastic decision (escape/scatter, absorb/scatter) is sampled against
the probabilities of a **frozen proposal** `q` (a detached copy of the parameters); the
fitted parameters `theta` enter only through a likelihood-ratio weight `p_theta / q` folded
into `event.w`.  This is the project's kind-1 reweighting estimator applied per event: the
sampled path is frozen as `theta` varies (the thresholds read `q`, not `theta`), only the
weight moves, so `d/dtheta E[observable]` is exact and, at a fixed PRNG key, autodiff ==
finite difference (no REINFORCE jump variance).  In production `proposal` defaults to the
detached current parameters (self-normalised: forward weight 1, gradient d log p); a test
passes an explicit fixed `proposal` so its finite-difference check sees a frozen path.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from adonis.fsi.base import FSIModel
from adonis.fsi.toy.component_c_r1 import _distance_to_boundary, _deflect, hg_sample_cos
from adonis.core.validation import check_gradient, GradCheck, TestResult
from adonis.params import PhysicsParams

M_PI = 139.57  # MeV; put the degraded pion back on-shell after energy loss


def _rates(params):
    """(total interaction rate [1/fm], P(absorb | interact)) from the FSI knobs."""
    lam = params.fsi_sigma_scatter + params.fsi_sigma_abs
    return lam, params.fsi_sigma_abs / lam


@dataclass(frozen=True)
class CascadeConfig:
    R: float = 3.0          # nucleus radius [fm]
    n_bounces: int = 16
    mom_loss: float = 0.20  # fractional |p_pi| lost per quasi-elastic scatter
    g_scatter: float = 0.40  # Henyey-Greenstein forward asymmetry
    seed: int = 0            # default internal PRNG seed (overridable per apply())


class ToyCascadeFSI(FSIModel):
    """Per-event, differentiable intranuclear cascade for the produced pion."""

    def __init__(self, cfg: CascadeConfig = CascadeConfig()):
        self.cfg = cfg

    def apply(self, params, event, key=None, proposal=None):
        cfg = self.cfg
        sg = jax.lax.stop_gradient
        # frozen proposal q (decision thresholds read this); theta enters only via the ratio
        proposal = params if proposal is None else proposal
        lam_cur, absp_cur = _rates(params)
        lam_q, absp_q = _rates(sg(proposal))

        n = event.p_pi.shape[0]
        key = jax.random.PRNGKey(cfg.seed) if key is None else key

        # initial pion |p| and direction (geometry + kinematics are detached)
        p3 = event.p_pi[:, 1:]
        pmag = sg(jnp.linalg.norm(p3, axis=1))
        has_p = pmag > 1e-6
        direction = sg(jnp.where(has_p[:, None], p3 / (pmag[:, None] + 1e-30),
                                 jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0]), p3.shape)))
        pos = jnp.zeros((n, 3))

        w_fsi = jnp.ones(n)
        alive = jnp.ones(n)       # 1.0 = still in medium, 0.0 = escaped or absorbed
        absorbed = jnp.zeros(n)   # 1.0 = pion absorbed (removed)

        for sub in jax.random.split(key, cfg.n_bounces):
            kr, kc, ks, kb = jax.random.split(sub, 4)
            live = alive > 0.5
            d = _distance_to_boundary(pos, direction, cfg.R)
            reach_q = jnp.exp(-d * lam_q)                       # P(escape) under the proposal
            reach_p = jnp.exp(-d * lam_cur)                     # P(escape) under theta

            # reach-vs-scatter: SAMPLE against the frozen proposal, REWEIGHT by p_theta/q
            escapes = live & (jax.random.uniform(kr, (n,)) < reach_q)
            ratio = jnp.where(escapes, reach_p / (reach_q + 1e-30),
                              (1.0 - reach_p) / (1.0 - reach_q + 1e-30))
            w_fsi = w_fsi * jnp.where(live, ratio, 1.0)
            alive = alive * jnp.where(escapes, 0.0, 1.0)

            # the interacting fraction chooses absorb vs scatter (proposal threshold, reweight)
            interacts = alive > 0.5
            is_abs = interacts & (jax.random.uniform(kc, (n,)) < absp_q)
            cratio = jnp.where(is_abs, absp_cur / (absp_q + 1e-30),
                               (1.0 - absp_cur) / (1.0 - absp_q + 1e-30))
            w_fsi = w_fsi * jnp.where(interacts, cratio, 1.0)
            absorbed = jnp.where(is_abs, 1.0, absorbed)
            alive = alive * jnp.where(is_abs, 0.0, 1.0)

            # scatterers deflect (detached) and lose a fixed momentum fraction
            scatters = alive > 0.5
            cos_a = sg(hg_sample_cos(jax.random.uniform(ks, (n,)), cfg.g_scatter))
            beta = jax.random.uniform(kb, (n,)) * 2 * jnp.pi
            new_dir = sg(_deflect(direction, cos_a, beta))
            direction = jnp.where(scatters[:, None], new_dir, direction)
            pmag = sg(jnp.where(scatters, pmag * (1.0 - cfg.mom_loss), pmag))
            pos = sg(pos + (0.5 * d * scatters)[:, None] * direction)

        # post-FSI pion 4-momentum (detached): survivors keep degraded |p| along their
        # final direction (on-shell); absorbed pions are removed (zeroed, pid -> 0).
        keep = (1.0 - absorbed)[:, None]
        p3_out = sg(direction * pmag[:, None]) * keep
        E_out = sg(jnp.sqrt(pmag ** 2 + M_PI ** 2)) * keep[:, 0]
        p_pi_out = sg(jnp.concatenate([E_out[:, None], p3_out], axis=1))
        pid_pi_out = jnp.where(absorbed > 0.5, 0, event.pid_pi)

        return event._replace(p_pi=p_pi_out, pid_pi=pid_pi_out, w=event.w * w_fsi)

    # --- self-test contract -------------------------------------------------- #
    def closure_test(self, key=None, n=4000, n_keys=24, **kw) -> TestResult:
        """Differentiability through the cascade: d/d(sigma_sc, sigma_abs) of a final-state
        observable (weighted CC0pi count + mean escaped |p_pi|) -- autodiff vs FD, averaged
        over common-random-number keys.  With detached decision thresholds each key's
        estimator is exact reweighting, so the match is tight."""
        key = jax.random.PRNGKey(7) if key is None else key
        kev, kfit = jax.random.split(key)
        event = toy_event_batch(kev, n)
        base = PhysicsParams()
        theta0 = jnp.array([float(base.fsi_sigma_scatter), float(base.fsi_sigma_abs)])

        def loss(theta, k):
            p = base._replace(fsi_sigma_scatter=theta[0], fsi_sigma_abs=theta[1])
            ev = self.apply(p, event, key=k, proposal=base)   # frozen proposal -> exact FD
            absorbed = (ev.pid_pi == 0)
            cc0pi = jnp.sum(ev.w * absorbed)
            pmag = jnp.linalg.norm(ev.p_pi[:, 1:], axis=1)
            ppi = jnp.sum(ev.w * (~absorbed) * pmag)
            return cc0pi + 1e-3 * ppi

        gc: GradCheck = check_gradient(loss, theta0, kfit, eps=1e-3, tol=2e-2, n_keys=n_keys)
        return TestResult(
            f"{type(self).__name__}.closure", "closure", bool(gc.passed), False,
            f"d/d(sig_sc,sig_abs): autodiff {gc.autodiff} vs FD {gc.finite_diff} "
            f"max_rel {gc.max_rel_err:.2e} (tol 2e-2, n_keys={n_keys})",
            {"autodiff": gc.autodiff.tolist(), "finite_diff": gc.finite_diff.tolist(),
             "max_rel_err": gc.max_rel_err},
        )


def toy_event_batch(key, n, e_pi_lo=120.0, e_pi_hi=450.0):
    """A minimal synthetic produced-pion EventRecord for the FSI closure/joint tests.

    Only the pion kinematics and the weight matter to the cascade; the leptonic legs are
    filled with placeholders.  Pions are isotropic with |p| uniform in [e_pi_lo, e_pi_hi]
    MeV, weight 1.  pid_pi = 211 (pi+), so absorbed events (-> pid 0) are identifiable.
    """
    from adonis.core.event import EventRecord
    kdir, kp = jax.random.split(key)
    cos_t = jax.random.uniform(kdir, (n,), minval=-1.0, maxval=1.0)
    phi = jax.random.uniform(kp, (n,)) * 2 * jnp.pi
    sin_t = jnp.sqrt(jnp.maximum(1 - cos_t ** 2, 0.0))
    dirs = jnp.stack([sin_t * jnp.cos(phi), sin_t * jnp.sin(phi), cos_t], axis=1)
    pmag = jax.random.uniform(jax.random.fold_in(key, 1), (n,), minval=e_pi_lo, maxval=e_pi_hi)
    p3 = dirs * pmag[:, None]
    E = jnp.sqrt(pmag ** 2 + M_PI ** 2)
    p_pi = jnp.concatenate([E[:, None], p3], axis=1)
    z = jnp.zeros((n, 4))
    o1 = jnp.ones(n)
    return EventRecord(
        k=z, kp=z, p_struck=z, p_pi=p_pi, p_N=z, w=o1,
        channel=jnp.zeros(n, jnp.int32), pid_pi=jnp.full(n, 211, jnp.int32),
        pid_N=jnp.full(n, 2212, jnp.int32), pid_Ni=jnp.full(n, 2112, jnp.int32),
        W=jnp.full(n, 1232.0), Q2_adj=jnp.full(n, 1.0e5),
    )
