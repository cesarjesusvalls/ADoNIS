"""Real differentiable intranuclear NUCLEON cascade -- the proton/neutron FSI needed for the
TKI observables (the leading-proton rescattering that distorts delta_pT / delta_alphaT / p_N).

Port of the ACHILLES NucleonNucleon GiBUU ELASTIC cross section (`NucleonNucleon.cc::NNElastic`,
piecewise in p_lab [GeV], pp/nn vs np) + two-body kinematics off a Fermi-moving background
nucleon, with Pauli blocking on BOTH outgoing nucleons (ACHILLES rejects the interaction if
either is blocked).  Nucleons are not absorbed; they rescatter (adding a high-imbalance tail)
or are Pauli-blocked (no interaction).  Same transport as the pion cascade (continuum
mean-free-path through rho(r), Local Fermi gas).  Inelastic NN->NDelta->NNpi is left to the
primary/pion sector (negligible for the leading nucleon's TKI smearing in the CC0pi/CC1pi
samples).  Geometry/kinematics are sampled against a frozen proposal (kind-1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

from adonis.fsi import oset_xsec as ox
from adonis.fsi.cascade_real import (_load_density, _rho_species, _kf_local,
                                     _sample_fermi_nucleon, _two_body_cm_scatter, MB_TO_FM2)

M_N = ox.M_N
M_N_GEV = M_N / 1000.0


def nn_elastic_sigma(sqrts_mev, same_iso):
    """NN elastic cross section [mb] (ACHILLES NNElastic, GiBUU).  sqrts in MeV; `same_iso`
    True for pp/nn, False for np.  Piecewise in p_lab [GeV]."""
    sqrts = sqrts_mev / 1000.0                       # GeV
    mn = M_N_GEV
    thr = sqrts ** 2 - 4.0 * mn ** 2
    plab = jnp.where(thr > 0, sqrts / (2.0 * mn) * jnp.sqrt(jnp.clip(thr, 0.0, None)), 0.0)
    thr_safe = jnp.clip(thr, 1e-6, None)
    L = jnp.log(sqrts ** 2)

    def same():
        return jnp.where(plab < 0.425, 5.12 * mn / thr_safe + 1.67,
               jnp.where(plab < 0.8, 23.5 + 1000.0 * (plab - 0.7) ** 4,
               jnp.where(plab < 2.0, 1250.0 / (plab + 50.0) - 4.0 * (plab - 1.3) ** 2,
               jnp.where(plab < 6.0, 77.0 / (plab + 1.5),
                         11.84 - 1.617 * L + 0.1359 * L ** 2))))

    def diff():
        return jnp.where(plab < 0.525, 17.05 * mn / thr_safe - 6.83,
               jnp.where(plab < 0.8, 33.0 + 196.0 * jnp.abs(plab - 0.95) ** 2.5,
               jnp.where(plab < 2.0, 31.0 / jnp.sqrt(jnp.clip(plab, 1e-6, None)),
               jnp.where(plab < 6.0, 77.0 / (plab + 1.5),
                         11.84 - 1.617 * L + 0.1359 * L ** 2))))

    sig = jnp.where(same_iso, same(), diff())
    return jnp.where(thr > 0, jnp.clip(sig, 0.0, None), 0.0)


@dataclass(frozen=True)
class NucleonCascadeConfig:
    nucleus: str = "c12_density.txt"
    step: float = 0.08
    max_steps: int = 220
    seed: int = 0


def propagate_nucleon(pos0, p_N0, is_proton0, cfg: NucleonCascadeConfig, key, protfrac=0.0):
    """Transport a batch of leading nucleons. pos0 (N,3) fm, p_N0 (N,4) MeV, is_proton0 (N,) bool.
    Returns (p_N_final (N,4), n_scatter (N,)).  The nucleon identity is conserved by elastic
    scattering; we follow the leading nucleon (the struck background nucleon becomes a secondary,
    not tracked here)."""
    _load_density(cfg.nucleus)          # warm cache EAGERLY (avoid tracer leak inside jit)
    return _propagate_nucleon_scan(pos0, p_N0, is_proton0, cfg, key, protfrac)


from functools import partial as _partial


@_partial(jax.jit, static_argnums=(3,))
def _propagate_nucleon_scan(pos0, p_N0, is_proton0, cfg: NucleonCascadeConfig, key, protfrac):
    rgrid, rho, radius = _load_density(cfg.nucleus)
    n = pos0.shape[0]
    keys = jax.random.split(key, cfg.max_steps)

    def body(carry, step_key):
        pos, p_N, isp, alive, nsc = carry
        r = jnp.linalg.norm(pos, axis=1)
        alive = alive & (r <= radius)
        rho_p = _rho_species(r, rgrid, rho)
        rho_tot = 2.0 * rho_p
        kf = _kf_local(rho_p)

        kN, step_key = jax.random.split(step_key)
        kpt, kN = jax.random.split(kN)
        zoa = (1.0 - protfrac) / 2.0                 # background proton fraction Z/A
        bg_is_proton = jax.random.uniform(kpt, (n,)) < zoa
        p_bg = jax.vmap(_sample_fermi_nucleon)(kf, jax.random.split(kN, n))

        same_iso = (isp == bg_is_proton)
        P = p_N + p_bg
        s = P[:, 0] ** 2 - jnp.sum(P[:, 1:] ** 2, axis=1)
        sqrts = jnp.sqrt(jnp.clip(s, (2 * M_N) ** 2, None))
        sigma = nn_elastic_sigma(sqrts, same_iso) * MB_TO_FM2
        p_int = -jnp.expm1(-rho_tot * sigma * cfg.step)

        kI, kS = jax.random.split(step_key)
        interacts = alive & (jax.random.uniform(kI, (n,)) < p_int)

        def scat_one(p_lead, p_bg_i, kf_i, k):
            p_out = _two_body_cm_scatter(p_lead, p_bg_i, M_N, k)
            p_rec = (p_lead + p_bg_i) - p_out
            blocked = (jnp.linalg.norm(p_out[1:]) < kf_i) | (jnp.linalg.norm(p_rec[1:]) < kf_i)
            return p_out, blocked
        p_out, blocked = jax.vmap(scat_one)(p_N, p_bg, kf, jax.random.split(kS, n))
        do_scatter = interacts & ~blocked
        p_N = jnp.where(do_scatter[:, None], p_out, p_N)
        nsc = nsc + do_scatter.astype(jnp.int32)

        v3 = p_N[:, 1:]
        d = v3 / jnp.clip(jnp.linalg.norm(v3, axis=1, keepdims=True), 1e-9, None)
        pos = jnp.where(alive[:, None], pos + cfg.step * d, pos)
        return (pos, p_N, isp, alive, nsc), None

    init = (pos0, p_N0, is_proton0, jnp.ones(n, bool), jnp.zeros(n, jnp.int32))
    (pos, p_N, isp, alive, nsc), _ = jax.lax.scan(body, init, keys)
    return p_N, nsc


class NucleonFSI:
    """FSIModel add-on: propagate the leading outgoing NUCLEON (p_N) through the nucleus with the
    NN-elastic cascade.  Composable with RealCascadeFSI (pion) -- apply both for full CC1pi+ FSI.
    protfrac = (N_n - N_p)/A (0 for 12C, 0.1 for 40Ar)."""

    def __init__(self, cfg: NucleonCascadeConfig = NucleonCascadeConfig(), protfrac: float = 0.0):
        self.cfg = cfg
        self.protfrac = float(protfrac)

    def apply(self, params, event, key=None):
        from adonis.fsi.cascade_real import sample_vertex
        key = jax.random.PRNGKey(self.cfg.seed + 1) if key is None else key
        kv, kp = jax.random.split(key)
        n = event.p_N.shape[0]
        is_proton = jnp.asarray(np.asarray(event.pid_N) == 2212)
        pos0 = sample_vertex(kv, n, self.cfg.nucleus)
        p_N, nsc = propagate_nucleon(pos0, event.p_N, is_proton, self.cfg, kp, self.protfrac)
        return event._replace(p_N=p_N)
