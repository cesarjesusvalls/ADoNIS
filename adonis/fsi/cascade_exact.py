"""BIT-EXACT (non-differentiable) port of the ACHILLES nucleon cascade (Cascade.cc BaseAlgorithm),
for DIAGNOSIS ONLY -- to isolate whether the QE-FSI residual is the discrete-Glauber approximation
(2-generation, best-knockout-only) or a real remaining discrepancy.

The single-particle walk is identical to the differentiable discrete-Glauber cascade (closest
background nucleon inside the Cylinder b^2 < sigma/(10 pi) in a 0.04 fm slab, NN-elastic isotropic
CM, Pauli on both outgoing, formation zone E_in*hbarc/|mN^2-p_in.p_out|).  The ONLY difference here
is FULL multi-particle recursion: EVERY knocked-out nucleon re-cascades (sharing the consumed
background set), as in ACHILLES UpdateKicked -- not just 2 generations of the single best knockout.

Physics (sigma, kf, scatter, formation zone) is mirrored from the discrete-Glauber path so that the
ONLY thing that changes between the two is the cascade algorithm.  Pure numpy; not differentiable.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from adonis.fsi import oset_xsec as ox
from adonis.fsi.cascade_real import _load_density, _rho_species, _kf_local
from adonis.fsi.cascade_discrete import sample_nucleons

M_N = ox.M_N
HBARC = ox.HBARC
MB_TO_FM2 = 0.1            # 1 mb = 0.1 fm^2  (ACHILLES xsec/10)


def _nn_sigma(sqrts_mev, same_iso):
    """NN elastic sigma [mb] -- numpy mirror of nucleon_cascade.nn_elastic_sigma (GiBUU)."""
    sqrts = sqrts_mev / 1000.0; mn = M_N / 1000.0
    thr = sqrts ** 2 - 4.0 * mn ** 2
    plab = np.where(thr > 0, sqrts / (2.0 * mn) * np.sqrt(np.clip(thr, 0.0, None)), 0.0)
    thr_s = np.clip(thr, 1e-6, None); L = np.log(sqrts ** 2)
    same = np.where(plab < 0.425, 5.12 * mn / thr_s + 1.67,
           np.where(plab < 0.8, 23.5 + 1000.0 * (plab - 0.7) ** 4,
           np.where(plab < 2.0, 1250.0 / (plab + 50.0) - 4.0 * (plab - 1.3) ** 2,
           np.where(plab < 6.0, 77.0 / (plab + 1.5), 11.84 - 1.617 * L + 0.1359 * L ** 2))))
    diff = np.where(plab < 0.525, 17.05 * mn / thr_s - 6.83,
           np.where(plab < 0.8, 33.0 + 196.0 * np.abs(plab - 0.95) ** 2.5,
           np.where(plab < 2.0, 31.0 / np.sqrt(np.clip(plab, 1e-6, None)),
           np.where(plab < 6.0, 77.0 / (plab + 1.5), 11.84 - 1.617 * L + 0.1359 * L ** 2))))
    sig = np.where(same_iso, same, diff)
    return np.where(thr > 0, np.clip(sig, 0.0, None), 0.0)


def _boost_to_lab(p4cm, P):
    """Boost CM 4-vec (...,4) to the lab frame whose parent is P (...,4)."""
    rsq = np.sqrt(np.clip(P[..., 0] ** 2 - np.sum(P[..., 1:] ** 2, axis=-1), 1e-9, None))
    E = (P[..., 0] * p4cm[..., 0] + np.sum(P[..., 1:] * p4cm[..., 1:], axis=-1)) / rsq
    c1 = (p4cm[..., 0] + E) / (rsq + P[..., 0])
    return np.concatenate([E[..., None], p4cm[..., 1:] + c1[..., None] * P[..., 1:]], axis=-1)


def _scatter_iso(p_lead, p_tgt, rng):
    """Elastic NN scatter, isotropic in CM (ACHILLES NucleonNucleon::GenerateMomentum).  Returns the
    outgoing LEADING nucleon 4-mom (n,4); recoil = p_lead + p_tgt - out."""
    P = p_lead + p_tgt
    s = P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, axis=1)
    sqrts = np.sqrt(np.clip(s, (2 * M_N) ** 2, None))
    pcm = np.sqrt(np.clip(sqrts ** 2 / 4.0 - M_N ** 2, 0.0, None))
    ct = 2.0 * rng.random(len(p_lead)) - 1.0; st = np.sqrt(np.clip(1 - ct ** 2, 0.0, None))
    ph = 2 * np.pi * rng.random(len(p_lead))
    d = np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)
    out_cm = np.concatenate([(sqrts / 2.0)[:, None], pcm[:, None] * d], axis=1)
    return _boost_to_lab(out_cm, P)


def _fz(p_in, p_out):
    dot4 = p_in[:, 0] * p_out[:, 0] - np.sum(p_in[:, 1:] * p_out[:, 1:], axis=1)
    return p_in[:, 0] * HBARC / np.clip(np.abs(M_N ** 2 - dot4), 1e-6, None)


def run_exact_nucleon_cascade(p_lead0, isp0, cfg, seed, K=8):
    """Bit-exact nucleon cascade with FULL secondary recursion.  p_lead0 (n,4) leading nucleon,
    isp0 (n,) proton mask.  Returns the highest-momentum final-state PROTON per event (n,4)
    (the analysis then applies the CC0pi-Np window).  K = max simultaneously-tracked particles."""
    n = p_lead0.shape[0]
    knd, kvtx = jax.random.split(jax.random.PRNGKey(seed), 2)
    npos, nmom, nisp = sample_nucleons(knd, n, cfg)
    npos = np.asarray(npos); nmom = np.asarray(nmom); nisp = np.asarray(nisp)
    rgrid, rho, _rhoN, radius = _load_density(cfg.nucleus, getattr(cfg, "density_n", None)); radius = float(radius)
    A = nisp.shape[1]; ar = np.arange(n)
    rnuc = np.linalg.norm(npos, axis=2)
    kf_n = np.asarray(_kf_local(_rho_species(jnp.asarray(rnuc), rgrid, rho)))   # (n,A)
    rng = np.random.default_rng(seed + 1)

    consumed = np.zeros((n, A), bool)
    pos = np.zeros((n, K, 3)); mom = np.zeros((n, K, 4)); active = np.zeros((n, K), bool)
    fz = np.zeros((n, K)); isp = np.zeros((n, K), bool)
    # production vertex = the struck NEUTRON's config nucleon (nu_mu n -> mu p).  CONSUME it from the
    # background: ACHILLES turns the struck nucleon INTO the outgoing proton, so it is no longer a
    # scatterer -> the leading proton walks through the remaining A-1 nucleons (11, not 12).
    rsel = rng.random((n, A)); rsel = np.where(~nisp, rsel, -1.0); vtx = np.argmax(rsel, axis=1)
    consumed[ar, vtx] = True
    pos[:, 0, :] = npos[ar, vtx]; mom[:, 0, :] = p_lead0; active[:, 0] = True; isp[:, 0] = isp0
    best_p = np.zeros((n, 4))                                  # highest-momentum final proton

    def _record_final(slot_mom, slot_isp, mask):
        nonlocal best_p
        pm = np.linalg.norm(slot_mom[:, 1:], axis=1)
        better = mask & slot_isp & (pm > np.linalg.norm(best_p[:, 1:], axis=1))
        best_p = np.where(better[:, None], slot_mom, best_p)

    for _step in range(cfg.max_steps):
        if not active.any():
            break
        beta_all = np.linalg.norm(mom[:, :, 1:], axis=2) / np.clip(mom[:, :, 0], 1e-9, None)
        max_beta = np.maximum(np.where(active, beta_all, 0.0).max(axis=1), 1e-6)     # (n,) AdaptiveStep
        timeStep = cfg.step / max_beta
        for k in range(K):
            act = active[:, k]
            if not act.any():
                continue
            mk = mom[:, k, :]; pk = pos[:, k, :]
            bk = np.linalg.norm(mk[:, 1:], axis=1) / np.clip(mk[:, 0], 1e-9, None)
            seglen = bk * timeStep                                                   # spatial step
            dk = mk[:, 1:] / np.clip(np.linalg.norm(mk[:, 1:], axis=1, keepdims=True), 1e-9, None)
            in_fz = fz[:, k] > 0
            do_int = act & ~in_fz
            rel = npos - pk[:, None, :]
            par = np.sum(rel * dk[:, None, :], axis=2)
            perp2 = np.sum(rel ** 2, axis=2) - par ** 2
            in_slab = (par > 0) & (par <= seglen[:, None]) & (~consumed) & do_int[:, None]
            Pp = mk[:, None, :] + nmom
            s = Pp[:, :, 0] ** 2 - np.sum(Pp[:, :, 1:] ** 2, axis=2)
            sqrts = np.sqrt(np.clip(s, (2 * M_N) ** 2, None))
            sig = np.clip(_nn_sigma(sqrts, isp[:, k][:, None] == nisp), 0.0, None)
            incyl = in_slab & (perp2 < sig * MB_TO_FM2 / np.pi)
            big = np.where(incyl, perp2, np.inf)
            j = np.argmin(big, axis=1)
            has_hit = np.isfinite(big[ar, j]) & do_int
            pN_j = nmom[ar, j]; kf_j = kf_n[ar, j]
            p_out = _scatter_iso(mk, pN_j, rng); recoil = (mk + pN_j) - p_out
            blocked = (np.linalg.norm(p_out[:, 1:], axis=1) < kf_j) | (np.linalg.norm(recoil[:, 1:], axis=1) < kf_j)
            scat = has_hit & ~blocked
            fz_lead = _fz(mk, p_out); fz_ko = _fz(mk, recoil)
            # apply scatter to particle k; spawn recoil; consume struck background nucleon
            mom[scat, k, :] = p_out[scat]; fz[scat, k] = fz_lead[scat]
            for ev in np.nonzero(scat)[0]:
                consumed[ev, j[ev]] = True
                free = np.nonzero(~active[ev])[0]
                if len(free):
                    sl = free[0]
                    active[ev, sl] = True; mom[ev, sl, :] = recoil[ev]; pos[ev, sl, :] = npos[ev, j[ev]]
                    fz[ev, sl] = fz_ko[ev]; isp[ev, sl] = nisp[ev, j[ev]]
            # propagate particle k; decrement fz
            pos[act, k, :] = pk[act] + seglen[act, None] * dk[act]
            dec = act & (fz[:, k] > 0)
            fz[dec, k] -= timeStep[dec]
            # escape: outside radius & moving outward -> final state (capture if KE < 10 MeV)
            r_new = np.linalg.norm(pos[:, k, :], axis=1)
            outward = np.sum(pos[:, k, :] * dk, axis=1) > 0
            escaped = act & (r_new > radius) & outward
            captured = escaped & ((mom[:, k, 0] - M_N - 10.0) <= 0.0)               # too soft -> captured
            _record_final(mom[:, k, :], isp[:, k], escaped & ~captured)
            active[escaped, k] = False
    # particles still propagating at maxSteps: take them as final (ACHILLES would force-resolve)
    for k in range(K):
        _record_final(mom[:, k, :], isp[:, k], active[:, k])
    return best_p
