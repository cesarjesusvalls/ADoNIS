"""SHARED-STATE discrete event cascade: one nucleus per event, all hadrons co-evolve
(ACHILLES Cascade::Evolve semantics; see docs/logbook/cc1pi_t2k.md #10).

The factorized chain (DiscreteCascadeFSI then DiscreteNucleonFSI with independent resampled
backgrounds and a bounded candidate set) reproduces every component at the few-% level but
composes them differently from ACHILLES's single evolving event state -- the residual +7%
CC1pi carbon excess.  Here the RES pion and nucleon start at the TRUE struck vertex of ONE
sampled QMC configuration, share the consumed-nucleon mask, and their products (NN recoils,
pi-scatter recoils, NN->NDelta decay nucleons + created pions) spawn into fixed particle
slots and cascade themselves (created pions can be re-absorbed).

Slots (n, K_PART): species code per slot: 0 = empty/dead, 1 = pion (charge in chg: +1/0/-1),
2 = nucleon (chg: 1 proton / 0 neutron).  v1 is nominal-only (no kind-1 records).
Physics reused verbatim from the validated single-particle kernels: oset abs (+isospin
partition), charge-resolved MB elastic + DCC angles, piN conversion, NN elastic (GiBUU),
NN->NDelta (Dmitriev-Sushkov) with isotropic Delta production/decay, Pauli on outgoing
nucleons, ACHILLES formation zones (nucleons only), sphere escape (in-event = internal)."""
from __future__ import annotations

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from adonis.fsi import oset_xsec as ox
from adonis.fsi.mb import cascade_mb
from adonis.fsi import nn_inelastic as nni
from adonis.fsi.nucleon_cascade import nn_elastic_sigma
from adonis.fsi.cascade_real import (_load_density, _rho_species, _kf_local, MB_TO_FM2,
                                     _CH_MASS)
from adonis.fsi.cascade_discrete import (_formation_zone, sample_nucleons,
                                         DiscreteCascadeConfig)

M_N = ox.M_N
K_PART = 8


def _boost4(p4, beta, sign=+1.0):
    """Boost p4 (n,4) by sign*beta (n,3)."""
    b = sign * beta
    b2 = jnp.sum(b ** 2, axis=-1)
    g = 1.0 / jnp.sqrt(jnp.clip(1 - b2, 1e-12, None))
    bp = jnp.sum(b * p4[..., 1:], axis=-1)
    E = g * (p4[..., 0] + bp)
    p3 = p4[..., 1:] + ((g - 1) * bp / jnp.clip(b2, 1e-30, None) + g * p4[..., 0])[..., None] * b
    return jnp.concatenate([E[..., None], p3], axis=-1)


def _two_body(P4, mA, mB, cth, phi):
    """Isotropic CM split of P4 (n,4) into masses (mA, mB) at angles (cth, phi); lab outputs."""
    ss = jnp.clip(P4[:, 0] ** 2 - jnp.sum(P4[:, 1:] ** 2, axis=1), (mA + mB) ** 2 * 1.0001, None)
    rss = jnp.sqrt(ss)
    EA = (ss + mA ** 2 - mB ** 2) / (2 * rss)
    pf = jnp.sqrt(jnp.clip(EA ** 2 - mA ** 2, 0.0, None))
    sth = jnp.sqrt(jnp.clip(1 - cth ** 2, 0, None))
    d = jnp.stack([sth * jnp.cos(phi), sth * jnp.sin(phi), cth], axis=1)
    pa = jnp.concatenate([EA[:, None], pf[:, None] * d], axis=1)
    pb = jnp.concatenate([(rss - EA)[:, None], -pf[:, None] * d], axis=1)
    beta = P4[:, 1:] / P4[:, [0]]
    return _boost4(pa, beta), _boost4(pb, beta)


def _dcc_scatter(p_pi, pN, m_out, cos_cm, phi):
    """Pion elastic scatter with the DCC cos_cm (mirrors cascade_real._two_body_cm_scatter
    semantics: cos measured wrt the incoming pion direction in the piN CM)."""
    P = p_pi + pN
    beta = P[:, 1:] / P[:, [0]]
    pi_cm = _boost4(p_pi, beta, -1.0)
    k = jnp.linalg.norm(pi_cm[:, 1:], axis=1)
    zhat = pi_cm[:, 1:] / jnp.clip(k[:, None], 1e-9, None)
    # orthonormal frame around zhat
    a = jnp.where((jnp.abs(zhat[:, 2]) < 0.9)[:, None],
                  jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0]), zhat.shape),
                  jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), zhat.shape))
    e1 = jnp.cross(zhat, a); e1 = e1 / jnp.clip(jnp.linalg.norm(e1, axis=1, keepdims=True), 1e-9, None)
    e2 = jnp.cross(zhat, e1)
    sth = jnp.sqrt(jnp.clip(1 - cos_cm ** 2, 0, None))
    d = cos_cm[:, None] * zhat + sth[:, None] * (jnp.cos(phi)[:, None] * e1 + jnp.sin(phi)[:, None] * e2)
    ss = jnp.clip(P[:, 0] ** 2 - jnp.sum(P[:, 1:] ** 2, axis=1), 1.0, None)
    rss = jnp.sqrt(ss)
    Epi = (ss + m_out ** 2 - M_N ** 2) / (2 * rss)
    kf_ = jnp.sqrt(jnp.clip(Epi ** 2 - m_out ** 2, 0.0, None))
    pi_out_cm = jnp.concatenate([Epi[:, None], kf_[:, None] * d], axis=1)
    pi_out = _boost4(pi_out_cm, beta)
    return pi_out, (p_pi + pN) - pi_out


@partial(jax.jit, static_argnums=(6,))
def _evolve(p0, spec0, chg0, npos, nmom, nisp, cfg: DiscreteCascadeConfig, key, consumed0,
            pos_init):
    """Co-evolve the particle slots.  p0 (n,K,4); spec0 (n,K) int 0/1/2; chg0 (n,K) int.
    Returns final (p, spec, chg, made_pi_n (n,), nabs (n,))."""
    rgrid, rho, radius = _load_density(cfg.nucleus)
    n, K, _ = p0.shape
    A = nisp.shape[1]
    arn = jnp.arange(n)
    keys = jax.random.split(key, cfg.max_steps)

    def body(state):
        i, p, spec, chg, pos, fz, consumed, absorbed_pi, inted = state
        sk = keys[i]
        alive = spec > 0
        mom = jnp.linalg.norm(p[..., 1:], axis=-1)
        dhat = p[..., 1:] / jnp.clip(mom[..., None], 1e-9, None)
        # escape: pions before their FIRST interaction follow the ACHILLES external_test
        # z>=radius PLANE rule (validated in the factorized chain vs the in-event absorption
        # fractions); everything else escapes on the sphere (outward).
        outward = jnp.sum(pos * dhat, axis=-1) > 0
        ext_pi = (spec == 1) & ~inted
        esc_plane = ext_pi & (pos[..., 2] >= radius)
        esc_sphere = ~ext_pi & (jnp.linalg.norm(pos, axis=-1) > radius) & outward
        esc = (esc_plane | esc_sphere) & alive
        spec = jnp.where(esc, -spec, spec).astype(jnp.int32)        # negative = escaped (final)
        alive = spec > 0
        # formation zone (nucleons)
        beta = mom / jnp.clip(p[..., 0], 1e-9, None)
        fz = jnp.where((fz > 0) & alive, fz - cfg.step / jnp.clip(beta, 1e-6, None), fz)
        can_int = (fz <= 0.0) & alive
        # slab geometry per slot: (n,K,A)
        rel = npos[:, None, :, :] - pos[:, :, None, :]
        par = jnp.sum(rel * dhat[:, :, None, :], axis=-1)
        perp2 = jnp.sum(rel ** 2, axis=-1) - par ** 2
        cand = (par > 0) & (par <= cfg.step) & (~consumed[:, None, :]) & can_int[:, :, None]
        # ---- per-slot total sigma (n,K,A) ----
        Pp = p[:, :, None, :] + nmom[:, None, :, :]
        s_pair = jnp.clip(Pp[..., 0] ** 2 - jnp.sum(Pp[..., 1:] ** 2, axis=-1), 1.0, None)
        W = jnp.sqrt(s_pair)
        is_pi = spec == 1
        is_nuc = spec == 2
        # pion sigmas
        pi_ch_idx = jnp.clip(1 - chg, 0, 2)                     # charge +1/0/-1 -> idx 0/1/2
        vpart = p[..., 1:] / jnp.clip(p[..., 0:1], 1e-9, None)
        vN = nmom[..., 1:] / nmom[..., 0:1]
        vrel = jnp.clip(jnp.linalg.norm(vpart[:, :, None, :] - vN[:, None, :, :], axis=-1), 1e-3, None)
        rnuc = jnp.linalg.norm(npos, axis=-1)
        rho_t = 2.0 * _rho_species(rnuc, rgrid, rho)
        kf_bg = _kf_local(_rho_species(rnuc, rgrid, rho))
        m_pi_k = _CH_MASS[pi_ch_idx]
        sa = jnp.clip(ox.abs_cross_section(p[..., 0:1] * jnp.ones((1, 1, A)),
                                           m_pi_k[..., None] * jnp.ones((1, 1, A)),
                                           mom[..., None] * jnp.ones((1, 1, A)),
                                           vrel, jnp.clip(kf_bg, 1e-6, None)[:, None, :],
                                           jnp.clip(rho_t, 1e-9, None)[:, None, :]), 0.0, None)
        nuc_i = jnp.where(nisp, 0, 1).astype(jnp.int32)
        sio = cascade_mb.jax_channel_sigmas_resolved(
            W.reshape(-1), jnp.broadcast_to(pi_ch_idx[:, :, None], (n, K, A)).reshape(-1),
            jnp.broadcast_to(nuc_i[:, None, :], (n, K, A)).reshape(-1)).reshape(n, K, A, 3)
        ss_pi = jnp.clip(jnp.sum(sio, axis=-1), 0.0, None)
        si_pi = jnp.clip(cascade_mb.jax_conversion_sigma(
            W.reshape(-1), jnp.broadcast_to(pi_ch_idx[:, :, None], (n, K, A)).reshape(-1),
            jnp.broadcast_to(nuc_i[:, None, :], (n, K, A)).reshape(-1)).reshape(n, K, A), 0.0, None)
        like = ((pi_ch_idx == 0)[:, :, None] & nisp[:, None, :]) | ((pi_ch_idx == 2)[:, :, None] & (~nisp[:, None, :]))
        sa = sa * jnp.where(like, 5.0 / 6.0, 1.0)
        sig_pi = sa + ss_pi + si_pi
        # nucleon sigmas
        same_iso = (chg == 1)[:, :, None] == nisp[:, None, :]
        sig_el_n = jnp.clip(nn_elastic_sigma(W, same_iso), 0.0, None)
        if cfg.nn_inelastic:
            pcm_n = jnp.sqrt(jnp.clip(s_pair / 4.0 - M_N ** 2, 1e-6, None)) / 1000.0
            sig_in_n = jnp.clip(nni.sigma_nn_ndelta(W / 1000.0, pcm_n, same_iso), 0.0, None)
        else:
            sig_in_n = jnp.zeros_like(sig_el_n)
        sig_nuc = sig_el_n + sig_in_n
        sig = jnp.where(is_pi[:, :, None], sig_pi, jnp.where(is_nuc[:, :, None], sig_nuc, 0.0))
        # interaction roll
        sfm = jnp.clip(sig * MB_TO_FM2, 1e-12, None)
        if cfg.cylinder:
            prob = jnp.where(cand & (perp2 < sfm / jnp.pi), 1.0, 0.0)
        else:
            prob = jnp.where(cand, jnp.exp(-jnp.pi * perp2 / sfm), 0.0)
        k_u = jax.random.fold_in(sk, 1)
        passes = cand & (jax.random.uniform(k_u, (n, K, A)) < prob)
        metric = jnp.where(passes, perp2, jnp.inf)
        j = jnp.argmin(metric, axis=-1)                          # (n,K)
        has_hit = jnp.isfinite(jnp.min(metric, axis=-1))
        # resolve ONE interaction per particle per step; if two slots hit the SAME background
        # nucleon this step, keep the lower slot (rare; declared)
        jj = jnp.where(has_hit, j, A)
        dup = jnp.zeros((n, K), bool)
        seen = jnp.full((n, A + 1), -1, jnp.int32)
        for kk in range(K):
            taken = seen[arn, jj[:, kk]] >= 0
            dup = dup.at[:, kk].set(taken & has_hit[:, kk])
            seen = seen.at[arn, jj[:, kk]].set(jnp.where(has_hit[:, kk] & ~taken, kk, seen[arn, jj[:, kk]]))
        has_hit = has_hit & ~dup
        gi = (arn[:, None], j)
        pN_j = nmom[gi[0], gi[1]]
        kf_loc = _kf_local(_rho_species(jnp.linalg.norm(npos, axis=-1), rgrid, rho))
        kf_j = kf_loc[gi[0], gi[1]]
        W_j = W[arn[:, None], jnp.arange(K)[None, :], j]
        sa_j = sa[arn[:, None], jnp.arange(K)[None, :], j]
        si_pi_j = si_pi[arn[:, None], jnp.arange(K)[None, :], j]
        sig_pi_j = sig_pi[arn[:, None], jnp.arange(K)[None, :], j]
        sin_n_j = sig_in_n[arn[:, None], jnp.arange(K)[None, :], j]
        sig_n_j = sig_nuc[arn[:, None], jnp.arange(K)[None, :], j]
        # ---- branch picks ----
        u_br = jax.random.uniform(jax.random.fold_in(sk, 2), (n, K))
        pi_abs_pick = is_pi & has_hit & (u_br < sa_j / jnp.clip(sig_pi_j, 1e-12, None))
        # ---- ABSORPTION final state (ACHILLES PionAbsorption::GenerateMomentum): pion +
        # struck + CLOSEST charge-allowed partner -> 2 nucleons, isotropic in the 3-body CM;
        # Pauli on BOTH products REJECTS the absorption (pion continues) -- this rejection is
        # the low-T_pi physics the v1 kernel missed.
        qpi = chg                                                # pion charge (+1/0/-1)
        struck_p = nisp[gi[0], gi[1]].astype(jnp.int32)          # (n,K)
        forced_n = (qpi + struck_p) > 1
        forced_p = (qpi + struck_p) < 0
        d2p = jnp.sum((npos[:, None, :, :] - npos[gi[0], gi[1]][:, :, None, :]) ** 2, axis=-1)  # (n,K,A)
        bad = ((jnp.arange(A)[None, None, :] == j[..., None]) | consumed[:, None, :]
               | (forced_n[..., None] & nisp[:, None, :]) | (forced_p[..., None] & ~nisp[:, None, :]))
        d2p = jnp.where(bad, jnp.inf, d2p)
        pj = jnp.argmin(d2p, axis=-1)                            # partner index (n,K)
        no_partner = ~jnp.isfinite(jnp.min(d2p, axis=-1))        # no charge-allowed partner -> no abs
        pN_part = nmom[arn[:, None], pj]
        P3 = p + pN_j + pN_part
        s3 = jnp.clip(P3[..., 0] ** 2 - jnp.sum(P3[..., 1:] ** 2, axis=-1), (2 * M_N) ** 2, None)
        Estar = jnp.sqrt(s3) / 2.0
        pstar = jnp.sqrt(jnp.clip(Estar ** 2 - M_N ** 2, 0.0, None))
        cth_ab = 2 * jax.random.uniform(jax.random.fold_in(sk, 13), (n, K)) - 1
        phi_ab = 2 * jnp.pi * jax.random.uniform(jax.random.fold_in(sk, 14), (n, K))
        sth_ab = jnp.sqrt(jnp.clip(1 - cth_ab ** 2, 0, None))
        d_ab = jnp.stack([sth_ab * jnp.cos(phi_ab), sth_ab * jnp.sin(phi_ab), cth_ab], axis=-1)
        beta3 = P3[..., 1:] / P3[..., 0:1]
        pa3 = jnp.concatenate([Estar[..., None], pstar[..., None] * d_ab], axis=-1)
        pb3 = jnp.concatenate([Estar[..., None], -pstar[..., None] * d_ab], axis=-1)
        pa3 = _boost4(pa3.reshape(n * K, 4), beta3.reshape(n * K, 3)).reshape(n, K, 4)
        pb3 = _boost4(pb3.reshape(n * K, 4), beta3.reshape(n * K, 3)).reshape(n, K, 4)
        kf_pi_pos = _kf_local(_rho_species(jnp.linalg.norm(pos, axis=-1), rgrid, rho))   # (n,K)
        abs_blocked = ((jnp.linalg.norm(pa3[..., 1:], axis=-1) < kf_pi_pos)
                       | (jnp.linalg.norm(pb3[..., 1:], axis=-1) < kf_j))
        if not cfg.pauli:
            abs_blocked = abs_blocked & False
        pi_abs = pi_abs_pick & ~abs_blocked & ~no_partner
        pi_conv = is_pi & has_hit & ~pi_abs_pick & (u_br < (sa_j + si_pi_j) / jnp.clip(sig_pi_j, 1e-12, None))
        pi_scat = is_pi & has_hit & ~pi_abs_pick & ~pi_conv
        n_inel = is_nuc & has_hit & (u_br < sin_n_j / jnp.clip(sig_n_j, 1e-12, None))
        n_el = is_nuc & has_hit & ~n_inel
        # ---- PION SCATTER (charge-resolved out channel + DCC angle) ----
        sio_j = sio[arn[:, None], jnp.arange(K)[None, :], j]    # (n,K,3)
        u_oc = jax.random.uniform(jax.random.fold_in(sk, 3), (n, K, 1))
        probs_oc = sio_j / jnp.clip(jnp.sum(sio_j, axis=-1, keepdims=True), 1e-12, None)
        out_ch = jnp.clip(jnp.sum((u_oc > jnp.cumsum(probs_oc, axis=-1)).astype(jnp.int32), axis=-1), 0, 2)
        nuc_idx_j = jnp.where(nisp[gi[0], gi[1]], 0, 1)
        chan_idx = pi_ch_idx * 6 + nuc_idx_j * 3 + out_ch
        u_a = jax.random.uniform(jax.random.fold_in(sk, 4), (n, K))
        cos_cm = cascade_mb.jax_sample_cos_cm(W_j.reshape(-1), u_a.reshape(-1),
                                              chan_idx.reshape(-1)).reshape(n, K)
        phi_s = 2 * jnp.pi * jax.random.uniform(jax.random.fold_in(sk, 5), (n, K))
        # vectorize the pion scatter over slots via reshape
        pi_out, rec_pi = _dcc_scatter(p.reshape(n * K, 4), pN_j.reshape(n * K, 4),
                                      _CH_MASS[out_ch].reshape(n * K), cos_cm.reshape(n * K),
                                      phi_s.reshape(n * K))
        pi_out = pi_out.reshape(n, K, 4); rec_pi = rec_pi.reshape(n, K, 4)
        pi_blocked = jnp.linalg.norm(rec_pi[..., 1:], axis=-1) < kf_j
        if not cfg.pauli:
            pi_blocked = pi_blocked & False
        pi_scat = pi_scat & ~pi_blocked
        # recoil nucleon charge: q_struck + q_pi_in - q_pi_out
        q_rec_pi = nisp[gi[0], gi[1]].astype(jnp.int32) + (1 - pi_ch_idx) - (1 - out_ch)
        # ---- NN ELASTIC ----
        cth_e = 2 * jax.random.uniform(jax.random.fold_in(sk, 6), (n, K)) - 1
        phi_e = 2 * jnp.pi * jax.random.uniform(jax.random.fold_in(sk, 7), (n, K))
        nA, nB = _two_body((p + pN_j).reshape(n * K, 4), M_N, M_N,
                           cth_e.reshape(n * K), phi_e.reshape(n * K))
        nA = nA.reshape(n, K, 4); nB = nB.reshape(n, K, 4)
        el_blocked = ((jnp.linalg.norm(nA[..., 1:], axis=-1) < kf_j)
                      | (jnp.linalg.norm(nB[..., 1:], axis=-1) < kf_j))
        if not cfg.pauli:
            el_blocked = el_blocked & False
        n_el = n_el & ~el_blocked
        # ---- NN INELASTIC (N Delta -> N N pi) ----
        rs_j = jnp.sqrt(jnp.clip((p + pN_j)[..., 0] ** 2
                                 - jnp.sum((p + pN_j)[..., 1:] ** 2, axis=-1), (2 * M_N) ** 2, None))
        u_m = jax.random.uniform(jax.random.fold_in(sk, 8), (n, K))
        m_d = jnp.clip(nni.sample_delta_mass((rs_j / 1000.0).reshape(-1), u_m.reshape(-1)).reshape(n, K) * 1000.0,
                       M_N + 135.0, rs_j - M_N - 1.0)
        cth_i = 2 * jax.random.uniform(jax.random.fold_in(sk, 9), (n, K)) - 1
        phi_i = 2 * jnp.pi * jax.random.uniform(jax.random.fold_in(sk, 10), (n, K))
        nI, pD = _two_body((p + pN_j).reshape(n * K, 4), M_N, m_d.reshape(n * K),
                           cth_i.reshape(n * K), phi_i.reshape(n * K))
        cth_d = 2 * jax.random.uniform(jax.random.fold_in(sk, 11), (n, K)) - 1
        phi_d = 2 * jnp.pi * jax.random.uniform(jax.random.fold_in(sk, 12), (n, K))
        nD, piD = _two_body(pD, M_N, 138.04, cth_d.reshape(n * K), phi_d.reshape(n * K))
        nI = nI.reshape(n, K, 4); nD = nD.reshape(n, K, 4); piD = piD.reshape(n, K, 4)
        in_blocked = ((jnp.linalg.norm(nI[..., 1:], axis=-1) < kf_j)
                      | (jnp.linalg.norm(nD[..., 1:], axis=-1) < kf_j))
        if not cfg.pauli:
            in_blocked = in_blocked & False
        n_inel = n_inel & ~in_blocked
        # ---- apply primary-slot updates ----
        interact = pi_abs | pi_conv | pi_scat | n_el | n_inel
        # pion: absorbed/converted -> slot dies (abs final state ignored for the pion slot;
        # the absorption products are NOT tracked in v1 -- the CC0pi chains keep using the
        # factorized kernel; this kernel targets the pion-SURVIVAL signals)
        p_old = p                                                # pre-update momenta for formation zones
        spec = jnp.where(pi_abs | pi_conv, 0, spec).astype(jnp.int32)
        absorbed_pi = absorbed_pi | jnp.any(pi_abs & (jnp.arange(K) == 0)[None, :], axis=1)
        p = jnp.where(pi_scat[..., None], pi_out, p)
        chg = jnp.where(pi_scat, (1 - out_ch).astype(jnp.int32), chg).astype(jnp.int32)
        p = jnp.where(n_el[..., None], nA, p)
        p = jnp.where(n_inel[..., None], nI, p)
        fz = jnp.where(n_el, _formation_zone(p_old.reshape(n * K, 4), nA.reshape(n * K, 4)).reshape(n, K), fz)
        fz = jnp.where(n_inel, _formation_zone(p_old.reshape(n * K, 4), nI.reshape(n * K, 4)).reshape(n, K), fz)
        # consume struck background nucleons (+ the absorption partner)
        hit_onehot = jax.nn.one_hot(j, A, dtype=bool) & interact[..., None]
        part_onehot = jax.nn.one_hot(pj, A, dtype=bool) & pi_abs[..., None]
        consumed = consumed | jnp.any(hit_onehot, axis=1) | jnp.any(part_onehot, axis=1)
        # ---- spawn products into free slots (priority: lower slot index) ----
        def spawn(p, spec, chg, fz, pos, new_p, new_spec, new_chg, new_fz, new_pos, want):
            free = spec == 0
            # rank free slots; assign the FIRST free slot per event to the first wanting product
            for kk in range(K):
                w_k = want[:, kk]
                # find first free slot index
                free_idx = jnp.argmax(free, axis=1)
                any_free = jnp.any(free, axis=1)
                do_ = w_k & any_free
                p = p.at[arn, free_idx].set(jnp.where(do_[:, None], new_p[:, kk], p[arn, free_idx]))
                spec = spec.at[arn, free_idx].set(jnp.where(do_, new_spec[:, kk], spec[arn, free_idx]).astype(jnp.int32))
                chg = chg.at[arn, free_idx].set(jnp.where(do_, new_chg[:, kk], chg[arn, free_idx]).astype(jnp.int32))
                fz = fz.at[arn, free_idx].set(jnp.where(do_, new_fz[:, kk], fz[arn, free_idx]))
                pos = pos.at[arn, free_idx].set(jnp.where(do_[:, None], new_pos[:, kk], pos[arn, free_idx]))
                free = free.at[arn, free_idx].set(jnp.where(do_, False, free[arn, free_idx]))
            return p, spec, chg, fz, pos
        pos_j = npos[gi[0], gi[1]]
        # absorption products: charges from conservation (nprot_out = qpi + struck + partner);
        # assign: pa3 proton if nprot_out >= 1, pb3 proton if nprot_out == 2
        npr_out = qpi + struck_p + nisp[arn[:, None], pj].astype(jnp.int32)
        p, spec, chg, fz, pos = spawn(p, spec, chg, fz, pos,
                                      pa3, jnp.full((n, K), 2, jnp.int32),
                                      (npr_out >= 1).astype(jnp.int32),
                                      jnp.zeros((n, K)), pos_j, pi_abs)
        p, spec, chg, fz, pos = spawn(p, spec, chg, fz, pos,
                                      pb3, jnp.full((n, K), 2, jnp.int32),
                                      (npr_out >= 2).astype(jnp.int32),
                                      jnp.zeros((n, K)), pos_j, pi_abs)
        # pi-scatter recoil nucleon
        p, spec, chg, fz, pos = spawn(p, spec, chg, fz, pos,
                                      rec_pi, jnp.full((n, K), 2, jnp.int32),
                                      jnp.clip(q_rec_pi, 0, 1),
                                      _formation_zone(p_old.reshape(n * K, 4), rec_pi.reshape(n * K, 4)).reshape(n, K),
                                      pos_j, pi_scat)
        # NN elastic recoil
        p, spec, chg, fz, pos = spawn(p, spec, chg, fz, pos,
                                      nB, jnp.full((n, K), 2, jnp.int32),
                                      nisp[gi[0], gi[1]].astype(jnp.int32),
                                      _formation_zone(p_old.reshape(n * K, 4), nB.reshape(n * K, 4)).reshape(n, K),
                                      pos_j, n_el)
        # NN inelastic: Delta-decay nucleon + created pion (charges: approximate bookkeeping --
        # decay nucleon inherits the struck isospin, pion charge balances; declared)
        p, spec, chg, fz, pos = spawn(p, spec, chg, fz, pos,
                                      nD, jnp.full((n, K), 2, jnp.int32),
                                      nisp[gi[0], gi[1]].astype(jnp.int32),
                                      _formation_zone(p_old.reshape(n * K, 4), nD.reshape(n * K, 4)).reshape(n, K),
                                      pos_j, n_inel)
        p, spec, chg, fz, pos = spawn(p, spec, chg, fz, pos,
                                      piD, jnp.full((n, K), 1, jnp.int32),
                                      jnp.zeros((n, K), jnp.int32),
                                      jnp.zeros((n, K)), pos_j, n_inel)
        inted = inted | interact
        # ---- advance ----
        alive = spec > 0
        mom2 = jnp.linalg.norm(p[..., 1:], axis=-1)
        dhat2 = p[..., 1:] / jnp.clip(mom2[..., None], 1e-9, None)
        pos = pos + cfg.step * dhat2 * alive[..., None]
        return (i + 1, p, spec, chg, pos, fz, consumed, absorbed_pi, inted)

    def cond(state):
        i, p, spec, chg, pos, fz, consumed, _, inted = state
        alive = spec > 0
        mom = jnp.linalg.norm(p[..., 1:], axis=-1)
        dhat = p[..., 1:] / jnp.clip(mom[..., None], 1e-9, None)
        outward = jnp.sum(pos * dhat, axis=-1) > 0
        inert = (jnp.linalg.norm(pos, axis=-1) > radius) & outward
        return (i < cfg.max_steps) & jnp.any(alive & ~inert)

    fz0 = jnp.zeros((n, K))
    state0 = (jnp.int32(0), p0, spec0, chg0, pos_init, fz0, consumed0, jnp.zeros(n, bool),
              jnp.zeros((n, K_PART), bool))
    _, p, spec, chg, pos, fz, consumed, absorbed_pi, _ = jax.lax.while_loop(cond, body, state0)
    return p, jnp.abs(spec), chg, absorbed_pi


def evolve_event(p_pi, p_N, pid_pi, pid_Ni, pid_N, cfg, key):
    """Public entry: RES pion + primary nucleon co-evolve through ONE shared nucleus.
    p_pi, p_N (n,4) MeV; pid_pi in {211,111,-211}; pid_Ni = struck nucleon pid (vertex);
    pid_N = primary outgoing nucleon pid.  Returns dict with the final particle slots:
    p (n,K,4), spec (n,K) {0 empty, 1 pion, 2 nucleon}, chg (n,K) (pion: +1/0/-1;
    nucleon: 1 p / 0 n), absorbed_pi (n,)."""
    n = p_pi.shape[0]
    kn, kv, kp = jax.random.split(key, 3)
    npos, nmom, nisp = sample_nucleons(kn, n, cfg)
    A = nisp.shape[1]
    struck_isp = (jnp.asarray(pid_Ni) == 2212)
    rsel = jnp.where(nisp == struck_isp[:, None], jax.random.uniform(kv, (n, A)), -1.0)
    vtx = jnp.argmax(rsel, axis=1)
    pos_v = npos[jnp.arange(n), vtx]
    consumed0 = jax.nn.one_hot(vtx, A).astype(bool)
    p0 = jnp.zeros((n, K_PART, 4))
    p0 = p0.at[:, 0].set(jnp.asarray(p_pi)).at[:, 1].set(jnp.asarray(p_N))
    spec0 = jnp.zeros((n, K_PART), jnp.int32)
    spec0 = spec0.at[:, 0].set(1).at[:, 1].set(2)
    chg0 = jnp.zeros((n, K_PART), jnp.int32)
    pi_q = jnp.where(jnp.asarray(pid_pi) == 211, 1, jnp.where(jnp.asarray(pid_pi) == -211, -1, 0))
    chg0 = chg0.at[:, 0].set(pi_q).at[:, 1].set((jnp.asarray(pid_N) == 2212).astype(jnp.int32))
    pos_init = jnp.broadcast_to(pos_v[:, None, :], (n, K_PART, 3))
    p, spec, chg, absorbed_pi = _evolve(p0, spec0, chg0, npos, nmom, nisp, cfg, kp,
                                        consumed0, pos_init)
    return dict(p=p, spec=spec, chg=chg, absorbed_pi=absorbed_pi)
