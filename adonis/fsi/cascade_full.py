"""Faithful multi-particle differentiable cascade engine (standalone, in development).

Design (docs/logbook/cascade_engine.md): BFS over a shared nucleus. A PARTICLE is one Markov walk
segment of fixed identity; it propagates (multi-step, elastic = same identity) until a terminal
(escape/absorb/convert) or a transmuting interaction (charge-exchange/inelastic), then emits its
spawned secondaries.  Generations are processed breadth-first with FIXED-shape buffers so the whole
thing is jit/vmap-able; per-particle kind-1 reweight records keep it differentiable.

This file builds the mechanism (problem #2: variable particle count in fixed shapes) FIRST, with a
PLUGGABLE single-particle kernel, then the physics kernels are dropped in (P1b+).  The kernel contract:

    kernel(part, nucleus, key) -> (term, spawn)
      part   : ParticleBatch (n, P, ...) of the current generation (alive mask says which are real)
      term   : ParticleBatch (n, P, ...) -- each input particle's TERMINAL state + fate (for observables)
      spawn  : ParticleBatch (n, P*SPAWN, ...) -- secondaries emitted (alive mask says which are real)

Fields per particle (all leading dim (n, P)):
    species : 0 = pion, 1 = nucleon
    charge  : pion charge idx (0:+,1:0,2:-) | nucleon isospin (1=p,0=n) -- interpretation by species
    p4      : (..,4) (E,px,py,pz) MeV
    pos     : (..,3) fm
    fz      : (..,) formation zone [fm]
    alive   : (..,) real-slot mask
    w       : (..,) particle weight (kind-1; 1.0 at nominal)
    fate    : (..,) terminal code -- 0 alive/none, 1 escaped, 2 absorbed, 3 converted (set by kernel)
"""
from __future__ import annotations
import os
import jax, jax.numpy as jnp
from adonis.fsi.cascade_discrete import (_CH_PID, sample_nucleons, MB_TO_FM2, pion_branch_reweight,
                                         fsi_pion_reweight, nucleon_scat_reweight, fsi_nucleon_reweight,
                                         fsi_nncex_reweight)

PION, NUCLEON = 0, 1
FATE_NONE, FATE_ESCAPE, FATE_ABSORB, FATE_CONVERT = 0, 1, 2, 3
_ORIG_PRIM_PI = 2          # pool origin tag for the RES PRIMARY pion (0=RES recoil/QE chain, 1=pi-knockout)
_TRACK_OFFSET = 1000       # daughter track_ids start here (> any primary track_id); see make_pool_stepper
# Engine DEFAULTS (cascade_nucleus): refill + waiting-queue are ON by default so every consumer gets the
# persistent-refill engine and the keep-overflow-particles correctness (the plan's allowed change).
_DEFAULT_NW = 2048         # refill working-set width (events in flight).  n_w=None -> min(_DEFAULT_NW, n);
                           # n_w=0 forces lock-step (the bit-exact reference).  Tuned by the (workers,n_w,P) study.
_DEFAULT_QCAP = 64         # particle waiting-queue width.  q_cap=None -> this; keeps overflow particles
                           # (stack overflow sofl -> 0); q_cap=0 disables (legacy drop-on-overflow).
_HARD_STEPS = 100000       # ACHILLES cMaxSteps-style ABSOLUTE step ceiling.  Physical termination is
                           # escape/capture/absorption/path-budget; reaching this ceiling means a particle
                           # never terminated -> a RUNAWAY, which must NOT be silently truncated.  The pool
                           # RAISES (run_cascade_pool) if it is ever hit, making the anomaly visible.


def _raise_if_runaway(counter, where):
    """Host-side guard: raise if the cascade hit the _HARD_STEPS ceiling (a particle never terminated).
    No-op when `counter` is a tracer (under jit) -- the eager forward/gen path is where it matters."""
    if isinstance(counter, (jax.Array,)) and getattr(counter, "shape", None) == ():
        try:
            c = int(counter)
        except Exception:
            return                                                   # traced -> skip host check
        if c > 0:
            raise RuntimeError(
                f"cascade {where}: hit the {_HARD_STEPS}-step hard ceiling for {c} particle/event(s) -- "
                "no physical termination (escape/capture/absorption/path-budget) fired.  This should never "
                "happen; investigate (runaway particle) rather than accept a silently truncated cascade.")


def setup_nucleus(p_pi, pid_pi, pid_Ni, cfg, key):
    """Sample the nucleus background + struck vertex (material from cfg) EXACTLY as DiscreteCascadeFSI.apply,
    so the engine's primary-pion segment reproduces the production chain bit-for-bit.  Returns nucleus + pion init."""
    kn, kv, kp = jax.random.split(key, 3)
    n = p_pi.shape[0]
    npos, nmom, nisp = sample_nucleons(kn, n, cfg)
    A = nisp.shape[1]
    struck_isp = (pid_Ni == 2212)
    rsel = jnp.where(nisp == struck_isp[:, None], jax.random.uniform(kv, (n, A)), -1.0)
    vtx = jnp.argmax(rsel, axis=1)
    pos0 = npos[jnp.arange(n), vtx]
    consumed0 = jax.nn.one_hot(vtx, A).astype(bool)
    ch0 = jnp.where(pid_pi == 211, 0, jnp.where(pid_pi == -211, 2, 1)).astype(jnp.int32)
    return dict(npos=npos, nmom=nmom, nisp=nisp, pos0=pos0, consumed0=consumed0, ch0=ch0, kp=kp)


def empty_batch(n, P):
    """An all-dead particle buffer of shape (n, P).  origin = gen-0 ancestor (0 = RES/QE nucleon chain,
    1 = pion-scatter-knockout chain), gen = BFS depth -- PROVENANCE for offline debugging."""
    return dict(species=jnp.zeros((n, P), jnp.int32), charge=jnp.zeros((n, P), jnp.int32),
                p4=jnp.zeros((n, P, 4)), pos=jnp.zeros((n, P, 3)), fz=jnp.zeros((n, P)),
                nsc=jnp.zeros((n, P), jnp.int32),                     # pion scatter count (pool: beam-vs-internal escape)
                alive=jnp.zeros((n, P), bool), w=jnp.ones((n, P)), fate=jnp.zeros((n, P), jnp.int32),
                origin=jnp.zeros((n, P), jnp.int32), gen=jnp.zeros((n, P), jnp.int32),
                track_id=jnp.zeros((n, P), jnp.int32),                # MC-truth: unique id (tracking.py)
                parent_id=jnp.full((n, P), -1, jnp.int32),            # MC-truth: spawning track (-1 = primary)
                pkey=jnp.zeros((n, P, 2), jnp.uint32),                # P-INVARIANT per-particle RNG key (lineage-derived)
                lstep=jnp.zeros((n, P), jnp.int32),                   # per-particle local step counter (RNG fold + safety cap)
                lpath=jnp.zeros((n, P)),                              # per-particle accumulated PATH length [fm] (physics cap)
                gtime=jnp.zeros((n, P), jnp.int32))                   # ABSOLUTE cascade time (= global step at P>=occ);
                                                                       # processing priority -> P-invariant claim order


def _take(b, idx):
    """Gather along the particle axis (n, P) -> (n, K) with idx (n, K)."""
    n = idx.shape[0]; ar = jnp.arange(n)[:, None]
    return dict(species=b["species"][ar, idx], charge=b["charge"][ar, idx], p4=b["p4"][ar, idx],
                pos=b["pos"][ar, idx], fz=b["fz"][ar, idx], alive=b["alive"][ar, idx],
                w=b["w"][ar, idx], fate=b["fate"][ar, idx])


def compact(b, P_out, sort_priority=False):
    """Compact live particles of a (n, M) batch to the front of a (n, P_out) buffer.  Returns
    (compacted_batch, n_overflow); overflow = live particles that did not fit in P_out (dropped, never
    silently).  sort_priority=False -> position order (cumsum).  sort_priority=True -> pack the ACTIVE
    cascade stack in ACHILLES processing order: gtime cohort, then ascending CREATION index (track_id)."""
    alive = b["alive"]; n, M = alive.shape
    if not sort_priority:
        # stable rank of each live particle within its event (0-based); dead get a large rank
        rank = jnp.cumsum(alive.astype(jnp.int32), axis=1) - 1
    else:
        # lexsort keys (LAST = primary): alive-first, then ABSOLUTE TIME gtime asc, then CREATION index
        # track_id asc.  gtime (not per-particle lstep) is the cascade-synchronized cohort so daughters act
        # with their cohort, not ahead of it; WITHIN a cohort, ascending track_id reproduces ACHILLES's
        # within-timestep order (Cascade.cc:353 iterates kickedIdxs = std::set<size_t> ascending particle
        # index = creation order).  Integer keys -> no float-precision dependence; track_id is unique per
        # alive particle (primaries 0/1; daughters _TRACK_OFFSET + step*3 + chan) -> no position fallback.
        order = jnp.lexsort((b["track_id"], b["gtime"], (~alive).astype(jnp.int32)), axis=-1)
        rank = jnp.argsort(order, axis=-1)
    rank = jnp.where(alive, rank, M + P_out)                      # dead -> out of range
    n_overflow = jnp.sum((alive & (rank >= P_out)).astype(jnp.int32))
    out = empty_batch(n, P_out)
    ar = jnp.broadcast_to(jnp.arange(n)[:, None], (n, M))
    dst = jnp.where(rank < P_out, rank, P_out)                    # P_out = scratch drop slot (will be sliced off)
    for k in ("species", "charge", "fate", "origin", "gen", "track_id", "parent_id", "nsc", "lstep", "gtime"):
        if k not in b:                                            # optional (e.g. BFS kernel spawns omit nsc)
            continue
        out[k] = jnp.zeros((n, P_out + 1), b[k].dtype).at[ar, dst].set(b[k], mode="drop")[:, :P_out]
    for k in ("fz", "w", "lpath"):
        out[k] = jnp.zeros((n, P_out + 1)).at[ar, dst].set(b[k], mode="drop")[:, :P_out]
    out["alive"] = jnp.zeros((n, P_out + 1), bool).at[ar, dst].set(b["alive"], mode="drop")[:, :P_out]
    out["p4"] = jnp.zeros((n, P_out + 1, 4)).at[ar, dst].set(b["p4"], mode="drop")[:, :P_out]
    out["pos"] = jnp.zeros((n, P_out + 1, 3)).at[ar, dst].set(b["pos"], mode="drop")[:, :P_out]
    if "pkey" in b:                                              # P-invariant per-particle RNG key (n,P,2)
        out["pkey"] = jnp.zeros((n, P_out + 1, 2), b["pkey"].dtype).at[ar, dst].set(b["pkey"], mode="drop")[:, :P_out]
    return out, n_overflow


def make_pool_stepper(su, cfg, with_rec=False, with_seg=False):
    """Build the pooled-engine physics stepper (S2b): advance every slot of the (n, M) stack ONE step,
    dispatched by species (NUCLEON -> _nucleon_step, PION -> _pion_step), with the consumed mask threaded
    SLOT-SERIALLY (slot m+1 sees m's depletion).  Both per-step bodies run on every slot and are selected
    by species (the v1 2x-eval tradeoff; the dead-slot waste, the dominant 10-24x factor, is gone).  Each
    slot emits up to 2 NUCLEON spawns + 1 PION spawn: a nucleon slot -> (1 knockout N, 1 NN-created pion);
    a pion slot -> (up to 2 absorption/recoil N, 0 pion).  Returns stepper(stack, key, consumed) ->
    (stack2, terminal (n,M) bool [escaped final-state particles], spawn ParticleBatch (n,3M), consumed)."""
    from adonis.fsi.cascade_discrete import _nucleon_step, _pion_step, _load_density
    npos0, nmom0, nisp0 = su["npos"], su["nmom"], su["nisp"]   # default background; per-call `bg` overrides it
    rgrid, rhoP, rhoN, radius = _load_density(cfg.nucleus, cfg.density_n)
    _dead = lambda n: (jnp.zeros((n, 4)), jnp.zeros((n, 3)), jnp.zeros(n), jnp.zeros(n, jnp.int32), jnp.zeros(n, bool))

    def stepper(stack, key, consumed, step=0, bg=None, dt_evt=None):
        # bg=(npos,nmom,nisp) per-call -> lets run_cascade_pool swap the background when an event slot is
        # refilled (persistent-refill engine); bg=None uses the closed-over default (legacy callers).
        # dt_evt: per-event (n,) time step = step/beta_max for the ACHILLES-faithful time-sync clock
        # (cfg.time_step); None -> the steppers fall back to a fixed Dt=step.  Ignored for distance-sync.
        npos, nmom, nisp = (npos0, nmom0, nisp0) if bg is None else bg
        n, M = stack["alive"].shape
        _dt_e = dt_evt if dt_evt is not None else jnp.full(n, cfg.step)   # (n,) per-event Dt
        # `key` is the PER-EVENT step key (n,2) = fold_in(fold_in(base, evt_id), nstep) built by
        # run_cascade_pool; fold by slot index m -> a unique per-(event,step,slot) key.  This makes a
        # refilled event draw identical randoms regardless of which slot/global-step runs it.

        def slot(consumed, m):
            p4 = stack["p4"][:, m]; pos = stack["pos"][:, m]; fz = stack["fz"][:, m]
            nsc = stack["nsc"][:, m]; chg = stack["charge"][:, m]; al = stack["alive"][:, m]
            sp = stack["species"][:, m]
            is_N = (sp == NUCLEON) & al; is_pi = (sp == PION) & al
            d3 = p4[:, 1:]; dhat = d3 / jnp.clip(jnp.linalg.norm(d3, axis=1, keepdims=True), 1e-9, None)
            # P-INVARIANT RNG: key by the particle's OWN lineage key (pkey) + its OWN local step count
            # (lstep), NOT by slot index m or the global nstep.  A particle thus draws identical randoms
            # whether it is processed in-stack or after the wait queue, at any stack width P.
            pkey_m = stack["pkey"][:, m]; lstep_m = stack["lstep"][:, m]
            step_key = jax.vmap(lambda pk, ls: jax.random.fold_in(pk, ls))(pkey_m, lstep_m)   # (n,2)
            _kk = jax.vmap(lambda k: jax.random.split(k))(step_key)        # (n,2,2)
            kN, kP = _kk[:, 0], _kk[:, 1]                                  # per-particle nucleon/pion keys (n,2)
            # NUCLEON branch (charge = isospin, 1=p); inactive slots produce no consumption/spawn.
            (p4n, posn, _dn, fzn, alnN, qln), escN, _rc, _do, koN, pinN, consumedN, nstat = _nucleon_step(
                p4, pos, dhat, fz, chg.astype(bool), is_N, npos, nmom, nisp, consumed,
                rgrid, rhoP, rhoN, radius, cfg, kN, dt_evt=_dt_e)
            # PION branch (charge = pion index 0/1/2); scatter continues, abs/conv removed.
            (p4p, posp, _dp, chp, nscp, alnP), escP, is_abs, is_conv, s1, s2, consumedP, pstat = _pion_step(
                p4, pos, dhat, chg, nsc, is_pi, npos, nmom, nisp, consumed,
                rgrid, rhoP, rhoN, radius, cfg, kP, dt_evt=_dt_e)
            # kind-1 FSI reweight sufficient statistics (mirrors the legacy brec/srec per-step records):
            #   pion: record every geometric hit (has_hit) -> branch code + sigma components (sa,ss,si).
            #   nucleon: record every in-slab candidate step (perp2_c<1e5) -> hit flag + a_nom=pi b^2/sigma.
            # Computed ONLY when with_rec (the differentiable/tuning path) -> forward generation pays nothing.
            if with_rec:
                (p_hh, p_bc, p_sa, p_ss_el, p_ss, p_si,               # branch stats @ the STRUCK candidate
                 p_perp2_c, p_sa_c, p_ss_el_c, p_ss_c, p_si_c) = pstat   # survival stats @ the CLOSEST one
                n_hh, n_perp2, n_sig, n_iso, n_finel, n_inel, n_swap = nstat  # _nucleon_step granular stats
                a_nom = jnp.pi * n_perp2 / jnp.clip(n_sig * MB_TO_FM2, 1e-12, None)
                p_sig_c = jnp.clip(p_sa_c + p_ss_c + p_si_c, 1e-12, None)
                p_a_nom = jnp.pi * p_perp2_c / jnp.clip(p_sig_c * MB_TO_FM2, 1e-12, None)
                # PION: record every IN-SLAB CANDIDATE step (perp2_c < 1e5), not just the hits -- the no-hit
                # steps are exactly where the sigma_tot (mean-free-path) response lives.  Mirrors the nucleon.
                # NON-FINITE sigma is EXCLUDED, and that is not a patch: the Oset absorption sigma is NaN
                # whenever the pion has E < m_pi (ReducedHalfWidth re-derives |p| as sqrt(E^2-m^2), and the
                # RES pion is built with the mpi0 KINEMATIC mass while the cascade uses the per-charge
                # PHYSICAL mass -> negative under the root for |p| < ~35 MeV/c).  The WALK then cannot
                # interact there (prob = exp(-pi b^2/NaN) = NaN, and `u < NaN` is False), and it cannot
                # interact at ANY theta either -- scaling NaN is still NaN.  Such a candidate carries zero
                # theta-dependence, so recording it would inject NaN into the weight for no physics.
                # (ACHILLES has the identical re-derivation + PID mass, so the FORWARD behaviour is
                # faithful; the soft-pion sigma itself is a separate open question -- see logbook.)
                p_fin = jnp.isfinite(p_a_nom) & jnp.isfinite(p_sa_c) & jnp.isfinite(p_ss_c) \
                    & jnp.isfinite(p_si_c) & jnp.isfinite(p_ss_el_c)
                rec_slot = (is_pi & (p_perp2_c < 1e5) & p_fin, p_bc, p_sa, p_ss_el, p_ss, p_si,
                            p_hh, p_a_nom, p_sa_c, p_ss_el_c, p_ss_c, p_si_c,
                            is_N & (n_perp2 < 1e5), n_hh, a_nom, n_iso, n_finel, n_inel, n_swap)  # nucleon
            consumed = consumedN | consumedP                          # only the active species adds bits
            p4_2 = jnp.where(is_N[:, None], p4n, jnp.where(is_pi[:, None], p4p, p4))
            pos_2 = jnp.where(is_N[:, None], posn, jnp.where(is_pi[:, None], posp, pos))
            fz_2 = jnp.where(is_N, fzn, fz)                           # only the nucleon updates fz
            nsc_2 = jnp.where(is_pi, nscp, nsc)                       # only the pion updates nsc
            # nucleon charge can now change: NN elastic charge-exchange / inelastic leading channel charge
            # (qln from _nucleon_step; == incident charge when no scatter).  Pion charge oscillates (chp).
            chg_2 = jnp.where(is_N, qln, jnp.where(is_pi, chp, chg))
            al_2 = jnp.where(is_N, alnN, jnp.where(is_pi, alnP, al))
            # PER-PARTICLE termination (P-INVARIANT; both caps key off the particle's OWN state, never the
            # global step count -> no P-dependent truncation; capped = dropped, not a final state):
            #   (1) PHYSICS: accumulated path length lpath >= path_budget_R * radius.  _dstep is the
            #       distance swept THIS step = beta*step (time-sync) or step (distance-sync), from the
            #       PRE-step momentum (matches the slab/advance inside _nucleon_step/_pion_step).
            #   (2) SAFETY: lstep >= max_steps -- a guaranteed-fire backstop (lstep always +1's) for the
            #       degenerate beta~0 case where lpath cannot advance.
            _beta_m = jnp.linalg.norm(d3, axis=1) / jnp.clip(p4[:, 0], 1e-9, None)
            _dstep_m = _beta_m * _dt_e if cfg.time_step else jnp.full_like(_beta_m, cfg.step)
            _lpath_2 = stack["lpath"][:, m] + _dstep_m * al.astype(stack["lpath"].dtype)
            al_2 = al_2 & (_lpath_2 < cfg.path_budget_R * radius) & ((lstep_m + 1) < cfg.max_steps)
            term = (is_N & escN) | (is_pi & escP)                    # escaped = final-state (collected)
            # per-slot fate (for the primary-pion latch in run_cascade_pool; output stays escape-only):
            # nucleon/pion escape -> ESCAPE, pion absorbed -> ABSORB, pion converted -> CONVERT.
            fate_2 = jnp.where((is_N & escN) | (is_pi & escP), FATE_ESCAPE,
                     jnp.where(is_pi & is_abs, FATE_ABSORB,
                     jnp.where(is_pi & is_conv, FATE_CONVERT, FATE_NONE))).astype(jnp.int32)
            new = (p4_2, pos_2, fz_2, nsc_2, chg_2, al_2, term, fate_2)
            # nucleon-slot knockout / pion-slot 1st product -> nuc1; pion-slot 2nd product -> nuc2.
            nuc1 = tuple(jnp.where(is_N, kn, jnp.where(is_pi, s1k, dk)) if kn.ndim == 1
                         else jnp.where(is_N[:, None], kn, jnp.where(is_pi[:, None], s1k, dk))
                         for kn, s1k, dk in zip(koN, s1, _dead(n)))
            nuc2 = tuple(jnp.where(is_pi, s2k, dk) if s2k.ndim == 1
                         else jnp.where(is_pi[:, None], s2k, dk)
                         for s2k, dk in zip(s2, _dead(n)))
            pio = tuple(jnp.where(is_N, pk, dk) if pk.ndim == 1
                        else jnp.where(is_N[:, None], pk, dk)
                        for pk, dk in zip(pinN, _dead(n)))
            seg_slot = None
            if with_seg:
                # per-slot SEGMENT record: channel + segment-start |p| + product (pid,|p|) set, mirroring
                # the ACHILLES VERTEXDUMP.  channel: pion {1 elastic,2 charge-ex,3 abs,4 conv}, nucleon
                # {1 elastic NN->NN, 2 inelastic NN->NNpi}, 0 = no interaction this step.  Assembled (per
                # channel) by the Python runner; here we export the raw pieces.
                _PIDPI = jnp.array([211, 111, -211])
                pi_scat = is_pi & (nscp > nsc)
                made_pi = is_N & pinN[4]                          # nucleon inelastic (created a pion)
                chan = jnp.where(made_pi, 2,
                        jnp.where(is_N & (_do > 0), 1,
                        jnp.where(is_pi & is_abs, 3,
                        jnp.where(is_pi & is_conv, 4,
                        jnp.where(pi_scat & (chp == chg), 1,
                        jnp.where(pi_scat & (chp != chg), 2, 0)))))).astype(jnp.int32)
                incp = jnp.linalg.norm(p4[:, 1:], axis=1)         # segment-start |p| (constant in flight)
                inc_pid = jnp.where(sp == NUCLEON, jnp.where(chg == 1, 2212, 2112),  # PRE-step pid
                                    _PIDPI[jnp.clip(chg, 0, 2)])
                cont_pid = jnp.where(sp == NUCLEON, jnp.where(chg_2 == 1, 2212, 2112),
                                     _PIDPI[jnp.clip(chg_2, 0, 2)])
                n1_pid = jnp.where(nuc1[3] == 1, 2212, 2112); n1_p = jnp.linalg.norm(nuc1[0][:, 1:], axis=1)
                n2_pid = jnp.where(nuc2[3] == 1, 2212, 2112); n2_p = jnp.linalg.norm(nuc2[0][:, 1:], axis=1)
                pio_pid = _PIDPI[jnp.clip(pio[3], 0, 2)]; pio_p = jnp.linalg.norm(pio[0][:, 1:], axis=1)
                cont_p = jnp.linalg.norm(p4_2[:, 1:], axis=1)
                seg_slot = (chan, incp, inc_pid, stack["parent_id"][:, m], stack["gen"][:, m],
                            stack["track_id"][:, m], cont_pid, cont_p, n1_pid, n1_p, nuc1[4].astype(jnp.int32),
                            n2_pid, n2_p, nuc2[4].astype(jnp.int32), pio_pid, pio_p, pio[4].astype(jnp.int32))
            out_slot = (new, nuc1, nuc2, pio) + ((rec_slot,) if with_rec else ()) + ((seg_slot,) if with_seg else ())
            return consumed, out_slot

        consumed, scanned = jax.lax.scan(slot, consumed, jnp.arange(M))
        new, nuc1, nuc2, pio = scanned[:4]
        T = lambda x: jnp.moveaxis(x, 0, 1)                          # (M, n, ...) -> (n, M, ...)
        p4s, poss, fzs, nscs, chgs, als, terms, fates = new
        # accumulated path: +beta*step (time-sync) or +step (distance-sync) per processed step, from the
        # PRE-step momentum -- the same _dstep the slot cap uses, vectorised over the whole stack.
        _beta_all = jnp.linalg.norm(stack["p4"][:, :, 1:], axis=2) / jnp.clip(stack["p4"][:, :, 0], 1e-9, None)
        _dstep_all = _beta_all * _dt_e[:, None] if cfg.time_step else jnp.full_like(_beta_all, cfg.step)
        stack2 = {**stack, "p4": T(p4s), "pos": T(poss), "fz": T(fzs), "nsc": T(nscs),
                  "charge": T(chgs), "alive": T(als), "fate": T(fates),
                  "lstep": stack["lstep"] + stack["alive"].astype(jnp.int32),   # +1 per processed (alive) step
                  "lpath": stack["lpath"] + _dstep_all * stack["alive"].astype(stack["lpath"].dtype),
                  "gtime": stack["gtime"] + stack["alive"].astype(jnp.int32)}   # absolute time advances in lockstep
        terminal = T(terms)
        if with_rec:                                                 # per-slot (n,M) kind-1 record this step
            rk = ("pi_m", "pi_bc", "pi_sa", "pi_ss_el", "pi_ss", "pi_si",          # pi_m = slot mask
                  "pi_hh", "pi_a", "pi_sa_c", "pi_ss_el_c", "pi_ss_c", "pi_si_c",  # survival stats
                  "nu_m", "nu_hh", "nu_a", "nu_iso", "nu_finel", "nu_inel", "nu_swap")
            rec = {k: T(v) for k, v in zip(rk, scanned[4])}
        if with_seg:                                                 # per-slot (n,M) SEGMENT record this step
            sk = ("chan", "incp", "inc_pid", "parent_id", "gen", "track_id", "cont_pid", "cont_p",
                  "n1_pid", "n1_p", "n1_al", "n2_pid", "n2_p", "n2_al", "pio_pid", "pio_p", "pio_al")
            seg = {k: T(v) for k, v in zip(sk, scanned[4 + (1 if with_rec else 0)])}

        def _spawn(species, packs):                                  # packs: list of (p4,pos,fz,q,al) (M,n,...)
            p4_, pos_, fz_, q_, al_ = (jnp.concatenate([T(p[i]) for p in packs], axis=1) for i in range(5))
            return species, p4_, pos_, fz_, q_.astype(jnp.int32), al_
        sp_n, np4, npos_, nfz, nq, nal = _spawn(NUCLEON, [nuc1, nuc2])
        sp_p, pp4, ppos, pfz, pq, pal = _spawn(PION, [pio])
        spawn = empty_batch(n, 3 * M)
        spawn["species"] = jnp.concatenate([jnp.full((n, 2 * M), NUCLEON, jnp.int32),
                                            jnp.full((n, M), PION, jnp.int32)], axis=1)
        spawn["charge"] = jnp.concatenate([nq, pq], axis=1)
        spawn["p4"] = jnp.concatenate([np4, pp4], axis=1)
        spawn["pos"] = jnp.concatenate([npos_, ppos], axis=1)
        spawn["fz"] = jnp.concatenate([nfz, pfz], axis=1)
        spawn["alive"] = jnp.concatenate([nal, pal], axis=1)
        # P-INVARIANT daughter RNG key (PHYSICS, not just provenance): derive from the PARENT's lineage
        # key + the daughter channel + the parent's local step at spawn -> unique & order/P-independent.
        _pslot = jnp.tile(jnp.arange(M), 3)                          # (3M,) parent slot per spawn col
        _chan = jnp.repeat(jnp.arange(3), M)                         # 0=nuc1, 1=nuc2, 2=pion daughter channel
        _ppk = stack["pkey"][:, _pslot]                             # (n,3M,2) parent lineage key
        _pls = stack["lstep"][:, _pslot]                            # (n,3M) parent local step at spawn
        _dk = lambda pk, c, ls: jax.random.fold_in(jax.random.fold_in(pk, c), ls)
        spawn["pkey"] = jax.vmap(jax.vmap(_dk))(_ppk, jnp.broadcast_to(_chan, (n, 3 * M)), _pls)
        spawn["lstep"] = jnp.zeros((n, 3 * M), jnp.int32)            # daughter's own step count starts at 0
        spawn["gtime"] = stack["gtime"][:, _pslot] + 1              # but acts at the NEXT absolute time (cohort-synced)
        # track_id = CREATION index (ALWAYS set; the within-cohort processing order keys off it -- see
        # compact sort_priority).  Unique per (step,slot,channel): `step` is a SCALAR global step (lock-step)
        # or a PER-EVENT (n,) nstep (refill); both monotone in creation time, so ascending track_id == the
        # order daughters were born == ACHILLES's ascending particle index.  _pslot/_chan reuse the pkey cols.
        sa = jnp.asarray(step)
        if sa.ndim == 0:
            tid = (_TRACK_OFFSET + (step * M + _pslot) * 3 + _chan).astype(jnp.int32)
            spawn["track_id"] = jnp.broadcast_to(tid[None, :], (n, 3 * M))
        else:
            tid = (_TRACK_OFFSET + (sa[:, None] * M + _pslot[None, :]) * 3 + _chan[None, :]).astype(jnp.int32)
            spawn["track_id"] = tid
        if with_seg:                                                 # G4-like provenance (physics-inert):
            # daughter gen = parent gen + 1; parent_id = parent track_id.  origin is NOT touched (the
            # physics' RES primary-pion latch depends on its default-0 propagation).
            spawn["gen"] = stack["gen"][:, _pslot] + 1
            spawn["parent_id"] = stack["track_id"][:, _pslot]
        out = (stack2, terminal, spawn, consumed)
        if with_rec:
            out = out + (rec,)
        if with_seg:
            out = out + (seg,)
        return out

    return stepper


def pool_reconcile(stack, terminal, spawn, M, wait=None, Q=0):
    """The POOLED engine's per-step in/out (docs/logbook/cascade_pool_engine.md): drop the slots that
    terminated this step (escape/absorb/convert), KEEP the survivors, INSERT the particles created this
    step, and re-pack to the fixed width M -- counting any that don't fit as overflow.  This is exactly
    compact(concat(survivors, spawned), M), so it reuses the validated compaction primitive.
      stack    : ParticleBatch (n, M)   -- the current stack (post-step state)
      terminal : (n, M) bool            -- slots that reached a terminal this step
      spawn    : ParticleBatch (n, K)   -- particles created this step (alive mask = which are real)
      M        : int                    -- fixed active-stack width
      wait     : ParticleBatch (n, Q) or None -- the FIFO waiting buffer (particles that didn't fit in M)
      Q        : int                    -- waiting-buffer width (0 = no queue; legacy drop-on-overflow)
    With Q>0 the combined survivors+wait+spawn are packed into M active + Q waiting (drain order: survivors
    keep their slots, then the waiting buffer re-enters before brand-new spawns); only > M+Q is dropped.
    This ADDS previously-dropped particles (correctness) without changing the active ordering when Q=0.
    Returns (new_stack (n, M), new_wait (n, Q) or None, overflow (scalar))."""
    stack = {**stack, "alive": stack["alive"] & ~terminal}
    if Q > 0 and wait is not None:
        combined = {k: jnp.concatenate([stack[k], wait[k], spawn[k]], axis=1) for k in stack}
        full, ndrop = compact(combined, M + Q, sort_priority=True)  # pack to M+Q in (lstep,sid) order
        active = {k: v[:, :M] for k, v in full.items()}            # the M highest-priority -> processed next
        new_wait = {k: v[:, M:M + Q] for k, v in full.items()}     # the rest wait (re-enter by priority)
        return active, new_wait, ndrop
    combined = {k: jnp.concatenate([stack[k], spawn[k]], axis=1) for k in stack}
    active, ndrop = compact(combined, M, sort_priority=True)      # Q=0: keep the M highest-priority
    return active, None, ndrop


def _rec_scatter(bufs, cnt, mask, vals, cap):
    """Append per-event the masked slots' `vals` into persistent (n, cap) buffers at a running write
    index `cnt` (the scratch-slot `cap` is out of range -> mode='drop' discards overflow + invalid).
    Within a step the masked slots get distinct indices (cumsum order); across steps gidx>=cnt, so no
    collision with prior writes.  Returns (new_bufs, new_cnt, overflow_this_step)."""
    n, M = mask.shape; ar = jnp.arange(n)[:, None]
    order = jnp.cumsum(mask.astype(jnp.int32), axis=1) - 1
    gidx = cnt[:, None] + order
    valid = mask & (gidx < cap)
    dst = jnp.where(valid, gidx, cap)
    new_bufs = [b.at[ar, dst].set(v, mode="drop") for b, v in zip(bufs, vals)]
    overflow = jnp.sum((mask & (gidx >= cap)).astype(jnp.int32)).astype(jnp.int32)
    return new_bufs, (cnt + jnp.sum(mask.astype(jnp.int32), axis=1)).astype(cnt.dtype), overflow


def _empty_fsi_record(n, Kp, Kn):
    """Per-event kind-1 FSI reweight buffers (defaults give per-slot LR=1).  bc is the granular pion channel
    code {0 el,1 cex,2 abs,3 conv}; ss_el = elastic part of the total scatter sigma ss (ss_cex = ss-ss_el).
    PION slots are per IN-SLAB CANDIDATE STEP (hit or not) -- pi_hh/pi_a/*_c carry the sigma_tot (hit/no-hit)
    response, without which a common rescale of the four pion sigmas is a spurious flat direction.
    Defaults: a=50 (p0->0 => no-hit factor 1), sa_c/ss_c=1, si_c=0 => g=1 at nominal."""
    return dict(bc=jnp.zeros((n, Kp), jnp.int32), sa=jnp.ones((n, Kp)),
                ss_el=jnp.ones((n, Kp)), ss=jnp.ones((n, Kp)),
                si=jnp.zeros((n, Kp)),
                pi_hh=jnp.zeros((n, Kp), bool), pi_a=jnp.full((n, Kp), 50.0),
                sa_c=jnp.ones((n, Kp)), ss_el_c=jnp.ones((n, Kp)), ss_c=jnp.ones((n, Kp)),
                si_c=jnp.zeros((n, Kp)), nh=jnp.zeros(n, jnp.int32),
                hh=jnp.zeros((n, Kn), bool), a=jnp.full((n, Kn), 50.0),
                iso=jnp.zeros((n, Kn), jnp.int32), finel=jnp.zeros((n, Kn)),
                inel=jnp.zeros((n, Kn), bool), swap=jnp.zeros((n, Kn), bool), ns=jnp.zeros(n, jnp.int32))


def pool_fsi_reweight(record, sabs, sscat, *, s_piN_elastic=None, s_piN_cex=None, s_conv=1.0,
                      s_NN_elastic=None, s_NN_inelastic=None, f_NN_cex=0.5):
    """Joint kind-1 FSI reweight for the pool.  Pion branch-split + nucleon (per-iso el/inel).  Pure in the
    scales; == 1 at nominal; == the in-walk weight at any theta.  BACKWARD-COMPATIBLE: pool_fsi_reweight(
    record, sabs, sscat) reproduces the legacy reweight bit-for-bit (all granular knobs default to sabs/sscat).
    GRANULAR knobs (differentiable_knobs.md Group A): pion s_piN_elastic/s_piN_cex/s_conv (s_pi_abs==sabs);
    nucleon s_NN_elastic/s_NN_inelastic each a length-3 per-iso {pp,pn,nn} scale (default = sscat each)."""
    s_el = sscat if s_piN_elastic is None else s_piN_elastic
    s_cex = sscat if s_piN_cex is None else s_piN_cex
    wp = fsi_pion_reweight((record["bc"], record["sa"], record["ss_el"], record["ss"], record["si"],
                           record["pi_hh"], record["pi_a"], record["sa_c"], record["ss_el_c"],
                           record["ss_c"], record["si_c"], record["nh"]), sabs, s_el, s_cex, s_conv)
    nse = jnp.broadcast_to(jnp.asarray(sscat), (3,)) if s_NN_elastic is None else jnp.asarray(s_NN_elastic)
    nsi = jnp.broadcast_to(jnp.asarray(sscat), (3,)) if s_NN_inelastic is None else jnp.asarray(s_NN_inelastic)
    wn = fsi_nucleon_reweight((record["hh"], record["a"], record["iso"], record["finel"], record["inel"],
                              record["ns"]), nse, nsi)
    wc = fsi_nncex_reweight((record["hh"], record["iso"], record["inel"], record["swap"], record["ns"]),
                            f_NN_cex)                            # NN charge-exchange fraction (nominal 0.5)
    return wp * wn * wc


# SEGMENT-LOG fields written by the in-engine logger (see run_cascade_pool log_cap path).  One row per
# cascade SEGMENT (a particle's free-flight episode ending in an interaction or a terminal escape):
#   chan: 0 transmit/escape | pion {1 elastic,2 charge-ex,3 abs,4 conv} | nucleon {1 elastic,2 inelastic}
#   inc_pid/incp: the particle's pid + segment-start |p|;  track_id/parent_id/gen: provenance (G4-like);
#   cont_*: the continuing particle post-interaction;  n1/n2/pio: the ejected daughters (pid,|p|,alive).
_LOG_I = ("chan", "inc_pid", "parent_id", "gen", "track_id", "cont_pid", "n1_pid", "n1_al",
          "n2_pid", "n2_al", "pio_pid", "pio_al")
_LOG_F = ("incp", "cont_p", "n1_p", "n2_p", "pio_p")


def _take_rows(d, idx):
    """Gather rows (leading axis) of a pytree-of-arrays dict at integer index `idx`."""
    return {k: v[idx] for k, v in d.items()}


def _round_betamax(stk, wait, round_gt, betamax):
    """ACHILLES AdaptiveStep, P-invariantly: per-event beta_max = max beta over ALL the event's alive
    particles (stack UNION wait -- a P-INDEPENDENT set), SNAPSHOTTED once per gtime-round.  A new round is
    detected when the event's min(gtime) over alive particles advances past `round_gt`; at that instant the
    cascade's lock-step has brought every alive particle to that min gtime, so the max is the true round
    beta_max at ANY P (P only changes how the set is split between stack and wait).  Within a round the
    stored beta_max is reused (so an already-processed fast particle still counts).  Returns (round_gt',
    beta_max').  dt_evt = step/beta_max -> a particle ALONE has beta_max=its own beta -> moves a full step
    (no freeze, no arbitrary cap -- the same slow-particle protection ACHILLES gets for free)."""
    def _bg(b):
        be = jnp.linalg.norm(b["p4"][..., 1:], axis=-1) / jnp.clip(b["p4"][..., 0], 1e-9, None)
        return be, b["alive"], b["gtime"]
    be_s, al_s, gt_s = _bg(stk)
    be_w, al_w, gt_w = _bg(wait)
    be = jnp.concatenate([be_s, be_w], axis=1)
    al = jnp.concatenate([al_s, al_w], axis=1)
    gt = jnp.concatenate([gt_s, gt_w], axis=1)
    g_min = jnp.min(jnp.where(al, gt, jnp.int32(1 << 30)), axis=1)
    bmax_new = jnp.max(jnp.where(al, be, 0.0), axis=1)
    new_round = g_min > round_gt
    betamax = jnp.where(new_round, jnp.clip(bmax_new, 1e-6, None), betamax)
    round_gt = jnp.where(new_round, g_min, round_gt)
    return round_gt, betamax


def run_cascade_pool(init, stepper, key, state0, M, max_steps, M_out=24, prim_origin=-999,
                     rec_caps=None, log_cap=None, bg=None, q_cap=0,
                     pending=None, n_w=None, per_event_cap=None, time_sync=False, step=0.04):
    """POOLED engine loop (docs/logbook/cascade_pool_engine.md): ONE fixed-size (W, M) particle stack
    stepped once per step; the in/out reconcile (pool_reconcile) runs INSIDE the step.
      stepper(stack, key, state) -> (stack2 (W,M), terminal (W,M) bool, spawn (W,K), state2[, rec|seg])
    Escaped terminals (the cascade FINAL STATE) accumulate into a fixed (W, M_out) output batch.  RNG is
    PER EVENT (key folded by evt_id+nstep) so an event is reproducible regardless of slot/step.

    TWO modes share the per-step body `_apply_step` (single source of truth):
      * NO-REFILL (pending=None, default): the working set IS the n events, run lock-step until all done or
        `max_steps`.  Bit-identical to the pre-refill engine -- the path every existing caller uses.
      * REFILL (pending given): the working set holds `n_w` event-slots fed from a PENDING POOL of all
        N_total events; when a slot's event finishes (no live particle) OR hits its per-event step cap, its
        accumulators are FLUSHED into global (N_total,...) buffers at its evt_id and the slot is REFILLED
        from a cursor.  `per_event_cap` replaces the global max_steps.  Removes the lock-step waste (a
        single long event no longer makes all events step to max_steps) and keeps the (W,M) tensor small.
        The ONLY behavioural change vs no-refill is none: at n_w=N_total it is bit-exact (gate); at
        n_w<N_total the per-event outputs are identical (compare by evt_id), only occupancy/wall changes.
    `prim_origin` (RES): origin tag of the PRIMARY pion -> its terminal fate is latched per event.
    rec_caps=(Kp,Kn): accumulate the per-event kind-1 FSI reweight record.  log_cap=L: in-engine segment
    logger.  rec_caps and log_cap are mutually exclusive.
    Returns (out_batch, stack_overflow, out_overflow, prim_fate[, fsi_record | (log, counts, log_overflow)])."""
    with_rec = rec_caps is not None
    do_log = log_cap is not None
    assert not (with_rec and do_log), "rec_caps and log_cap are mutually exclusive"
    Kp, Kn = rec_caps if with_rec else (1, 1)
    L = int(log_cap) if do_log else 1                                # dummy (.,1) buffers when not logging
    Q = int(q_cap) if q_cap and q_cap > 0 else 0                     # particle waiting-buffer width (0 = drop)
    cap = int(per_event_cap) if per_event_cap else int(max_steps)    # per-event step cap (refill mode)

    def _logbuf(W):
        return {**{k: jnp.zeros((W, L), jnp.int32) for k in _LOG_I},
                **{k: jnp.zeros((W, L), jnp.float32) for k in _LOG_F}}

    def _apply_step(stk, state, kk, step_i, wait, out, prim, rb, log, wptr, sofl, oofl, rofl, logofl, bg_w, ar,
                    round_gt, betamax):
        """ONE pooled step on the working set + accumulation (out, prim latch, rec, seg log) + reconcile.
        Scalars sofl/oofl/rofl/logofl accumulate globally; out/prim/rb/log/wptr are per-slot.  Returns the
        updated working-set state; identical maths regardless of refill mode.  time_sync: ACHILLES
        AdaptiveStep per-event dt = step/beta_max (snapshotted per gtime-round over stack UNION wait)."""
        if time_sync:
            round_gt, betamax = _round_betamax(stk, wait, round_gt, betamax)
            # DECOUPLED diagnostic (ADONIS_DECOUPLED=1): fixed Dt=step (each particle moves beta*step from
            # its OWN beta, no beta_max coupling) instead of ACHILLES adaptive Dt=step/beta_max.  Same
            # beta_i:beta_j ratios, different absolute step -> isolates "coupled vs decoupled time-sync".
            dt_evt = jnp.full_like(betamax, step) if os.environ.get("ADONIS_DECOUPLED") == "1" else step / betamax
        else:
            dt_evt = None
        # step_i is passed in EVERY path now (not just logging): track_id (creation index, used by the
        # within-cohort processing order) needs the real step counter -- a stale step=0 would collapse all
        # daughters to the same id and break the ACHILLES-order tiebreak.
        if bg_w is not None:
            _step = stepper(stk, kk, state, step_i, bg=bg_w, dt_evt=dt_evt)
        else:
            _step = stepper(stk, kk, state, step_i, dt_evt=dt_evt)
        stk2, terminal, spawn, state2 = _step[:4]
        extra = _step[4] if len(_step) >= 5 else None
        term_batch = {**stk2, "alive": terminal}
        out2, oo = compact({k: jnp.concatenate([out[k], term_batch[k]], axis=1) for k in out}, M_out)
        isprim = (stk2["origin"] == prim_origin) & (stk2["species"] == PION) & (stk2["fate"] != FATE_NONE)
        anyp = jnp.any(isprim, axis=1); j = jnp.argmax(isprim, axis=1)
        prim = jnp.where(anyp & (prim == FATE_NONE), stk2["fate"][ar, j], prim)
        if with_rec:
            rec = extra
            (bc, sa, ss_el, ss, si, pi_hh, pi_a, sa_c, ss_el_c, ss_c, si_c), nh, op = _rec_scatter(
                [rb["bc"], rb["sa"], rb["ss_el"], rb["ss"], rb["si"],
                 rb["pi_hh"], rb["pi_a"], rb["sa_c"], rb["ss_el_c"], rb["ss_c"], rb["si_c"]],
                rb["nh"], rec["pi_m"],
                [rec["pi_bc"], rec["pi_sa"], rec["pi_ss_el"], rec["pi_ss"], rec["pi_si"],
                 rec["pi_hh"], rec["pi_a"], rec["pi_sa_c"], rec["pi_ss_el_c"], rec["pi_ss_c"],
                 rec["pi_si_c"]], Kp)
            (hh, a, iso, finel, inel, swap), ns, on = _rec_scatter(
                [rb["hh"], rb["a"], rb["iso"], rb["finel"], rb["inel"], rb["swap"]], rb["ns"], rec["nu_m"],
                [rec["nu_hh"], rec["nu_a"], rec["nu_iso"], rec["nu_finel"], rec["nu_inel"], rec["nu_swap"]], Kn)
            rb = dict(bc=bc, sa=sa, ss_el=ss_el, ss=ss, si=si,
                      pi_hh=pi_hh, pi_a=pi_a, sa_c=sa_c, ss_el_c=ss_el_c, ss_c=ss_c, si_c=si_c,
                      nh=nh, hh=hh, a=a, iso=iso, finel=finel, inel=inel, swap=swap, ns=ns)
            rofl = (rofl + op + on).astype(rofl.dtype)
        if do_log:
            seg = extra
            ev = (seg["chan"] > 0) | terminal                        # interactions (1-4) + escapes (0)
            evi = ev.astype(jnp.int32)
            off = jnp.cumsum(evi, axis=1) - evi                      # exclusive prefix -> per-event slot order
            tgt0 = wptr[:, None] + off
            logofl = logofl + (ev & (tgt0 >= L)).sum().astype(logofl.dtype)
            tgt = jnp.where(ev & (tgt0 < L), tgt0, L)                # OOB / non-events dropped by mode='drop'
            log = dict(log)
            for k in _LOG_I + _LOG_F:
                log[k] = log[k].at[ar[:, None], tgt].set(seg[k].astype(log[k].dtype), mode="drop")
            wptr = wptr + evi.sum(axis=1).astype(wptr.dtype)
        if Q > 0:                                                    # particle waiting-queue: keep overflow
            newstk, newwait, so = pool_reconcile(stk2, terminal, spawn, M, wait, Q)
        else:                                                        # drop overflow (wait stays dummy)
            newstk, _nw, so = pool_reconcile(stk2, terminal, spawn, M)
            newwait = wait
        sofl = sofl + so.astype(sofl.dtype); oofl = oofl + oo.astype(oofl.dtype)
        return newstk, state2, newwait, out2, prim, rb, log, wptr, sofl, oofl, rofl, logofl, round_gt, betamax

    # ---------------- NO-REFILL: the working set IS the n events (bit-exact to the pre-refill engine) ----
    if pending is None:
        n = init["alive"].shape[0]; ar = jnp.arange(n)
        # INITIAL overflow (more primaries than M, e.g. RES pion+recoil at M=1) must go to the wait QUEUE,
        # not be dropped -- else the engine is not P-invariant (the initial compact silently discarded the
        # 2nd primary at small M).  Mirror pool_reconcile: pack init into M active + Q waiting.
        if Q > 0:
            _full, _ = compact(init, M + Q, sort_priority=True)
            stack = {k: v[:, :M] for k, v in _full.items()}
            wait0 = {k: v[:, M:M + Q] for k, v in _full.items()}
        else:
            stack, _ = compact(init, M, sort_priority=True)
            wait0 = empty_batch(n, max(Q, 1))
        out0 = empty_batch(n, M_out); rec0 = _empty_fsi_record(n, Kp, Kn)
        log0 = _logbuf(n); wptr0 = jnp.zeros(n, jnp.int32); logofl0 = jnp.int32(0)
        evt_id0 = jnp.arange(n, dtype=jnp.int32); nstep0 = jnp.zeros(n, jnp.int32)

        # The per-particle caps (stepper: physical lpath budget + lstep safety backstop) guarantee every
        # particle terminates, so the loop normally ends via 'no particle alive' (early-exit) at ANY P.  The
        # _HARD_STEPS ceiling is a pure runaway tripwire: reaching it means a particle never terminated, and
        # run_cascade_pool RAISES below rather than silently truncating (-> visible anomaly, per design).
        def cond(st):
            return (st[0] < _HARD_STEPS) & (jnp.any(st[1]["alive"]) | jnp.any(st[12]["alive"]))

        def body(st):
            (i, stk, state, out, sofl, oofl, prim, rb, rofl, log, wptr, logofl, wait, evt_id, nstep,
             round_gt, betamax) = st
            kk = jax.vmap(lambda e, s: jax.random.fold_in(jax.random.fold_in(key, e), s))(evt_id, nstep)
            (newstk, state2, newwait, out2, prim2, rb2, log2, wptr2, sofl2, oofl2, rofl2, logofl2,
             round_gt2, betamax2) = _apply_step(
                stk, state, kk, i, wait, out, prim, rb, log, wptr, sofl, oofl, rofl, logofl, bg, ar,
                round_gt, betamax)
            return (i + jnp.int32(1), newstk, state2, out2, sofl2, oofl2, prim2, rb2, rofl2,
                    log2, wptr2, logofl2, newwait, evt_id, nstep + jnp.int32(1), round_gt2, betamax2)

        init_st = (jnp.int32(0), stack, state0, out0, jnp.int32(0), jnp.int32(0),
                   jnp.full(n, FATE_NONE, jnp.int32), rec0, jnp.int32(0), log0, wptr0, logofl0, wait0,
                   evt_id0, nstep0, jnp.full(n, -1, jnp.int32), jnp.ones(n))
        (_i_fin, stack, _, out, sofl, oofl, prim, rb, rofl, log, wptr, logofl, _wait,
         _evt, _ns, _rg, _bm) = jax.lax.while_loop(cond, body, init_st)
        _runaway = jnp.where(_i_fin >= _HARD_STEPS,
                             jnp.sum((jnp.any(stack["alive"], axis=1) | jnp.any(_wait["alive"], axis=1))
                                     .astype(jnp.int32)), jnp.int32(0))
        _raise_if_runaway(_runaway, "lock-step")
        if do_log:
            return out, sofl, oofl, prim, (log, wptr, logofl)
        if with_rec:
            return out, sofl, oofl, prim, (rb, rofl)
        return out, sofl, oofl, prim

    # ---------------- REFILL: working set of n_w slots fed from the pending pool of N_total events -------
    Ntot = pending["stack"]["alive"].shape[0]
    W = min(int(n_w) if n_w else Ntot, Ntot); ar = jnp.arange(W)
    # INITIAL overflow (events with >M primaries, e.g. RES pion+recoil at M=1) must ride in the WAIT queue,
    # not be dropped -- mirror the no-refill init.  pstack = M active per event, pwait = Q waiting per event.
    # (Without this the RES recoil nucleon is silently dropped at refill setup -> N(p) halved.)
    if Q > 0:
        _pfull, _ = compact(pending["stack"], M + Q, sort_priority=True)
        pstack = {k: v[:, :M] for k, v in _pfull.items()}
        pwait = {k: v[:, M:M + Q] for k, v in _pfull.items()}
    else:
        pstack, _ = compact(pending["stack"], M, sort_priority=True)
        pwait = empty_batch(Ntot, 1)
    pbg = (pending["npos"], pending["nmom"], pending["nisp"])
    pcons = pending["consumed0"]
    # global per-event buffers (filled by flush-on-finish, indexed by evt_id)
    g_out = empty_batch(Ntot, M_out); g_prim = jnp.full(Ntot, FATE_NONE, jnp.int32)
    g_rb = _empty_fsi_record(Ntot, Kp, Kn); g_log = _logbuf(Ntot); g_wptr = jnp.zeros(Ntot, jnp.int32)
    # initial working set = first W events
    idx0 = jnp.arange(W, dtype=jnp.int32)
    stk0 = _take_rows(pstack, idx0); cons0 = pcons[idx0]
    bg0 = (pbg[0][idx0], pbg[1][idx0], pbg[2][idx0])
    out0 = empty_batch(W, M_out); rb0 = _empty_fsi_record(W, Kp, Kn)
    log0 = _logbuf(W); wptr0 = jnp.zeros(W, jnp.int32)
    prim0 = jnp.full(W, FATE_NONE, jnp.int32)
    wait0 = _take_rows(pwait, idx0) if Q > 0 else empty_batch(W, max(Q, 1))   # recoil rides in wait from t=0
    cursor0 = jnp.int32(W); evt0 = idx0; nstep0 = jnp.zeros(W, jnp.int32)

    def rcond(st):
        (cursor, stk, cons, bgw, evt, nstep, wait, out, prim, rb, log, wptr,
         sofl, oofl, rofl, logofl, go, gp, grb, glog, gwp, round_gt, betamax, forced) = st
        return (cursor < Ntot) | jnp.any(stk["alive"]) | jnp.any(wait["alive"])

    def rbody(st):
        (cursor, stk, cons, bgw, evt, nstep, wait, out, prim, rb, log, wptr,
         sofl, oofl, rofl, logofl, go, gp, grb, glog, gwp, round_gt, betamax, forced) = st
        kk = jax.vmap(lambda e, s: jax.random.fold_in(jax.random.fold_in(key, e), s))(jnp.maximum(evt, 0), nstep)
        (newstk, cons2, newwait, out2, prim2, rb2, log2, wptr2, sofl2, oofl2, rofl2, logofl2,
         round_gt2, betamax2) = _apply_step(
            stk, cons, kk, nstep, wait, out, prim, rb, log, wptr, sofl, oofl, rofl, logofl, bgw, ar,
            round_gt, betamax)
        nstep2 = nstep + jnp.int32(1)
        still_alive = jnp.any(newstk["alive"], axis=1) | jnp.any(newwait["alive"], axis=1)
        hit_cap = (nstep2 >= cap)                                     # _HARD_STEPS per-event ceiling
        forced = (forced + jnp.sum((hit_cap & still_alive & (evt >= 0)).astype(jnp.int32))).astype(jnp.int32)
        finished = (~still_alive) | hit_cap                          # runaway tripwire -> _raise_if_runaway
        flush = finished & (evt >= 0)                                 # only real (non-idle) slots flush
        # FLUSH: scatter per-slot accumulators to the global buffers at evt_id (idle/non-finished -> Ntot, dropped)
        gi = jnp.where(flush, evt, Ntot)
        go = {k: go[k].at[gi].set(out2[k], mode="drop") for k in go}
        gp = gp.at[gi].set(prim2, mode="drop")
        grb = {k: grb[k].at[gi].set(rb2[k], mode="drop") for k in grb}
        glog = {k: glog[k].at[gi].set(log2[k], mode="drop") for k in glog}
        gwp = gwp.at[gi].set(wptr2, mode="drop")
        # REFILL flushed slots from the cursor (creation order); idle slots (evt<0) are skipped
        rank = jnp.cumsum(flush.astype(jnp.int32)) - flush.astype(jnp.int32)
        new_id = (cursor + rank).astype(jnp.int32)
        take = flush & (new_id < Ntot)                                # this slot gets a fresh event
        cursor2 = (cursor + jnp.sum(take.astype(jnp.int32))).astype(jnp.int32)
        gidx = jnp.where(take, new_id, 0)                             # safe gather index
        rstk = _take_rows(pstack, gidx); rbg = (pbg[0][gidx], pbg[1][gidx], pbg[2][gidx]); rcons = pcons[gidx]
        rwait = _take_rows(pwait, gidx) if Q > 0 else empty_batch(W, max(Q, 1))   # fresh event's WAIT (its recoil)
        def _sel(new, old, mask, md=None):                            # per-slot pick (mask along axis 0)
            m = mask.reshape((-1,) + (1,) * (old.ndim - 1))
            return jnp.where(m, new, old)
        idle = finished & (~take)                                     # finished but no pending left -> go dead
        stk3 = {k: _sel(rstk[k], newstk[k], take) for k in newstk}
        stk3["alive"] = jnp.where(idle[:, None], False, stk3["alive"])  # drop any cap-cutoff survivors
        bg3 = tuple(_sel(rbg[t], bgw[t], take) for t in range(3))
        cons3 = _sel(rcons, cons2, take)
        # wait: refilled slot -> fresh event's wait (carries its recoil); finished+idle -> empty; else keep.
        _ewait = empty_batch(W, max(Q, 1))
        wait3 = {k: _sel(rwait[k], _sel(_ewait[k], newwait[k], finished), take) for k in newwait}
        # reset per-slot accumulators on EVERY finished slot (taken or now-idle); evt/nstep updated
        e_out = empty_batch(W, M_out); e_rb = _empty_fsi_record(W, Kp, Kn); e_log = _logbuf(W)
        out3 = {k: _sel(e_out[k], out2[k], finished) for k in out2}
        rb3 = {k: _sel(e_rb[k], rb2[k], finished) for k in rb2}
        log3 = {k: _sel(e_log[k], log2[k], finished) for k in log2}
        prim3 = jnp.where(finished, FATE_NONE, prim2)
        wptr3 = jnp.where(finished, 0, wptr2)
        evt3 = jnp.where(finished, jnp.where(take, new_id, jnp.int32(-1)), evt)
        nstep3 = jnp.where(finished, jnp.int32(0), nstep2)
        # reset the per-event time-sync snapshot on EVERY finished slot (a refilled slot is a NEW event,
        # so its first step must recompute beta_max): round_gt -> -1 triggers new_round next step.
        round_gt3 = jnp.where(finished, jnp.int32(-1), round_gt2)
        betamax3 = jnp.where(finished, 1.0, betamax2)
        return (cursor2, stk3, cons3, bg3, evt3, nstep3, wait3, out3, prim3, rb3, log3, wptr3,
                sofl2, oofl2, rofl2, logofl2, go, gp, grb, glog, gwp, round_gt3, betamax3, forced)

    init_st = (cursor0, stk0, cons0, bg0, evt0, nstep0, wait0, out0, prim0, rb0, log0, wptr0,
               jnp.int32(0), jnp.int32(0), jnp.int32(0), jnp.int32(0), g_out, g_prim, g_rb, g_log, g_wptr,
               jnp.full(W, -1, jnp.int32), jnp.ones(W), jnp.int32(0))
    out_st = jax.lax.while_loop(rcond, rbody, init_st)
    (_, _, _, _, _, _, _, _, _, _, _, _, sofl, oofl, rofl, logofl, go, gp, grb, glog, gwp,
     _rg, _bm, _forced) = out_st
    _raise_if_runaway(_forced, "refill")
    if do_log:
        return go, sofl, oofl, gp, (glog, gwp, logofl)
    if with_rec:
        return go, sofl, oofl, gp, (grb, rofl)
    return go, sofl, oofl, gp


def _cascade_pool(channel, p_pi, p_N, Npid, su, cfg, knuc, n, rec_caps=None, log_cap=None,
                  n_w=0, q_cap=0, per_event_cap=None):
    """POOLED-engine realization of cascade_nucleus (QE + RES), mapping the flat (n,M_out) terminal
    buffer back to the rich (pterm, nterms, overflow, created) schema.
    log_cap=L: run the SAME cascade with the in-engine SEGMENT logger on (with_seg stepper +
    run_cascade_pool log_cap) and return (log, counts, log_overflow) directly -- the single entry point
    for the cascade-vertex/segment matrix (gen_cascade_segments), so there is NO duplicated cascade.
      QE : gen-0 stack = the struck->proton (1 NUCLEON slot); pterm = QE "none"; created = leading
           surviving pion.
      RES: gen-0 stack = the PRIMARY pion (PION slot, origin-tagged) + the RES recoil nucleon; the pion's
           own scatter-recoils / absorption protons / created pions spawn natively during the walk.
           pterm = the primary pion's outcome (escape -> its pid/p4; absorbed -> pid 0; converted -> -1,
           via the latched fate); created = leading surviving NON-primary pion."""
    ar = jnp.arange(n)
    if channel == "qe":
        g0 = empty_batch(n, 1)
        g0["alive"] = jnp.ones((n, 1), bool)
        g0["species"] = jnp.full((n, 1), NUCLEON, jnp.int32)
        g0["charge"] = (Npid == 2212).astype(jnp.int32)[:, None]
        g0["p4"] = p_N[:, None, :]; g0["pos"] = su["pos0"][:, None, :]
        g0["track_id"] = jnp.zeros((n, 1), jnp.int32)                  # primary id (inert unless logging)
        _base = jax.vmap(lambda e: jax.random.fold_in(knuc, e))(jnp.arange(n))   # (n,2) per-event base key
        g0["pkey"] = jax.vmap(lambda b: jax.random.fold_in(b, 0))(_base)[:, None, :]  # primary idx 0 (n,1,2)
        prim_origin = -999
    else:                                                              # RES: primary pion + recoil nucleon
        g0 = empty_batch(n, 2)
        g0["alive"] = jnp.ones((n, 2), bool)
        g0["species"] = jnp.array([PION, NUCLEON], jnp.int32)[None, :] * jnp.ones((n, 1), jnp.int32)
        g0["charge"] = jnp.stack([su["ch0"], (Npid == 2212).astype(jnp.int32)], axis=1)
        g0["p4"] = jnp.stack([p_pi, p_N], axis=1)
        g0["pos"] = jnp.broadcast_to(su["pos0"][:, None, :], (n, 2, 3))
        g0["origin"] = jnp.array([_ORIG_PRIM_PI, 0], jnp.int32)[None, :] * jnp.ones((n, 1), jnp.int32)
        g0["track_id"] = jnp.array([0, 1], jnp.int32)[None, :] * jnp.ones((n, 1), jnp.int32)  # pion=0, recoil=1
        _base = jax.vmap(lambda e: jax.random.fold_in(knuc, e))(jnp.arange(n))   # (n,2) per-event base key
        g0["pkey"] = jnp.stack([jax.vmap(lambda b: jax.random.fold_in(b, 0))(_base),   # pion primary idx 0
                                jax.vmap(lambda b: jax.random.fold_in(b, 1))(_base)], axis=1)  # recoil idx 1 (n,2,2)
        prim_origin = _ORIG_PRIM_PI
    # n_w>0 (refill): run the persistent-refill engine -- a working set of n_w event-slots fed from the
    # pending pool of all n events (g0 + per-event nucleus background).  n_w=0 -> lock-step.  q_cap>0
    # enables the particle waiting-queue in EITHER path (keeps overflow particles).
    pend = dict(stack=g0, consumed0=su["consumed0"], npos=su["npos"], nmom=su["nmom"],
                nisp=su["nisp"]) if n_w and n_w > 0 else None
    nw = n_w if pend is not None else None
    if log_cap is not None:                                            # SEGMENT-LOGGER path: same cascade, logger on
        stepper = make_pool_stepper(su, cfg, with_seg=True)
        _o, _so, _oo, _pf, logtuple = run_cascade_pool(
            g0, stepper, knuc, su["consumed0"], M=1, max_steps=cfg.max_steps, M_out=24,
            prim_origin=prim_origin, log_cap=log_cap, pending=pend, n_w=nw, q_cap=q_cap,
            per_event_cap=per_event_cap, time_sync=cfg.time_step, step=cfg.step)
        return logtuple                                               # (log dict, counts (n,), log_overflow)
    stepper = make_pool_stepper(su, cfg, with_rec=rec_caps is not None)
    _rc = run_cascade_pool(g0, stepper, knuc, su["consumed0"], M=1, max_steps=cfg.max_steps,
                           M_out=24, prim_origin=prim_origin, rec_caps=rec_caps, pending=pend, n_w=nw,
                           q_cap=q_cap, per_event_cap=per_event_cap, time_sync=cfg.time_step, step=cfg.step)
    if rec_caps is not None:
        out, sofl, oofl, prim_fate, (fsi_rec, _rofl) = _rc      # joint per-event kind-1 FSI record
    else:
        out, sofl, oofl, prim_fate = _rc; fsi_rec = None
    sp = out["species"]; chg = out["charge"]; al = out["alive"]; p4o = out["p4"]
    nterms = [dict(species=sp, pid=jnp.where((sp == NUCLEON) & (chg == 1), 2212, 2112),
                   charge=chg,                                       # raw charge idx: nucleon isospin
                                                                    # (1=p) | pion charge (0:+,1:0,2:-).
                                                                    # ADDITIVE -- pid unchanged (golden-safe);
                                                                    # lets consumers count pions by charge.
                   p4=p4o, alive=al, origin=out["origin"], gen=out["gen"])]
    is_surv_pi = (sp == PION) & al                                     # escaped pions (output is escape-only)
    if channel == "qe":
        pim = jnp.linalg.norm(p4o[:, :, 1:], axis=2) * is_surv_pi
        jpi = jnp.argmax(pim, axis=1); has_pi = pim[ar, jpi] > 0.0
        created = dict(pid=jnp.where(has_pi, _CH_PID[chg[ar, jpi]], 0), p4=p4o[ar, jpi],
                       w=jnp.ones((n,)), alive=has_pi)
        pterm = dict(species=jnp.zeros((n,), jnp.int32), pid=jnp.zeros((n,), jnp.int32),
                     p4=jnp.zeros((n, 4)), charge=jnp.zeros((n,), jnp.int32), w=jnp.ones((n,)),
                     alive=jnp.ones((n,), bool), nsc=jnp.zeros((n,), jnp.int32))
        return pterm, nterms, sofl + oofl, created, fsi_rec
    # RES: split surviving pions into primary (origin tag) vs created
    is_prim = is_surv_pi & (out["origin"] == _ORIG_PRIM_PI)            # escaped primary pion (>=0 per event)
    jp = jnp.argmax(is_prim, axis=1); esc_prim = jnp.any(is_prim, axis=1)
    prim_ch = chg[ar, jp]
    # pterm pid: escaped -> charge->pid; converted -> -1 (vetoes); absorbed/none -> 0.
    pterm_pid = jnp.where(prim_fate == FATE_ESCAPE, _CH_PID[prim_ch],
                jnp.where(prim_fate == FATE_CONVERT, -1, 0)).astype(jnp.int32)
    pterm = dict(species=jnp.full((n,), PION, jnp.int32), pid=pterm_pid,
                 p4=jnp.where(esc_prim[:, None], p4o[ar, jp], 0.0), charge=prim_ch,
                 w=jnp.ones((n,)), alive=jnp.ones((n,), bool), nsc=jnp.zeros((n,), jnp.int32))
    is_cr = is_surv_pi & (out["origin"] != _ORIG_PRIM_PI)             # cascade-created surviving pion
    cim = jnp.linalg.norm(p4o[:, :, 1:], axis=2) * is_cr
    jc = jnp.argmax(cim, axis=1); has_cr = cim[ar, jc] > 0.0
    created = dict(pid=jnp.where(has_cr, _CH_PID[chg[ar, jc]], 0), p4=p4o[ar, jc],
                   w=jnp.ones((n,)), alive=has_cr)
    return pterm, nterms, sofl + oofl, created, fsi_rec


def cascade_nucleus(p_pi, p_N, pid_pi, pid_Ni, Npid, cfg, key, sabs=1.0, sscat=1.0,
                      channel="res", rec_caps=None, log_cap=None, n_w=None, q_cap=None, per_event_cap=None,
                      su_external=None):
    """Faithful engine, SHARED by RES (CC1pi) and QE (CC0pi).
    channel="res": a primary pion segment (+ its top-K knockouts) then a NUCLEON BFS over {RES recoil,
                   pion knockouts}; pterm = the surviving pion.
    channel="qe":  NO primary pion -- gen-0 nucleon = the QE proton (p_N); pterm = "no pion" (pid 0).
    Both share the nucleon BFS (top-K knockouts) + the created-pion (NN->NDelta->Npi) re-entry, so the
    meson veto (no surviving pion for CC0pi / exactly one pi+ for CC1pi) is handled uniformly.
    rec_caps=(Kp,Kn) (pool only): also return the joint per-event kind-1 FSI reweight record as a 5th
    element (for pool_fsi_reweight / the differentiable blueprint).  Default None -> 4-tuple as before.
    log_cap=L: run the SAME cascade with the in-engine SEGMENT logger and return (log, counts, overflow)
    -- the single entry point for the cascade-vertex/segment matrix (no duplicated cascade).
    ENGINE DEFAULTS (refill + waiting-queue ON for every consumer):
      n_w   : None -> refill with working set min(_DEFAULT_NW, n);  0 -> lock-step (bit-exact reference);
              int>0 -> refill with min(n_w, n).
      q_cap : None -> _DEFAULT_QCAP (keep overflow particles, sofl->0);  0 -> legacy drop-on-overflow.
      per_event_cap: None -> cfg.max_steps (per-slot step cap in the refill path).
    Returns (pterm, nucleon_terminals_per_gen, overflow, created[, fsi_record]) | (log, counts, overflow)."""
    # su_external (ablation harness): run on an EXTERNALLY supplied nucleus (e.g. ACHILLES's exact per-event
    # background + struck vertex) instead of sampling our own -- isolates transport from input generation.
    # Must provide npos,nmom,nisp,pos0,consumed0,ch0,kp (same keys setup_nucleus returns).
    su = setup_nucleus(p_pi, pid_pi, pid_Ni, cfg, key) if su_external is None else su_external
    n = p_pi.shape[0]
    _kpi, knuc, _kpi2 = jax.random.split(su["kp"], 3)     # knuc drives the pool (RNG stream preserved)
    # Resolve engine defaults: refill ON (n_w window) + waiting-queue ON (q_cap) for ALL consumers.
    nw_eff = 0 if n_w == 0 else (min(_DEFAULT_NW, n) if n_w is None else min(int(n_w), n))
    q_eff = _DEFAULT_QCAP if q_cap is None else int(q_cap)
    # POOLED engine (the single cascade core): ONE fixed-size stack stepped once/step, in/out reconcile
    # inside the step; persistent-refill working set + keep-overflow queue by default (n_w=0 -> lock-step).
    # Validated vs ACHILLES + bit-exact gates (docs/logbook/cascade_persistent_refill_plan.md).
    if log_cap is not None:
        return _cascade_pool(channel, p_pi, p_N, Npid, su, cfg, knuc, n, log_cap=log_cap,
                             n_w=nw_eff, q_cap=q_eff, per_event_cap=per_event_cap)
    res5 = _cascade_pool(channel, p_pi, p_N, Npid, su, cfg, knuc, n, rec_caps=rec_caps,
                         n_w=nw_eff, q_cap=q_eff, per_event_cap=per_event_cap)
    return res5 if rec_caps is not None else res5[:4]
