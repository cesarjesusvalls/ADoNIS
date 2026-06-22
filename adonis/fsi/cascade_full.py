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
import jax, jax.numpy as jnp
from adonis.fsi.cascade_discrete import (_propagate_discrete, _propagate_nucleon_discrete, _CH_PID,
                                         sample_nucleons, DiscreteCascadeConfig, _formation_zone,
                                         MB_TO_FM2, pion_branch_reweight, nucleon_scat_reweight)
from adonis.fsi.tracking import GEN_STRIDE as _GEN_STRIDE

PION, NUCLEON = 0, 1
FATE_NONE, FATE_ESCAPE, FATE_ABSORB, FATE_CONVERT = 0, 1, 2, 3
_ORIG_PRIM_PI = 2          # pool origin tag for the RES PRIMARY pion (0=RES recoil/QE chain, 1=pi-knockout)


def pion_segment(p4, pos, ch, consumed, npos, nmom, nisp, cfg, key, sabs=1.0, sscat=1.0):
    """Propagate one pion (batch (n,)) through its FULL segment via the validated _propagate_discrete.
    Returns (term, sec) flat dicts: term = the pion's terminal state + fate; sec = its leading recoil
    proton (alive only where a proton recoil exists).  Charge oscillation is internal to the segment."""
    out = _propagate_discrete(pos, p4, ch, npos, nmom, nisp, cfg, key, consumed,
                              jnp.asarray(sabs, float), jnp.asarray(sscat, float))
    (p_pi, ch_out, absorbed, conv, nsc, best_abs, w_fsi, nseg, n_trunc, brec,
     (best_rec, best_rec_pos, best_rec_fz), scat_ko_all, traj, abs_pos, best_abs2) = out
    fate = jnp.where(absorbed, FATE_ABSORB, jnp.where(conv, FATE_CONVERT, FATE_ESCAPE))
    pid = jnp.where(absorbed, 0, jnp.where(conv, -1, _CH_PID[ch_out]))     # 0 abs, -1 conv, else pi pid
    n = p4.shape[0]
    term = dict(species=jnp.full((n,), PION, jnp.int32), charge=ch_out, pid=pid, p4=p_pi,
                fate=fate, w=w_fsi, nsc=nsc, traj=traj)               # traj = per-step (pos,p4,alive) or None
    has_rec = jnp.linalg.norm(best_rec[:, 1:], axis=1) > 1.0
    sec = dict(species=jnp.full((n,), NUCLEON, jnp.int32), charge=jnp.ones((n,), jnp.int32),  # proton
               p4=best_rec, pos=best_rec_pos, fz=best_rec_fz, alive=has_rec, w=w_fsi)
    rp4, rpos, rfz, rchg = scat_ko_all                                    # (n,K,4),(n,K,3),(n,K),(n,K) all-species recoils
    # FIX: the pion-ABSORPTION proton (piNN->NN, ACHILLES final state) was dropped -- feed it as an
    # extra knockout slot so it is counted and re-cascades (the old factorized chain used last_abs_proton).
    # BOTH pion-absorption protons (piNN->NN) fed as extra knockout slots, each with its ACHILLES
    # formation zone SetFormationZone(pion, product); they re-cascade and are counted like ACHILLES.
    a1_alive = jnp.linalg.norm(best_abs[:, 1:], axis=1) > 1.0
    a2_alive = jnp.linalg.norm(best_abs2[:, 1:], axis=1) > 1.0
    fz1 = _formation_zone(p_pi, best_abs); fz2 = _formation_zone(p_pi, best_abs2)
    p4K = jnp.concatenate([rp4, best_abs[:, None, :], best_abs2[:, None, :]], axis=1)
    posK = jnp.concatenate([rpos, abs_pos[:, None, :], abs_pos[:, None, :]], axis=1)
    fzK = jnp.concatenate([rfz, fz1[:, None], fz2[:, None]], axis=1)
    # charge per knockout slot: scatter recoils carry their species (rchg); the two absorption products
    # are protons (piNN->NN gives the proton knockouts, charge 1).
    o1 = jnp.ones((rp4.shape[0], 1), jnp.int32)
    chgK = jnp.concatenate([rchg.astype(jnp.int32), o1, o1], axis=1)
    aliveK = jnp.concatenate([jnp.linalg.norm(rp4[:, :, 1:], axis=2) > 1.0,
                              a1_alive[:, None], a2_alive[:, None]], axis=1)
    secK = dict(p4=p4K, pos=posK, fz=fzK, chg=chgK, alive=aliveK, w=w_fsi)
    return term, sec, secK


def nucleon_segment(p4, pos, isp, fz, consumed, npos, nmom, nisp, cfg, key, sscat=1.0):
    """Propagate one nucleon (batch (n,)) through its segment via _propagate_nucleon_discrete.  isp is
    the proton-mask (bool).  Returns (term, sec): term = the nucleon's terminal state (nucleons always
    escape -- no absorption); sec = its leading knockout proton (alive only where one exists).  made_pi
    flags an NN->NDelta->NNpi pion (v1: flag only; the pion 4-vec spawn needs a kernel extension)."""
    out = _propagate_nucleon_discrete(pos, p4, isp, npos, nmom, nisp, cfg, key, fz, consumed,
                                      jnp.asarray(sscat, float))
    p_N, nsc, best_ko, best_ko_pos, best_ko_fz, w_scat, srec, made_pi, ko_all, pi_made, _nt, _ns, traj, consumed_out = out
    bpi, bpich, bpipos, bpifz = pi_made                                   # leading NN-created pion
    n = p4.shape[0]
    term = dict(species=jnp.full((n,), NUCLEON, jnp.int32), charge=isp.astype(jnp.int32),
                pid=jnp.where(isp, 2212, 2112), p4=p_N, fate=jnp.full((n,), FATE_ESCAPE, jnp.int32),
                w=w_scat, nsc=nsc, made_pi=made_pi, traj=traj,           # traj = per-step (pos,p4,alive) or None
                consumed=consumed_out,                                   # final consumed mask (BFS depletion, #2)
                pi4=bpi, pich=bpich, pipos=bpipos, pifz=bpifz)           # created-pion (4-vec, charge idx, vertex, fz)
    has_ko = jnp.linalg.norm(best_ko[:, 1:], axis=1) > 1.0
    sec = dict(species=jnp.full((n,), NUCLEON, jnp.int32), charge=jnp.ones((n,), jnp.int32),  # proton
               p4=best_ko, pos=best_ko_pos, fz=best_ko_fz, alive=has_ko, w=w_scat)
    kp4, kpos, kfz, kchg = ko_all                                         # (n,K,4),(n,K,3),(n,K),(n,K) all-species knockouts
    secK = dict(p4=kp4, pos=kpos, fz=kfz, chg=kchg.astype(jnp.int32),     # chg: 1=proton, 0=neutron (re-cascade by species)
                alive=jnp.linalg.norm(kp4[:, :, 1:], axis=2) > 1.0, w=w_scat)
    return term, sec, secK


def setup_carbon(p_pi, pid_pi, pid_Ni, cfg, key):
    """Sample the carbon background + struck vertex EXACTLY as DiscreteCascadeFSI.apply, so the engine's
    primary-pion segment reproduces the production chain bit-for-bit.  Returns the nucleus + pion init."""
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
                parent_id=jnp.full((n, P), -1, jnp.int32))            # MC-truth: spawning track (-1 = primary)


def _take(b, idx):
    """Gather along the particle axis (n, P) -> (n, K) with idx (n, K)."""
    n = idx.shape[0]; ar = jnp.arange(n)[:, None]
    return dict(species=b["species"][ar, idx], charge=b["charge"][ar, idx], p4=b["p4"][ar, idx],
                pos=b["pos"][ar, idx], fz=b["fz"][ar, idx], alive=b["alive"][ar, idx],
                w=b["w"][ar, idx], fate=b["fate"][ar, idx])


def compact(b, P_out):
    """Compact live particles of a (n, M) batch to the front of a (n, P_out) buffer (BFS refill /
    overflow handling).  Returns (compacted_batch, n_overflow) where n_overflow counts live particles
    that did not fit in P_out (dropped, never silently)."""
    alive = b["alive"]; n, M = alive.shape
    # stable rank of each live particle within its event (0-based); dead get a large rank
    rank = jnp.cumsum(alive.astype(jnp.int32), axis=1) - 1
    rank = jnp.where(alive, rank, M + P_out)                      # dead -> out of range
    n_overflow = jnp.sum((alive & (rank >= P_out)).astype(jnp.int32))
    out = empty_batch(n, P_out)
    ar = jnp.broadcast_to(jnp.arange(n)[:, None], (n, M))
    dst = jnp.where(rank < P_out, rank, P_out)                    # P_out = scratch drop slot (will be sliced off)
    for k in ("species", "charge", "fate", "origin", "gen", "track_id", "parent_id", "nsc"):
        if k not in b:                                            # optional (e.g. BFS kernel spawns omit nsc)
            continue
        out[k] = jnp.zeros((n, P_out + 1), b[k].dtype).at[ar, dst].set(b[k], mode="drop")[:, :P_out]
    for k in ("fz", "w"):
        out[k] = jnp.zeros((n, P_out + 1)).at[ar, dst].set(b[k], mode="drop")[:, :P_out]
    out["alive"] = jnp.zeros((n, P_out + 1), bool).at[ar, dst].set(b["alive"], mode="drop")[:, :P_out]
    out["p4"] = jnp.zeros((n, P_out + 1, 4)).at[ar, dst].set(b["p4"], mode="drop")[:, :P_out]
    out["pos"] = jnp.zeros((n, P_out + 1, 3)).at[ar, dst].set(b["pos"], mode="drop")[:, :P_out]
    return out, n_overflow


def make_kernel(npos, nmom, nisp, consumed0, cfg, sabs=1.0, sscat=1.0):
    """Build the per-generation kernel: species-dispatched propagation of a (n,P) buffer through the
    SHARED nucleus (npos,nmom,nisp).  v1: each slot runs BOTH segment fns, selects by species (2x
    waste, simple); the shared consumed mask is the generation's starting state for ALL slots
    (parallel-consumption approximation -- flagged in the logbook).  One leading secondary per slot."""
    def kernel(buf, key, consumed_cur):
        n, P = buf["alive"].shape
        pk = jax.random.split(key, P)

        def slot(i):
            p4 = buf["p4"][:, i]; pos = buf["pos"][:, i]; chg = buf["charge"][:, i]
            fz = buf["fz"][:, i]; alive = buf["alive"][:, i]; sp = buf["species"][:, i]
            tp, sp_, _ = pion_segment(p4, pos, chg, consumed_cur, npos, nmom, nisp, cfg, pk[i], sabs, sscat)
            tn, sn, _ = nucleon_segment(p4, pos, chg.astype(bool), fz, consumed_cur, npos, nmom, nisp, cfg, pk[i], sscat)
            is_pi = (sp == PION)
            # terminal (record): select by species; dead slots -> FATE_NONE, not alive
            seg_w = jnp.where(is_pi, tp["w"], tn["w"])               # kind-1 segment reweight (1 at nominal)
            term = dict(species=sp, pid=jnp.where(is_pi, tp["pid"], tn["pid"]),
                        p4=jnp.where(is_pi[:, None], tp["p4"], tn["p4"]),
                        fate=jnp.where(alive, jnp.where(is_pi, tp["fate"], tn["fate"]), FATE_NONE),
                        w=buf["w"][:, i] * seg_w, alive=alive)
            # leading secondary (a proton): select by species; alive only if the parent was alive + it exists
            salive = alive & jnp.where(is_pi, sp_["alive"], sn["alive"])
            sec = dict(species=jnp.full((n,), NUCLEON, jnp.int32), charge=jnp.ones((n,), jnp.int32),
                       pid=jnp.full((n,), 2212, jnp.int32),
                       p4=jnp.where(is_pi[:, None], sp_["p4"], sn["p4"]),
                       pos=jnp.where(is_pi[:, None], sp_["pos"], sn["pos"]),
                       fz=jnp.where(is_pi, sp_["fz"], sn["fz"]),
                       fate=jnp.zeros((n,), jnp.int32),
                       w=jnp.where(is_pi, sp_["w"], sn["w"]) * buf["w"][:, i], alive=salive)
            return term, sec
        terms = [slot(i) for i in range(P)]
        # stack slots back to (n, P) for terminals; (n, P) secondaries (1 per slot)
        def stk(key_, lst):
            return jnp.stack([d[key_] for d in lst], axis=1)
        tcat = {k: stk(k, [t[0] for t in terms]) for k in ("species", "pid", "p4", "fate", "w", "alive")}
        scat = {k: stk(k, [t[1] for t in terms]) for k in ("species", "charge", "p4", "pos", "fz", "fate", "w", "alive")}
        # secondaries carry no charge-as-pion-idx confusion (all protons); add missing 'charge' to term
        tcat["charge"] = jnp.zeros((n, P), jnp.int32)
        return tcat, scat, consumed_cur                          # v1 legacy: no cross-gen depletion (passthrough)
    return kernel


def run_cascade(init, kernel, key, consumed0, P=10, max_gen=6):
    """BFS over generations with fixed-shape buffers.

    init     : ParticleBatch (n, P0) of generation-0 particles (the pre-FSI interaction products).
    kernel   : (part, key, consumed) -> (term, spawn, consumed_out)  (nucleus captured in the closure).
    consumed0: the (n, A) background-consumed mask at gen 0 (the struck vertex); threaded + DEPLETED
               across generations so a nucleon hit in one generation is not re-hit in the next (#2).
    Returns (terminals_list, total_overflow): one ParticleBatch (n, P) per generation.
    """
    n = init["alive"].shape[0]
    cur, _ = compact(init, P)                                     # normalize to width P
    terminals = []
    overflow = jnp.int32(0)
    consumed = consumed0
    gkeys = jax.random.split(key, max_gen)
    for g in range(max_gen):
        term, spawn, consumed = kernel(cur, gkeys[g], consumed)
        terminals.append(term)
        if "track_id" in spawn:                                  # MC-truth: unique id per gen-(g+1) secondary
            M = spawn["alive"].shape[1]
            spawn["track_id"] = jnp.broadcast_to(
                (_GEN_STRIDE * (g + 1) + jnp.arange(M, dtype=jnp.int32))[None, :], (n, M))
        nxt, ofl = compact(spawn, P)                             # pack secondaries into the next generation
        overflow = overflow + ofl
        cur = nxt
    return terminals, overflow


def make_pool_stepper(su, cfg, with_rec=False):
    """Build the pooled-engine physics stepper (S2b): advance every slot of the (n, M) stack ONE step,
    dispatched by species (NUCLEON -> _nucleon_step, PION -> _pion_step), with the consumed mask threaded
    SLOT-SERIALLY (slot m+1 sees m's depletion).  Both per-step bodies run on every slot and are selected
    by species (the v1 2x-eval tradeoff; the dead-slot waste, the dominant 10-24x factor, is gone).  Each
    slot emits up to 2 NUCLEON spawns + 1 PION spawn: a nucleon slot -> (1 knockout N, 1 NN-created pion);
    a pion slot -> (up to 2 absorption/recoil N, 0 pion).  Returns stepper(stack, key, consumed) ->
    (stack2, terminal (n,M) bool [escaped final-state particles], spawn ParticleBatch (n,3M), consumed)."""
    from adonis.fsi.cascade_discrete import _nucleon_step, _pion_step, _load_density
    npos, nmom, nisp = su["npos"], su["nmom"], su["nisp"]
    rgrid, rhoP, rhoN, radius = _load_density(cfg.nucleus, cfg.density_n)
    _dead = lambda n: (jnp.zeros((n, 4)), jnp.zeros((n, 3)), jnp.zeros(n), jnp.zeros(n, jnp.int32), jnp.zeros(n, bool))

    def stepper(stack, key, consumed):
        n, M = stack["alive"].shape
        keys = jax.random.split(key, M)

        def slot(consumed, m):
            p4 = stack["p4"][:, m]; pos = stack["pos"][:, m]; fz = stack["fz"][:, m]
            nsc = stack["nsc"][:, m]; chg = stack["charge"][:, m]; al = stack["alive"][:, m]
            sp = stack["species"][:, m]
            is_N = (sp == NUCLEON) & al; is_pi = (sp == PION) & al
            d3 = p4[:, 1:]; dhat = d3 / jnp.clip(jnp.linalg.norm(d3, axis=1, keepdims=True), 1e-9, None)
            kN, kP = jax.random.split(keys[m])
            # NUCLEON branch (charge = isospin, 1=p); inactive slots produce no consumption/spawn.
            (p4n, posn, _dn, fzn, alnN), escN, _rc, _do, koN, pinN, consumedN, nstat = _nucleon_step(
                p4, pos, dhat, fz, chg.astype(bool), is_N, npos, nmom, nisp, consumed,
                rgrid, rhoP, rhoN, radius, cfg, kN)
            # PION branch (charge = pion index 0/1/2); scatter continues, abs/conv removed.
            (p4p, posp, _dp, chp, nscp, alnP), escP, is_abs, is_conv, s1, s2, consumedP, pstat = _pion_step(
                p4, pos, dhat, chg, nsc, is_pi, npos, nmom, nisp, consumed,
                rgrid, rhoP, rhoN, radius, cfg, kP)
            # kind-1 FSI reweight sufficient statistics (mirrors the legacy brec/srec per-step records):
            #   pion: record every geometric hit (has_hit) -> branch code + sigma components (sa,ss,si).
            #   nucleon: record every in-slab candidate step (perp2_c<1e5) -> hit flag + a_nom=pi b^2/sigma.
            # Computed ONLY when with_rec (the differentiable/tuning path) -> forward generation pays nothing.
            if with_rec:
                p_hh, p_bc, p_sa, p_ss, p_si = pstat                  # _pion_step stats
                n_hh, n_perp2, n_sig = nstat                          # _nucleon_step stats
                a_nom = jnp.pi * n_perp2 / jnp.clip(n_sig * MB_TO_FM2, 1e-12, None)
                rec_slot = (is_pi & p_hh, p_bc, p_sa, p_ss, p_si,    # pion hit mask + branch stats
                            is_N & (n_perp2 < 1e5), n_hh, a_nom)     # nucleon candidate mask + hit + a_nom
            consumed = consumedN | consumedP                          # only the active species adds bits
            p4_2 = jnp.where(is_N[:, None], p4n, jnp.where(is_pi[:, None], p4p, p4))
            pos_2 = jnp.where(is_N[:, None], posn, jnp.where(is_pi[:, None], posp, pos))
            fz_2 = jnp.where(is_N, fzn, fz)                           # only the nucleon updates fz
            nsc_2 = jnp.where(is_pi, nscp, nsc)                       # only the pion updates nsc
            chg_2 = jnp.where(is_pi, chp, chg)                        # pion charge oscillates
            al_2 = jnp.where(is_N, alnN, jnp.where(is_pi, alnP, al))
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
            out_slot = (new, nuc1, nuc2, pio, rec_slot) if with_rec else (new, nuc1, nuc2, pio)
            return consumed, out_slot

        consumed, scanned = jax.lax.scan(slot, consumed, jnp.arange(M))
        new, nuc1, nuc2, pio = scanned[:4]
        T = lambda x: jnp.moveaxis(x, 0, 1)                          # (M, n, ...) -> (n, M, ...)
        p4s, poss, fzs, nscs, chgs, als, terms, fates = new
        stack2 = {**stack, "p4": T(p4s), "pos": T(poss), "fz": T(fzs), "nsc": T(nscs),
                  "charge": T(chgs), "alive": T(als), "fate": T(fates)}
        terminal = T(terms)
        if with_rec:                                                 # per-slot (n,M) kind-1 record this step
            rk = ("pi_hh", "pi_bc", "pi_sa", "pi_ss", "pi_si", "nu_m", "nu_hh", "nu_a")
            rec = {k: T(v) for k, v in zip(rk, scanned[4])}

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
        if with_rec:
            return stack2, terminal, spawn, consumed, rec
        return stack2, terminal, spawn, consumed

    return stepper


def pool_reconcile(stack, terminal, spawn, M):
    """The POOLED engine's per-step in/out (docs/logbook/cascade_pool_engine.md): drop the slots that
    terminated this step (escape/absorb/convert), KEEP the survivors, INSERT the particles created this
    step, and re-pack to the fixed width M -- counting any that don't fit as overflow.  This is exactly
    compact(concat(survivors, spawned), M), so it reuses the validated compaction primitive.
      stack    : ParticleBatch (n, M)   -- the current stack (post-step state)
      terminal : (n, M) bool            -- slots that reached a terminal this step
      spawn    : ParticleBatch (n, K)   -- particles created this step (alive mask = which are real)
      M        : int                    -- fixed stack width
    Returns (new_stack (n, M), overflow (scalar))."""
    stack = {**stack, "alive": stack["alive"] & ~terminal}
    combined = {k: jnp.concatenate([stack[k], spawn[k]], axis=1) for k in stack}
    return compact(combined, M)


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
    """Per-event kind-1 FSI reweight buffers in legacy brec/srec layout (defaults give per-slot LR=1)."""
    return dict(bc=jnp.zeros((n, Kp), jnp.int32), sa=jnp.ones((n, Kp)), ss=jnp.ones((n, Kp)),
                si=jnp.zeros((n, Kp)), nh=jnp.zeros(n, jnp.int32),
                hh=jnp.zeros((n, Kn), bool), a=jnp.full((n, Kn), 50.0), ns=jnp.zeros(n, jnp.int32))


def pool_fsi_reweight(record, sabs, sscat):
    """Joint kind-1 FSI reweight for the pool (mirrors legacy): pion branch-split (sabs,sscat) x nucleon
    sigma_scatter (sscat).  Pure in (sabs,sscat); == 1 at nominal; == the in-walk weight at any theta."""
    wp = pion_branch_reweight((record["bc"], record["sa"], record["ss"], record["si"], record["nh"]),
                              sabs, sscat)
    wn = nucleon_scat_reweight((record["hh"], record["a"], record["ns"]), sscat)
    return wp * wn


def run_cascade_pool(init, stepper, key, state0, M, max_steps, M_out=24, prim_origin=-999, rec_caps=None):
    """POOLED engine loop (docs/logbook/cascade_pool_engine.md): ONE fixed-size (n, M) particle stack
    stepped once per step; the in/out reconcile (pool_reconcile) runs INSIDE the step, vs the BFS's
    per-generation compact across max_gen separate full-max_steps passes (~10-24x dead-slot waste).
      stepper(stack, key, state) -> (stack2 (n,M), terminal (n,M) bool, spawn (n,K), state2, rec)
        advances every live slot ONE step and returns: the post-step stack, the newly-terminal flag, the
        particles created this step (spawn ParticleBatch), the threaded step state (e.g. the consumed
        mask), and per-slot kind-1 FSI record fields `rec` (consumed only when rec_caps is set).
    Escaped terminals (the cascade FINAL STATE) are accumulated into a fixed (n, M_out) output batch each
    step.  `prim_origin` (RES): the origin tag of the PRIMARY pion -- its terminal FATE (escape/absorb/
    convert) is latched per event; -999 (default) never matches -> a no-op for QE/tests.
    rec_caps=(Kp,Kn): if set, accumulate the per-event kind-1 FSI reweight record (pion hits<=Kp,
    nucleon candidate steps<=Kn) and return it as a 5th element; if None, return the 4-tuple as before.
    Returns (out_batch, stack_overflow, out_overflow, prim_fate[, fsi_record])."""
    n = init["alive"].shape[0]; ar = jnp.arange(n)
    stack, _ = compact(init, M)
    out0 = empty_batch(n, M_out)
    with_rec = rec_caps is not None
    Kp, Kn = rec_caps if with_rec else (1, 1)
    rec0 = _empty_fsi_record(n, Kp, Kn)

    def cond(st):
        return (st[0] < max_steps) & jnp.any(st[1]["alive"])

    def body(st):
        i, stk, state, out, sofl, oofl, prim, rb, rofl = st
        _step = stepper(stk, jax.random.fold_in(key, i), state)      # 4- or 5-tuple (rec optional)
        stk2, terminal, spawn, state2 = _step[:4]
        rec = _step[4] if len(_step) == 5 else None
        term_batch = {**stk2, "alive": terminal}
        out2, oo = compact({k: jnp.concatenate([out[k], term_batch[k]], axis=1) for k in out}, M_out)
        isprim = (stk2["origin"] == prim_origin) & (stk2["species"] == PION) & (stk2["fate"] != FATE_NONE)
        anyp = jnp.any(isprim, axis=1); j = jnp.argmax(isprim, axis=1)
        prim = jnp.where(anyp & (prim == FATE_NONE), stk2["fate"][ar, j], prim)
        if with_rec:                                                  # accumulate kind-1 FSI records
            (bc, sa, ss, si), nh, op = _rec_scatter(
                [rb["bc"], rb["sa"], rb["ss"], rb["si"]], rb["nh"], rec["pi_hh"],
                [rec["pi_bc"], rec["pi_sa"], rec["pi_ss"], rec["pi_si"]], Kp)
            (hh, a), ns, on = _rec_scatter([rb["hh"], rb["a"]], rb["ns"], rec["nu_m"],
                                           [rec["nu_hh"], rec["nu_a"]], Kn)
            rb = dict(bc=bc, sa=sa, ss=ss, si=si, nh=nh, hh=hh, a=a, ns=ns)
            rofl = (rofl + op + on).astype(rofl.dtype)
        newstk, so = pool_reconcile(stk2, terminal, spawn, M)
        return (i + jnp.int32(1), newstk, state2, out2,
                sofl + so.astype(sofl.dtype), oofl + oo.astype(oofl.dtype), prim, rb, rofl)

    init_st = (jnp.int32(0), stack, state0, out0, jnp.int32(0), jnp.int32(0),
               jnp.full(n, FATE_NONE, jnp.int32), rec0, jnp.int32(0))
    _, stack, _, out, sofl, oofl, prim, rb, rofl = jax.lax.while_loop(cond, body, init_st)
    if with_rec:
        return out, sofl, oofl, prim, (rb, rofl)
    return out, sofl, oofl, prim


def _nucleon_kernel(npos, nmom, nisp, cfg, sscat=1.0):
    """v2 nucleon-only generation kernel: run nucleon_segment on every slot (NO 2x-run-both, NO pion).
    Each slot emits its TOP-K all-species knockouts -> spawn buffer (n, P*K).  Threads the DEPLETING
    consumed mask BOTH within a generation (slot i+1 sees slots 0..i's consumption -- intra-generation,
    #2b) AND across generations (next gen starts from this gen's cumulative consumed -- #2), so a struck
    nucleon is never re-hit.  Slots are processed in index order (vs ACHILLES time order; removes the
    double-consumption overcount regardless of order).  Returns (term, spawn, consumed_out)."""
    def kernel(buf, key, consumed_cur):
        n, P = buf["alive"].shape
        pk = jax.random.split(key, P)

        def slot(i, consumed_in):
            alive = buf["alive"][:, i]
            tn, _, snK = nucleon_segment(buf["p4"][:, i], buf["pos"][:, i], buf["charge"][:, i].astype(bool),
                                         buf["fz"][:, i], consumed_in, npos, nmom, nisp, cfg, pk[i], sscat)
            pi_alive = alive & (jnp.linalg.norm(tn["pi4"][:, 1:], axis=1) > 1.0)   # this slot made a pion
            term = dict(species=jnp.full((n,), NUCLEON, jnp.int32), charge=buf["charge"][:, i], pid=tn["pid"],
                        p4=tn["p4"], fate=jnp.where(alive, FATE_ESCAPE, FATE_NONE),
                        w=buf["w"][:, i] * tn["w"], alive=alive, origin=buf["origin"][:, i], gen=buf["gen"][:, i],
                        track_id=buf["track_id"][:, i], parent_id=buf["parent_id"][:, i],   # MC-truth
                        p4_birth=buf["p4"][:, i], pos=buf["pos"][:, i],                      # birth state
                        pi4=tn["pi4"], pich=tn["pich"], pipos=tn["pipos"], pifz=tn["pifz"], pi_alive=pi_alive)
            K = snK["p4"].shape[1]
            secK = dict(p4=snK["p4"], pos=snK["pos"], fz=snK["fz"], chg=snK["chg"],
                        alive=snK["alive"] & alive[:, None],
                        w=snK["w"][:, None] * buf["w"][:, i][:, None] * jnp.ones((n, K)),
                        origin=buf["origin"][:, i][:, None] * jnp.ones((n, K), jnp.int32),   # inherit gen-0 ancestor
                        gen=(buf["gen"][:, i][:, None] + 1) * jnp.ones((n, K), jnp.int32),   # BFS depth + 1
                        parent_id=buf["track_id"][:, i][:, None] * jnp.ones((n, K), jnp.int32))  # spawning track
            return term, secK, tn["consumed"]                          # slot's depleted background
        # sequential slot loop threading consumed (intra-generation depletion, #2b): slot i runs against
        # the mask already depleted by slots 0..i-1, so two secondaries can't strike the same nucleon.
        res = []
        consumed_run = consumed_cur
        for i in range(P):
            term_i, secK_i, consumed_run = slot(i, consumed_run)
            res.append((term_i, secK_i))
        consumed_out = consumed_run                                     # cumulative after all slots (-> next gen)
        tcat = {k: jnp.stack([r[0][k] for r in res], axis=1) for k in
                ("species", "charge", "pid", "p4", "fate", "w", "alive", "origin", "gen", "track_id", "parent_id",
                 "p4_birth", "pos", "pi4", "pich", "pipos", "pifz", "pi_alive")}
        K = res[0][1]["p4"].shape[1]

        def sfield(k):                                                 # (n,P,K,...) -> (n, P*K, ...)
            arr = jnp.stack([r[1][k] for r in res], axis=1)
            return arr.reshape((n, P * K) + arr.shape[3:])
        _chg = sfield("chg").astype(jnp.int32)                          # 1=proton, 0=neutron (all-species re-cascade)
        scat = dict(species=jnp.full((n, P * K), NUCLEON, jnp.int32), charge=_chg,
                    pid=jnp.where(_chg == 1, 2212, 2112).astype(jnp.int32), fate=jnp.zeros((n, P * K), jnp.int32),
                    p4=sfield("p4"), pos=sfield("pos"), fz=sfield("fz"), w=sfield("w"), alive=sfield("alive"),
                    origin=sfield("origin").astype(jnp.int32), gen=sfield("gen").astype(jnp.int32),
                    parent_id=sfield("parent_id").astype(jnp.int32),
                    track_id=jnp.zeros((n, P * K), jnp.int32))         # track_id assigned in run_cascade (needs gen)
        return tcat, scat, consumed_out
    return kernel


def _cascade_pool_v2(channel, p_pi, p_N, Npid, su, cfg, knuc, n, P, rec_caps=None):
    """POOLED-engine realization of cascade_carbon_v2 (QE + RES), mapping the flat (n,M_out) terminal
    buffer back to the rich (pterm, nterms, overflow, created) schema.
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
        prim_origin = -999
    else:                                                              # RES: primary pion + recoil nucleon
        g0 = empty_batch(n, 2)
        g0["alive"] = jnp.ones((n, 2), bool)
        g0["species"] = jnp.array([PION, NUCLEON], jnp.int32)[None, :] * jnp.ones((n, 1), jnp.int32)
        g0["charge"] = jnp.stack([su["ch0"], (Npid == 2212).astype(jnp.int32)], axis=1)
        g0["p4"] = jnp.stack([p_pi, p_N], axis=1)
        g0["pos"] = jnp.broadcast_to(su["pos0"][:, None, :], (n, 2, 3))
        g0["origin"] = jnp.array([_ORIG_PRIM_PI, 0], jnp.int32)[None, :] * jnp.ones((n, 1), jnp.int32)
        prim_origin = _ORIG_PRIM_PI
    stepper = make_pool_stepper(su, cfg, with_rec=rec_caps is not None)
    _rc = run_cascade_pool(g0, stepper, knuc, su["consumed0"], M=P, max_steps=cfg.max_steps,
                           M_out=24, prim_origin=prim_origin, rec_caps=rec_caps)
    if rec_caps is not None:
        out, sofl, oofl, prim_fate, (fsi_rec, _rofl) = _rc      # joint per-event kind-1 FSI record
    else:
        out, sofl, oofl, prim_fate = _rc; fsi_rec = None
    sp = out["species"]; chg = out["charge"]; al = out["alive"]; p4o = out["p4"]
    nterms = [dict(species=sp, pid=jnp.where((sp == NUCLEON) & (chg == 1), 2212, 2112),
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


def cascade_carbon_v2(p_pi, p_N, pid_pi, pid_Ni, Npid, cfg, key, P=12, max_gen=6, sabs=1.0, sscat=1.0,
                      channel="res", rec_caps=None):
    """Faithful engine, SHARED by RES (CC1pi) and QE (CC0pi).
    channel="res": a primary pion segment (+ its top-K knockouts) then a NUCLEON BFS over {RES recoil,
                   pion knockouts}; pterm = the surviving pion.
    channel="qe":  NO primary pion -- gen-0 nucleon = the QE proton (p_N); pterm = "no pion" (pid 0).
    Both share the nucleon BFS (top-K knockouts) + the created-pion (NN->NDelta->Npi) re-entry, so the
    meson veto (no surviving pion for CC0pi / exactly one pi+ for CC1pi) is handled uniformly.
    rec_caps=(Kp,Kn) (pool only): also return the joint per-event kind-1 FSI reweight record as a 5th
    element (for pool_fsi_reweight / the differentiable blueprint).  Default None -> 4-tuple as before.
    Returns (pterm, nucleon_terminals_per_gen, overflow, created[, fsi_record])."""
    su = setup_carbon(p_pi, pid_pi, pid_Ni, cfg, key)
    n = p_pi.shape[0]
    kpi, knuc, kpi2 = jax.random.split(su["kp"], 3)
    if getattr(cfg, "engine", "bfs") == "pool":
        # POOLED engine: ONE fixed-size stack stepped once/step, in/out reconcile inside the step (vs the
        # BFS's max_gen full-max_steps passes).  More faithful (true step-order consumption + ALL created
        # pions propagated from their creation point; for RES the primary pion is a gen-0 PION stack slot
        # whose own scatter-recoils/knockouts spawn natively -- no pre-segment extraction).  NOT bit-
        # identical to the BFS -> validated vs ACHILLES (docs/logbook/cascade_pool_engine.md).
        res5 = _cascade_pool_v2(channel, p_pi, p_N, Npid, su, cfg, knuc, n, P, rec_caps=rec_caps)
        return res5 if rec_caps is not None else res5[:4]
    if rec_caps is not None:
        raise NotImplementedError("rec_caps (kind-1 FSI record) requires engine='pool'")
    if channel == "res":
        pt, _, precK = pion_segment(p_pi, su["pos0"], su["ch0"], su["consumed0"], su["npos"], su["nmom"],
                                    su["nisp"], cfg, kpi, sabs, sscat)
        pterm = dict(species=jnp.full((n,), PION, jnp.int32), pid=pt["pid"], p4=pt["p4"],
                     charge=pt["charge"], w=pt["w"], alive=jnp.ones((n,), bool), nsc=pt["nsc"])
        # gen-0 nucleons: the RES recoil nucleon (slot 0) + ALL top-K pion knockout protons (slots 1..K)
        K = precK["p4"].shape[1]
        g0 = empty_batch(n, 1 + K)
        g0["alive"] = jnp.concatenate([jnp.ones((n, 1), bool), precK["alive"]], 1)
        g0["species"] = jnp.full((n, 1 + K), NUCLEON, jnp.int32)
        g0["charge"] = jnp.concatenate([(Npid == 2212).astype(jnp.int32)[:, None], precK["chg"]], 1)  # real recoil species (#2)
        g0["p4"] = jnp.concatenate([p_N[:, None, :], precK["p4"]], 1)
        g0["pos"] = jnp.concatenate([su["pos0"][:, None, :], precK["pos"]], 1)
        g0["fz"] = jnp.concatenate([jnp.zeros((n, 1)), precK["fz"]], 1)
        g0["w"] = jnp.concatenate([jnp.ones((n, 1)), precK["w"][:, None] * jnp.ones((n, K))], 1)
        g0["origin"] = jnp.concatenate([jnp.zeros((n, 1), jnp.int32), jnp.ones((n, K), jnp.int32)], 1)  # 0=RES recoil, 1=pi-knockout
        g0["track_id"] = jnp.broadcast_to(jnp.arange(1 + K, dtype=jnp.int32)[None, :], (n, 1 + K))  # gen-0 ids 0..K
    else:                                                              # QE (CC0pi): no primary pion
        pterm = dict(species=jnp.zeros((n,), jnp.int32), pid=jnp.zeros((n,), jnp.int32),
                     p4=jnp.zeros((n, 4)), charge=jnp.zeros((n,), jnp.int32), w=jnp.ones((n,)),
                     alive=jnp.ones((n,), bool), nsc=jnp.zeros((n,), jnp.int32))
        g0 = empty_batch(n, 1)
        g0["alive"] = jnp.ones((n, 1), bool)
        g0["species"] = jnp.full((n, 1), NUCLEON, jnp.int32)
        g0["charge"] = (Npid == 2212).astype(jnp.int32)[:, None]
        g0["p4"] = p_N[:, None, :]
        g0["pos"] = su["pos0"][:, None, :]
        g0["fz"] = jnp.zeros((n, 1))
        g0["w"] = jnp.ones((n, 1))
        g0["track_id"] = jnp.zeros((n, 1), jnp.int32)                  # gen-0 track id 0 (QE proton)
    kernel = _nucleon_kernel(su["npos"], su["nmom"], su["nisp"], cfg, sscat)
    nterms, ofl = run_cascade(g0, kernel, knuc, su["consumed0"], P=P, max_gen=max_gen)
    # CREATED-PION RE-ENTRY: gather the leading NN-created pion per event across the nucleon BFS, then
    # cascade it as a pion (it can survive as a pi+ and BE the signal pion when the primary died).
    ar = jnp.arange(n)
    bpi = jnp.zeros((n, 4)); bm = jnp.zeros(n); bch = jnp.ones((n,), jnp.int32)
    bpos = su["pos0"]; bfz = jnp.zeros(n)
    for g in nterms:
        pm = jnp.linalg.norm(g["pi4"][:, :, 1:], axis=2) * g["pi_alive"]
        j = jnp.argmax(pm, axis=1); gm = pm[ar, j]; upd = gm > bm
        bpi = jnp.where(upd[:, None], g["pi4"][ar, j], bpi); bm = jnp.where(upd, gm, bm)
        bch = jnp.where(upd, g["pich"][ar, j], bch)
        bpos = jnp.where(upd[:, None], g["pipos"][ar, j], bpos); bfz = jnp.where(upd, g["pifz"][ar, j], bfz)
    has_created = bm > 0.0
    cpt, _, _ = pion_segment(bpi, bpos, bch, su["consumed0"], su["npos"], su["nmom"], su["nisp"], cfg, kpi2, sabs, sscat)
    created = dict(pid=jnp.where(has_created, cpt["pid"], 0), p4=cpt["p4"], w=cpt["w"], alive=has_created)
    return pterm, nterms, ofl, created


def cascade_carbon(p_pi, p_N, pid_pi, pid_Ni, Npid, cfg, key, P=10, max_gen=6, sabs=1.0, sscat=1.0):
    """Full faithful cascade on carbon: gen-0 = the primary pion + primary recoil nucleon at the struck
    vertex; BFS re-cascades all (leading, v1) secondaries through the SHARED nucleus.  Returns
    (terminals_per_gen, overflow)."""
    su = setup_carbon(p_pi, pid_pi, pid_Ni, cfg, key)
    n = p_pi.shape[0]
    g0 = empty_batch(n, 2)
    g0["alive"] = jnp.ones((n, 2), bool)
    g0["species"] = jnp.stack([jnp.full((n,), PION, jnp.int32), jnp.full((n,), NUCLEON, jnp.int32)], 1)
    g0["charge"] = jnp.stack([su["ch0"], (Npid == 2212).astype(jnp.int32)], 1)
    g0["p4"] = jnp.stack([p_pi, p_N], 1)
    g0["pos"] = jnp.stack([su["pos0"], su["pos0"]], 1)
    kernel = make_kernel(su["npos"], su["nmom"], su["nisp"], su["consumed0"], cfg, sabs, sscat)
    return run_cascade(g0, kernel, su["kp"], su["consumed0"], P=P, max_gen=max_gen)
