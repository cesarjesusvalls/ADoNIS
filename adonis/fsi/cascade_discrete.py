"""Differentiable DISCRETE-Glauber pion cascade -- a faithful JAX port of the ACHILLES Cascade
(src/Achilles/Cascade.cc Evolve/BaseAlgorithm + RunCascade.cc CrossSection mode), validated to
reproduce the ACHILLES pi+-12C transparency oracle to ~5% (scratch/discrete_transport.py).

WHY discrete, not the continuum mean-free-path (cascade_real): ACHILLES gives each background
nucleon a transverse interaction reach via prob = exp(-pi b^2/(sigma/10)) over EXPLICIT nucleons
(impact parameter b), so sqrt(sigma/pi) ~ 2.2 fm of reach.  The 1-D continuum MFP ignores this
and is ~2x too low.  Here the pion walks (step dist ~0.04 fm) through A nucleons placed from the
ACHILLES QMC configurations (data/configurations/QMC_configs.out.gz), interacting with the
first (smallest-b) nucleon in the slab that passes its exp(-pi b^2/sigma) roll, branching
abs/scatter by sigma_abs/sigma_tot, with DCC scatter angle + Pauli blocking on the recoil, and
CONSUMING the struck nucleon.  Cross sections are the bit-exact Oset abs + ANL-Osaka DCC scatter
ports (proven 1.0000 / 1.003 vs instrumented ACHILLES).

Differentiability (kind-1 reweighting): the trajectory is SAMPLED against a frozen proposal (the
cross sections at detached parameters); the Oset/DCC knobs enter via a per-event likelihood
weight, so d/d(knob) E[obs] is exact and the sampled final state is preserved.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from adonis.fsi import oset_xsec as ox
from adonis.fsi.mb import cascade_mb
from adonis.fsi.cascade_real import (_load_density, _rho_species, _kf_local, _two_body_cm_scatter,
                                     _boost, MB_TO_FM2, _CH_MASS, _CH_PID)
from adonis.fsi.absorption_modes import kernel_tables as _abs_kernel_tables
from adonis.fsi.nucleon_cascade import nn_elastic_sigma     # NN-elastic sigma (used by _nucleon_step)
from adonis.fsi import nn_inelastic as nni                  # NN -> N Delta -> N N pi (nn_inelastic)

# pion-absorption proton-count distribution + partner species, indexed by ch*2+struck_p (ch 0:pi+
# 1:pi0 2:pi-; struck_p 1=proton).  Faithful ACHILLES isospin partition (Nucl.Phys. A568) -- replaces
# the old geometric nearest-partner pick that biased proton multiplicity for neutron-rich targets.
_ABS_W_NP, _ABS_PART_NP = _abs_kernel_tables()      # (6,3) numpy constants

M_N = ox.M_N                                          # average nucleon mass (Constant::mN = (mp+mn)/2)
from adonis.constants import mp as _MP_PHYS, mn as _MN_PHYS   # PHYSICAL per-species (ACHILLES particle 4-vec E)
_M_ETA = 548.0                # eta mass [MeV] (ACHILLES Constants mEta) for the piN->etaN' conversion baryon (#5)
HBARC = ox.HBARC
_CFG = {}


def _formation_zone(p_in, p_out):
    """ACHILLES Particle::SetFormationZone: fz = E_in * hbarc / |mN^2 - p_in.p_out|  [fm].
    p_in (n,4) = incoming nucleon momentum, p_out (n,4) = outgoing.  Forward scatters (p_in~p_out)
    -> |mN^2 - p_in.p_out| -> 0 -> large fz (free-streams out); wide scatters -> small fz."""
    dot4 = p_in[:, 0] * p_out[:, 0] - jnp.sum(p_in[:, 1:] * p_out[:, 1:], axis=1)
    return p_in[:, 0] * HBARC / jnp.clip(jnp.abs(M_N ** 2 - dot4), 1e-6, None)


def _load_qmc_configs(nmax=36000, name="QMC_configs.out.gz"):
    """Load A-nucleon configurations (positions [fm], isospin proton-mask, per-config weight).
    nmax=36000 = ALL configs in QMC_configs.out.gz (ACHILLES uses the full set); verified the
    transparency is unchanged vs the old 20000 cap (first-20k and full-36k have identical rms radius
    and both sample ∝ weight)."""
    # Cache as NUMPY (not jnp): a jnp array first created inside a jit trace would leak the
    # tracer context to later traces (cf. cascade_mb._jax_grids_resolved); asarray per-call is free.
    if name not in _CFG:
        path = Path(__file__).resolve().parents[2].parent / "Achilles" / "data" / "configurations" / name
        with gzip.open(path, "rt") as f:
            # header: [A Nconfigs maxWgt minWgt] (ACHILLES Configuration.cc:30-37).  A is read HERE,
            # not hardcoded -> QMC (A=12, C) and RMF (A=40, Ar) share this parser unchanged.
            hdr = f.readline().split()
            A = int(hdr[0]); ncfg = int(hdr[1]); nread = min(nmax, ncfg)
            iso = np.zeros((nread, A), bool); pos = np.zeros((nread, A, 3)); wt = np.zeros(nread)
            for c in range(nread):
                for i in range(A):
                    t = f.readline().split()
                    iso[c, i] = float(t[0]) > 0
                    pos[c, i] = [float(t[1]), float(t[2]), float(t[3])]
                wt[c] = float(f.readline()); f.readline()
        _CFG[name] = (pos, iso, wt / wt.sum(), A)
    pos, iso, w, A = _CFG[name]
    return jnp.asarray(pos), jnp.asarray(iso), jnp.asarray(w), A


_KSLAB = 3                   # # of nearest in-slab nucleons whose cross sections are evaluated per step
                             # (fast_xsec).  sigma for non-in-slab nucleons is never used (prob is masked
                             # to in_slab); >=K in one 0.04 fm slab is ~never, so this is bit-exact.


def pion_branch_reweight(brec, sabs, sscat):
    """Kind-1 branching reweight from compressed walk records brec =
    (branch (n,K) int {0 scatter, 1 abs, 2 conversion}, sa, ss, si (n,K), n_hits (n,)).
    Per-hit likelihood ratio p_branch(theta)/p_branch(nominal) with
    p_abs = sabs*sa/D, p_scat = sscat*ss/D, p_conv = si/D, D = sabs*sa + sscat*ss + si
    (conversion sigma unscaled).  Pure in (sabs, sscat); == in-propagation w_fsi; reduces to
    the previous two-branch formula where si = 0."""
    bc, sa, ss, si, nh = brec
    valid = jnp.arange(sa.shape[1])[None, :] < nh[:, None]
    ss = jnp.clip(ss, 1e-6, None)
    D0 = sa + ss + si
    Dk = sabs * sa + sscat * ss + si
    num = jnp.where(bc == 1, sabs * sa, jnp.where(bc == 2, si, sscat * ss))
    den = jnp.where(bc == 1, sa, jnp.where(bc == 2, si, ss))
    br = (num / Dk) / jnp.clip(den / D0, 1e-12, None)
    return jnp.prod(jnp.where(valid, br, 1.0), axis=1)


def fsi_pion_reweight(brec, s_abs, s_el, s_cex, s_conv):
    """GRANULAR pion FSI reweight -- BRANCH x SURVIVAL, one slot per IN-SLAB CANDIDATE STEP (hit or not).

    brec = (code4 {0 el,1 cex,2 abs,3 conv}, sa, ss_el, ss, si  [at the STRUCK candidate, branch factor],
            hh (hit flag), a (= pi*perp2/(sigma_tot*MB_TO_FM2) at the CLOSEST in-slab candidate),
            sa_c, ss_el_c, ss_c, si_c [at that candidate, for the sigma_tot response], np_ (n,) slot count).

    A sigma scale moves TWO things and the reweight must carry both (the nucleon reweight always did;
    the pion one carried only the first -- see docs/logbook/info_content.md):
      (1) BRANCH, given an interaction:  s_realized * D0/D,
          D0 = sa+ss+si,  D = s_abs*sa + s_el*ss_el + s_cex*(ss-ss_el) + s_conv*si.
      (2) SURVIVAL -- WHETHER it interacts.  The walk draws the hit with prob = exp(-a) per candidate
          (_pion_step), and sigma_tot -> g*sigma_tot maps a -> a/g, so with g = D_c/D0_c at that candidate
            hit  : pk/p0        no-hit : (1-pk)/(1-p0),     p0 = exp(-a), pk = exp(-a/g).
    Under a COMMON rescale s: g = s, the branch factor is 1 but the survival factor is NOT -- which is the
    whole point (a common rescale changes the mean free path).  == 1 at all-nominal (g=1 -> pk=p0)."""
    code, sa, ss_el, ss, si, hh, a, sa_c, ss_el_c, ss_c, si_c, np_ = brec
    valid = jnp.arange(sa.shape[1])[None, :] < np_[:, None]
    # On a NO-HIT slot the branch stats are taken at j = argmin over an all-inf metric -> meaningless (and
    # possibly non-finite).  jnp.where picks the right VALUE, but a non-finite untaken branch still poisons
    # the reverse-mode GRADIENT (nan * 0 = nan), so neutralize the branch inputs off-hit up front.
    hitf = hh.astype(bool)
    sa = jnp.where(hitf, sa, 1.0); ss_el = jnp.where(hitf, ss_el, 1.0)
    ss = jnp.where(hitf, ss, 1.0); si = jnp.where(hitf, si, 0.0)
    # (1) branch, at the struck candidate (only meaningful on a hit)
    ss_cex = jnp.clip(ss - ss_el, 0.0, None)
    D0 = sa + ss + si
    D = s_abs * sa + s_el * ss_el + s_cex * ss_cex + s_conv * si
    s_real = jnp.where(code == 0, s_el, jnp.where(code == 1, s_cex, jnp.where(code == 2, s_abs, s_conv)))
    branch = s_real * D0 / jnp.clip(D, 1e-12, None)
    # (2) survival, at the closest in-slab candidate: sigma_tot scale g = D_c/D0_c
    ss_cex_c = jnp.clip(ss_c - ss_el_c, 0.0, None)
    D0_c = jnp.clip(sa_c + ss_c + si_c, 1e-12, None)
    D_c = s_abs * sa_c + s_el * ss_el_c + s_cex * ss_cex_c + s_conv * si_c
    g = jnp.clip(D_c / D0_c, 1e-6, None)
    p0 = jnp.clip(jnp.exp(-a), 1e-6, 1.0 - 1e-6)
    pk = jnp.clip(jnp.exp(-a / g), 1e-6, 1.0 - 1e-6)
    surv = jnp.where(hh, pk / p0, (1.0 - pk) / (1.0 - p0))
    per = jnp.where(hh, surv * branch, surv)
    return jnp.prod(jnp.where(valid, per, 1.0), axis=1)


def fsi_nucleon_reweight(srec, s_el, s_inel):
    """GRANULAR per-candidate nucleon FSI reweight.  srec = (hh (n,K) hit, a_nom (n,K)=pi b^2/(sigma_tot fm^2),
    iso (n,K) {0 pp,1 pn,2 nn}, finel (n,K)=sigma_in/sigma_tot, inel (n,K) realized-inelastic, ns (n,)).
    s_el,s_inel are length-3 (per-iso) scales.  At each candidate the total-sigma scale is
      g = s_el[iso]*(1-finel) + s_inel[iso]*finel    (sigma_tot(s)=g*sigma_tot_0),
    so a_nom(s)=a_nom/g; the Gaussian hit factor is exp(-a_nom/g)/exp(-a_nom) for a hit (else the no-hit
    complement), and a HIT carries the el/inel sub-branch factor s_realized/g (s_realized=s_inel[iso] if
    inel else s_el[iso]).  ==1 at all-nominal; reduces BIT-EXACTLY to nucleon_scat_reweight(.,sscat) when
    s_el=s_inel=sscat (g=sscat, s_realized/g=1)."""
    hh, a_nom, iso, finel, inel, ns = srec
    se = jnp.asarray(s_el); si = jnp.asarray(s_inel)
    valid = jnp.arange(a_nom.shape[1])[None, :] < ns[:, None]
    se_i = se[iso]; si_i = si[iso]                                # per-candidate per-iso scales
    g = jnp.clip(se_i * (1.0 - finel) + si_i * finel, 1e-6, None)
    p0 = jnp.clip(jnp.exp(-a_nom), 1e-6, 1.0 - 1e-6)
    pk = jnp.clip(jnp.exp(-a_nom / g), 1e-6, 1.0 - 1e-6)
    s_real = jnp.where(inel, si_i, se_i)
    hit_f = (pk / p0) * (s_real / g)                             # hit: Gaussian ratio x el/inel sub-branch
    br = jnp.where(valid, jnp.where(hh, hit_f, (1.0 - pk) / (1.0 - p0)), 1.0)
    return jnp.prod(br, axis=1)


def fsi_nncex_reweight(srec, f_cex):
    """NN-elastic charge-exchange FRACTION reweight (ACHILLES value 0.5).  srec = (hh, iso, inel, swap (n,K),
    ns (n,)).  Only pn elastic hits (iso==1, hh & ~inel) carry a meaningful swap; per such hit the realized
    branch likelihood ratio is f_cex/0.5 if swapped else (1-f_cex)/0.5.  ==1 at f_cex=0.5.  pp/nn swaps are
    no-ops (same species) -> not reweighted."""
    hh, iso, inel, swap, ns = srec
    valid = jnp.arange(iso.shape[1])[None, :] < ns[:, None]
    pn_el = hh & (~inel) & (iso == 1)
    br = jnp.where(valid & pn_el, jnp.where(swap, f_cex / 0.5, (1.0 - f_cex) / 0.5), 1.0)
    return jnp.prod(br, axis=1)


def nucleon_scat_reweight(srec, sscat):
    """Kind-1 sigma_scatter reweight from compressed nucleon-walk records srec =
    (hit (n,K), a_nom (n,K), n_slab (n,)) with a_nom = pi b^2/(sigma fm^2) of the closest
    in-slab nucleon at each candidate step.  Pure in sscat; bit-exact equal to the
    in-propagation w_scat."""
    hh, a_nom, ns = srec
    valid = jnp.arange(a_nom.shape[1])[None, :] < ns[:, None]
    p_nom = jnp.clip(jnp.exp(-a_nom), 1e-6, 1.0 - 1e-6)
    p_knb = jnp.clip(jnp.exp(-a_nom / sscat), 1e-6, 1.0 - 1e-6)
    br = jnp.where(valid, jnp.where(hh, p_knb / p_nom, (1.0 - p_knb) / (1.0 - p_nom)), 1.0)
    return jnp.prod(br, axis=1)


@dataclass(frozen=True)
class DiscreteCascadeConfig:
    nucleus: str = "c12_density.txt"       # proton density file (data/nuclear/)
    density_n: str = "c12_density.txt"     # neutron density file (= nucleus for N=Z nuclei, e.g. C)
    configs: str = "QMC_configs.out.gz"    # nucleon configuration file (QMC/RMF; A read from header)
    step: float = 0.05
    max_steps: int = 100000  # ABSOLUTE per-particle/per-event step ceiling (ACHILLES cMaxSteps = 100000).
                             # NOT a physics knob: the cascade terminates via escape/capture/absorption/
                             # path_budget_R far below this.  Reaching it = a runaway particle, which the
                             # pool RAISES on (run_cascade_pool _raise_if_runaway) rather than silently
                             # truncating.  Keep == cascade_full._HARD_STEPS.
    path_budget_R: float = 3.0  # PHYSICS termination: drop a particle once its accumulated path length
                             # exceeds path_budget_R * nuclear_radius (lpath >= R_budget).  Scheme-
                             # independent (distance- AND time-sync), no beta literal: a slow track just
                             # takes more steps to cover the same distance.  Tweak to widen/tighten reach.
    seed: int = 0
    time_step: bool = False  # stepping clock: False = fixed-DISTANCE march (every particle advances
                             # `step` fm/global-step -> distance-synchronized).  True = fixed-TIME march
                             # mirroring ACHILLES AdaptiveStep: `step` is read as Dt and every particle
                             # advances `beta*step` fm/global-step (its own beta), so fast tracks outrun
                             # slow ones -> time-synchronized.  Decoupled (own beta, no max_beta) so
                             # P-invariance holds; fz then decrements by the shared Dt (=step), keeping
                             # the fz-expiry distance beta*fz invariant.  See cascade_subcascade_sequencing.md.
    # interaction probability is ALWAYS the Gaussian model exp(-pi b^2/sigma) -- the single,
    # differentiable interaction-probability law (the non-differentiable Cylinder hard-disk and the
    # unreliable ACHILLES "Pion" model were removed in the pool-unification cleanup).
    fast_xsec: bool = True   # evaluate Oset/DCC cross sections only for the K nearest in-slab nucleons
                             # (scatter back into the (n,A) grid); bit-exact, ~4x cheaper per step.
    pauli: bool = True       # Pauli-block the outgoing nucleon(s) of scatter/absorption (ACHILLES
                             # FinalizeMomentum); set False for ablation (no-blocking) studies.
    algo: str = "step"       # "step": fixed-step Glauber march (reference).  "interaction": jump
                             # directly to the next interaction (same probability model, ~20x fewer
                             # iterations).  Statistically equivalent; validated against "step".
    nn_inelastic: bool = True  # NN -> N Delta -> N N pi in the NUCLEON cascade (ACHILLES
                             # NucleonNucleon GiBUU ResonanceMode: Decay; adonis/fsi/nn_inelastic).
                             # Degrades fast nucleons and CREATES a pion (meson-veto relevant).
                             # False = the previous elastic-only walk (bit-exact).
    early_exit: bool = True  # while_loop walk that stops once NO particle can interact again
                             # (dead, or outside the radius moving outward = inert).  BIT-EXACT in
                             # every returned output (same per-step keys; skipped steps are
                             # identity), gated in tests/test_cascade_early_exit.py.  max_steps
                             # then acts as a pure safety bound.  False = the reference lax.scan.
    track_steps: bool = False  # MC-truth: also stack the per-step (pos, p4, alive) trajectory from the
                             # scan -> returns traj for viz/diagnostics (adonis/fsi/tracking).  Forces the
                             # reference scan (not early_exit).  DEFAULT OFF -> production path unchanged.
    engine: str = "pool"     # cascade structure: "pool" = single per-step-reconciled particle stack
                             # (DEFAULT, validated single core); "bfs" = legacy generation-synchronized; true step-order
                             # consumption; see docs/logbook/cascade_pool_engine.md).  WIP behind switch.
    recap_ke: float = 10.0   # escaping nucleon with KE < recap_ke [MeV] is RECAPTURED (set to rest).
                             # FAITHFUL to ACHILLES Cascade::Escaped (src/Achilles/Cascade.cc, called
                             # every step, UNGATED by PotentialProp): `constexpr double potential = 10.0;
                             # energy = E - mN - potential; if(|pos|>radius){ KE<10 -> captured }`.  This
                             # 10 MeV optical-potential capture is SEPARATE from (and always on, unlike)
                             # the PotentialProp:True Hamiltonian capture.  Mirrors ACHILLES's hard-coded
                             # 10.0 -> keep in sync with it; set 0.0 only for no-capture ablations.


def sample_nucleons(key, n, cfg: DiscreteCascadeConfig):
    """Pick n configurations ~ weight; assign each nucleon an isotropic local-Fermi-gas momentum.
    Returns npos (n,A,3), nmom (n,A,4), nisp (n,A) proton-mask."""
    pos, iso, w, A = _load_qmc_configs(name=cfg.configs)
    rgrid, rhoP, rhoN, _ = _load_density(cfg.nucleus, cfg.density_n)
    kc, kd, km = jax.random.split(key, 3)
    idx = jax.random.choice(kc, pos.shape[0], (n,), p=w)
    npos = pos[idx]; nisp = iso[idx]
    r = jnp.linalg.norm(npos, axis=2)
    # ACHILLES Local FG: PER-SPECIES k_F -- protons from rho_p, neutrons from rho_n (Nucleus.cc:212-238).
    # For N=Z (carbon) rho_p == rho_n bitwise -> unchanged.
    kf = _kf_local(jnp.where(nisp, _rho_species(r, rgrid, rhoP), _rho_species(r, rgrid, rhoN)))
    d = jax.random.normal(kd, (n, A, 3)); d = d / jnp.linalg.norm(d, axis=2, keepdims=True)
    pm = kf * jax.random.uniform(km, (n, A)) ** (1.0 / 3.0)
    m_sp = jnp.where(nisp, _MP_PHYS, _MN_PHYS)               # PHYSICAL per-species (ACHILLES Nucleus.cc:145)
    p3 = d * pm[:, :, None]; E = jnp.sqrt(m_sp ** 2 + pm ** 2)
    nmom = jnp.concatenate([E[:, :, None], p3], axis=2)
    return npos, nmom, nisp


# --- per-event RNG (persistent-refill engine) -------------------------------------------------------
# The step physics is keyed PER EVENT so a refilled event draws the SAME randoms regardless of which
# working-set slot / global step processes it (docs/logbook/cascade_persistent_refill_plan.md).  `key`
# into _nucleon_step/_pion_step is therefore an (n,2) array (one PRNG key per event), not a shared (2,)
# key.  These helpers vmap the per-event draws; distributions are unchanged (each event gets an
# independent stream), only WHICH draws each event sees differs from the old shared-key scheme.
def _ev_split(keys, k):
    """Per-event split: keys (n,2) -> (n,k,2)."""
    return jax.vmap(lambda key: jax.random.split(key, k))(keys)


def _ev_uniform(keys, shape=()):
    """Per-event uniform: keys (n,2) -> (n,*shape)."""
    return jax.vmap(lambda key: jax.random.uniform(key, shape))(keys)


def _ev_fold_uniform(keys, data, shape=()):
    """Per-event fold_in(data) then uniform: keys (n,2) -> (n,*shape)."""
    return jax.vmap(lambda key: jax.random.uniform(jax.random.fold_in(key, data), shape))(keys)


def _nucleon_step(p4, pos, dhat, fz, isp, alive, npos, nmom, nisp, consumed,
                  rgrid, rhoP, rhoN, radius, cfg, key, dt_evt=None):
    """ONE step of the NUCLEON cascade for one particle per event (n,) -- the per-step physics of
    `_propagate_nucleon_discrete.body` (escape/recapture, formation zone, in-slab geometry, elastic
    scatter + per-species Pauli, NN->NDelta->NN'pi inelastic + channel charges), re-expressed for the
    POOLED engine: the knockout (recoil | inelastic 2nd nucleon) and the created pion are returned as
    IMMEDIATE spawns (no best_ko top-K deferral), and the updated `consumed` mask is returned for
    slot-serialized depletion.  RNG usage matches body exactly (split(key,3) + fold_in(sk,101..108)),
    so iterating this with the same per-step keys reproduces the bfs leading trajectory bit-for-bit.
    Returns: (p4', pos', dhat', fz', alive'), terminal, recap, do, (ko4,kopos,kofz,koisp,koal),
             (pi4,pipos,pifz,pich,pial), consumed', (has_hit, perp2_c, sig_c)."""
    n, A = nisp.shape; ar = jnp.arange(n)
    outward = jnp.sum(pos * dhat, axis=1) > 0
    escaping = (jnp.linalg.norm(pos, axis=1) > radius) & outward
    # ACHILLES Cascade::Escaped (every step, ungated): captured if E - mN_avg - 10 < 0.  CRITICAL mass
    # convention: E uses the PHYSICAL per-species mass (the escaping particle's 4-vec; neutron E with
    # mn=939.565), while the subtracted threshold uses the AVERAGE mN (Constant::mN=938.919).  ADoNIS
    # nucleon p4[:,0] carries avg M_N, so recompute E from |p| with the physical species mass; using avg
    # for E too over-captured neutrons in |p| in (132.9, 137.4) MeV -> the <137 MeV first-bin deficit.
    _e_phys = jnp.sqrt(jnp.where(isp, _MP_PHYS, _MN_PHYS) ** 2 + jnp.sum(p4[:, 1:] ** 2, axis=1))
    recap = escaping & ((_e_phys - M_N) < cfg.recap_ke)
    p4 = jnp.where(recap[:, None], jnp.array([M_N, 0.0, 0.0, 0.0]), p4)
    alive = alive & ~escaping
    beta = jnp.linalg.norm(p4[:, 1:], axis=1) / jnp.clip(p4[:, 0], 1e-9, None)
    # STEPPING CLOCK: distance-sync -> every particle sweeps `step` fm/step, fz -= step/beta (proper time).
    # time-sync (ACHILLES AdaptiveStep) -> the per-event time step dt_evt = step/beta_max (beta_max over the
    # event's alive particles, supplied by the pool) sets the clock: sweep beta*dt_evt fm/step (fast tracks
    # outrun slow ones; a particle ALONE has beta_max=beta -> sweeps `step`, never freezes), fz -= dt_evt.
    # Both schemes keep the fz-expiry distance = beta*fz invariant.
    if cfg.time_step:
        _dt = jnp.full_like(beta, cfg.step) if dt_evt is None else dt_evt
        _dstep = beta * _dt                           # distance swept this step = beta * Dt
        timeStep = _dt                                # fz decrement = shared per-event Dt
    else:
        _dstep = jnp.full_like(beta, cfg.step)        # fixed distance
        timeStep = cfg.step / jnp.clip(beta, 1e-6, None)
    can_int = fz <= 0.0
    rel = npos - pos[:, None, :]
    par = jnp.sum(rel * dhat[:, None, :], axis=2)
    perp2 = jnp.sum(rel ** 2, axis=2) - par ** 2
    in_slab = (par > 0) & (par <= _dstep[:, None]) & (~consumed) & alive[:, None]
    Pp = p4[:, None, :] + nmom
    s = Pp[:, :, 0] ** 2 - jnp.sum(Pp[:, :, 1:] ** 2, axis=2)
    sqrts = jnp.sqrt(jnp.clip(s, (2 * M_N) ** 2, None))
    same_iso = isp[:, None] == nisp
    # ACHILLES NNElastic.cc:184 uses the PER-PAIR average physical mass (mp for pp, mn for nn, avg for
    # pn) in threshold/plab + the low-plab mn/threshold terms -- not the global average.
    _m_pair_gev = 0.5 * (jnp.where(isp, _MP_PHYS, _MN_PHYS)[:, None]
                         + jnp.where(nisp, _MP_PHYS, _MN_PHYS)) / 1000.0
    sig_el = jnp.clip(nn_elastic_sigma(sqrts, same_iso, _m_pair_gev), 0.0, None)
    if cfg.nn_inelastic:
        pcm = jnp.sqrt(jnp.clip(s / 4.0 - M_N ** 2, 1e-6, None)) / 1000.0
        sig_in = jnp.clip(nni.sigma_nn_ndelta(sqrts / 1000.0, pcm, same_iso), 0.0, None)
    else:
        sig_in = jnp.zeros_like(sig_el)
    sig = sig_el + sig_in
    # Gaussian interaction probability (the single, differentiable model)
    prob = jnp.where(in_slab, jnp.exp(-jnp.pi * perp2 / jnp.clip(sig * MB_TO_FM2, 1e-12, None)), 0.0)
    _ks3 = _ev_split(key, 3); sk, ku, ks = _ks3[:, 0], _ks3[:, 1], _ks3[:, 2]   # per-event keys (n,2)
    passes = in_slab & (_ev_uniform(ku, (A,)) < prob)
    big = jnp.where(passes, perp2, jnp.inf)
    j = jnp.argmin(big, axis=1)
    has_hit = jnp.isfinite(big[ar, j]) & alive & can_int
    perp2_is = jnp.where(in_slab, perp2, jnp.inf); cidx = jnp.argmin(perp2_is, axis=1)
    has_slab = jnp.any(in_slab, axis=1)
    perp2_c = jnp.where(has_slab, perp2_is[ar, cidx], 1e6); sig_c = sig[ar, cidx]
    # GRANULAR nucleon-FSI record: pair-isospin of the closest in-slab candidate {0 pp,1 pn,2 nn} and the
    # inelastic fraction finel = sig_in/sig_tot there -> lets fsi_nucleon_reweight scale per-iso elastic and
    # inelastic NN sigma independently (s_NN_elastic{pp,pn,nn}, s_NN_inelastic{pp,pn,nn}).
    _nisp_c = nisp[ar, cidx]
    iso_c = jnp.where(isp & _nisp_c, 0, jnp.where((~isp) & (~_nisp_c), 2, 1)).astype(jnp.int32)
    finel_c = jnp.clip(sig_in[ar, cidx] / jnp.clip(sig_c, 1e-12, None), 0.0, 1.0)
    pN_j = nmom[ar, j]
    rnuc = jnp.linalg.norm(npos, axis=2)
    kf_n = _kf_local(jnp.where(nisp, _rho_species(rnuc, rgrid, rhoP), _rho_species(rnuc, rgrid, rhoN)))
    kf_j = kf_n[ar, j]                                            # recoil (struck nucleon) k_F at the struck vertex
    _rnuc_j = rnuc[ar, j]
    kf_p_j = _kf_local(_rho_species(_rnuc_j, rgrid, rhoP))        # (charge-exchange recoil uses the struck vertex)
    kf_n_j = _kf_local(_rho_species(_rnuc_j, rgrid, rhoN))
    # LEADING outgoing Pauli k_F: evaluate at the LEADING's OWN position (|pos|), NOT the struck nucleon's
    # position -- ACHILLES PauliBlocking(paOut) uses k_F at the outgoing's own position.  For large-sigma
    # (slow-proton, near-threshold) scatters the struck nucleon sits up to ~the impact parameter (~2 fm)
    # away at a different density, so using its k_F leaked sub-k_F leading outgoing -> slow-proton
    # over-interaction (5sigma at 125-250 MeV vs ACHILLES).
    _r_lead = jnp.linalg.norm(pos, axis=1)
    _kfp_l = _kf_local(_rho_species(_r_lead, rgrid, rhoP)); _kfn_l = _kf_local(_rho_species(_r_lead, rgrid, rhoN))

    # ACHILLES NN ELASTIC CHARGE EXCHANGE: the two outgoing are {id1,id2} OR the id-SWAPPED {id2,id1}, each
    # at HALF the elastic sigma (NucleonNucleon.cc:62-65).  a-role = p_out (continues in the slot),
    # b-role = recoil; the swap exchanges their isospin (-> mass, Pauli k_F species, charge).  No-op for
    # pp/nn (isp == struck).  Per-event coin (fold 109), independent of the other channel draws.
    _swap_cx = _ev_fold_uniform(sk, 109) < 0.5
    _struck_isp = nisp[ar, j]
    lead_isp_out = jnp.where(_swap_cx, _struck_isp, isp)         # continuing nucleon's OUTGOING isospin
    rec_isp_out = jnp.where(_swap_cx, isp, _struck_isp)          # recoil's OUTGOING isospin
    kf_lead = jnp.where(lead_isp_out, _kfp_l, _kfn_l)            # leading Pauli k_F (its own position, swapped species)
    kf_rec = jnp.where(rec_isp_out, kf_p_j, kf_n_j)             # recoil Pauli k_F (struck vertex, swapped species)
    # 2->2 final-state masses PHYSICAL per-species (ACHILLES GenerateMomentum ma/mb), swap-aware.
    m_lead_phys = jnp.where(lead_isp_out, _MP_PHYS, _MN_PHYS)
    m_struck_phys = jnp.where(rec_isp_out, _MP_PHYS, _MN_PHYS)

    def scat_one(p_lead, pN_i, kf_out, kf_rec_, m1, m2, k):
        p_o = _two_body_cm_scatter(p_lead, pN_i, m1, k, m_recoil=m2)
        p_r = (p_lead + pN_i) - p_o
        return p_o, (jnp.linalg.norm(p_o[1:]) < kf_out) | (jnp.linalg.norm(p_r[1:]) < kf_rec_)
    p_out, blocked = jax.vmap(scat_one)(p4, pN_j, kf_lead, kf_rec, m_lead_phys, m_struck_phys, ks)
    if not cfg.pauli:
        blocked = blocked & False
    sig_in_j = sig_in[ar, j]; sig_el_j = sig_el[ar, j]
    u_br = _ev_fold_uniform(sk, 101)
    chose_inel = has_hit & (u_br < sig_in_j / jnp.clip(sig_el_j + sig_in_j, 1e-12, None))
    Pj = p4 + pN_j
    rs_j = jnp.sqrt(jnp.clip(Pj[:, 0] ** 2 - jnp.sum(Pj[:, 1:] ** 2, axis=1), (2 * M_N) ** 2, None))
    u_m = _ev_fold_uniform(sk, 102)
    m_d = jnp.clip(nni.sample_delta_mass(rs_j / 1000.0, u_m) * 1000.0, M_N + 135.0, rs_j - M_N - 1.0)
    cth1 = 2 * _ev_fold_uniform(sk, 103) - 1.0
    phi1 = 2 * jnp.pi * _ev_fold_uniform(sk, 104)
    cth2 = 2 * _ev_fold_uniform(sk, 105) - 1.0
    phi2 = 2 * jnp.pi * _ev_fold_uniform(sk, 106)
    # Channel charges computed BEFORE the splits so the NN->N Delta-> N N pi products carry PHYSICAL
    # per-species/charge masses (ACHILLES decays the Delta via DecayHandler -> ParticleInfo masses).
    # Fold keys 107/108 are order-independent, so this move is bit-neutral vs the draw sequence.
    q_pair = isp.astype(jnp.int32) + nisp[ar, j].astype(jnp.int32)
    u107 = _ev_fold_uniform(sk, 107)
    u108 = _ev_fold_uniform(sk, 108)
    dch = jnp.where(q_pair == 2, jnp.where(u107 < 0.75, 2, 1),
            jnp.where(q_pair == 1, jnp.where(u107 < 0.5, 1, 0),
                                   jnp.where(u107 < 0.25, 0, -1)))
    pi_q = jnp.where(dch == 2, 1,
            jnp.where(dch == 1, jnp.where(u108 < 1.0 / 3.0, 1, 0),
            jnp.where(dch == 0, jnp.where(u108 < 2.0 / 3.0, 0, -1), -1)))
    _mN1 = jnp.where((q_pair - dch) == 1, _MP_PHYS, _MN_PHYS)   # nucleon recoiling against the Delta
    _mN2 = jnp.where((dch - pi_q) == 1, _MP_PHYS, _MN_PHYS)     # nucleon from the Delta decay
    _mpi_dec = _CH_MASS[(1 - pi_q)]                             # pion from the Delta decay (per-charge phys)

    def _split2(P4, mA, mB, cth_, phi_):
        ss = jnp.clip(P4[:, 0] ** 2 - jnp.sum(P4[:, 1:] ** 2, axis=1), (mA + mB) ** 2 * 1.0001, None)
        rss = jnp.sqrt(ss)
        EA = (ss + mA ** 2 - mB ** 2) / (2 * rss)
        pf = jnp.sqrt(jnp.clip(EA ** 2 - mA ** 2, 0.0, None))
        sth_ = jnp.sqrt(jnp.clip(1 - cth_ ** 2, 0, None))
        d_ = jnp.stack([sth_ * jnp.cos(phi_), sth_ * jnp.sin(phi_), cth_], axis=1)
        pa = jnp.concatenate([EA[:, None], pf[:, None] * d_], axis=1)
        pb = jnp.concatenate([(rss - EA)[:, None], -pf[:, None] * d_], axis=1)
        beta_ = P4[:, 1:] / P4[:, [0]]
        b2_ = jnp.sum(beta_ ** 2, axis=1); g_ = 1 / jnp.sqrt(jnp.clip(1 - b2_, 1e-12, None))
        def lab(p4_):
            bp_ = jnp.sum(beta_ * p4_[:, 1:], axis=1)
            E = g_ * (p4_[:, 0] + bp_)
            p3 = p4_[:, 1:] + ((g_ - 1) * bp_ / jnp.clip(b2_, 1e-30, None) + g_ * p4_[:, 0])[:, None] * beta_
            return jnp.concatenate([E[:, None], p3], axis=1)
        return lab(pa), lab(pb)

    pN1, pD = _split2(Pj, _mN1, m_d, cth1, phi1)
    pN2, _pPiX = _split2(pD, _mN2, _mpi_dec, cth2, phi2)
    # Pauli-block the two inelastic outgoing nucleons per species; the LEADING one (faster -> continues
    # from |pos|) is blocked at the LEADING's position (like the elastic kf_lead), the other (knockout at
    # the struck vertex) at the struck k_F -- same position fix as the elastic channel.
    _kfp_lead = _kf_local(_rho_species(_r_lead, rgrid, rhoP)); _kfn_lead = _kf_local(_rho_species(_r_lead, rgrid, rhoN))
    nl_is1 = jnp.linalg.norm(pN1[:, 1:], axis=1) >= jnp.linalg.norm(pN2[:, 1:], axis=1)   # pN1 is leading
    _n1p = (q_pair - dch) == 1; _n2p = (dch - pi_q) == 1
    kf_N1 = jnp.where(nl_is1, jnp.where(_n1p, _kfp_lead, _kfn_lead), jnp.where(_n1p, kf_p_j, kf_n_j))
    kf_N2 = jnp.where(~nl_is1, jnp.where(_n2p, _kfp_lead, _kfn_lead), jnp.where(_n2p, kf_p_j, kf_n_j))
    in_blocked = ((jnp.linalg.norm(pN1[:, 1:], axis=1) < kf_N1)
                  | (jnp.linalg.norm(pN2[:, 1:], axis=1) < kf_N2))
    if not cfg.pauli:
        in_blocked = in_blocked & False
    is_inel = chose_inel & ~in_blocked
    lead_in = jnp.where(nl_is1[:, None], pN1, pN2)
    pi_chidx = (1 - pi_q).astype(jnp.int32)
    do = has_hit & ~chose_inel & ~blocked
    recoil = (p4 + pN_j) - p_out
    bg_proton = rec_isp_out                                      # elastic recoil isospin (charge-exchange aware)
    fz_new = _formation_zone(p4, p_out)
    nl_is1 = jnp.linalg.norm(pN1[:, 1:], axis=1) >= jnp.linalg.norm(pN2[:, 1:], axis=1)
    inel_nl = jnp.where(nl_is1[:, None], pN2, pN1)
    inel_nl_q = jnp.where(nl_is1, dch - pi_q, q_pair - dch)
    ko_cand = jnp.where(is_inel[:, None], inel_nl, recoil)
    ko_q = jnp.where(is_inel, inel_nl_q, bg_proton.astype(jnp.int32))
    # CONTINUING (leading) nucleon's new charge: elastic charge-exchange -> lead_isp_out; inelastic -> the
    # LEADING nucleon's channel charge (the partner of the knockout inel_nl_q); else unchanged.  Previously
    # the leading kept its incident charge always (no NN charge exchange + inelastic leading mis-charged).
    lead_inel_q = jnp.where(nl_is1, q_pair - dch, dch - pi_q)
    lead_q_new = jnp.where(do, lead_isp_out.astype(jnp.int32),
                           jnp.where(is_inel, lead_inel_q, isp.astype(jnp.int32))).astype(jnp.int32)
    ko_fz = jnp.where(is_inel, _formation_zone(p4, inel_nl), _formation_zone(p4, recoil))
    ko_alive = (do | is_inel) & (jnp.linalg.norm(ko_cand[:, 1:], axis=1) > 1.0)
    ko_pos = npos[ar, j]
    # created pion spawn (inelastic only)
    pi_alive = is_inel & (jnp.linalg.norm(_pPiX[:, 1:], axis=1) > 1.0)
    pi_fz = _formation_zone(p4, _pPiX)
    pi_pos = npos[ar, j]
    # leading update + consumed depletion + fz + advance
    p4 = jnp.where(do[:, None], p_out, jnp.where(is_inel[:, None], lead_in, p4))
    consumed = consumed | (jax.nn.one_hot(j, A, dtype=bool) & (do | is_inel)[:, None])
    fz = jnp.where((fz > 0.0) & alive, fz - timeStep, fz)
    fz = jnp.where(do, fz_new, jnp.where(is_inel, _formation_zone(p4, lead_in), fz))
    d3 = p4[:, 1:]; dhat = d3 / jnp.clip(jnp.linalg.norm(d3, axis=1, keepdims=True), 1e-9, None)
    pos = pos + _dstep[:, None] * dhat * alive[:, None]    # beta*step (time-sync) or step (distance-sync)
    return ((p4, pos, dhat, fz, alive, lead_q_new), escaping, recap, do.astype(jnp.int32),
            (ko_cand, ko_pos, ko_fz, ko_q, ko_alive),
            (_pPiX, pi_pos, pi_fz, pi_chidx, pi_alive), consumed,
            jax.lax.stop_gradient((has_hit, perp2_c, sig_c, iso_c, finel_c, is_inel, do & _swap_cx)))


def _pion_step(p4, pos, dhat, ch, nsc, alive, npos, nmom, nisp, consumed,
               rgrid, rhoP, rhoN, radius, cfg, key, dt_evt=None):
    """ONE step of the PION cascade for one pion per event (n,) -- a line-for-line extraction of
    `_propagate_discrete.body` (algo="step"), re-expressed for the POOLED engine: the absorption
    products (piNN->NN, up to 2 protons), the scatter recoil, and the eta-N' conversion baryon are
    returned as IMMEDIATE NUCLEON spawns (no best_abs/best_rec top-K deferral); the scattered pion
    continues in place (charge oscillates) and the absorbed/converted pion is removed.  RNG usage
    matches body exactly (split(sk,7) + fold_in(ka,211/212) + per-event splits), so iterating this with
    the same per-step keys reproduces the bfs leading PION trajectory (p_pi, ch, nsc, absorbed, conv,
    pos) bit-for-bit.  ch = pion charge INDEX (0=pi+,1=pi0,2=pi-).
    Returns: (p4', pos', dhat', ch', nsc', alive'), escaping, is_abs, is_conv,
             (s1_p4,s1_pos,s1_fz,s1_q,s1_al), (s2_p4,s2_pos,s2_fz,s2_q,s2_al), consumed', srec."""
    assert cfg.algo == "step", "pool pion step implements the 'step' algo (production path) only"
    n, A = nisp.shape; ar = jnp.arange(n)
    p_pi = p4
    # ----- escape (Cascade.cc:532-553): beam pion (nsc==0) -> z>=radius PLANE; scattered -> sphere -----
    # The POOL has no early-exit "inert" skip (unlike _propagate_discrete's while_loop), so a pion that
    # has LEFT the nucleus (|pos|>radius, moving outward -> rho=0 ahead, can never re-enter) must be
    # escaped explicitly here for ALL nsc; otherwise an un-scattered (nsc==0) pion leaving in any
    # direction but +z never triggers esc_plane/esc_sphere and idles to max_steps (BFS treats these as
    # inert-survived).  inert subsumes esc_sphere; esc_plane keeps the +z beam-transparency convention.
    ext = nsc == 0
    outward = jnp.sum(pos * dhat, axis=1) > 0
    esc_plane = ext & (pos[:, 2] >= radius)
    inert = (jnp.linalg.norm(pos, axis=1) > radius) & outward          # outside & outward -> will escape
    escaping = esc_plane | inert
    alive = alive & ~escaping
    # STEPPING CLOCK (see _nucleon_step): time-sync -> sweep beta*dt_evt (dt_evt=step/beta_max from the pool);
    # else fixed step.  Pions carry NO formation zone (ACHILLES skips IsPion in InFormationZone) -> slab+advance only.
    _beta_pi = jnp.linalg.norm(p_pi[:, 1:], axis=1) / jnp.clip(p_pi[:, 0], 1e-9, None)
    if cfg.time_step:
        _dt = jnp.full_like(_beta_pi, cfg.step) if dt_evt is None else dt_evt
        _dstep = _beta_pi * _dt
    else:
        _dstep = jnp.full_like(_beta_pi, cfg.step)
    rel = npos - pos[:, None, :]
    par = jnp.sum(rel * dhat[:, None, :], axis=2)
    perp2 = jnp.sum(rel ** 2, axis=2) - par ** 2
    cand = (par > 0) & (~consumed) & alive[:, None] & (par <= _dstep[:, None])
    pE = p_pi[:, 0]; pmom = jnp.linalg.norm(p_pi[:, 1:], axis=1); m_pi = _CH_MASS[ch]
    vpi = p_pi[:, 1:] / pE[:, None]
    rnuc = jnp.linalg.norm(npos, axis=2)
    kf_n = _kf_local(jnp.where(nisp, _rho_species(rnuc, rgrid, rhoP), _rho_species(rnuc, rgrid, rhoN)))

    def _xsec(nm, npo, nip):
        vN = nm[..., 1:] / nm[..., 0:1]
        vrel = jnp.clip(jnp.linalg.norm(vpi[:, None, :] - vN, axis=-1), 1e-3, None)
        rnpo = jnp.linalg.norm(npo, axis=-1)
        rho_t = _rho_species(rnpo, rgrid, rhoP) + _rho_species(rnpo, rgrid, rhoN)
        kf = _kf_local(jnp.where(nip, _rho_species(rnpo, rgrid, rhoP), _rho_species(rnpo, rgrid, rhoN)))
        Pp = p_pi[:, None, :] + nm
        Wl = jnp.sqrt(jnp.clip(Pp[..., 0] ** 2 - jnp.sum(Pp[..., 1:] ** 2, axis=-1), 1.0, None))
        sal = jnp.clip(ox.abs_cross_section(pE[:, None] + 0 * Wl, m_pi[:, None] + 0 * Wl,
                       pmom[:, None] + 0 * Wl, vrel, jnp.clip(kf, 1e-6, None),
                       jnp.clip(rho_t, 1e-9, None)), 0.0, None)
        K_ = Wl.shape[1]
        nuc_i = jnp.where(nip, 0, 1).astype(jnp.int32)
        sio = cascade_mb.jax_channel_sigmas_resolved(Wl.reshape(-1),
                  jnp.broadcast_to(ch[:, None], (n, K_)).reshape(-1),
                  nuc_i.reshape(-1)).reshape(n, K_, 3)
        ssl = jnp.clip(jnp.sum(sio, axis=-1), 0.0, None)
        sil = jnp.clip(cascade_mb.jax_conversion_sigma(Wl.reshape(-1),
                  jnp.broadcast_to(ch[:, None], (n, K_)).reshape(-1),
                  nuc_i.reshape(-1)).reshape(n, K_), 0.0, None)
        return sal, ssl, sil, sio, Wl

    if cfg.fast_xsec:
        score = jnp.where(cand, -perp2, -jnp.inf)
        _, idx = jax.lax.top_k(score, _KSLAB)
        gi = (ar[:, None], idx)
        sa_k, ss_k, si_k, sio_k, W_k = _xsec(nmom[ar[:, None], idx], npos[ar[:, None], idx], nisp[ar[:, None], idx])
        sa = jnp.zeros((n, A)).at[gi].set(sa_k); ss = jnp.zeros((n, A)).at[gi].set(ss_k)
        si = jnp.zeros((n, A)).at[gi].set(si_k); sig_io = jnp.zeros((n, A, 3)).at[gi].set(sio_k).reshape(n * A, 3)
        W = jnp.zeros((n, A)).at[gi].set(W_k)
    else:
        sa, ss, si, sio, W = _xsec(nmom, npos, nisp); sig_io = sio.reshape(n * A, 3)
    like_charge = ((ch == 0)[:, None] & nisp) | ((ch == 2)[:, None] & (~nisp))
    sa = sa * jnp.where(like_charge, 5.0 / 6.0, 1.0)
    sig = sa + ss + si
    _sfm = jnp.clip(sig * MB_TO_FM2, 1e-12, None)
    # Gaussian interaction probability (the single, differentiable model)
    prob = jnp.where(cand, jnp.exp(-jnp.pi * perp2 / _sfm), 0.0)
    _ks7 = _ev_split(key, 7)            # per-event keys (n,2) each
    sk, ku, kc, kf, ka, kab, knp = (_ks7[:, i] for i in range(7))
    passes = cand & (_ev_uniform(ku, (A,)) < prob)
    metric = jnp.where(passes, perp2, jnp.inf)
    j = jnp.argmin(metric, axis=1)
    has_hit = jnp.isfinite(metric[ar, j]) & alive
    sa_j = sa[ar, j]; si_j = si[ar, j]; sig_j = sig[ar, j]
    # CLOSEST in-slab candidate (mirrors _nucleon_step's perp2_c/sig_c).  The pion's INTERACTION
    # PROBABILITY is prob = exp(-pi*perp2/(sigma_tot*MB_TO_FM2)) -- it responds to a sigma scale, but the
    # per-hit branch record (sa_j..si_j, taken at the STRUCK candidate) cannot express that: under a common
    # rescale the branch LR is identically 1.  These are the sufficient statistics for the hit/no-hit
    # factor, recorded on EVERY in-slab step (hit or not), exactly as the nucleon record does.
    perp2_is = jnp.where(cand, perp2, jnp.inf)
    cidx = jnp.argmin(perp2_is, axis=1)
    has_slab = jnp.any(cand, axis=1)
    perp2_c = jnp.where(has_slab, perp2_is[ar, cidx], 1e6)
    sa_c = sa[ar, cidx]; si_c = si[ar, cidx]; sig_c = sig[ar, cidx]
    ss_c = sig_c - sa_c - si_c
    ss_el_c = sig_io.reshape(n, A, 3)[ar, cidx][ar, ch]          # elastic (out==in) part at the candidate
    W_j = W[ar, j]; pN_j = nmom[ar, j]; kf_j = kf_n[ar, j]
    kf_p_j = _kf_local(_rho_species(rnuc[ar, j], rgrid, rhoP))
    kf_n_j = _kf_local(_rho_species(rnuc[ar, j], rgrid, rhoN))
    p_abs = sa_j / jnp.clip(sig_j, 1e-12, None)
    p_conv = si_j / jnp.clip(sig_j, 1e-12, None)
    u_br = _ev_uniform(kc)
    chose_abs = has_hit & (u_br < p_abs)
    chose_conv = has_hit & ~chose_abs & (u_br < p_abs + p_conv)
    # ----- absorption final state (isospin partition; per-species Pauli) -----
    struck_p = nisp[ar, j].astype(jnp.int32)
    d2 = jnp.sum((npos - npos[ar, j][:, None, :]) ** 2, axis=2)
    self_used = (jnp.arange(A)[None, :] == j[:, None]) | consumed
    d2p = jnp.where(self_used | (~nisp), jnp.inf, d2)
    d2n = jnp.where(self_used | nisp, jnp.inf, d2)
    pj_p = jnp.argmin(d2p, axis=1); has_p = jnp.isfinite(d2p[ar, pj_p])
    pj_n = jnp.argmin(d2n, axis=1); has_n = jnp.isfinite(d2n[ar, pj_n])
    idx2 = ch * 2 + struck_p
    Wabs = jnp.asarray(_ABS_W_NP)[idx2]; PARTabs = jnp.asarray(_ABS_PART_NP)[idx2]
    avail = jnp.where(PARTabs == 1, has_p[:, None], jnp.where(PARTabs == 0, has_n[:, None], False))
    Wm = jnp.where(avail, Wabs, 0.0); wtot = jnp.sum(Wm, axis=1, keepdims=True)
    Wm = Wm / jnp.clip(wtot, 1e-12, None)
    u_np = _ev_uniform(knp)
    nprot_out = jnp.clip(jnp.sum((u_np[:, None] > jnp.cumsum(Wm, axis=1)).astype(jnp.int32), axis=1), 0, 2)
    has_mode = wtot[:, 0] > 0
    part_is_p = PARTabs[ar, nprot_out] == 1
    pN_p = jnp.where(part_is_p[:, None], nmom[ar, pj_p], nmom[ar, pj_n])
    pos_hit = pos
    rA = jnp.linalg.norm(pos_hit, axis=1); rB = jnp.linalg.norm(npos[ar, j], axis=1)
    kfPA = _kf_local(_rho_species(rA, rgrid, rhoP)); kfNA = _kf_local(_rho_species(rA, rgrid, rhoN))
    kfPB = _kf_local(_rho_species(rB, rgrid, rhoP)); kfNB = _kf_local(_rho_species(rB, rgrid, rhoN))

    def abs_one(p_pi_i, pNj_i, pNp_i, npr, kfpa, kfna, kfpb, kfnb, k):
        P = p_pi_i + pNj_i + pNp_i
        s = P[0] ** 2 - jnp.sum(P[1:] ** 2)
        sqrts = jnp.sqrt(jnp.clip(s, (2 * M_N) ** 2, None))
        Estar = sqrts / 2.0
        pstar = jnp.sqrt(jnp.clip(Estar ** 2 - M_N ** 2, 0.0, None))
        k1, k2, k3 = jax.random.split(k, 3)
        cth = 2.0 * jax.random.uniform(k1) - 1.0
        sth = jnp.sqrt(jnp.clip(1 - cth ** 2, 0.0, None)); phi = 2 * jnp.pi * jax.random.uniform(k2)
        dirn = jnp.array([sth * jnp.cos(phi), sth * jnp.sin(phi), cth])
        beta = P[1:] / P[0]
        pa = _boost(jnp.concatenate([Estar[None], pstar * dirn]), beta)
        pb = _boost(jnp.concatenate([Estar[None], -pstar * dirn]), beta)
        ma = jnp.linalg.norm(pa[1:]); mb = jnp.linalg.norm(pb[1:])
        a_is_p = jax.random.uniform(k3) < 0.5
        A_is_p = (npr >= 2) | ((npr == 1) & a_is_p)
        B_is_p = (npr >= 2) | ((npr == 1) & (~a_is_p))
        kfA = jnp.where(A_is_p, kfpa, kfna); kfB = jnp.where(B_is_p, kfpb, kfnb)
        blocked = (ma < kfA) | (mb < kfB)
        # ACHILLES re-cascades BOTH absorption nucleons (Cascade.cc particles_out[0],[1]) regardless of
        # charge: a neutron product can knock out a proton downstream.  Emit both full 4-vecs + their
        # true charges, NOT proton-only slots (zeroing neutrons dropped secondary proton knockouts).
        return pa, pb, A_is_p.astype(jnp.int32), B_is_p.astype(jnp.int32), blocked
    abs_pa, abs_pb, abs_qa, abs_qb, abs_blocked = jax.vmap(abs_one)(p_pi, pN_j, pN_p, nprot_out,
                                                         kfPA, kfNA, kfPB, kfNB, kab)   # kab (n,2) per-event
    if not cfg.pauli:
        abs_blocked = abs_blocked & False
    is_abs = chose_abs & ~abs_blocked & has_mode
    # ----- scatter: out-pion charge, DCC angle, per-species Pauli recoil -----
    sig_io_j = sig_io.reshape(n, A, 3)[ar, j]
    probs = sig_io_j / jnp.clip(jnp.sum(sig_io_j, axis=1, keepdims=True), 1e-12, None)
    u = _ev_uniform(kf, (1,))
    out_ch = jnp.clip(jnp.sum((u > jnp.cumsum(probs, axis=1)).astype(jnp.int32), axis=1), 0, 2).astype(jnp.int32)
    nuc_idx = jnp.where(nisp[ar, j], 0, 1)
    chan_idx = ch * 6 + nuc_idx * 3 + out_ch
    cos_cm = cascade_mb.jax_sample_cos_cm(W_j, _ev_uniform(ka), chan_idx)
    _rec_is_p = (struck_p + out_ch - ch) == 1                  # recoil nucleon charge (charge-exchange)
    kf_rec_pi = jnp.where(_rec_is_p, kf_p_j, kf_n_j)
    m_rec_pi = jnp.where(_rec_is_p, _MP_PHYS, _MN_PHYS)        # recoil nucleon mass PHYSICAL per-species

    def scat_one(p_pi_i, pN_i, out_i, kf_i, mrec, cc, k):
        p_out = _two_body_cm_scatter(p_pi_i, pN_i, _CH_MASS[out_i], k, cos_cm=cc, m_recoil=mrec)
        p_rec = (p_pi_i + pN_i) - p_out
        return p_out, jnp.linalg.norm(p_rec[1:]) < kf_i
    p_out, blocked = jax.vmap(scat_one)(p_pi, pN_j, out_ch, kf_rec_pi, m_rec_pi, cos_cm, sk)   # sk (n,2) per-event
    if not cfg.pauli:
        blocked = blocked & False
    # ----- conversion (piN -> etaN'): emit the N' baryon (eta neutral -> q_bary = q_pi + q_struck) -----
    is_conv = chose_conv
    q_bary = (1 - ch) + struck_p
    eta_ok = is_conv & ((q_bary == 0) | (q_bary == 1))
    Pcv = p_pi + pN_j
    scv = Pcv[:, 0] ** 2 - jnp.sum(Pcv[:, 1:] ** 2, axis=1)
    rscv = jnp.sqrt(jnp.clip(scv, (M_N + _M_ETA) ** 2, None))
    EN = (scv + M_N ** 2 - _M_ETA ** 2) / (2.0 * rscv)
    pst = jnp.sqrt(jnp.clip(EN ** 2 - M_N ** 2, 0.0, None))
    ccv = 2.0 * _ev_fold_uniform(ka, 211) - 1.0
    scv_ = jnp.sqrt(jnp.clip(1 - ccv ** 2, 0.0, None)); phcv = 2 * jnp.pi * _ev_fold_uniform(ka, 212)
    dcv = jnp.stack([scv_ * jnp.cos(phcv), scv_ * jnp.sin(phcv), ccv], axis=1)
    beta = Pcv[:, 1:] / Pcv[:, [0]]; b2 = jnp.sum(beta ** 2, axis=1); gcv = 1 / jnp.sqrt(jnp.clip(1 - b2, 1e-12, None))
    Ncm = jnp.concatenate([EN[:, None], pst[:, None] * dcv], axis=1)
    bpcv = jnp.sum(beta * Ncm[:, 1:], axis=1)
    p3cv = Ncm[:, 1:] + ((gcv - 1) * bpcv / jnp.clip(b2, 1e-30, None) + gcv * Ncm[:, 0])[:, None] * beta
    p_bary = jnp.concatenate([(gcv * (Ncm[:, 0] + bpcv))[:, None], p3cv], axis=1)
    is_scat = has_hit & ~chose_abs & ~chose_conv & ~blocked
    p_rec = (p_pi + pN_j) - p_out
    q_rec = struck_p + out_ch - ch
    rcand = jnp.where(is_conv[:, None], p_bary, p_rec)
    q_rcand = jnp.where(is_conv, q_bary, q_rec)
    fz_rec = _formation_zone(p_pi, rcand)
    # ----- spawns: slot1 = abs nucleon A | scatter recoil | conv baryon ; slot2 = abs nucleon B -----
    # Both absorption nucleons (A,B) carry their TRUE charge (abs_qa/abs_qb) and full 4-vec; neutron
    # products re-cascade (ACHILLES particles_out[0],[1]) instead of being dropped.
    s1_p4 = jnp.where(is_abs[:, None], abs_pa, jnp.where(is_scat[:, None], p_rec, jnp.where(eta_ok[:, None], p_bary, 0.0)))
    s1_q = jnp.where(is_abs, abs_qa, jnp.where(is_scat, q_rec, jnp.where(eta_ok, q_bary, 0))).astype(jnp.int32)
    s1_pos = jnp.where(is_abs[:, None], pos_hit, npos[ar, j])
    s1_fz = jnp.where(is_abs, _formation_zone(p_pi, abs_pa), jnp.where(is_scat | eta_ok, fz_rec, 0.0))
    s1_al = (jnp.linalg.norm(s1_p4[:, 1:], axis=1) > 1.0) & (is_abs | is_scat | eta_ok)
    s2_p4 = jnp.where(is_abs[:, None], abs_pb, 0.0)
    s2_q = jnp.where(is_abs, abs_qb, 0).astype(jnp.int32)
    s2_pos = pos_hit
    s2_fz = _formation_zone(p_pi, abs_pb)
    s2_al = (jnp.linalg.norm(s2_p4[:, 1:], axis=1) > 1.0) & is_abs
    # ----- pion state update (scatter continues; abs/conv removed) -----
    p_pi = jnp.where(is_scat[:, None], p_out, p_pi)
    ch = jnp.where(is_scat, out_ch, ch)
    nsc = nsc + is_scat.astype(jnp.int32)
    alive = alive & ~is_abs & ~is_conv
    interacted = is_abs | is_scat | is_conv
    consumed = consumed | (jax.nn.one_hot(j, A, dtype=bool) & interacted[:, None])
    d3 = p_pi[:, 1:]; dhat = d3 / jnp.clip(jnp.linalg.norm(d3, axis=1, keepdims=True), 1e-9, None)
    pos = pos + _dstep[:, None] * dhat * alive[:, None]    # beta*step (time-sync) or step (distance-sync)
    ss_j = sig_j - sa_j - si_j
    ss_el_j = sig_io_j[ar, ch]                                  # elastic (out==in) scatter sigma at the hit
    # GRANULAR kind-1 channel code {0 elastic, 1 charge-exchange, 2 absorption, 3 conversion} (was the
    # 3-code {0 scatter,1 abs,2 conv}).  Scatter splits into elastic/cex by the sampled out-pion charge
    # (out_ch==ch -> elastic).  ss_el recorded alongside ss (total scatter) -> ss_cex = ss - ss_el; this
    # lets fsi_pion_reweight scale s_piN_elastic / s_piN_cex / s_pi_abs / s_conv independently.
    code4 = jnp.where(chose_abs, 2, jnp.where(chose_conv, 3,
                      jnp.where(out_ch == ch, 0, 1))).astype(jnp.int32)
    return ((p_pi, pos, dhat, ch, nsc, alive), escaping, is_abs, is_conv,
            (s1_p4, s1_pos, s1_fz, s1_q, s1_al), (s2_p4, s2_pos, s2_fz, s2_q, s2_al), consumed,
            jax.lax.stop_gradient((has_hit, code4, sa_j, ss_el_j, ss_j, si_j,
                                   perp2_c, sa_c, ss_el_c, ss_c, si_c)))

