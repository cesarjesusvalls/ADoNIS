"""Full differentiable final state for CC single-pion production.

The angle-integrated fold (`dcc_fold_full`) returns only (W, Q2, weight): the pion solid
angle is integrated inside the hadron tensor.  This fold UN-INTEGRATES that angle -- it
samples the pion CM direction (cos theta*, phi*), evaluates the DIFFERENTIAL hadron
current at that angle (`angular_diff`), and builds the full lab-frame final state
(`final_state`): outgoing lepton, pion, recoil nucleon.

Estimator (kind-1 reweighting, exactly as the rest of the project): every continuous
quantity is sampled from a FIXED detached proposal; only the elementary weight carries
the knobs, so gradients are exact.  The pion angle is sampled uniformly over the solid
angle, so by construction

    E_Omega[ 4*pi * sum_c mult_c * L.w_diff_c(Omega) ] = sum_c mult_c * L.W_int_c
                                                       = the dcc_fold_full hadronic factor,

i.e. the differential fold reproduces dcc_fold_full's dsigma/dW, dsigma/dQ2 in expectation
(validated in validate_final_state.py).

Per event:
  1. sample lepton (E', theta), nucleon (p, E_rm ~ S(p,E)), pion (cos theta*, phi*), lep azimuth;
  2. off-shell struck nucleon E = mqe - E_rm; true q; W = (q+p_struck)^2;
  3. on-shell rebalanced Q2_adj (current_init) for the amplitude/cut;
  4. cuts W in [1076.957,2000], Q2_adj in [0,5 GeV^2];
  5. amplitude(W, Q2_adj) -> per-channel zmtx -> differential current at (theta*,phi*)
     -> L_{mu nu} W^{mu nu}_diff;
  6. weight = (E'/E) sin(theta) (k_pi/W) (4 pi) sum_c mult_c L.w_diff_c, zeroed outside cuts;
  7. two-body piN decay -> lab pion + nucleon; uniform lepton azimuth.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from adonis.primary.dcc.amplitudes import DCCKnobs
from adonis.nuclear.spectral import load_spectral, SpectralSampler
from adonis.primary.dcc.structure import HadronStructure
from adonis.primary.dcc.lepton import cm_lepton_momenta, lepton_tensor_cc, lepton_tensor_em, contract
from adonis.primary.dcc.form_factors import axial_reweight_dipole
from adonis.constants import MQE, M_PI, W_THR, W_MAX, Q2_MAX
from adonis.primary.dcc.differential import (precompute_diff_coeffs, angular_factor,
                           build_zmtx_batched, differential_current, differential_tensor)
from adonis.primary.dcc.final_state import cm_basis, two_body_lab, rotate_about_z
from adonis.primary.dcc.fold_integrated import pion_cm_momentum
from adonis.core.event import EventRecord

E_NU_DEFAULT = 1500.0

# PDG ids per CC channel (matches hadron_xsec.CC_CHANNELS order)
_PID_NI = np.array([2112, 2112, 2212])     # initial nucleon
_PID_N = np.array([2212, 2112, 2212])      # final nucleon
_PID_PI = np.array([111, 211, 211])        # final pion


def _diff_coeffs(hs: HadronStructure):
    """Per-channel angle-independent differential coefficients, cached on the hs object."""
    if getattr(hs, "_diff_pre", None) is None:
        hs._diff_pre = [precompute_diff_coeffs(hs.twoJ, hs.twoL, hs.twoI, tcrz=c.tcrz,
                                               tiz=c.tiz, tpinz=c.tpinz, tpiz=c.tpiz)
                        for c in hs.channels]
    return hs._diff_pre


def sample_final_state(key, n=200000, hs: HadronStructure | None = None,
                       sf="pke12p_tot.data", e_nu=E_NU_DEFAULT, ep_lo=50.0, ep_hi=1480.0,
                       theta_max_deg=180.0, m_pi=M_PI, m_N=MQE, nuclear=None, m_lep=0.0,
                       current="CC", theta_min_deg=0.0, weight_ep_volume=False):
    """Draw the FIXED detached proposal (all kinematics + the precomputed angular kernels,
    lepton tensor, cuts, phase-space factors, lab final-state momenta).  Nothing here
    depends on the physics knobs -- so `weight_from_sample(knobs, S)` can be re-evaluated
    (and differentiated) cheaply over many knob values without re-sampling.  This split is
    the kind-1 reweighting estimator made explicit (and is what fitting will reuse).

    `nuclear` is a swappable NuclearModel (default: spectral function `sf`); it supplies the
    struck-nucleon (p_vec, E_removal) from the same RNG key, so the default path is
    bit-identical to before."""
    hs = hs or HadronStructure()
    if nuclear is None:
        from adonis.nuclear.spectral import SpectralFunction
        nuclear = SpectralFunction(sf)
    pre = _diff_coeffs(hs)
    sg = jax.lax.stop_gradient

    klep, kth, ksf, kpi_a, kpi_p, kphi, kch = jax.random.split(key, 7)
    Ep = sg(ep_lo + (ep_hi - ep_lo) * jax.random.uniform(klep, (n,)))
    th_lo = jnp.deg2rad(theta_min_deg)
    theta = sg(th_lo + (jnp.deg2rad(theta_max_deg) - th_lo) * jax.random.uniform(kth, (n,)))
    cos_ts = sg(2.0 * jax.random.uniform(kpi_a, (n,)) - 1.0)        # pion cos(theta*)
    phi_s = sg(2.0 * jnp.pi * jax.random.uniform(kpi_p, (n,)))       # pion phi*
    phi_lep = sg(2.0 * jnp.pi * jax.random.uniform(kphi, (n,)))      # lepton azimuth

    omega = e_nu - Ep
    # outgoing-lepton 3-momentum magnitude: |p'| = sqrt(E'^2 - m_lep^2). m_lep=0 reproduces
    # the massless path bit-for-bit; m_lep>0 (e.g. the muon) shifts q, W, Q2 self-consistently.
    # The CC lepton tensor form is unchanged: the (1∓γ5) projectors kill the m_lep terms, so
    # the mass enters ONLY through this kinematic magnitude.
    plep = jnp.sqrt(jnp.clip(Ep ** 2 - m_lep ** 2, 0.0, None))
    qx = -plep * jnp.sin(theta)
    qz = e_nu - plep * jnp.cos(theta)
    q_vec2 = qx ** 2 + qz ** 2
    e_nu_a = jnp.broadcast_to(jnp.asarray(e_nu, float), (n,))     # scalar OR per-event flux
    k_lab = jnp.stack([e_nu_a, jnp.zeros((n,)), jnp.zeros((n,)), e_nu_a], axis=-1)
    kp_lab = jnp.stack([Ep, plep * jnp.sin(theta), jnp.zeros((n,)), plep * jnp.cos(theta)], axis=-1)

    p_vec, E_rm = nuclear.sample_nucleon(ksf, n)
    p2 = jnp.sum(p_vec ** 2, axis=1)
    p_struck = jnp.concatenate([(MQE - E_rm)[:, None], p_vec], axis=1)

    tot = jnp.stack([omega, qx, jnp.zeros((n,)), qz], axis=-1) + p_struck
    W = sg(jnp.sqrt(jnp.clip(tot[:, 0] ** 2 - jnp.sum(tot[:, 1:] ** 2, axis=1), 1.0, None)))
    T_N = jnp.sqrt(p2 + MQE ** 2) - MQE
    Q2_adj = sg(q_vec2 - (omega - E_rm - T_N) ** 2)
    cut = (W > W_THR) & (W < W_MAX) & (Q2_adj > 0.0) & (Q2_adj < Q2_MAX) & (Ep >= m_lep)

    k_cm, kp_cm = cm_lepton_momenta(k_lab, kp_lab, p_struck)
    is_em = (current == "EM")
    Lmn = sg(lepton_tensor_em(k_cm, kp_cm) if is_em else lepton_tensor_cc(k_cm, kp_cm))
    e1, e2, e3, P = cm_basis(k_lab, kp_lab, p_struck)
    # EM photon propagator 1/Q^4 (true leptonic Q^2); CC W-propagator is ~constant -> 1.
    Q2_lep = q_vec2 - omega ** 2
    em_prop = (1.0 / jnp.clip(Q2_lep, 1.0, None) ** 2) if is_em else 1.0

    # per-channel angular kernels at the sampled pion angle (detached numpy -> jnp)
    theta_pi = np.arccos(np.asarray(cos_ts)); phi_np = np.asarray(phi_s)
    Kfac = [jnp.asarray(angular_factor(theta_pi, phi_np, pre[c])) for c in range(len(hs.channels))]

    # lab final state (knob-independent) + leptonic/phase-space prefactor.
    # The leptonic factor is |p'|/E_nu (the standard |k'|/|k| flux/phase-space ratio), NOT
    # E'/E_nu: for a massless lepton |p'|=E' so this is unchanged, but for the muon the
    # momentum magnitude `plep` (not the energy Ep) is what enters d^3p'/(2E').
    # KINEMATIC pion mass (mpi0 to match ACHILLES) for the final-state on-shell decay; the
    # amplitude-internal m_pi (=fpio 138.04, passed in as `m_pi`) stays in S["m_pi"] for build_zmtx.
    from adonis.primary.dcc.conventions import kin_m_pi, M_PIP
    m_pi_kin = kin_m_pi(M_PIP)
    p_pi, p_N = two_body_lab(P, e1, e2, e3, jnp.clip(W, 1.0, None), cos_ts, phi_s, m_pi_kin, m_N)
    prefac = (plep / e_nu) * jnp.sin(theta) * (pion_cm_momentum(W) / jnp.clip(W, 1.0, None)) * em_prop
    # flux-folding: when ep_hi is per-event (= E_nu), the lepton-energy proposal volume
    # (ep_hi - ep_lo) varies per event and must enter the weight (it is a constant absorbed by
    # the bridge for the fixed-energy default, so guarded off there).
    if weight_ep_volume:
        prefac = prefac * (jnp.asarray(ep_hi, float) - ep_lo)
    return dict(
        hs=hs, n=n, m_pi=m_pi, m_N=m_N, kch=kch,
        Wc=jnp.clip(W, 1.0, None), Q2c=jnp.clip(Q2_adj, 1.0, None), W=W, Q2_adj=Q2_adj,
        cut=cut, prefac=sg(prefac), Lmn=Lmn, Kfac=Kfac,
        mult=jnp.asarray([c.mult for c in hs.channels]),
        k_lab=sg(k_lab), kp_lab=sg(rotate_about_z(kp_lab, phi_lep)),
        p_pi=sg(rotate_about_z(p_pi, phi_lep)), p_N=sg(rotate_about_z(p_N, phi_lep)),
        p_struck=sg(rotate_about_z(p_struck, phi_lep)))


def weight_from_sample(knobs: DCCKnobs, S, use_spline=True):
    """Knob-dependent per-event weight (N,) and per-channel L.W (N,n_ch) from a fixed
    sample S (sample_final_state).  Differentiable in `knobs`; cheap to re-evaluate.
    use_spline=False uses the faster bilinear amplitude interp (for closures/fits)."""
    hs = S["hs"]
    amp_fn = hs.amp.amplitudes_spline if use_spline else hs.amp.amplitudes_bilinear
    vec, isv, axial = amp_fn(S["Wc"], S["Q2c"], knobs)
    r_ax = axial_reweight_dipole(S["Q2c"], knobs.axial_MA)
    LWc = []
    for c, ch in enumerate(hs.channels):
        zmtx = build_zmtx_batched(vec, isv, axial, S["Wc"], S["Q2c"], hs.twoJ, hs.twoL,
                                  hs.twoI, mode=ch.mode, itiz=ch.itiz, m_N=S["m_N"],
                                  m_pi=S["m_pi"], r_axial=r_ax)
        zj = differential_current(zmtx, S["Kfac"][c], hs._diff_pre[c]["ixi1_map"])
        LWc.append(contract(S["Lmn"], differential_tensor(zj)))
    LWc = jnp.stack(LWc, axis=-1)
    LW = jnp.sum(S["mult"] * LWc, axis=-1)
    w = jnp.where(S["cut"], S["prefac"] * (4.0 * jnp.pi) * LW, 0.0)
    return w, LWc


def assemble_event(S, w, LWc):
    """Build the EventRecord from a sample S, its weight w and per-channel L.W: sample the
    per-event channel identity (detached, prop. to mult x max(L.W,0)) and attach PDGs."""
    sg = jax.lax.stop_gradient
    probs = sg(S["mult"] * jnp.clip(LWc, 0.0, None))
    psum = jnp.sum(probs, axis=-1, keepdims=True)
    cdf = jnp.cumsum(jnp.where(psum > 0, probs / psum, 1.0 / probs.shape[-1]), axis=-1)
    u = jax.random.uniform(S["kch"], (S["n"], 1))
    chan = jnp.clip(sg(jnp.sum((u > cdf).astype(jnp.int32), axis=-1)), 0, probs.shape[-1] - 1)
    return EventRecord(
        k=S["k_lab"], kp=S["kp_lab"], p_struck=S["p_struck"], p_pi=S["p_pi"], p_N=S["p_N"],
        w=w, channel=chan, pid_pi=jnp.asarray(_PID_PI)[chan], pid_N=jnp.asarray(_PID_N)[chan],
        pid_Ni=jnp.asarray(_PID_NI)[chan], W=S["W"], Q2_adj=S["Q2_adj"])


def fold_final_state(knobs: DCCKnobs, key, n=200000, hs: HadronStructure | None = None,
                     sf="pke12p_tot.data", e_nu=E_NU_DEFAULT, ep_lo=50.0, ep_hi=1480.0,
                     theta_max_deg=180.0, m_pi=M_PI, m_N=MQE):
    """Generate `n` CC single-pion events with a full differentiable lab-frame final state.
    Returns an EventRecord (weight differentiable in knobs; kinematics detached)."""
    S = sample_final_state(key, n, hs, sf, e_nu, ep_lo, ep_hi, theta_max_deg, m_pi, m_N)
    w, LWc = weight_from_sample(knobs, S)
    return assemble_event(S, w, LWc)


# --------------------------------------------------------------------------- #
#  DCCSinglePion -- the Channel object (sample / weight / event_record contract)
# --------------------------------------------------------------------------- #
from adonis.core.process import Channel              # noqa: E402
from adonis.params import GenConfig                  # noqa: E402
from adonis.flux.mono import Monochromatic           # noqa: E402
from adonis.nuclear.spectral import SpectralFunction  # noqa: E402


class DCCSinglePion(Channel):
    """CC single-pion production (ANL-Osaka DCC).  Holds the static config + swappable
    flux / nuclear model; exposes the sample/reweight contract."""

    def __init__(self, cfg: GenConfig = GenConfig(), hs: HadronStructure | None = None,
                 flux=None, nuclear=None):
        self.cfg = cfg
        self.hs = hs or HadronStructure(n_theta=cfg.n_theta, n_phi=cfg.n_phi, spline=cfg.spline)
        self.flux = flux or Monochromatic(cfg.e_nu)
        self.nuclear = nuclear or SpectralFunction(cfg.sf)

    def sample(self, key, n):
        c = self.cfg
        return sample_final_state(key, n, hs=self.hs, sf=c.sf, e_nu=self.flux.e_nu_nominal,
                                  ep_lo=c.ep_lo, ep_hi=c.ep_hi, theta_max_deg=c.theta_max_deg,
                                  m_pi=c.m_pi, m_N=c.m_N, nuclear=self.nuclear, m_lep=c.m_lep)

    def weight(self, params, sample):
        return weight_from_sample(params, sample, use_spline=self.cfg.spline)

    def event_record(self, params, sample):
        w, LWc = self.weight(params, sample)
        return assemble_event(sample, w, LWc)

    # -- per-module self-tests ------------------------------------------------- #
    def closure_test(self, key=None, n=20_000, tol=1e-3, eps=2e-3, **kw):
        """Standalone differentiability: d(total xsec)/dM_A, autodiff vs central FD.

        The proposal is sampled ONCE (detached) and reweighted at M_A +/- eps, so
        this is the exact kind-1 gradient gate -- they must agree to ~`tol`."""
        from adonis.core.validation import grad_closure
        key = jax.random.PRNGKey(0) if key is None else key
        S = self.sample(key, n)

        def total_xsec(MA):
            w, _ = self.weight(DCCKnobs(axial_MA=MA), S)
            return jnp.sum(w)

        return grad_closure(total_xsec, f"{type(self).__name__}.closure",
                            x0=1.0, eps=eps, tol=tol)

    def oracle_test(self, oracle, key=None, n=200_000, chunk=50_000,
                    observables=("cos_theta_star", "ppi_mag", "Q2", "lepton_costheta"),
                    gate=("cos_theta_star", "ppi_mag", "Q2"), group=None,
                    chi2_ndf_max=3.0, params=None, **kw):
        """Physics validity vs the ACHILLES oracle: per-observable chi2/ndf of the
        normalised model final state against `oracle` (path or loaded npz dict).

        Model events are generated in chunks and histogrammed on an EXACT integer
        coarsening of the oracle's fine edges (avoids rebinning aliasing).  Passes
        when every observable in `gate` has chi2/ndf <= `chi2_ndf_max`.  `W` is
        intentionally not gated by default (a known near-threshold residual)."""
        from adonis.core.validation import TestResult, chi2_ndf
        from adonis import observables as obs
        od = np.load(oracle) if isinstance(oracle, (str, bytes)) else oracle
        key = jax.random.PRNGKey(5000) if key is None else key
        params = DCCKnobs() if params is None else params
        group = group or {"W": 6, "Q2": 5, "cos_theta_star": 5, "ppi_mag": 5,
                          "phi_star": 2, "lepton_energy": 5, "lepton_costheta": 5,
                          "nucleon_mom": 6}

        names = [o for o in observables if f"edges_{o}" in od]
        edges, o_sw, o_sw2 = {}, {}, {}
        for nm in names:
            G = group.get(nm, 1)
            fe = od[f"edges_{nm}"]
            assert (len(fe) - 1) % G == 0, f"{nm}: fine bins not divisible by {G}"
            edges[nm] = fe[::G]
            o_sw[nm] = od[f"sw_{nm}"].reshape(-1, G).sum(1)
            o_sw2[nm] = od[f"sw2_{nm}"].reshape(-1, G).sum(1)

        m_sw = {nm: np.zeros(len(edges[nm]) - 1) for nm in names}
        m_sw2 = {nm: np.zeros(len(edges[nm]) - 1) for nm in names}
        done = 0
        ci = 0
        while done < n:
            nc = min(chunk, n - done)
            ev = self.event_record(params, self.sample(jax.random.fold_in(key, ci), nc))
            w = np.asarray(ev.w)
            for nm in names:
                v = np.asarray(obs.OBSERVABLES[nm](ev))
                m_sw[nm] += np.histogram(v, bins=edges[nm], weights=w)[0]
                m_sw2[nm] += np.histogram(v, bins=edges[nm], weights=w ** 2)[0]
            done += nc; ci += 1

        metrics = {}
        for nm in names:
            chi2, ndf = chi2_ndf(m_sw[nm], np.sqrt(m_sw2[nm]), o_sw[nm], np.sqrt(o_sw2[nm]))
            metrics[nm] = {"chi2": chi2, "ndf": ndf, "chi2_ndf": chi2 / max(ndf, 1)}
        gated = [nm for nm in gate if nm in metrics]
        passed = all(metrics[nm]["chi2_ndf"] <= chi2_ndf_max for nm in gated)
        detail = "  ".join(f"{nm} {metrics[nm]['chi2_ndf']:.2f}" for nm in names)
        return TestResult(f"{type(self).__name__}.oracle", "oracle", bool(passed),
                          False, f"chi2/ndf  {detail}  (gate<={chi2_ndf_max})", metrics)
