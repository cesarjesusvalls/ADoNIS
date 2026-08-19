"""A stochastic detector: smear every particle's momentum and direction, on-shell.

The section-5 unfolding study needs a RECONSTRUCTED sample to unfold.  This builds it the way an
experiment does -- particle by particle, before any selection -- so that the reco selection, the reco
observables, efficiency and purity all fall out of the SAME selection code applied to smeared inputs.
Nothing about the selection or the observables is reimplemented for reco; that is the whole point:

    B_reco = smear_chunk(B, spec, chunk=i)
    true_sel, true_obs, w0, _ = select_full(B,      sd)      # truth   phase space
    reco_sel, reco_obs, _,  _ = select_full(B_reco, sd)      # reco    phase space

NO GRADIENT SURVIVES THIS, by design.  Smearing is a fixed draw, not a parameter: it decides which reco
bin an event lands in, and the fit differentiates only the WEIGHTS attached to those fixed indices.  So
the smeared arrays are plain numpy and never enter the jax path.

DETERMINISTIC, and deterministic PER CHUNK.  The seed is (spec.seed, chunk index), never a running
generator, so a chunk smears identically whether it is processed first, last, or alone in a shard -- the
detector cannot depend on how the job was sharded.  That also means the reco sample needs no storage:
it is a pure function of the bank plus an integer.

What is smeared, and what is not:
  * |p| -> |p| * (1 + sigma_p * g), g ~ N(0,1), floored just above zero;
  * direction -> rotated by a polar angle |N(0, sigma_th)| about a uniformly random azimuth, i.e. a cone
    about the true direction, which smears theta and phi together.  phi matters as much as theta here:
    delta-alpha_T and delta-p_T are transverse-plane quantities.
  * E is REBUILT from the smeared |p| and the particle's own invariant mass (m^2 = E^2 - |p|^2 from the
    true 4-vector), so every smeared particle stays exactly on shell and no PID table is needed.
  * PID and charge are NOT smeared -- the meson veto in the CC0pi definition is a topology statement, so
    this detector has perfect particle identification and imperfect kinematics.  A mis-ID model is a
    separate effect and would confound the unfolding result it is meant to demonstrate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEG = np.pi / 180.0


@dataclass(frozen=True)
class SmearSpec:
    """The detector.  Defaults are the section-5 working point: 20% on momentum, 10 degrees on angle."""
    sigma_p: float = 0.20            # fractional momentum resolution
    sigma_theta_deg: float = 10.0    # 3-D angular resolution (cone half-angle, 1 sigma)
    seed: int = 20260813
    # PION DETECTION EFFICIENCY.  Below `pi_eff_p_max` MeV/c a charged pion is SEEN with probability
    # `pi_eff`; otherwise it is missed and the event is reconstructed as if it were not there.  This is
    # the one effect that creates a real CC0pi BACKGROUND: a true CC1pi event whose only pion is missed
    # passes the CC0pi reco selection.  With perfect PID (the defaults) a topological CC0pi sample has
    # purity exactly 1.0 -- truth and reco selections coincide event for event -- so the cross-section
    # knobs, which reweight only the background, have nothing to act on and their systematic is
    # identically zero.  That is not a small mis-estimate; it is a missing term.
    pi_eff_p_max: float = 0.0        # MeV/c; 0 disables (perfect PID, the previous behaviour)
    pi_eff: float = 1.0              # probability of SEEING a pion below pi_eff_p_max

    @classmethod
    def parse(cls, d):
        d = dict(d or {})
        extra = set(d) - {"sigma_p", "sigma_theta_deg", "seed", "pi_eff_p_max", "pi_eff"}
        if extra:
            raise ValueError(f"detector: unknown key(s) {sorted(extra)}")
        return cls(**d)


def _rotate_into(dirs, cos_delta, sin_delta, phi):
    """Rotate each unit vector in `dirs` by polar angle delta about a random azimuth phi.

    Built from an orthonormal frame (u, v, dirs) per particle.  The frame is constructed against
    whichever axis the direction is LEAST aligned with, so `u` never degenerates to a zero vector for
    particles travelling along a coordinate axis -- forward-going muons sit exactly there, and a naive
    cross-with-z would leave them unsmeared.
    """
    n = dirs.shape[0]
    ref = np.zeros((n, 3))
    ref[np.arange(n), np.argmin(np.abs(dirs), axis=1)] = 1.0     # least-aligned axis
    u = np.cross(dirs, ref)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(dirs, u)                                        # already unit: |d|=|u|=1, d _|_ u
    perp = np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v
    return cos_delta[:, None] * dirs + sin_delta[:, None] * perp


def _smear_p4(p4, rng, spec):
    """Smear an (N,4) array of (E, px, py, pz), returning a new array.  Zero-momentum rows pass through:
    they have no direction to rotate, and dividing by |p| would produce NaN that then propagates silently
    into every observable built from them."""
    p4 = np.asarray(p4, dtype=np.float64)
    if p4.size == 0:
        return p4.copy()
    p = p4[:, 1:]
    pmag = np.linalg.norm(p, axis=1)
    ok = pmag > 0
    m2 = np.maximum(p4[:, 0] ** 2 - pmag ** 2, 0.0)              # each particle's own mass, kept exactly

    n = p4.shape[0]
    scale = 1.0 + spec.sigma_p * rng.standard_normal(n)
    pmag_r = np.maximum(pmag * scale, 1e-6)                      # a negative |p| is not a measurement
    delta = np.abs(rng.standard_normal(n)) * spec.sigma_theta_deg * DEG
    phi = rng.uniform(0.0, 2.0 * np.pi, n)

    out = p4.copy()
    if ok.any():
        dirs = p[ok] / pmag[ok][:, None]
        dirs_r = _rotate_into(dirs, np.cos(delta[ok]), np.sin(delta[ok]), phi[ok])
        out[ok, 1:] = dirs_r * pmag_r[ok][:, None]
        out[ok, 0] = np.sqrt(m2[ok] + pmag_r[ok] ** 2)
    return out


def smear_chunk(B, spec: SmearSpec, chunk: int):
    """A reconstructed view of bank chunk `B`.

    Returns a dict that SHARES every array with `B` except `k_lep` and `fs_p4`, which are smeared copies.
    Sharing is deliberate: the ragged offsets, PIDs, charges and weights are identical by construction, so
    copying them would only invite the two views to drift apart.  Callers must not mutate the result.
    """
    rng = np.random.default_rng([spec.seed, int(chunk)])
    out = dict(B)
    out["k_lep"] = _smear_p4(B["k_lep"], rng, spec)
    out["fs_p4"] = _smear_p4(B["fs_p4"], rng, spec)

    # MISSED PIONS.  A pion that is not seen must be invisible to the PID accounting, which is what
    # makes the event reconstruct as CC0pi.  The final state is RAGGED -- flat arrays plus per-event
    # offsets -- so entries cannot be deleted without rebuilding the offsets of every downstream
    # consumer; setting the PID to 0 is equivalent and local, because the meson veto is
    # np.isin(fs_pid, MESONS) and 0 is in no list.  fs_pid is COPIED here: smear_chunk otherwise shares
    # every array but k_lep and fs_p4 with the truth bank, and mutating it in place would silently
    # change the TRUTH selection too, which is the one thing this must not do.
    #
    # The threshold is on the TRUE momentum: whether a detector can see a particle is a property of the
    # particle, not of the number the reconstruction happened to draw for it.
    #
    # The draw happens AFTER both _smear_p4 calls and only when the effect is enabled, so the kinematic
    # smearing is bit-identical to the perfect-PID case and the two can be compared directly.
    if spec.pi_eff_p_max > 0.0 and spec.pi_eff < 1.0:
        pid = np.asarray(B["fs_pid"]).copy()
        ptrue = np.linalg.norm(np.asarray(B["fs_p4"], dtype=np.float64)[:, 1:], axis=1)
        cand = (np.abs(pid) == 211) & (ptrue < spec.pi_eff_p_max)
        missed = cand & (rng.random(pid.shape[0]) >= spec.pi_eff)
        pid[missed] = 0
        out["fs_pid"] = pid
    return out
