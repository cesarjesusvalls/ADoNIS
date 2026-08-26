"""A stochastic detector: smear every particle's momentum and direction, on-shell.

Builds a reconstructed sample the way an experiment does -- particle by particle, before any selection --
so reco selection, observables, efficiency and purity all fall out of the same selection code applied to
smeared inputs:

    B_reco = smear_chunk(B, spec, chunk=i)
    true_sel, true_obs, w0, _ = select_full(B,      sd)      # truth phase space
    reco_sel, reco_obs, _,  _ = select_full(B_reco, sd)      # reco  phase space

Smearing is a fixed draw, not a parameter: it decides which reco bin an event lands in, and the fit
differentiates only the weights attached to those fixed indices, so the smeared arrays are plain numpy
and never enter the jax path. Deterministic per chunk: the seed is (spec.seed, chunk index), never a
running generator, so the reco sample needs no storage -- it is a pure function of the bank plus an
integer, independent of processing order.

What is smeared:
  * |p| -> |p| * (1 + sigma_p * g), g ~ N(0,1), floored just above zero.
  * direction -> rotated by a polar angle |N(0, sigma_th)| about a uniformly random azimuth (a cone
    about the true direction); phi matters as much as theta since delta-alpha_T and delta-p_T are
    transverse-plane quantities.
  * E is rebuilt from the smeared |p| and the particle's own invariant mass, so every particle stays
    exactly on shell.
  * PID and charge are not smeared: the meson veto in the CC0pi definition is a topology statement, so
    this detector has perfect particle ID and imperfect kinematics only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEG = np.pi / 180.0


@dataclass(frozen=True)
class SmearSpec:
    """The detector.  Nominal working point: 20% on momentum, 10 degrees on angle."""
    sigma_p: float = 0.20            # fractional momentum resolution
    sigma_theta_deg: float = 10.0    # 3-D angular resolution (cone half-angle, 1 sigma)
    seed: int = 20260813
    # Pion detection efficiency: below `pi_eff_p_max` MeV/c a pion is seen with probability `pi_eff`,
    # otherwise missed (the event reconstructs as if it weren't there) -- the one effect giving CC0pi a
    # real background. With perfect PID (the defaults) purity is exactly 1.0, and the cross-section
    # knobs, which reweight only the background, have nothing to act on.
    pi_eff_p_max: float = 0.0        # MeV/c; 0 disables the effect (perfect PID)
    pi_eff: float = 1.0              # probability of seeing a pion below pi_eff_p_max

    @classmethod
    def parse(cls, d):
        d = dict(d or {})
        extra = set(d) - {"sigma_p", "sigma_theta_deg", "seed", "pi_eff_p_max", "pi_eff"}
        if extra:
            raise ValueError(f"detector: unknown key(s) {sorted(extra)}")
        return cls(**d)


def _rotate_into(dirs, cos_delta, sin_delta, phi):
    """Rotate each unit vector in `dirs` by polar angle delta about a random azimuth phi.

    Built from an orthonormal frame (u, v, dirs) per particle, using whichever axis the direction is
    least aligned with, so `u` never degenerates for particles along a coordinate axis (e.g.
    forward-going muons).
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
    """Smear an (N,4) array of (E, px, py, pz), returning a new array.  Zero-momentum rows pass through
    unchanged: they have no direction to rotate, and dividing by |p| would produce NaN."""
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

    Shares every array with `B` except `k_lep` and `fs_p4` (smeared copies): offsets, PIDs, charges and
    weights are identical by construction, so copying them would only invite drift. Callers must not
    mutate the result.
    """
    rng = np.random.default_rng([spec.seed, int(chunk)])
    out = dict(B)
    out["k_lep"] = _smear_p4(B["k_lep"], rng, spec)
    out["fs_p4"] = _smear_p4(B["fs_p4"], rng, spec)

    # A missed pion must be invisible to PID accounting, so the event reconstructs as CC0pi. The final
    # state is ragged, so entries can't be deleted without rebuilding offsets; setting PID to 0 is
    # equivalent and local (the meson veto is np.isin(fs_pid, MESONS), and 0 matches no list). fs_pid is
    # copied, not shared, so this never touches the truth selection.
    #
    # The threshold is on true momentum, a property of the particle, and the draw happens after both
    # smearing calls, so the kinematics stay bit-identical to the perfect-PID case.
    if spec.pi_eff_p_max > 0.0 and spec.pi_eff < 1.0:
        pid = np.asarray(B["fs_pid"]).copy()
        ptrue = np.linalg.norm(np.asarray(B["fs_p4"], dtype=np.float64)[:, 1:], axis=1)
        cand = (np.abs(pid) == 211) & (ptrue < spec.pi_eff_p_max)
        missed = cand & (rng.random(pid.shape[0]) >= spec.pi_eff)
        pid[missed] = 0
        out["fs_pid"] = pid
    return out
