"""Isotropic two-body decay in the CM, boosted to the lab -- the shared core of the qe/ee/qe_nc samplers.

qe.py (nu N -> mu N'), ee.py (e N -> e' N') and qe_nc.py (nu N -> nu N') each MC-sample an isotropic
two-body final state the SAME way: form s from the total lab 4-momentum, get the CM momentum pcm from the
Kallen function, draw (cos theta*, phi) isotropically, and boost back.  This is that computation once, in
numpy (the samplers are numpy), using adonis.kinematics.boost for the CM->lab step -- so a fix to the
kinematics lands in one place, and the future NC-QE nuclear sampler builds on the same primitive.
"""
import numpy as np

from adonis.kinematics import boost as _boost

_TWO_PI = 2 * np.pi


def isotropic_two_body_cm(P, m1, m2, u_cos, u_phi):
    """Decay a system of total LAB 4-momentum P (n,4) into back-to-back bodies of mass (m1, m2).

    u_cos, u_phi in [0,1) parametrise body 1's (cos theta*, phi) isotropically in the CM.  Returns
    (k1, k2, pcm, sqrts, s, lam): k1 (mass m1) and k2 (mass m2) as lab 4-vectors, the CM momentum pcm,
    sqrt(s), the invariant s, and the Kallen lambda.  The caller forms the two-body phase-space factor
    (2*pi*pcm/(sqrts*16 pi^2)) and the validity mask (s > (m1+m2)^2, lam > 0) -- kept OUT of here because
    each sampler ANDs it with its own channel-specific cuts.
    """
    P = np.asarray(P)
    s = P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, axis=1)
    sqrts = np.sqrt(np.clip(s, 1e-9, None))
    s2, s3 = m1 ** 2, m2 ** 2
    E1 = sqrts / 2 * (1 + s2 / s - s3 / s); E2 = sqrts / 2 * (1 + s3 / s - s2 / s)
    lam = np.sqrt(np.clip((s - s2 - s3) ** 2 - 4 * s2 * s3, 0, None)); pcm = lam / (2 * sqrts)
    cts = 2 * np.asarray(u_cos) - 1; sts = np.sqrt(np.clip(1 - cts ** 2, 0, None)); php = _TWO_PI * np.asarray(u_phi)
    dirn = np.stack([sts * np.cos(php), sts * np.sin(php), cts], axis=1)
    beta = P[:, 1:] / P[:, 0:1]
    k1 = np.asarray(_boost(np.concatenate([E1[:, None], pcm[:, None] * dirn], axis=1), beta))
    k2 = np.asarray(_boost(np.concatenate([E2[:, None], -pcm[:, None] * dirn], axis=1), beta))
    return k1, k2, pcm, sqrts, s, lam
