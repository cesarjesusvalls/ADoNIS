"""Ground-truth test of the isotropic 3-body sampler's Q^2 distribution.
At FIXED beam energy, struck proton at rest, amps2 IGNORED (pure phase space):
  (A) FLAT-DALITZ reference: uniform in (m12^2, m23^2) over the Dalitz region + uniform SO(3)
      orientation = exactly uniform dPhi3.  Each accepted event has equal weight.
  (B) ISOTROPIC sampler (res_xsec / adonis_free_res internals): two isotropic 2-body splits,
      weighted by J3 = 1/density.  dPhi3/dQ2 = histogram weighted by J3.
If (B)'s NORMALISED dPhi3/dQ2 != (A)'s, the isotropic sampler is biased in Q^2 (the suspected bug).
"""
import numpy as np

M_MU, M_P, M_PIP = 105.7, 938.27, 139.57018
ENU = 1000.0
k_nu = np.array([ENU, 0, 0, ENU]); p_st = np.array([M_P, 0., 0., 0.]); P = k_nu + p_st
s = P[0] ** 2 - P[1:] @ P[1:]; sqrts = np.sqrt(s)
TWO_PI = 2 * np.pi


def sqlam(s_, a, b):
    x = (s_ - a - b) ** 2 - 4 * a * b
    return np.where(x > 0, np.sqrt(np.clip(x, 0, None)) / s_, 0.0)


def boost_to_lab(p4cm):
    rsq = sqrts
    E = (P[0] * p4cm[:, 0] + p4cm[:, 1:] @ P[1:]) / rsq
    c1 = (p4cm[:, 0] + E) / (rsq + P[0])
    return np.concatenate([E[:, None], p4cm[:, 1:] + c1[:, None] * P[1:]], axis=1)


def rand_rot(n, rng):
    u = rng.random((n, 3))
    q = np.stack([np.sqrt(1 - u[:, 0]) * np.sin(TWO_PI * u[:, 1]),
                  np.sqrt(1 - u[:, 0]) * np.cos(TWO_PI * u[:, 1]),
                  np.sqrt(u[:, 0]) * np.sin(TWO_PI * u[:, 2]),
                  np.sqrt(u[:, 0]) * np.cos(TWO_PI * u[:, 2])], axis=1)
    w, x, y, z = q.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1).reshape(n, 3, 3)


def flat_dalitz(n, rng):
    m1, m2, m3 = M_MU, M_P, M_PIP                       # mu, N, pi
    s12 = rng.uniform((m1 + m2) ** 2, (sqrts - m3) ** 2, n)   # M^2(mu,N)
    s23 = rng.uniform((m2 + m3) ** 2, (sqrts - m1) ** 2, n)   # M^2(N,pi)
    E1 = (s + m1 ** 2 - s23) / (2 * sqrts)
    E3 = (s + m3 ** 2 - s12) / (2 * sqrts)
    E2 = sqrts - E1 - E3
    p1 = np.sqrt(np.clip(E1 ** 2 - m1 ** 2, 0, None))
    p2 = np.sqrt(np.clip(E2 ** 2 - m2 ** 2, 0, None))
    p3 = np.sqrt(np.clip(E3 ** 2 - m3 ** 2, 0, None))
    ct13 = (p2 ** 2 - p1 ** 2 - p3 ** 2) / (2 * p1 * p3)
    ok = (E2 >= m2) & (np.abs(ct13) <= 1) & (p1 > 0) & (p3 > 0)
    st13 = np.sqrt(np.clip(1 - ct13 ** 2, 0, None))
    v1 = np.stack([np.zeros(n), np.zeros(n), p1], axis=1)
    v3 = np.stack([p3 * st13, np.zeros(n), p3 * ct13], axis=1)
    v2 = -(v1 + v3)
    R = rand_rot(n, rng)
    v1 = np.einsum('nij,nj->ni', R, v1)                 # only need the muon (particle 1)
    kmu = np.concatenate([E1[:, None], v1], axis=1)
    kmu = boost_to_lab(kmu)
    q = k_nu - kmu; Q2 = (np.sum(q[:, 1:] ** 2, axis=1) - q[:, 0] ** 2) / 1e6
    return Q2[ok]


def isotropic(n, rng):
    u = rng.random((n, 5))
    mpi, mNf, mmu = M_PIP, M_P, M_MU
    s23max = (sqrts - mpi) ** 2; s23min = (mmu + mNf) ** 2
    s23 = s23min + (s23max - s23min) * u[:, 0]; rs23 = np.sqrt(np.clip(s23, 1e-9, None))
    EmuN = (s + s23 - mpi ** 2) / (2 * sqrts); pA = sqrts * sqlam(s, s23, mpi ** 2) / 2
    ctA = 2 * u[:, 1] - 1; stA = np.sqrt(np.clip(1 - ctA ** 2, 0, None)); phA = TWO_PI * u[:, 2]
    dA = np.stack([stA * np.cos(phA), stA * np.sin(phA), ctA], axis=1)
    muN_cm = np.concatenate([EmuN[:, None], pA[:, None] * dA], axis=1)
    p_muN = boost_to_lab(muN_cm)
    I2W_A = 2.0 / np.pi / np.clip(sqlam(s, s23, mpi ** 2), 1e-12, None)
    Emu = (s23 + mmu ** 2 - mNf ** 2) / (2 * rs23); pB = rs23 * sqlam(s23, mmu ** 2, mNf ** 2) / 2
    ctB = 2 * u[:, 3] - 1; stB = np.sqrt(np.clip(1 - ctB ** 2, 0, None)); phB = TWO_PI * u[:, 4]
    dB = np.stack([stB * np.cos(phB), stB * np.sin(phB), ctB], axis=1)
    mu_cm = np.concatenate([Emu[:, None], pB[:, None] * dB], axis=1)
    k_mu = boost_to_lab_parent(mu_cm, p_muN)
    I2W_B = 2.0 / np.pi / np.clip(sqlam(s23, mmu ** 2, mNf ** 2), 1e-12, None)
    density = (2 * np.pi) ** 5 * I2W_A * I2W_B / (s23max - s23min)
    J3 = np.where(density > 0, 1.0 / np.clip(density, 1e-300, None), 0.0)
    valid = (s23max > s23min) & (sqlam(s, s23, mpi ** 2) > 0) & (sqlam(s23, mmu ** 2, mNf ** 2) > 0)
    q = k_nu - k_mu; Q2 = (np.sum(q[:, 1:] ** 2, axis=1) - q[:, 0] ** 2) / 1e6
    return Q2[valid], J3[valid]


def isotropic_full(n, rng):
    """isotropic sampler returning full lab momenta (k_mu, p_N, p_pi) + Q2, J3, valid."""
    u = rng.random((n, 5))
    mpi, mNf, mmu = M_PIP, M_P, M_MU
    s23max = (sqrts - mpi) ** 2; s23min = (mmu + mNf) ** 2
    s23 = s23min + (s23max - s23min) * u[:, 0]; rs23 = np.sqrt(np.clip(s23, 1e-9, None))
    EmuN = (s + s23 - mpi ** 2) / (2 * sqrts); pA = sqrts * sqlam(s, s23, mpi ** 2) / 2
    ctA = 2 * u[:, 1] - 1; stA = np.sqrt(np.clip(1 - ctA ** 2, 0, None)); phA = TWO_PI * u[:, 2]
    dA = np.stack([stA * np.cos(phA), stA * np.sin(phA), ctA], axis=1)
    muN_cm = np.concatenate([EmuN[:, None], pA[:, None] * dA], axis=1)
    pi_cm = np.concatenate([np.sqrt(mpi ** 2 + pA ** 2)[:, None], -pA[:, None] * dA], axis=1)
    p_muN = boost_to_lab(muN_cm); p_pi = boost_to_lab(pi_cm)
    I2W_A = 2.0 / np.pi / np.clip(sqlam(s, s23, mpi ** 2), 1e-12, None)
    Emu = (s23 + mmu ** 2 - mNf ** 2) / (2 * rs23); pB = rs23 * sqlam(s23, mmu ** 2, mNf ** 2) / 2
    ctB = 2 * u[:, 3] - 1; stB = np.sqrt(np.clip(1 - ctB ** 2, 0, None)); phB = TWO_PI * u[:, 4]
    dB = np.stack([stB * np.cos(phB), stB * np.sin(phB), ctB], axis=1)
    mu_cm = np.concatenate([Emu[:, None], pB[:, None] * dB], axis=1)
    N_cm = np.concatenate([np.sqrt(mNf ** 2 + pB ** 2)[:, None], -pB[:, None] * dB], axis=1)
    k_mu = boost_to_lab_parent(mu_cm, p_muN); p_N = boost_to_lab_parent(N_cm, p_muN)
    I2W_B = 2.0 / np.pi / np.clip(sqlam(s23, mmu ** 2, mNf ** 2), 1e-12, None)
    density = (2 * np.pi) ** 5 * I2W_A * I2W_B / (s23max - s23min)
    J3 = np.where(density > 0, 1.0 / np.clip(density, 1e-300, None), 0.0)
    valid = (s23max > s23min) & (sqlam(s, s23, mpi ** 2) > 0) & (sqlam(s23, mmu ** 2, mNf ** 2) > 0)
    q = k_nu - k_mu; Q2 = (np.sum(q[:, 1:] ** 2, axis=1) - q[:, 0] ** 2) / 1e6
    return Q2, J3, k_mu, p_N, p_pi, valid


def boost_to_lab_parent(p4cm, parent):
    rsq = np.sqrt(np.clip(parent[:, 0] ** 2 - np.sum(parent[:, 1:] ** 2, axis=1), 1e-9, None))
    E = (parent[:, 0] * p4cm[:, 0] + np.sum(parent[:, 1:] * p4cm[:, 1:], axis=1)) / rsq
    c1 = (p4cm[:, 0] + E) / (rsq + parent[:, 0])
    return np.concatenate([E[:, None], p4cm[:, 1:] + c1[:, None] * parent[:, 1:]], axis=1)


def phi3_analytic():
    import numpy as _np
    m1, m2, m3 = M_MU, M_P, M_PIP
    def phi2(M2, a2, b2):
        lam = (M2 - a2 - b2) ** 2 - 4 * a2 * b2
        return _np.where(lam > 0, (1.0 / (8 * _np.pi)) * _np.sqrt(_np.clip(lam, 0, None)) / M2, 0.0)
    s23g = _np.linspace((m2 + m3) ** 2, (sqrts - m1) ** 2, 4_000_000)
    integ = phi2(s, s23g, m1 ** 2) * phi2(s23g, m2 ** 2, m3 ** 2) / (2 * _np.pi)
    return _np.sum(0.5 * (integ[1:] + integ[:-1]) * _np.diff(s23g))


def flat_dalitz_amps2(n, rng):
    """flat-Dalitz returning full lab 4-vectors + accept mask, for amps2 weighting."""
    m1, m2, m3 = M_MU, M_P, M_PIP
    s12 = rng.uniform((m1 + m2) ** 2, (sqrts - m3) ** 2, n)
    s23 = rng.uniform((m2 + m3) ** 2, (sqrts - m1) ** 2, n)
    E1 = (s + m1 ** 2 - s23) / (2 * sqrts); E3 = (s + m3 ** 2 - s12) / (2 * sqrts); E2 = sqrts - E1 - E3
    p1 = np.sqrt(np.clip(E1 ** 2 - m1 ** 2, 0, None)); p2 = np.sqrt(np.clip(E2 ** 2 - m2 ** 2, 0, None)); p3 = np.sqrt(np.clip(E3 ** 2 - m3 ** 2, 0, None))
    ct13 = (p2 ** 2 - p1 ** 2 - p3 ** 2) / (2 * p1 * p3); st13 = np.sqrt(np.clip(1 - ct13 ** 2, 0, None))
    ok = (E2 >= m2) & (np.abs(ct13) <= 1) & (p1 > 0) & (p3 > 0)
    v1 = np.stack([np.zeros(n), np.zeros(n), p1], axis=1)
    v3 = np.stack([p3 * st13, np.zeros(n), p3 * ct13], axis=1); v2 = -(v1 + v3)
    R = rand_rot(n, rng)
    v1 = np.einsum('nij,nj->ni', R, v1); v2 = np.einsum('nij,nj->ni', R, v2); v3 = np.einsum('nij,nj->ni', R, v3)
    kmu = boost_to_lab(np.concatenate([E1[:, None], v1], axis=1))
    pN = boost_to_lab(np.concatenate([E2[:, None], v2], axis=1))
    ppi = boost_to_lab(np.concatenate([E3[:, None], v3], axis=1))
    return kmu, pN, ppi, ok


if __name__ == "__main__":
    import jax; jax.config.update("jax_enable_x64", True)
    from adonis.xsec.dcc_current import exclusive_amps2_batch
    rng = np.random.default_rng(0)
    N = 4_000_000
    pstk = np.tile(p_st, (N, 1)); knu = np.tile(k_nu, (N, 1))
    # ---- amps2-weighted integral both ways ----
    Phi3 = phi3_analytic()
    kmu, pN, ppi, ok = flat_dalitz_amps2(N, rng)
    a_f = np.zeros(N); idx = np.where(ok)[0]
    a_f[idx] = np.asarray(exclusive_amps2_batch(knu[idx], kmu[idx], pstk[idx], pN[idx], ppi[idx], +1, 211))
    a_f = np.where(np.isfinite(a_f), a_f, 0)
    int_flat = Phi3 * a_f[ok].mean()
    qi2, ji2, kmu2, pN2, ppi2, val2 = isotropic_full(N, rng)
    a_i = np.zeros(N); idx = np.where(val2)[0]
    a_i[idx] = np.asarray(exclusive_amps2_batch(knu[idx], kmu2[idx], pstk[idx], pN2[idx], ppi2[idx], +1, 211))
    a_i = np.where(np.isfinite(a_i), a_i, 0)
    int_iso = (a_i * ji2 * val2).mean()
    print(f"INT amps2*dPhi3 at Enu={ENU}:  flat-Dalitz={int_flat:.5e}  isotropic={int_iso:.5e}  iso/flat={int_iso/int_flat:.4f}")
    # also the pure-PS Q2 shape check
    qf = flat_dalitz(8_000_000, rng)
    qi, ji = isotropic(8_000_000, rng)
    bins = np.linspace(0, (qf.max() if len(qf) else 1), 21)
    hf, _ = np.histogram(qf, bins=bins)                       # flat-Dalitz: equal weight
    hi, _ = np.histogram(qi, bins=bins, weights=ji)           # isotropic: weighted by J3
    hf = hf / hf.sum(); hi = hi / hi.sum()                    # area-normalised shapes
    print(f"sqrt(s)={sqrts:.1f} MeV  Enu={ENU}  (Q2 in GeV^2)")
    print(" Q2 bin        flat-Dalitz   isotropic    iso/flat")
    ctr = 0.5 * (bins[1:] + bins[:-1])
    for i in range(len(ctr)):
        if hf[i] > 0:
            print(f"  {bins[i]:.3f}-{bins[i+1]:.3f}   {hf[i]:.4f}      {hi[i]:.4f}     {hi[i]/hf[i]:.3f}")
