"""MINIMAL STEP 1: is the irot0->irot1 shape break in the HADRON TENSOR or the LEPTON?

Compare a Lorentz INVARIANT (trace = eta_mu,nu W^{mu,nu}) of the angle-integrated hadron
tensor built two ways, NO lepton involved:
  irot0:  current_and_tensor(zmtx, kernel)            [q||z, analytic int dOmega]
  irot1:  sum_Omega conj(zj)⊗zj  via exclusive gather [real q-angle + dfun, quadrature]
Both are the SAME physical tensor, so trace (frame-invariant) must agree. Ratio vs Q^2:
  flat  -> tensors agree -> the break is in the lepton path;
  slope -> the dfun gather is the culprit.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import adonis.xsec.constants as C
from adonis.xsec.backend import MASS_PDG_PROTON
from adonis.primary.dcc.assembly import build_zmtx, angular_kernel, current_and_tensor
from adonis.primary.dcc.amplitudes import DCCAmplitudes, DCCKnobs
from adonis.primary.dcc.loader import load_cached
from adonis.xsec.dcc_current import exclusive_amps2_batch

_T = load_cached(); _AMP = DCCAmplitudes()
_2J, _2L, _2I = np.asarray(_T.pw_2J), np.asarray(_T.pw_2L), np.asarray(_T.pw_2I)
_KER = angular_kernel(_2J, _2L, _2I, tcrz=1.0, tiz=0.5, tpinz=1.5, tpiz=1.0, n_theta=16, n_phi=16)
ETA = np.array([1.0, -1.0, -1.0, -1.0])
E_NU = 1000.0; MPI = 139.57018


def _boost(p, beta):
    b2 = np.clip(np.sum(beta ** 2, 1, keepdims=True), 1e-30, 0.9999999)
    g = 1.0 / np.sqrt(1 - b2); bp = np.sum(beta * p[:, 1:], 1, keepdims=True)
    E = g[:, 0] * (p[:, 0] + bp[:, 0]); pv = p[:, 1:] + ((g - 1) * bp / b2 + g * p[:, 0:1]) * beta
    return np.concatenate([E[:, None], pv], 1)


def trace(W):  # eta_mu,nu W^{mu,nu} (Lorentz invariant), real
    return np.real(W[:, 0, 0] - W[:, 1, 1] - W[:, 2, 2] - W[:, 3, 3])


def boost_W0_to_lab(W0, knu, kmu, pst, pcm):
    """current_and_tensor builds W0 in F0 = (2CM, q along +z). Transform it to LAB:
    rotate F0->F_cm (q: z->real direction n in 2CM), then boost F_cm->lab (xlr).
    W0 is azimuthally symmetric about q, so the residual rotation about n is irrelevant."""
    from adonis.xsec.dcc_kinematics import boost_matrix_batch
    N = len(W0); q = knu - kmu
    E_on = np.sqrt(np.sum(pst[:, 1:] ** 2, 1) + C.mN ** 2)
    qsh = q.copy(); qsh[:, 0] = q[:, 0] + pst[:, 0] - E_on
    xlrs = boost_matrix_batch(pcm, to_cm=True); xlr = boost_matrix_batch(pcm, to_cm=False)
    qx2 = np.einsum('nij,nj->ni', xlrs, qsh)
    n = qx2[:, 1:] / np.linalg.norm(qx2[:, 1:], axis=1, keepdims=True)
    z = np.tile([0, 0, 1.0], (N, 1))
    e1 = np.cross(z, n); s = np.linalg.norm(e1, axis=1, keepdims=True)
    e1 = np.where(s > 1e-9, e1 / np.where(s > 1e-9, s, 1.0), np.tile([1.0, 0, 0], (N, 1)))
    e2 = np.cross(n, e1)
    R3 = np.stack([e1, e2, n], axis=2)                     # columns: R3[:,:,2]=n  => R3 z=n
    LamR = np.zeros((N, 4, 4)); LamR[:, 0, 0] = 1.0; LamR[:, 1:, 1:] = R3
    Lam = np.einsum('nij,njk->nik', xlr, LamR)             # boost . rotation
    return np.einsum('nim,nmk,njk->nij', Lam, W0, np.conj(Lam))  # Lam W0 Lam^T  (Lam real)


def Wbar_irot0(W, Q2adj):
    out = np.zeros((len(W), 4, 4), complex)
    for i in range(len(W)):
        vec, isv, axial = _AMP.amplitudes_spline_np(np.array([W[i]]), np.array([Q2adj[i]]), DCCKnobs())
        z = build_zmtx(vec[0], isv[0], axial[0], jnp.asarray(W[i]), jnp.asarray(Q2adj[i]),
                       _2J, _2L, _2I, mode=1, itiz=1, m_N=C.mN, m_pi=MPI)
        out[i] = np.asarray(current_and_tensor(z, _KER))
    return out


def Wbar_irot1(knu, kmu, pst, pcm, W, n_ct=10, n_phi=10):
    N = len(W)
    Epi = (W ** 2 + MPI ** 2 - C.mN ** 2) / (2 * W); EN = (W ** 2 + C.mN ** 2 - MPI ** 2) / (2 * W)
    kpi = np.sqrt(np.clip(Epi ** 2 - MPI ** 2, 0, None))
    ct, wct = np.polynomial.legendre.leggauss(n_ct); beta = pcm[:, 1:] / pcm[:, 0:1]
    out = np.zeros((N, 4, 4), complex); t0 = time.time()
    for i in range(n_ct):
        c = ct[i]; s = np.sqrt(max(1 - c * c, 0.0))
        for j in range(n_phi):
            ph = 2 * np.pi * j / n_phi; dx, dy, dz = s * np.cos(ph), s * np.sin(ph), c
            ppi = _boost(np.stack([Epi, kpi * dx, kpi * dy, kpi * dz], 1), beta)
            pN = _boost(np.stack([EN, -kpi * dx, -kpi * dy, -kpi * dz], 1), beta)
            zj = np.asarray(exclusive_amps2_batch(knu, kmu, pst, pN, ppi, +1, 211, return_zj=True))
            W4 = np.einsum('ecm,ecn->emn', np.conj(zj), zj)        # (N,4,4) sum over combos
            out += wct[i] * (2 * np.pi / n_phi) * np.where(np.isfinite(W4), W4, 0.0)
        print(f"    Omega {(i+1)*n_phi}/{n_ct*n_phi}  {time.time()-t0:.0f}s", flush=True)
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    rng = np.random.default_rng(0)
    Ep = 50.0 + (E_NU - 60.0) * rng.random(n); theta = np.deg2rad(180.0) * rng.random(n)
    omega = E_NU - Ep; qx = -Ep * np.sin(theta); qz = E_NU - Ep * np.cos(theta)
    qv2 = qx ** 2 + qz ** 2; Q2 = qv2 - omega ** 2
    knu = np.tile([E_NU, 0, 0, E_NU], (n, 1)).astype(float)
    kmu = np.stack([Ep, Ep * np.sin(theta), np.zeros(n), Ep * np.cos(theta)], -1)
    pst = np.tile([MASS_PDG_PROTON, 0, 0, 0], (n, 1)).astype(float)
    pcm = np.stack([omega, qx, np.zeros(n), qz], -1) + pst
    W = np.sqrt(np.clip(pcm[:, 0] ** 2 - np.sum(pcm[:, 1:] ** 2, 1), 1.0, None))
    Q2adj = qv2 - (omega + pst[:, 0] - np.sqrt(pst[:, 0] ** 2)) ** 2  # ~ Q2 (free p, tiny shift)
    Q2adj = np.clip(Q2adj, 1.0, None)
    cut = (W > 1076.957) & (W < 2000.0) & (Q2 > 0) & (Q2 < 1.2e6)
    idx = np.where(cut)[0]
    print(f"  {len(idx)}/{n} events; building Wbar both ways ...", flush=True)
    W0 = Wbar_irot0(W[idx], Q2adj[idx])
    W1 = Wbar_irot1(knu[idx], kmu[idx], pst[idx], pcm[idx], W[idx])
    t0, t1 = trace(W0), trace(W1)
    Q2g = Q2[idx] / 1e6

    # ---- full |M|^2 both ways: irot0 = A_cm . W0_cm ; irot1 = A_lab . W1_lab ---- #
    import jax.numpy as _jnp
    from adonis.xsec.leptonic import lepton_current as _lc
    import scripts.fold_bisect as _FB
    g = ETA[:, None] * ETA[None, :]
    kcm, kpcm = _FB.cm_lepton_momenta(knu[idx], kmu[idx], pst[idx])
    Acm = np.einsum('ecm,ecn->emn', np.asarray(_lc(_jnp.asarray(kcm), _jnp.asarray(kpcm))),
                    np.conj(np.asarray(_lc(_jnp.asarray(kcm), _jnp.asarray(kpcm)))))
    Lab = np.asarray(_lc(_jnp.asarray(knu[idx]), _jnp.asarray(kmu[idx])))
    Alab = np.einsum('ecm,ecn->emn', Lab, np.conj(Lab))
    Msq0 = np.real(np.sum(g[None] * Acm * np.conj(W0), axis=(-2, -1)))
    Msq1 = np.real(np.sum(g[None] * Alab * np.conj(W1), axis=(-2, -1)))
    cmbad = ~np.isfinite(np.sum(kcm, 1))     # cm_lepton_momenta failures

    print("\n  trace(Wbar) [hadron only, frame-invariant]  and  |M|^2 [+lepton]  ratios irot1/irot0:")
    bins = np.linspace(0, 1.0, 11)
    for b in range(len(bins) - 1):
        sel = (Q2g >= bins[b]) & (Q2g < bins[b + 1])
        mt = sel & (np.abs(t0) > 1e-30)
        mm = sel & np.isfinite(Msq0) & np.isfinite(Msq1) & (np.abs(Msq0) > 1e-300)
        if sel.sum() > 3:
            rt = np.median(t1[mt] / t0[mt]) if mt.sum() else np.nan
            rm = np.median(Msq1[mm] / Msq0[mm]) if mm.sum() else np.nan
            print(f"   Q2={0.5*(bins[b]+bins[b+1]):.2f}  n={sel.sum():5d}  "
                  f"trace1/0={rt:+.3f}   |M|^2 1/0={rm:+.3f}   cm_fail={int((sel&cmbad).sum())}")

    # ---- COMPONENT-LEVEL common-frame compare: boost W0 to LAB, vs W1 (lab) ---- #
    W0lab = boost_W0_to_lab(W0, knu[idx], kmu[idx], pst[idx], pcm[idx])
    # validate the boost: A_lab . W0lab must equal A_cm . W0 (Lorentz scalar)
    Msq0_lab = np.real(np.sum(g[None] * Alab * np.conj(W0lab), axis=(-2, -1)))
    vv = np.isfinite(Msq0) & np.isfinite(Msq0_lab) & (np.abs(Msq0) > 1e-300)
    print(f"\n  boost validation:  median(A_lab.W0lab / A_cm.W0) = "
          f"{np.median(Msq0_lab[vv]/Msq0[vv]):.4f}  (should be 1.000)")
    print(f"  => same-lepton hadron ratio  median(A_lab.W1 / A_lab.W0lab) at Q2<0.2 = "
          f"{np.median((Msq1[vv & (Q2g<0.2)])/(Msq0_lab[vv & (Q2g<0.2)])):.3f}")
    print("\n  W0(irot0)->lab  vs  W1(irot1,lab)  per-component median ratio (low-Q^2 events Q2<0.2):")
    lo = (Q2g < 0.2) & np.isfinite(W0lab[:, 0, 0].real) & (np.abs(W0lab[:, 1, 1].real) > 1e-300)
    comps = [(0, 0, "W00 time"), (1, 1, "W11 trans"), (2, 2, "W22 trans"), (3, 3, "W33 z"),
             (0, 3, "W03 time-z"), (0, 1, "W01"), (1, 3, "W13")]
    for a, c, nm in comps:
        v0 = W0lab[lo, a, c]; v1 = W1[lo, a, c]
        # report on the real part (dominant) with a magnitude-weighted median ratio
        mag = np.abs(v0) > 1e-6 * np.max(np.abs(W0lab[lo, 1, 1]))
        if mag.sum() > 3:
            r = np.real(v1[mag]) / np.real(v0[mag])
            print(f"   {nm:11s}: medianRatio(re)={np.median(r):+.3f}   "
                  f"<W0lab>={np.mean(np.real(v0)):+.3e}  <W1>={np.mean(np.real(v1)):+.3e}")


if __name__ == "__main__":
    main()
