"""FORWARD BISECTION (user's method): start from the known-good irot_q=0 angle-integrated
'fold' path (which the old diffpi fold proved reproduces ACHILLES), then swap ONE
simplification at a time toward what the current exclusive ADoNIS does, and watch which
swap breaks the perfect dsigma/dW, dsigma/dQ2 agreement.

Free proton, mono 1 GeV (matches Achilles/_resrun_out/nofsi_res_H_mono.hepmc). Single
channel p->p pi+ (pure I=3/2). Compares SHAPE (area-normalised) so a Q^2-DEPENDENT deficit
(0.5 at low Q^2 -> 1 at high) is visible, while a flat scale is not.

Toggles (env):
  FOLD_LEPTON   = massless | full     (massless = diffpi lepton_tensor_cc; full = m_mu spinors)
  FOLD_HADRON   = irot0    | excl     (irot0 = current_and_tensor angle-integrated;
                                       excl  = per-event exclusive gather, MC-integrated over Omega_pi)
Default massless+irot0 reproduces the WORKING fold.
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

import adonis.xsec.constants as C
from adonis.xsec.backend import MASS_PDG_PROTON
from adonis.primary.dcc.assembly import build_zmtx, angular_kernel, current_and_tensor
from adonis.primary.dcc.amplitudes import DCCAmplitudes, DCCKnobs
from adonis.primary.dcc.loader import load_cached
from adonis.xsec.dcc_current import _NORM, exclusive_amps2_batch

LEPTON = os.environ.get("FOLD_LEPTON", "massless")
HADRON = os.environ.get("FOLD_HADRON", "irot0")
E_NU = 1000.0
M_MU = 105.6583745
ACHHEPMC = Path("/Users/homelab/Lab/Playground/projects/DIFFGEN/Achilles/_resrun_out/nofsi_res_H_mono.hepmc")

_T = load_cached(); _AMP = DCCAmplitudes()
_2J, _2L, _2I = np.asarray(_T.pw_2J), np.asarray(_T.pw_2L), np.asarray(_T.pw_2I)
# proton channel p -> p pi+ : CC nu (tcrz=1), proton (tiz=+1/2), tpinz=3/2, pi+ (tpiz=1)
_KER = angular_kernel(_2J, _2L, _2I, tcrz=1.0, tiz=0.5, tpinz=1.5, tpiz=1.0,
                      n_theta=16, n_phi=16)

# --------- massless lepton tensor (diffpi lepton_tensor.py, q along z piN-CM) ---------- #
ETA = np.array([1.0, -1.0, -1.0, -1.0])
_EPS = np.zeros((4, 4, 4, 4))
import itertools
for _p in itertools.permutations(range(4)):
    s = 1
    pl = list(_p)
    for a in range(4):
        for b in range(a + 1, 4):
            if pl[a] > pl[b]:
                s = -s
    _EPS[_p] = s


def _mink(a, b): return a[..., 0] * b[..., 0] - np.sum(a[..., 1:] * b[..., 1:], -1)


def boost_to_rest(P, a):
    M = np.sqrt(np.clip(_mink(P, P), 1e-9, None))
    gamma = P[..., 0] / M
    beta = P[..., 1:] / P[..., 0:1]
    b2 = np.clip(np.sum(beta ** 2, -1), 1e-30, None)
    bda = np.sum(beta * a[..., 1:], -1)
    a0 = gamma * (a[..., 0] - bda)
    coef = ((gamma - 1.0) * bda / b2 - gamma * a[..., 0])[..., None]
    return np.concatenate([a0[..., None], a[..., 1:] + coef * beta], -1)


def cm_lepton_momenta(k, kp, p_struck):
    q = k - kp; P = q + p_struck
    k_r = boost_to_rest(P, k); kp_r = boost_to_rest(P, kp); q_r = boost_to_rest(P, q)
    e3 = q_r[..., 1:] / np.linalg.norm(q_r[..., 1:], axis=-1, keepdims=True)
    def resolve(v):
        vz = np.sum(v[..., 1:] * e3, -1); vT = np.linalg.norm(v[..., 1:] - vz[..., None] * e3, axis=-1)
        return np.stack([v[..., 0], vT, np.zeros_like(vT), vz], -1)
    return resolve(k_r), resolve(kp_r)


def lepton_tensor_cc_massless(k, kp):
    kk = _mink(k, kp)[..., None, None]
    sym = (k[..., :, None] * kp[..., None, :] + kp[..., :, None] * k[..., None, :] - np.diag(ETA)[None] * kk)
    asym = np.einsum("mnab,...a,...b->...mn", _EPS, k * ETA, kp * ETA)
    return 8.0 * (sym - 1j * asym)


def contract_LW(L, W):
    g = ETA[:, None] * ETA[None, :]
    return np.real(np.sum(g[None] * L * W, axis=(-2, -1)))


def pion_cm_mom(W):
    thr_hi = (C.mN + 139.57018) ** 2; thr_lo = (C.mN - 139.57018) ** 2
    lam = np.clip((W ** 2 - thr_hi) * (W ** 2 - thr_lo), 0.0, None)
    return np.sqrt(lam) / (2.0 * W)


def _trapz(h, x):
    return float(np.sum(0.5 * (h[1:] + h[:-1]) * np.diff(x)))


def _boost(p, beta):
    """Boost p (N,4) from the frame moving with velocity beta (N,3) in lab, to lab."""
    b2 = np.clip(np.sum(beta ** 2, 1, keepdims=True), 1e-30, 0.9999999)
    g = 1.0 / np.sqrt(1 - b2)
    bp = np.sum(beta * p[:, 1:], 1, keepdims=True)
    E = g[:, 0] * (p[:, 0] + bp[:, 0])
    pv = p[:, 1:] + ((g - 1) * bp / b2 + g * p[:, 0:1]) * beta
    return np.concatenate([E[:, None], pv], 1)


def inclusive_Msq_irot1(knu, kmu, pst, pcm, W, n_ct=10, n_phi=10):
    """Per-event INCLUSIVE |M|^2 = int dOmega_pi amps2_excl(Omega), using the irot_q=1 exclusive
    gather (real q-angle + dfun) + full m_mu lepton, integrated over the pion solid angle by
    Gauss-Legendre x uniform-phi quadrature (deterministic, no MC noise)."""
    N = len(W); mpi = 139.57018
    Epi = (W ** 2 + mpi ** 2 - C.mN ** 2) / (2 * W)
    EN = (W ** 2 + C.mN ** 2 - mpi ** 2) / (2 * W)
    kpi = np.sqrt(np.clip(Epi ** 2 - mpi ** 2, 0, None))
    ct, wct = np.polynomial.legendre.leggauss(n_ct)
    beta = pcm[:, 1:] / pcm[:, 0:1]
    out = np.zeros(N); t0 = time.time()
    for i in range(n_ct):
        c = ct[i]; s = np.sqrt(max(1 - c * c, 0.0))
        for j in range(n_phi):
            ph = 2 * np.pi * j / n_phi
            dx, dy, dz = s * np.cos(ph), s * np.sin(ph), c
            ppi_cm = np.stack([Epi, kpi * dx, kpi * dy, kpi * dz], 1)
            pN_cm = np.stack([EN, -kpi * dx, -kpi * dy, -kpi * dz], 1)
            ppi = _boost(ppi_cm, beta); pN = _boost(pN_cm, beta)
            a = np.asarray(exclusive_amps2_batch(knu, kmu, pst, pN, ppi, +1, 211))
            out += wct[i] * (2 * np.pi / n_phi) * np.where(np.isfinite(a), a, 0.0)
        print(f"    irot1 Omega {(i+1)*n_phi}/{n_ct*n_phi}  {time.time()-t0:.0f}s", flush=True)
    return out


def shape_err(v, w, edges):
    """Area-normalised shape + per-bin stat error (ADoNIS: sqrt(sum w^2); ACHILLES w=1 -> Poisson)."""
    c = 0.5 * (edges[1:] + edges[:-1]); bw = np.diff(edges)
    h, _ = np.histogram(v, bins=edges, weights=w)
    h2, _ = np.histogram(v, bins=edges, weights=w ** 2)
    d = h / bw; e = np.sqrt(h2) / bw
    area = _trapz(d, c)
    area = area if area > 0 else 1.0
    return c, d / area, e / area


def chi2_ratio(hm, em, ho, eo):
    m = (ho > 0) & (hm > 0)
    r = hm[m] / ho[m]
    sr = r * np.sqrt((em[m] / hm[m]) ** 2 + (eo[m] / ho[m]) ** 2)
    chi2 = float(np.sum((r - 1) ** 2 / sr ** 2)); ndf = int(m.sum())
    return r, sr, m, chi2, ndf


_GRID = None  # (Wg, Q2g, Wmn_grid[nq,nw,4,4])


def _build_grid():
    global _GRID
    if _GRID is not None:
        return _GRID
    Wg = np.asarray(_T.W); Q2g = np.asarray(_T.Q2)
    grid = np.zeros((len(Q2g), len(Wg), 4, 4), complex)
    t0 = time.time()
    for iq, qv in enumerate(Q2g):
        for iw, wv in enumerate(Wg):
            vec, isv, axial = _AMP.amplitudes_spline_np(np.array([wv]), np.array([qv]), DCCKnobs())
            z = build_zmtx(vec[0], isv[0], axial[0], jnp.asarray(wv), jnp.asarray(qv),
                           _2J, _2L, _2I, mode=1, itiz=1, m_N=C.mN, m_pi=139.57018)
            grid[iq, iw] = np.asarray(current_and_tensor(z, _KER))
        if iq % 5 == 0:
            print(f"    grid Q2 {iq+1}/{len(Q2g)}  {time.time()-t0:.0f}s", flush=True)
    _GRID = (Wg, Q2g, grid)
    return _GRID


def hadron_tensor_irot0(W, Q2):
    """Angle-integrated W^{mu,nu}[N,4,4] via current_and_tensor (irot_q=0), bilinear interp."""
    Wg, Q2g, grid = _build_grid()
    iw = np.clip(np.searchsorted(Wg, W) - 1, 0, len(Wg) - 2)
    iq = np.clip(np.searchsorted(Q2g, Q2) - 1, 0, len(Q2g) - 2)
    tw = (W - Wg[iw]) / (Wg[iw + 1] - Wg[iw]); tq = (Q2 - Q2g[iq]) / (Q2g[iq + 1] - Q2g[iq])
    tw = tw[:, None, None]; tq = tq[:, None, None]
    v00, v01, v10, v11 = grid[iq, iw], grid[iq, iw + 1], grid[iq + 1, iw], grid[iq + 1, iw + 1]
    return (1 - tq) * ((1 - tw) * v00 + tw * v01) + tq * ((1 - tw) * v10 + tw * v11)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    print(f"=== fold_bisect  LEPTON={LEPTON} HADRON={HADRON}  n={n} ===", flush=True)
    rng = np.random.default_rng(0)
    Ep = 50.0 + (E_NU - 60.0) * rng.random(n)
    theta = np.deg2rad(180.0) * rng.random(n)
    omega = E_NU - Ep
    qx = -Ep * np.sin(theta); qz = E_NU - Ep * np.cos(theta)
    qvec2 = qx ** 2 + qz ** 2
    Q2_lep = qvec2 - omega ** 2
    k_lab = np.tile([E_NU, 0, 0, E_NU], (n, 1)).astype(float)
    kp_lab = np.stack([Ep, Ep * np.sin(theta), np.zeros(n), Ep * np.cos(theta)], -1)
    _pm = C.mN if os.environ.get("FOLD_NOSHIFT", "0") == "1" else MASS_PDG_PROTON
    p_struck = np.tile([_pm, 0, 0, 0], (n, 1)).astype(float)
    # de Forest q-shift for the amplitude Q2 (ACHILLES current_init); W from true q
    E_on = np.sqrt(np.sum(p_struck[:, 1:] ** 2, 1) + C.mN ** 2)
    qp0 = omega + p_struck[:, 0] - E_on
    Q2_adj = qvec2 - qp0 ** 2
    tot = np.stack([omega, qx, np.zeros(n), qz], -1) + p_struck
    W = np.sqrt(np.clip(tot[:, 0] ** 2 - np.sum(tot[:, 1:] ** 2, 1), 1.0, None))
    cut = (W > 1076.957) & (W < 2000.0) & (Q2_adj > 0) & (Q2_adj < 5e6)
    idx = np.where(cut)[0]
    print(f"  {len(idx)}/{n} pass cuts; computing |M|^2 ({HADRON}/{LEPTON}) ...", flush=True)
    t0 = time.time()
    kpi = pion_cm_mom(W)
    Msq = np.zeros(n)                                            # per-event pion-angle-integrated |M|^2
    if HADRON == "irot0":
        Wmn = np.zeros((n, 4, 4), complex)
        Wmn[idx] = hadron_tensor_irot0(W[idx], np.clip(Q2_adj[idx], 1.0, None))
        k_cm, kp_cm = cm_lepton_momenta(k_lab, kp_lab, p_struck)
        if LEPTON == "massless":
            LW = contract_LW(lepton_tensor_cc_massless(k_cm, kp_cm), Wmn)
        else:
            from adonis.xsec.leptonic import lepton_current
            Lc = np.asarray(lepton_current(jnp.asarray(k_cm), jnp.asarray(kp_cm)))
            A = np.einsum('ecm,ecn->emn', Lc, np.conj(Lc))
            g = ETA[:, None] * ETA[None, :]
            LW = np.real(np.sum(g[None] * A * np.conj(Wmn), axis=(-2, -1)))
        Msq = LW / _NORM
    elif HADRON == "irot1":                                     # real-q-angle exclusive gather, full lepton, int dOmega_pi
        Msq[idx] = inclusive_Msq_irot1(k_lab[idx], kp_lab[idx], p_struck[idx], tot[idx], W[idx])
    print(f"    |M|^2 done  {time.time()-t0:.0f}s", flush=True)
    w = np.where(cut, (Ep / E_NU) * np.sin(theta) * (kpi / W) * Msq, 0.0)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    print(f"  sum(w)={w.sum():.4e}  kept {np.sum(w>0)}", flush=True)

    # ---- oracle (mono hepmc), shapes + propagated stat errors + chi2 ----
    aW, aQ2 = parse_oracle()
    We = np.linspace(1080, 1500, 26); Qe = np.linspace(0, 1.2e6, 26)
    cW, hmW, emW = shape_err(W, w, We);          _, hoW, eoW = shape_err(aW, np.ones(len(aW)), We)
    cQ, hmQ, emQ = shape_err(Q2_lep, w, Qe);     _, hoQ, eoQ = shape_err(aQ2 * 1e6, np.ones(len(aQ2)), Qe)
    rW, srW, mW, chi2W, ndfW = chi2_ratio(hmW, emW, hoW, eoW)
    rQ, srQ, mQ, chi2Q, ndfQ = chi2_ratio(hmQ, emQ, hoQ, eoQ)
    cQg = cQ / 1e6
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), height_ratios=[3, 1])
    ax[0,0].step(cW, hoW, where="mid", color="0.4", label="ACHILLES")
    ax[0,0].errorbar(cW, hmW, yerr=emW, fmt="o-", color="C3", ms=3, label=f"fold {HADRON}/{LEPTON}")
    ax[0,0].legend(); ax[0,0].set_title("dσ/dW shape")
    ax[1,0].axhspan(0.97,1.03,color="g",alpha=0.12); ax[1,0].axhline(1, ls="--", color="g")
    ax[1,0].errorbar(cW[mW], rW, yerr=srW, fmt="o", color="C3", ms=3)
    ax[1,0].set_ylim(0.3,1.5); ax[1,0].set_xlabel("W [MeV]"); ax[1,0].set_ylabel("fold/ACH")
    ax[1,0].text(0.05,0.85,f"χ²/ndf={chi2W:.1f}/{ndfW}={chi2W/max(ndfW,1):.2f}",transform=ax[1,0].transAxes)
    ax[0,1].step(cQg, hoQ, where="mid", color="0.4", label="ACHILLES")
    ax[0,1].errorbar(cQg, hmQ, yerr=emQ, fmt="o-", color="C3", ms=3, label="fold")
    ax[0,1].legend(); ax[0,1].set_title("dσ/dQ² shape")
    ax[1,1].axhspan(0.97,1.03,color="g",alpha=0.12); ax[1,1].axhline(1, ls="--", color="g")
    ax[1,1].errorbar(cQg[mQ], rQ, yerr=srQ, fmt="o", color="C3", ms=3)
    ax[1,1].set_ylim(0.3,1.5); ax[1,1].set_xlabel("Q² [GeV²]"); ax[1,1].set_ylabel("fold/ACH")
    ax[1,1].text(0.05,0.85,f"χ²/ndf={chi2Q:.1f}/{ndfQ}={chi2Q/max(ndfQ,1):.2f}",transform=ax[1,1].transAxes)
    fig.suptitle(f"FOLD bisect  hadron={HADRON} lepton={LEPTON}   "
                 f"χ²/ndf(W)={chi2W/max(ndfW,1):.2f}  χ²/ndf(Q²)={chi2Q/max(ndfQ,1):.2f}")
    fig.tight_layout(); out = f"figures/fold_bisect_{HADRON}_{LEPTON}.png"; fig.savefig(out, dpi=110)
    print(f"  wrote {out}", flush=True)
    print(f"  chi2/ndf  W={chi2W/max(ndfW,1):.2f}  Q2={chi2Q/max(ndfQ,1):.2f}   "
          f"Q2 ratio low(<0.2)={np.nanmean(rQ[cQg[mQ]<0.2]):.3f} high(>0.6)={np.nanmean(rQ[cQg[mQ]>0.6]):.3f}", flush=True)


def parse_oracle():
    """W=M(Npi), Q2[GeV^2] from the mono hepmc (knu=max-E pid14)."""
    W, Q2 = [], []
    nu, mu, pN, pPi = [], None, None, None
    def m2(p): return p[0]**2 - p[1]**2 - p[2]**2 - p[3]**2
    def flush():
        nonlocal nu, mu, pN, pPi
        if nu and mu is not None and pN is not None and pPi is not None:
            knu = max(nu, key=lambda p: p[0]); q = knu - mu
            Q2.append(-m2(q) / 1e6); W.append(np.sqrt(max(m2(pN + pPi), 0.0)))
        nu, mu, pN, pPi = [], None, None, None
    with open(ACHHEPMC) as fh:
        for line in fh:
            t = line[:2]
            if t == "E ": flush()
            elif t == "P ":
                f = line.split(); pid = int(f[3]); st = int(f[9])
                p4 = np.array([float(f[7]), float(f[4]), float(f[5]), float(f[6])])
                if pid == 14: nu.append(p4)
                elif pid == 13 and st == 1: mu = p4
                elif st == 1 and abs(pid) in (2112, 2212): pN = p4
                elif st == 1 and pid in (211, 111, -211): pPi = p4
    flush()
    return np.array(W), np.array(Q2)


if __name__ == "__main__":
    main()
