"""STANDALONE study (no production code touched): how much does a DEFENSIVE-MIXTURE spectral-function
proposal raise N_eff in the high-|p_struck| corner (the FSI-pion / high-delta_pT tail), and at what
bulk cost?  Differentiability is unaffected -- the proposal is frozen; only the variance changes.

Current SF proposal: (|p|,E) ~ |p|^2 S  (SpectralImportanceSampler).  The matrix element amps2 grows
in the high-|p| off-shell tail where |p|^2 S is small -> a few huge-weight events (N_eff~8 in the
FSI-pion bin).  Defensive mixture: draw |p| from (1-a)*[|p|^2 n_p] + a*[|p|^2 flat] (a fat tail that
over-covers high |p|); weight *= ratio = q_cur/q' so it stays unbiased.

Run: python -u scripts/sf_proposal_neff_test.py [N=300000]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax.numpy as jnp
from adonis.xsec import constants as C
from adonis.xsec.flux import T2KFlux, M_MU, M_P
from adonis.xsec.spectral import SpectralFunction, SpectralImportanceSampler
from adonis.xsec.backend import me_cross_section, MASS_PDG_NEUTRON
import adonis.xsec.flux as flux; flux.BEAM_MODE = "is"
_MN = C.mN; _TWO_PI = 2 * np.pi; N_NEUTRON = 6

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300000


def _boost(p4, beta):
    b2 = np.sum(beta ** 2, axis=1, keepdims=True); g = 1 / np.sqrt(np.clip(1 - b2, 1e-15, None))
    bp = np.sum(beta * p4[:, 1:], axis=1, keepdims=True)
    E = g * (p4[:, :1] + bp)
    fac = (g - 1) * np.where(b2 > 1e-15, bp / np.clip(b2, 1e-15, None), 0.0) + g * p4[:, :1]
    return np.concatenate([E, p4[:, 1:] + fac * beta], axis=1)


def _trapz(y, x):
    return float(np.sum((y[1:] + y[:-1]) * 0.5 * np.diff(x)))


def mixture_sf(samp, n, alpha, rng):
    """Draw (pvec,E_rm) from (1-alpha)*[|p|^2 n_p] + alpha*[|p|^2 flat], + the importance ratio
    q_cur/q' for the (unbiased) weight correction.  E|p and direction are shared with the current
    sampler, so the mixture is purely in the |p| marginal."""
    pf = samp.mom; n_p = np.maximum(np.interp(pf, pf, _np_marg(samp)), 0)
    d_cur = pf ** 2 * n_p; Z_cur = _trapz(d_cur, pf)
    d_tail = pf ** 2;        Z_tail = _trapz(d_tail, pf)
    # draw |p|
    use_tail = rng.random(n) < alpha
    up = rng.random(n)
    # current |p| via p_cdf
    pj = np.clip(np.searchsorted(samp.p_cdf, up), 1, len(pf) - 1)
    c0, c1 = samp.p_cdf[pj - 1], samp.p_cdf[pj]; fr = np.clip((up - c0) / (c1 - c0 + 1e-30), 0, 1)
    p_cur = pf[pj - 1] + fr * (pf[pj] - pf[pj - 1])
    # tail |p| ~ |p|^2 over [pf0,pfN]
    p3l, p3h = pf[0] ** 3, pf[-1] ** 3
    p_tail = (p3l + rng.random(n) * (p3h - p3l)) ** (1.0 / 3.0)
    pmag = np.where(use_tail, p_tail, p_cur)
    # E|p : interpolate the conditional inverse-CDF at pmag (bracket pf)
    bj = np.clip(np.searchsorted(pf, pmag), 1, len(pf) - 1); bf = (pmag - pf[bj - 1]) / (pf[bj] - pf[bj - 1] + 1e-30)
    ue = rng.random(n)
    def einterp(rows_idx):
        rows = samp.e_cdf[rows_idx]
        ei = np.clip(np.array([np.searchsorted(rows[k], ue[k]) for k in range(n)]), 1, len(samp.energy) - 1)
        d0 = rows[np.arange(n), ei - 1]; d1 = rows[np.arange(n), ei]
        ef = np.clip((ue - d0) / (d1 - d0 + 1e-30), 0, 1)
        return samp.energy[ei - 1] + ef * (samp.energy[ei] - samp.energy[ei - 1])
    E_rm = (1 - bf) * einterp(bj - 1) + bf * einterp(bj)
    # density ratio q_cur/q' at pmag (|p| dimension; E/direction shared -> cancel)
    dc = pmag ** 2 * np.maximum(np.interp(pmag, pf, n_p), 1e-30)
    qcur = dc / Z_cur; qp = (1 - alpha) * dc / Z_cur + alpha * (pmag ** 2) / Z_tail
    ratio = qcur / qp
    ct = 2 * rng.random(n) - 1; st = np.sqrt(np.clip(1 - ct ** 2, 0, None)); ph = _TWO_PI * rng.random(n)
    pvec = np.stack([pmag * st * np.cos(ph), pmag * st * np.sin(ph), pmag * ct], axis=1)
    return pvec, E_rm, ratio


def _np_marg(samp):
    # the |p| marginal n_p used by the sampler = p_cdf'(|p|)/|p|^2 ; reconstruct from p_cdf
    pf = samp.mom; dens = np.gradient(samp.p_cdf, pf)          # ~ |p|^2 n_p (normalized)
    return np.clip(dens / np.clip(pf ** 2, 1e-9, None), 0, None)


def run(alpha, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.random((N, 7))
    fx = T2KFlux(); minE = fx.seed_min_GeV()
    E_GeV, J_beam = fx.sample_beam(u[:, 4], minE); Enu = E_GeV * 1000.0
    k_nu = np.stack([Enu, np.zeros(N), np.zeros(N), Enu], axis=1)
    sf = SpectralFunction("data/Spectral_Functions/pke12n_tot.data"); samp = SpectralImportanceSampler(sf)
    if alpha == 0.0:
        pvec, E_rm = samp.sample(N, rng); ratio = np.ones(N)
    else:
        pvec, E_rm, ratio = mixture_sf(samp, N, alpha, rng)
    p_struck = np.concatenate([(_MN - E_rm)[:, None], pvec], axis=1)
    P = k_nu + p_struck; s = P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, axis=1); sqrts = np.sqrt(np.clip(s, 1e-9, None))
    s2, s3 = M_MU ** 2, M_P ** 2
    E1 = sqrts / 2 * (1 + s2 / s - s3 / s); E2 = sqrts / 2 * (1 + s3 / s - s2 / s)
    lam = np.sqrt(np.clip((s - s2 - s3) ** 2 - 4 * s2 * s3, 0, None)); pcm = lam / (2 * sqrts)
    cts = 2 * u[:, 5] - 1; sts = np.sqrt(np.clip(1 - cts ** 2, 0, None)); php = _TWO_PI * u[:, 6]
    dirn = np.stack([sts * np.cos(php), sts * np.sin(php), cts], axis=1)
    beta = P[:, 1:] / P[:, 0:1]
    k_mu = _boost(np.concatenate([E1[:, None], pcm[:, None] * dirn], axis=1), beta)
    p_out = _boost(np.concatenate([E2[:, None], -pcm[:, None] * dirn], axis=1), beta)
    J_2body = 2.0 * _TWO_PI * pcm / (sqrts * 16 * np.pi ** 2)
    mom_s = np.linalg.norm(pvec, axis=1)
    det_e = Enu ** 2 + mom_s ** 2 + 2 * pvec[:, 2] * Enu + (M_MU + M_P) ** 2
    emax = np.minimum(np.minimum(_MN + Enu - np.sqrt(np.clip(det_e, 0, None)), _MN - mom_s), 400.0)
    valid = (s > (M_MU + M_P) ** 2) & (lam > 0) & (E_rm > 2.5) & (E_rm < emax)
    me = np.asarray(me_cross_section(jnp.asarray(k_nu), jnp.asarray(k_mu), jnp.asarray(p_struck),
                                    jnp.asarray(p_out), spin_avg=0.5, had_mass=MASS_PDG_NEUTRON)["me_xsec"])
    w = np.where(valid, me * N_NEUTRON * J_2body * J_beam * ratio, 0.0)
    w = np.where(np.isfinite(w), w, 0.0)
    return w, mom_s


def neff(w):
    s2 = (w ** 2).sum(); return (w.sum() ** 2 / s2) if s2 > 0 else 0.0


def main():
    print(f"SF proposal N_eff study (N={N}/run, T2K nu+12C QE)\n", flush=True)
    print(f"{'alpha':>6} {'sigma[nb]':>12} {'Neff_all':>10} {'Neff(|p|>500)':>14} {'Neff(|p|>600)':>14}", flush=True)
    for a in (0.0, 0.05, 0.1, 0.2, 0.4):
        w, p = run(a)
        m5 = p > 500; m6 = p > 600
        print(f"{a:>6.2f} {w.sum()/N:>12.4e} {neff(w):>10.0f} {neff(w[m5]):>14.0f} {neff(w[m6]):>14.0f}", flush=True)
    print("\n(alpha=0 = current proposal.  sigma should stay ~constant = unbiased; Neff(|p|>500/600) is the FSI-pion corner.)")


if __name__ == "__main__":
    main()
