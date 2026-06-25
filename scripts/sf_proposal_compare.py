"""STANDALONE: compare several FROZEN spectral-function proposals by N_eff in the high-|p_struck| corner
(FSI-pion / high-delta_pT tail), all UNBIASED (sigma fixed) and differentiability-safe.

Proposals (q', frozen; weight *= q_cur/q'):
  current  : (|p|,E) ~ |p|^2 S                                  (production default)
  pmix(a)  : |p| from (1-a)|p|^2 n_p + a|p|^2-flat ; E ~ S(E|p)  (|p|-tail mixture)
  temper(g): |p| ~ |p|^2 n_p^g  (g<1 flattens the |p| tail)   ; E ~ S(E|p)
  2dmix(a) : (1-a)*current + a*[|p|^2-flat x E-uniform]         (broadens |p| AND removal-E)

Run: python -u scripts/sf_proposal_compare.py [N=200000]
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
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200000


def _boost(p4, beta):
    b2 = np.sum(beta ** 2, axis=1, keepdims=True); g = 1 / np.sqrt(np.clip(1 - b2, 1e-15, None))
    bp = np.sum(beta * p4[:, 1:], axis=1, keepdims=True)
    E = g * (p4[:, :1] + bp)
    fac = (g - 1) * np.where(b2 > 1e-15, bp / np.clip(b2, 1e-15, None), 0.0) + g * p4[:, :1]
    return np.concatenate([E, p4[:, 1:] + fac * beta], axis=1)


def _cdf(d):
    c = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]))]); return c / c[-1]


def _inv(cdf, grid, u):
    j = np.clip(np.searchsorted(cdf, u), 1, len(grid) - 1)
    f = np.clip((u - cdf[j - 1]) / (cdf[j] - cdf[j - 1] + 1e-30), 0, 1)
    return grid[j - 1] + f * (grid[j] - grid[j - 1])


class SFgrids:
    def __init__(self, sf):
        self.sf = sf
        s = SpectralImportanceSampler(sf); self.pf = s.mom; self.Ef = s.energy
        self.p_cdf = s.p_cdf; self.e_cdf = s.e_cdf
        PP, EE = np.meshgrid(self.pf, self.Ef, indexing="ij")
        self.S = np.clip(sf.batch(PP.ravel(), EE.ravel()).reshape(len(self.pf), len(self.Ef)), 0, None)
        dE = self.Ef[1] - self.Ef[0]
        self.n_p = self.S.sum(1) * dE                              # int S dE
        self.Zcur = float(np.sum(self.pf[:, None] ** 2 * self.S) * dE * (self.pf[1] - self.pf[0]))  # ~ int |p|^2 S
        self.Zp2 = float(np.sum(self.pf ** 2) * (self.pf[1] - self.pf[0]))
        self.dEtot = self.Ef[-1] - self.Ef[0]

    def E_given_p(self, pmag, ue):
        bj = np.clip(np.searchsorted(self.pf, pmag), 1, len(self.pf) - 1)
        bf = (pmag - self.pf[bj - 1]) / (self.pf[bj] - self.pf[bj - 1] + 1e-30)
        lo = _inv_rows(self.e_cdf[bj - 1], self.Ef, ue); hi = _inv_rows(self.e_cdf[bj], self.Ef, ue)
        return (1 - bf) * lo + bf * hi


def _inv_rows(rows, grid, u):
    n = len(u); ei = np.clip(np.array([np.searchsorted(rows[k], u[k]) for k in range(n)]), 1, len(grid) - 1)
    d0 = rows[np.arange(n), ei - 1]; d1 = rows[np.arange(n), ei]
    f = np.clip((u - d0) / (d1 - d0 + 1e-30), 0, 1)
    return grid[ei - 1] + f * (grid[ei] - grid[ei - 1])


def propose(g: SFgrids, spec, n, rng):
    """Return pvec(n,3), E_rm(n), ratio(n)=q_cur/q' for the unbiased weight."""
    mode = spec[0]; pf = g.pf
    if mode == "current":
        s = SpectralImportanceSampler(g.sf); pvec, E = s.sample(n, rng); return pvec, E, np.ones(n)
    if mode == "temper":
        gam = spec[1]
        dens = pf ** 2 * np.power(np.clip(g.n_p, 1e-30, None), gam); cdf = _cdf(dens)
        pmag = _inv(cdf, pf, rng.random(n))
        E = g.E_given_p(pmag, rng.random(n))
        # ratio = q_cur/q' = (|p|^2 n_p/Zc)/(|p|^2 n_p^g/Zg) = n_p^(1-g) Zg/Zc
        npm = np.maximum(np.interp(pmag, pf, g.n_p), 1e-30)
        Zc = float(np.sum(pf ** 2 * g.n_p)); Zg = float(np.sum(dens))
        ratio = np.power(npm, 1 - gam) * Zg / Zc
        ct = 2 * rng.random(n) - 1; st = np.sqrt(np.clip(1 - ct ** 2, 0, None)); ph = _TWO_PI * rng.random(n)
        return np.stack([pmag * st * np.cos(ph), pmag * st * np.sin(ph), pmag * ct], 1), E, ratio
    if mode in ("pmix", "2dmix"):
        a = spec[1]; tail = rng.random(n) < a
        # |p|: current marginal vs |p|^2-flat
        p_cur = _inv(g.p_cdf, pf, rng.random(n))
        p3l, p3h = pf[0] ** 3, pf[-1] ** 3; p_tail = (p3l + rng.random(n) * (p3h - p3l)) ** (1 / 3)
        pmag = np.where(tail, p_tail, p_cur)
        if mode == "pmix":
            E = g.E_given_p(pmag, rng.random(n))
            npm = np.maximum(np.interp(pmag, pf, g.n_p), 1e-30)
            Zc = float(np.sum(pf ** 2 * g.n_p)); Zt = float(np.sum(pf ** 2))
            qcur = pmag ** 2 * npm / Zc; qp = (1 - a) * qcur + a * pmag ** 2 / Zt
            ratio = qcur / qp
        else:  # 2dmix: tail E uniform over [Ef0,EfN]
            E_cur = g.E_given_p(pmag, rng.random(n))
            E_uni = g.Ef[0] + rng.random(n) * g.dEtot
            E = np.where(tail, E_uni, E_cur)
            S = np.clip(np.asarray(g.sf.batch(pmag, E)), 0, None)
            qcur = pmag ** 2 * S / g.Zcur; qtail = pmag ** 2 / (g.Zp2 * g.dEtot)
            ratio = qcur / ((1 - a) * qcur + a * qtail + 1e-300)
        ct = 2 * rng.random(n) - 1; st = np.sqrt(np.clip(1 - ct ** 2, 0, None)); ph = _TWO_PI * rng.random(n)
        return np.stack([pmag * st * np.cos(ph), pmag * st * np.sin(ph), pmag * ct], 1), E, ratio
    raise ValueError(mode)


def run(g, spec, seed=0):
    rng = np.random.default_rng(seed); u = rng.random((N, 7))
    fx = T2KFlux(); E_GeV, J_beam = fx.sample_beam(u[:, 4], fx.seed_min_GeV()); Enu = E_GeV * 1000.0
    k_nu = np.stack([Enu, np.zeros(N), np.zeros(N), Enu], 1)
    pvec, E_rm, ratio = propose(g, spec, N, rng)
    p_struck = np.concatenate([(_MN - E_rm)[:, None], pvec], 1)
    P = k_nu + p_struck; s = P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, 1); sqrts = np.sqrt(np.clip(s, 1e-9, None))
    s2, s3 = M_MU ** 2, M_P ** 2
    E1 = sqrts / 2 * (1 + s2 / s - s3 / s); E2 = sqrts / 2 * (1 + s3 / s - s2 / s)
    lam = np.sqrt(np.clip((s - s2 - s3) ** 2 - 4 * s2 * s3, 0, None)); pcm = lam / (2 * sqrts)
    cts = 2 * u[:, 5] - 1; sts = np.sqrt(np.clip(1 - cts ** 2, 0, None)); php = _TWO_PI * u[:, 6]
    dirn = np.stack([sts * np.cos(php), sts * np.sin(php), cts], 1); beta = P[:, 1:] / P[:, 0:1]
    k_mu = _boost(np.concatenate([E1[:, None], pcm[:, None] * dirn], 1), beta)
    p_out = _boost(np.concatenate([E2[:, None], -pcm[:, None] * dirn], 1), beta)
    J_2body = 2.0 * _TWO_PI * pcm / (sqrts * 16 * np.pi ** 2)
    mom_s = np.linalg.norm(pvec, axis=1)
    det_e = Enu ** 2 + mom_s ** 2 + 2 * pvec[:, 2] * Enu + (M_MU + M_P) ** 2
    emax = np.minimum(np.minimum(_MN + Enu - np.sqrt(np.clip(det_e, 0, None)), _MN - mom_s), 400.0)
    valid = (s > (M_MU + M_P) ** 2) & (lam > 0) & (E_rm > 2.5) & (E_rm < emax)
    me = np.asarray(me_cross_section(jnp.asarray(k_nu), jnp.asarray(k_mu), jnp.asarray(p_struck),
                                    jnp.asarray(p_out), spin_avg=0.5, had_mass=MASS_PDG_NEUTRON)["me_xsec"])
    pmu = np.linalg.norm(k_mu[:, 1:], axis=1); cz = k_mu[:, 3] / np.clip(pmu, 1e-9, None)
    muacc = (pmu > 250.0) & (pmu < 7000.0) & (cz > np.cos(np.deg2rad(70.0)))   # production muon cut
    w = np.where(valid & muacc & np.isfinite(me * ratio), me * N_NEUTRON * J_2body * J_beam * ratio, 0.0)
    return w, mom_s


def neff(w):
    s2 = (w ** 2).sum(); return (w.sum() ** 2 / s2) if s2 > 0 else 0.0


def main():
    sf = SpectralFunction("data/Spectral_Functions/pke12n_tot.data"); g = SFgrids(sf)
    print(f"SF proposal comparison (N={N}/run, T2K nu+12C QE)\n", flush=True)
    print(f"{'proposal':>14} {'sigma[nb]':>12} {'Neff_all':>9} {'Neff>500':>9} {'Neff>600':>9} {'Neff>700':>9}", flush=True)
    specs = [("current",), ("pmix", 0.2), ("pmix", 0.4)]
    for spec in specs:
        w, pp = run(g, spec, seed=0)
        lab = spec[0] + (f"({spec[1]})" if len(spec) > 1 else "")
        print(f"{lab:>14} {w.sum()/N:>12.4e} {neff(w):>9.0f} {neff(w[pp>500]):>9.0f} "
              f"{neff(w[pp>600]):>9.0f} {neff(w[pp>700]):>9.0f}", flush=True)
    print("\nsigma ~const => unbiased; higher Neff>X = better-sampled FSI-pion/high-dpT corner.")


if __name__ == "__main__":
    main()
