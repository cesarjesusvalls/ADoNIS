"""The unfolding fit: template parameters and physics knobs against reco bins.

    mu_i(theta, c) = sum_j A_ij c_j  +  B_i(theta)

The signal term is exactly linear in c, so its Jacobian is A itself; the background term's
dependence on the physics knobs theta needs one jvp per knob over the compact background bank.

Templates carry no prior -- an unfolded spectrum pulled toward the generator's prediction would not
be a measurement.  The physics knobs carry independent priors, making them a systematic rather than
a second signal model.  Uncertainties are Poisson on the reco bins, frozen at the data rather than
recomputed from the current prediction.
"""
from __future__ import annotations

import numpy as np

from adonis.stats.gaussian import gn_covariance
from scipy.optimize import least_squares

from adonis.analysis import knobs as K
from adonis.reweight import bank_reweight as BR
from adonis.reweight.reweight_model import nominal_knobs
from adonis.unfold import flux as FX


class UnfoldEngine:
    """Everything the fit needs, resident: the response, the background bank, and the priors."""

    def __init__(self, inp, prior_scale=1.0, det_prior=0.05, flux_sigma=None, flux_corr=None,
                 knob_prior=None, flux_edges=None):
        self.A = np.asarray(inp["A"], float)
        self.n_true_mc = np.asarray(inp["n_true"], float)
        self.bkg_bin = np.asarray(inp["bkg_bin"], np.int64)
        self.flux_edges = flux_edges
        self.nflux = int(inp.get("nflux", FX.n_flux(flux_edges)))
        self.bkg_fbin = np.asarray(inp["bkg_fbin"], np.int64)
        self.bkg_flat = self.bkg_bin * self.nflux + self.bkg_fbin
        self.bkg_M = inp.get("bkg_M", None)
        if isinstance(self.bkg_M, np.ndarray) and self.bkg_M.dtype == object:
            self.bkg_M = self.bkg_M.item()
        self.flux_sigma = FX.SIGMA if flux_sigma is None else float(flux_sigma)
        self.flux_corr = FX.CORR_LENGTH if flux_corr is None else float(flux_corr)
        self.flux_L = FX.prior_chol(self.flux_sigma, self.flux_corr, flux_edges)
        self.det_prior = float(det_prior)
        self.JB = BR.to_jax(inp["bkg_bank"])
        self.grids = BR.default_grids()
        self.nom = nominal_knobs()
        self.nreco, self.ntrue, _nf = self.A.shape
        self.th0 = np.asarray(K.theta_nominal(self.nom), float)
        _kp = K.PRIOR if knob_prior is None else knob_prior
        self.prior = np.asarray(_kp, float) * float(prior_scale)
        self.npar = self.ntrue + K.NPAR
        self._jvp = None

    def background(self, th):
        """B_ib(theta): background rate per (reco bin, flux bin).  Kept resolved in the flux index
        because a flux parameter scales background too."""
        w = np.asarray(BR.bank_weight(self.JB, K.knobs_of(np.asarray(th), self.nom), self.grids))
        return self._spread(w)

    def _spread(self, w):
        """Per-event weights -> (nreco, nflux).  One-hot bincount, or the sparse fractions."""
        if self.bkg_M is not None:
            return np.asarray(self.bkg_M.T @ w).reshape(self.nreco, self.nflux)
        return np.bincount(self.bkg_flat, weights=w,
                           minlength=self.nreco * self.nflux).reshape(self.nreco, self.nflux)

    def model(self, c, f, th, det=None):
        """mu_i = d_i * [ sum_jb A_ijb c_j f_b + sum_b B_ib(theta) f_b ].

        The detector dial d_i scales all of reco bin i, signal and background together -- a detector
        normalisation does not know what produced the event.  It is the only block acting in reco
        space; templates act in truth space, flux in true energy.
        """
        base = np.einsum("ijb,j,b->i", self.A, c, f) + self.background(th) @ f
        return base if det is None else det * base

    def x0(self):
        """Start at the generator's own prediction: everything at 1, knobs at nominal."""
        return np.ones(self.ntrue), np.ones(self.nflux), self.th0.copy(), np.ones(self.nreco)

    def _bkg_jac(self, th):
        """d B_ib / d theta_k -- (nreco, nflux, nknob), one jvp per knob over the background bank."""
        import jax
        import jax.numpy as jnp
        if self._jvp is None:
            self._jvp = jax.jit(lambda t, tang, JB: jax.jvp(
                lambda u: BR.bank_weight(JB, K.knobs_of(u, self.nom), self.grids), (t,), (tang,))[1])
        thj = jnp.asarray(th)
        out = np.zeros((self.nreco, self.nflux, K.NPAR))
        for k in range(K.NPAR):
            g = np.asarray(self._jvp(thj, jnp.zeros(K.NPAR).at[k].set(1.0), self.JB))
            out[:, :, k] = self._spread(g)
        return out

    def jac_blocks(self, c, f, th, knob_idx):
        """(d mu/d c, d mu/d f, d mu/d theta).  The first two are EXACT -- the model is bilinear in
        (c, f), so no autodiff is involved on either; only the background's theta dependence needs it."""
        Af = np.einsum("ijb,b->ij", self.A, f)
        B = self.background(th)
        dF = np.einsum("ijb,j->ib", self.A, c) + B
        dT = (np.einsum("ibk,b->ik", self._bkg_jac(th), f)[:, knob_idx]
              if len(knob_idx) else np.zeros((self.nreco, 0)))
        return Af, dF, dT

    def asimov(self, c_true=None, f_true=None, th_true=None, det_true=None):
        """Expected reco spectrum at a stated truth, plus its Poisson sigma."""
        c = np.ones(self.ntrue) if c_true is None else np.asarray(c_true, float)
        f = np.ones(self.nflux) if f_true is None else np.asarray(f_true, float)
        th = self.th0 if th_true is None else np.asarray(th_true, float)
        det = None if det_true is None else np.asarray(det_true, float)
        d = self.model(c, f, th, det)
        return d, np.sqrt(np.maximum(d, 1e-9))

    def dial_impact(self, sigma, th=None):
        """Per-knob impact on the prediction, in units of the data error:

            impact_k = || (dB/dtheta_k) * prior_k / sigma ||_2

        How far a one-sigma-prior move of knob k pushes the prediction, measured against the
        uncertainty that would notice.  Prior-weighted because the knobs are not commensurable --
        e.g. a fractional rate normalisation and an absolute energy shift only compare once each is
        in units of its own prior width.
        """
        J = self._bkg_jac(self.th0 if th is None else th).sum(axis=1)
        return np.linalg.norm(J * self.prior[None, :] / np.asarray(sigma)[:, None], axis=0)

    def select_dials(self, sigma, threshold=1.0, th=None, log=print):
        """The knobs worth floating: those whose prior-sized move shifts the prediction by at least
        `threshold` sigma.  Dropping a knob can only shrink the reported template errors, never
        inflate them, so a threshold cut is never conservative.
        """
        imp = self.dial_impact(sigma, th)
        idx = np.flatnonzero(imp >= threshold)
        if log:
            log(f"  dials: {len(idx)}/{K.NPAR} float (impact >= {threshold}): "
                f"{[K.PNAMES[k] for k in idx]}")
        return idx, imp

    def fit(self, data, sigma, knob_idx=None, free_flux=True, free_det=True, max_nfev=200, log=print):
        """Minimise chi2_data + priors over [c | f | theta_subset | d].

        Four blocks, four prior treatments:

            c      templates, in truth space, NO prior -- an unfolded spectrum pulled toward the
                   generator would not be a measurement.
            f      flux, in true energy, correlated prior (whitened by its Cholesky factor).
            theta  physics knobs (the `knob_idx` subset), independent priors.
            d      detector, in reco space, independent per-reco-bin priors.

        The detector block adds one parameter per reco bin; each carries its own prior, so the fit
        is not under-determined, but the added normalisation freedom does inflate the template
        errors.
        """
        knob_idx = np.arange(K.NPAR) if knob_idx is None else np.asarray(knob_idx, int)
        nk = len(knob_idx)
        nf = self.nflux if free_flux else 0
        nd = self.nreco if free_det else 0
        nt = self.ntrue
        pri = self.prior[knob_idx]
        Linv = np.linalg.inv(self.flux_L) if free_flux else None
        o_f, o_k, o_d = nt, nt + nf, nt + nf + nk

        def unpack(p):
            c = p[:nt]
            f = p[o_f:o_f + nf] if free_flux else np.ones(self.nflux)
            th = self.th0.copy()
            if nk:
                th[knob_idx] = p[o_k:o_k + nk]
            det = p[o_d:o_d + nd] if free_det else None
            return c, f, th, det

        def resid(p):
            c, f, th, det = unpack(p)
            r = [(self.model(c, f, th, det) - data) / sigma]
            if free_flux:
                r.append(Linv @ (f - 1.0))
            if nk:
                r.append((p[o_k:o_k + nk] - self.th0[knob_idx]) / pri)
            if free_det:
                r.append((p[o_d:o_d + nd] - 1.0) / self.det_prior)
            return np.concatenate(r)

        def jacf(p):
            c, f, th, det = unpack(p)
            Ac, Af, At = self.jac_blocks(c, f, th, knob_idx)
            base = np.einsum("ijb,j,b->i", self.A, c, f) + self.background(th) @ f
            sc = np.ones(self.nreco) if det is None else det
            top = [Ac * sc[:, None]] + ([Af * sc[:, None]] if free_flux else []) + [At * sc[:, None]]
            if free_det:
                top.append(np.diag(base))
            rows = [np.hstack(top) / sigma[:, None]]
            npar = nt + nf + nk + nd
            if free_flux:
                B = np.zeros((self.nflux, npar)); B[:, o_f:o_f + nf] = Linv; rows.append(B)
            if nk:
                B = np.zeros((nk, npar)); B[:, o_k:o_k + nk] = np.diag(1.0 / pri); rows.append(B)
            if free_det:
                B = np.zeros((nd, npar)); B[:, o_d:o_d + nd] = np.eye(nd) / self.det_prior; rows.append(B)
            return np.vstack(rows)

        lo = np.concatenate([np.zeros(nt), np.zeros(nf),
                             [K.phys_lo(K.PNAMES[k]) if K.phys_lo(K.PNAMES[k]) is not None else -np.inf
                              for k in knob_idx], np.zeros(nd)])
        hi = np.concatenate([np.full(nt + nf, np.inf),
                             [K.phys_hi(K.PNAMES[k]) if K.phys_hi(K.PNAMES[k]) is not None else np.inf
                              for k in knob_idx], np.full(nd, np.inf)])
        p0 = np.clip(np.concatenate([np.ones(nt + nf), self.th0[knob_idx], np.ones(nd)]),
                     lo + 1e-12, hi - 1e-12)

        r = least_squares(resid, p0, jac=jacf, bounds=(lo, hi), method="trf",
                          xtol=1e-14, ftol=1e-14, gtol=1e-10, max_nfev=max_nfev)
        J = jacf(r.x)
        cov = gn_covariance(J)
        c, f, th, det = unpack(r.x)
        chi2 = float(np.sum(((self.model(c, f, th, det) - data) / sigma) ** 2))
        err = np.sqrt(np.abs(np.diag(cov)))
        if log:
            log(f"  nfev={r.nfev} chi2_data={chi2:.3e} npar={len(r.x)}")
        return dict(x=r.x, c=c, f=f, th=th, det=det, knob_idx=knob_idx, cov=cov,
                    c_err=err[:nt], f_err=(err[o_f:o_f + nf] if free_flux else np.zeros(0)),
                    det_err=(err[o_d:o_d + nd] if free_det else np.zeros(0)),
                    chi2=chi2, ndof=self.nreco - nt - nf - nk, nfev=r.nfev, success=bool(r.success))
