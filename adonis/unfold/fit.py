"""The unfolding fit: 10 template parameters + 28 physics knobs against 60 reco bins.

    mu_i(theta, c) = sum_j A_ij c_j  +  B_i(theta)

The two halves are differentiated in completely different ways, because they ARE completely different:

  * the signal term is exactly linear in c, so its Jacobian is the response matrix A itself.  No autodiff,
    no finite differences, no approximation -- d mu_i / d c_j = A_ij, exactly, at every point.
  * the background term needs the per-event weight w(theta), so its Jacobian comes from one jvp per knob
    over the compact background bank -- the same machinery the Gate-I Jacobian uses.

Templates carry NO prior: an unfolded spectrum that has been pulled toward the generator's prediction is
not a measurement.  The knobs carry the Gate-I priors (20% multiplicative, 4 MeV on E_b), which is what
makes them a systematic rather than a second signal model.

Uncertainties are Poisson on the reco bins, frozen at the DATA, not recomputed from the current
prediction: a sigma that moves with the model biases the fit toward whichever direction inflates the
error, and at these occupancies (min 11.5 signal events per bin) that bias is not negligible.
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

    def __init__(self, inp, prior_scale=1.0, det_prior=0.05, flux_sigma=None, flux_corr=None):
        self.A = np.asarray(inp["A"], float)
        self.n_true_mc = np.asarray(inp["n_true"], float)     # generator truth per bin, the c = 1 reference
        self.bkg_bin = np.asarray(inp["bkg_bin"], np.int64)
        self.nflux = int(inp.get("nflux", FX.n_flux()))
        self.bkg_fbin = np.asarray(inp["bkg_fbin"], np.int64)
        self.bkg_flat = self.bkg_bin * self.nflux + self.bkg_fbin      # (reco, flux) -> one bincount
        # SOFT assignment stores a distribution over reco bins per background event instead of one
        # index.  The hard case is the one-hot version of the same matrix, so both go through _spread.
        self.bkg_M = inp.get("bkg_M", None)
        if isinstance(self.bkg_M, np.ndarray) and self.bkg_M.dtype == object:
            self.bkg_M = self.bkg_M.item()
        # PRIOR WIDTHS ARE ARGUMENTS.  These were literals -- 0.05 here, FX.SIGMA inside prior_chol --
        # so the only way to change the section's dominant systematic was to edit a module, and a saved
        # npz could not state which width produced it.  None keeps the module default.
        self.flux_sigma = FX.SIGMA if flux_sigma is None else float(flux_sigma)
        self.flux_corr = FX.CORR_LENGTH if flux_corr is None else float(flux_corr)
        self.flux_L = FX.prior_chol(self.flux_sigma, self.flux_corr)   # correlated prior, whitened below
        self.det_prior = float(det_prior)      # per-reco-bin detector normalisation, uncorrelated
        self.JB = BR.to_jax(inp["bkg_bank"])
        self.grids = BR.default_grids()
        self.nom = nominal_knobs()
        self.nreco, self.ntrue, _nf = self.A.shape
        self.th0 = np.asarray(K.theta_nominal(self.nom), float)
        self.prior = np.asarray(K.PRIOR, float) * float(prior_scale)
        self.npar = self.ntrue + K.NPAR
        self._jvp = None

    # ---- forward ------------------------------------------------------------------------------------ #
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

        The detector dial d_i scales the whole content of reco bin i -- signal and background together --
        because a detector normalisation uncertainty does not know what produced the event.  That makes
        it the only block that acts in RECO space; templates act in truth space and flux in true energy.
        """
        base = np.einsum("ijb,j,b->i", self.A, c, f) + self.background(th) @ f
        return base if det is None else det * base

    def x0(self):
        """Start at the generator's own prediction: everything at 1, knobs at nominal."""
        return np.ones(self.ntrue), np.ones(self.nflux), self.th0.copy(), np.ones(self.nreco)

    # ---- jacobian ----------------------------------------------------------------------------------- #
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
        Af = np.einsum("ijb,b->ij", self.A, f)                 # d mu_i / d c_j
        B = self.background(th)
        dF = np.einsum("ijb,j->ib", self.A, c) + B             # d mu_i / d f_b
        dT = (np.einsum("ibk,b->ik", self._bkg_jac(th), f)[:, knob_idx]
              if len(knob_idx) else np.zeros((self.nreco, 0)))
        return Af, dF, dT

    # ---- data --------------------------------------------------------------------------------------- #
    def asimov(self, c_true=None, f_true=None, th_true=None, det_true=None):
        """Expected reco spectrum at a stated truth, plus its Poisson sigma."""
        c = np.ones(self.ntrue) if c_true is None else np.asarray(c_true, float)
        f = np.ones(self.nflux) if f_true is None else np.asarray(f_true, float)
        th = self.th0 if th_true is None else np.asarray(th_true, float)
        det = None if det_true is None else np.asarray(det_true, float)
        d = self.model(c, f, th, det)
        return d, np.sqrt(np.maximum(d, 1e-9))

    # ---- which dials are worth floating --------------------------------------------------------------- #
    def dial_impact(self, sigma, th=None):
        """Per-knob impact on the prediction in units of the data error:

            impact_k = || (dB/dtheta_k) * prior_k / sigma ||_2

        how far a ONE-SIGMA-PRIOR move of knob k pushes the prediction, measured against the uncertainty
        that would notice.  Prior-weighted because the knobs are not commensurable: a 20% move of a rate
        normalisation and a 4 MeV move of E_b only compare once each is in units of its own range.
        """
        J = self._bkg_jac(self.th0 if th is None else th).sum(axis=1)      # flux at nominal = 1
        return np.linalg.norm(J * self.prior[None, :] / np.asarray(sigma)[:, None], axis=0)

    def select_dials(self, sigma, threshold=1.0, th=None, log=print):
        """The knobs worth floating: those whose prior-sized move shifts the prediction by at least
        `threshold` sigma.  Measured here, the 28 span four orders of magnitude -- kF_sf at 11.1 down to
        s_conv at 0.000 -- because they reweight only the background, and most describe physics this
        selection never sees.

        Note which way the error moves: dropping a knob can only SHRINK the template errors, so a cut is
        never conservative.  threshold=0.1 keeps 20 and costs nothing; threshold=1.0 keeps ~11 (a
        prior-sized move must be worth at least one sigma) and costs about 1% of the quoted error.
        """
        imp = self.dial_impact(sigma, th)
        idx = np.flatnonzero(imp >= threshold)
        if log:
            log(f"  dials: {len(idx)}/{K.NPAR} float (impact >= {threshold}): "
                f"{[K.PNAMES[k] for k in idx]}")
        return idx, imp

    # ---- fit ---------------------------------------------------------------------------------------- #
    def fit(self, data, sigma, knob_idx=None, free_flux=True, free_det=True, max_nfev=200, log=print):
        """Minimise chi2_data + priors over [c | f | theta_subset | d].

        FOUR blocks, four prior treatments -- which is the section in one function:

            c      templates, in TRUTH space, NO prior.  An unfolded spectrum pulled toward the
                   generator is not a measurement.
            f      flux, in TRUE ENERGY, CORRELATED 10% prior (whitened by its Cholesky factor).
            theta  cross-section knobs, independent Gate-I priors (20%, 4 MeV on E_b).
            d      detector, in RECO space, independent 5% priors -- one per reco bin.

        The detector block adds 60 parameters to a 60-bin fit, which looks under-determined and is not:
        every one of them carries its own prior, so each contributes a constraint alongside its
        parameter.  What it does do is inflate the template errors, because a per-bin normalisation
        freedom is exactly what the templates are trying to measure through.
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
                top.append(np.diag(base))                       # d mu_i / d d_i = base_i
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
        cov = gn_covariance(J)                                 # GN covariance; exact at an Asimov minimum
        c, f, th, det = unpack(r.x)
        chi2 = float(np.sum(((self.model(c, f, th, det) - data) / sigma) ** 2))
        err = np.sqrt(np.abs(np.diag(cov)))
        if log:
            log(f"  nfev={r.nfev} chi2_data={chi2:.3e} npar={len(r.x)}")
        return dict(x=r.x, c=c, f=f, th=th, det=det, knob_idx=knob_idx, cov=cov,
                    c_err=err[:nt], f_err=(err[o_f:o_f + nf] if free_flux else np.zeros(0)),
                    det_err=(err[o_d:o_d + nd] if free_det else np.zeros(0)),
                    chi2=chi2, ndof=self.nreco - nt - nf - nk, nfev=r.nfev, success=bool(r.success))
