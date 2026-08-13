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
from scipy.optimize import least_squares

from adonis.analysis import knobs as K
from adonis.reweight import bank_reweight as BR
from adonis.reweight.reweight_model import nominal_knobs


class UnfoldEngine:
    """Everything the fit needs, resident: the response, the background bank, and the priors."""

    def __init__(self, inp, prior_scale=1.0):
        self.A = np.asarray(inp["A"], float)
        self.n_true_mc = np.asarray(inp["n_true"], float)     # generator truth per bin, the c = 1 reference
        self.bkg_bin = np.asarray(inp["bkg_bin"], np.int64)
        self.JB = BR.to_jax(inp["bkg_bank"])
        self.grids = BR.default_grids()
        self.nom = nominal_knobs()
        self.nreco, self.ntrue = self.A.shape
        self.th0 = np.asarray(K.theta_nominal(self.nom), float)
        self.prior = np.asarray(K.PRIOR, float) * float(prior_scale)
        self.npar = self.ntrue + K.NPAR
        self._jvp = None

    # ---- forward ------------------------------------------------------------------------------------ #
    def background(self, th):
        w = np.asarray(BR.bank_weight(self.JB, K.knobs_of(np.asarray(th), self.nom), self.grids))
        return np.bincount(self.bkg_bin, weights=w, minlength=self.nreco)

    def model(self, x):
        c, th = self.split(x)
        return self.A @ c + self.background(th)

    def split(self, x):
        x = np.asarray(x, float)
        return x[:self.ntrue], x[self.ntrue:]

    def join(self, c, th):
        return np.concatenate([np.asarray(c, float), np.asarray(th, float)])

    def x0(self):
        """Start at the generator's own prediction: templates at 1, knobs at nominal."""
        return self.join(np.ones(self.ntrue), self.th0)

    # ---- jacobian ----------------------------------------------------------------------------------- #
    def _bkg_jac(self, th):
        """d B_i / d theta_k, one jvp per knob over the background bank."""
        import jax
        import jax.numpy as jnp
        if self._jvp is None:
            self._jvp = jax.jit(lambda t, tang, JB: jax.jvp(
                lambda u: BR.bank_weight(JB, K.knobs_of(u, self.nom), self.grids), (t,), (tang,))[1])
        th = jnp.asarray(th)
        out = np.zeros((self.nreco, K.NPAR))
        for k in range(K.NPAR):
            g = np.asarray(self._jvp(th, jnp.zeros(K.NPAR).at[k].set(1.0), self.JB))
            out[:, k] = np.bincount(self.bkg_bin, weights=g, minlength=self.nreco)
        return out

    def jac(self, x):
        _c, th = self.split(x)
        return np.hstack([self.A, self._bkg_jac(th)])          # exact in c, jvp in theta

    # ---- data --------------------------------------------------------------------------------------- #
    def asimov(self, c_true=None, th_true=None):
        """Expected reco spectrum at a stated truth, plus its Poisson sigma."""
        c = np.ones(self.ntrue) if c_true is None else np.asarray(c_true, float)
        th = self.th0 if th_true is None else np.asarray(th_true, float)
        d = self.A @ c + self.background(th)
        return d, np.sqrt(np.maximum(d, 1e-9))

    # ---- fit ---------------------------------------------------------------------------------------- #
    def fit(self, data, sigma, free_knobs=True, max_nfev=200, log=print):
        """Minimise chi2_data + prior.  Returns theta-hat, the covariance, and diagnostics.

        Residuals are stacked [data ; prior] so a linear least-squares solver sees the MAP problem
        directly; the prior block has a constant Jacobian, which is why the covariance below is exact
        rather than an estimate.
        """
        nk = K.NPAR if free_knobs else 0
        npar = self.ntrue + nk

        def resid(p):
            x = self.join(p[:self.ntrue], p[self.ntrue:] if free_knobs else self.th0)
            r = (self.model(x) - data) / sigma
            if not free_knobs:
                return r
            return np.concatenate([r, (p[self.ntrue:] - self.th0) / self.prior])

        def jacf(p):
            x = self.join(p[:self.ntrue], p[self.ntrue:] if free_knobs else self.th0)
            J = self.jac(x)[:, :self.ntrue + nk] / sigma[:, None]
            if not free_knobs:
                return J
            P = np.hstack([np.zeros((K.NPAR, self.ntrue)), np.diag(1.0 / self.prior)])
            return np.vstack([J, P])

        # Templates are scale factors on a rate: negative is unphysical, and the boundary is real.
        lo = np.concatenate([np.zeros(self.ntrue), [K.phys_lo(n) or -np.inf for n in K.PNAMES][:nk]])
        hi = np.concatenate([np.full(self.ntrue, np.inf), [K.phys_hi(n) or np.inf for n in K.PNAMES][:nk]])
        p0 = np.concatenate([np.ones(self.ntrue), self.th0[:nk]])
        p0 = np.clip(p0, lo + 1e-12, hi - 1e-12)

        r = least_squares(resid, p0, jac=jacf, bounds=(lo, hi), method="trf",
                          xtol=1e-14, ftol=1e-14, gtol=1e-10, max_nfev=max_nfev)
        J = jacf(r.x)
        cov = np.linalg.pinv(J.T @ J, rcond=1e-12)             # GN covariance; exact at an Asimov minimum
        chi2 = float(np.sum(((self.model(self.join(r.x[:self.ntrue],
                                                   r.x[self.ntrue:] if free_knobs else self.th0))
                              - data) / sigma) ** 2))
        if log:
            log(f"  nfev={r.nfev} status={r.status} chi2_data={chi2:.4g} "
                f"ndof={self.nreco - npar} |grad|={np.max(np.abs(r.grad)):.2e}")
        return dict(x=r.x, c=r.x[:self.ntrue], th=(r.x[self.ntrue:] if free_knobs else self.th0),
                    cov=cov, c_err=np.sqrt(np.diag(cov)[:self.ntrue]), chi2=chi2,
                    ndof=self.nreco - npar, nfev=r.nfev, success=bool(r.success))
