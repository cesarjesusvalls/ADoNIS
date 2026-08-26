"""One device-resident objective, shared by both minimisers (Gauss-Newton and MIGRAD).

Built from one traced function per sample; per-sample results are summed on the host rather than
fused into a single XLA program, and never captured as constants -- data and weights are arguments.
Dial-axis batches are all width B (zero-padded on the last block) so every dispatch shares one
compiled program.  `BinSpec._device()` caches its gather indices and must be warmed once per
sample, in the constructor, before anything is jitted -- caching inside a trace raises
`UnexpectedTracerError` on the next call.
"""
from __future__ import annotations

import numpy as np

from adonis.stats.gaussian import gn_covariance

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from adonis.analysis import knobs as _K


VJP_PASSES = 2.0
HVP_PASSES = 3.0


class FitKernel:
    """The fused objective over `subset` dials of `eng`, plus its derivative objects.

    Parameters
    ----------
    eng        MultiEngine exposing .samples[i].model_blocks_jax, .ds, .th0, .prior, .pnames
    subset     indices of the fitted dials, in the order the fit uses them
    mask       optional per-bin boolean over the full stacked binning; bins are dropped by zeroing
               their whitening weight.  Can only REMOVE bins: a bin with sigma = inf stays dead
               regardless, and `n_live` reports how many bins survived.
    jac_batch  tangent columns per dispatch (default: all n in one).  Changes wall-clock and peak
               memory, never the result.
    """

    def __init__(self, eng, subset, mask=None, jac_batch=None, th_fixed=None):
        self.eng = eng
        self.subset = [int(k) for k in subset]
        self.idx = np.asarray(self.subset, int)
        self.n = len(self.subset)
        self.pnames = [eng.pnames[k] for k in self.subset]
        self.th0 = np.asarray(eng.th0 if th_fixed is None else th_fixed, float)
        self.x0 = np.asarray(eng.th0, float)[self.idx].copy()
        self.pw = 1.0 / np.asarray(eng.prior, float)[self.idx]
        self.nbin_s = [int(sum(d["nbin"] for d in s.ds)) for s in eng.samples]
        self.nbin = int(sum(self.nbin_s))
        self.row0_s = np.cumsum([0] + self.nbin_s)
        self.mask = np.ones(self.nbin, bool) if mask is None else np.asarray(mask, bool).copy()
        if len(self.mask) != self.nbin:
            raise ValueError(f"mask has {len(self.mask)} bins, engine has {self.nbin}")

        self.B = self.n if not jac_batch else min(int(jac_batch), self.n)
        self.nblk = int(np.ceil(self.n / self.B))

        _idx = jnp.asarray(self.idx)
        self._thf = jnp.asarray(self.th0)

        def _mk_f(s):
            def f(x, thf):
                return jnp.concatenate(list(s.model_blocks_jax(thf.at[_idx].set(x))))
            return f
        self._f = [_mk_f(s) for s in eng.samples]

        for f in self._f:
            np.asarray(f(jnp.asarray(self.x0), self._thf))

        def _mk(f):
            def r(x, thf, D, R):
                return (f(x, thf) - D) * R

            def c(x, thf, D, R):
                rr = r(x, thf, D, R)
                return jnp.sum(rr * rr)

            def jc(x, thf, D, R, T):
                return jax.vmap(lambda t: jax.jvp(lambda u: r(u, thf, D, R), (x,), (t,))[1])(T)

            def hs(x, thf, D, R, V):
                return jax.vmap(lambda v: jax.jvp(
                    lambda u: jax.grad(c)(u, thf, D, R), (x,), (v,))[1])(V)

            return (jax.jit(r), jax.jit(c), jax.jit(jax.grad(c)), jax.jit(jc), jax.jit(f), jax.jit(hs),
                    jax.jit(jax.value_and_grad(c)))
        (self._j_r, self._j_c, self._j_g, self._j_J, self._j_m,
         self._j_H, self._j_vg) = (list(z) for z in zip(*[_mk(f) for f in self._f]))

        E = np.eye(self.n)
        self._T = []
        for lo in range(0, self.n, self.B):
            m = min(self.B, self.n - lo)
            blk = np.zeros((self.B, self.n))
            blk[:m] = E[lo:lo + m]
            self._T.append((jnp.asarray(blk), lo, m))

        self.refresh_data()
        self.reset_counts()

    def set_fixed(self, th_fixed):
        """Move the non-fitted dials (e.g. pin a scanned dial at a profile node); no recompilation,
        just a host-to-device copy.  The prior centre `x0` is left untouched."""
        self.th0 = np.asarray(th_fixed, float).copy()
        self._thf = jnp.asarray(self.th0)
        return self

    def refresh_data(self):
        """Re-read (data, sigma) from eng.ds -- call after set_closure_data() or a toy throw.

        Costs a host-to-device copy of a few hundred floats.  No recompilation: D and R are arguments of
        the jitted programs, not constants baked into them.
        """
        D, R, live = [], [], 0
        for i, s in enumerate(self.eng.samples):
            d_ = np.concatenate([d["data"] for d in s.ds])
            sg = np.concatenate([d["sigma"] for d in s.ds])
            mk = self.mask[self.row0_s[i]:self.row0_s[i + 1]]
            ok = mk & np.isfinite(sg) & (sg > 0)
            live += int(ok.sum())
            D.append(jnp.asarray(d_))
            R.append(jnp.asarray(np.where(ok, 1.0 / np.where(ok, sg, 1.0), 0.0)))
        self._D, self._R = D, R
        self.n_live = live
        return self

    def data_sigma(self):
        """Host copies of the stacked (data, sigma)."""
        return (np.concatenate([d["data"] for s in self.eng.samples for d in s.ds]),
                np.concatenate([d["sigma"] for s in self.eng.samples for d in s.ds]))

    def whiten(self):
        """The stacked whitening vector 1/sigma (0 on dead bins) -- the same w trf_fit builds."""
        return np.concatenate([np.asarray(r) for r in self._R])

    def reset_counts(self):
        self.counts = dict(chi2=0, vg=0, resid=0, grad=0, jac=0, model=0,
                           primal=0,
                           tangent=0,
                           tangent_eff=0,
                           vjp=0,
                           hvp=0,
                           hess=0)
        return self

    def visits(self):
        """Forward traversals of the event set: value-only calls plus value-and-gradient calls --
        the denominator for a "per model evaluation" figure comparable across callers that only
        need chi2 and callers that also need the gradient."""
        c = self.counts
        return int(c["chi2"] + c["vg"] + c["resid"] + c["model"] + c["jac"])

    def event_passes(self):
        """Passes over the resident event set: primal + tangent + VJP_PASSES*vjp + HVP_PASSES*hvp."""
        c = self.counts
        return float(c["primal"] + c["tangent"] + VJP_PASSES * c["vjp"]
                     + HVP_PASSES * c["hvp"])

    def model(self, x):
        """Binned model (unwhitened, prior-free) at x -- the device counterpart of eng.model()."""
        xj = jnp.asarray(np.asarray(x, float))
        out = np.concatenate([np.asarray(f(xj, self._thf)) for f in self._j_m])
        self.counts["model"] += 1
        self.counts["primal"] += 1
        return out

    def residuals(self, x):
        """(nbin + n,) -- whitened data residuals stacked with the prior residuals."""
        xj = jnp.asarray(np.asarray(x, float))
        r = np.concatenate([np.asarray(f(xj, self._thf, D, R))
                            for f, D, R in zip(self._j_r, self._D, self._R)])
        self.counts["resid"] += 1
        self.counts["primal"] += 1
        return np.concatenate([r, (np.asarray(x, float) - self.x0) * self.pw])

    def chi2(self, x):
        """Scalar objective: ||whitened data residual||^2 + ||prior residual||^2, computed inside the
        jit as sum(r*r) -- the same expression `residuals` returns squared, so the two paths cannot
        drift apart."""
        xj = jnp.asarray(np.asarray(x, float))
        c = float(sum(float(f(xj, self._thf, D, R))
                      for f, D, R in zip(self._j_c, self._D, self._R)))
        self.counts["chi2"] += 1
        self.counts["primal"] += 1
        dx = (np.asarray(x, float) - self.x0) * self.pw
        return c + float(dx @ dx)

    def grad(self, x):
        """(n,) gradient of chi2 by ONE reverse-mode VJP per sample -- O(1) in the dial count."""
        xj = jnp.asarray(np.asarray(x, float))
        g = sum(np.asarray(f(xj, self._thf, D, R))
                for f, D, R in zip(self._j_g, self._D, self._R))
        self.counts["grad"] += 1
        self.counts["vjp"] += 1
        return np.asarray(g) + 2.0 * (np.asarray(x, float) - self.x0) * self.pw ** 2

    def chi2_and_grad(self, x):
        """(chi2, grad) from ONE value_and_grad per sample.  For callers that always need both at the
        same point (e.g. HMC/NUTS); MIGRAD's line search calls `chi2` and `grad` separately instead."""
        xj = jnp.asarray(np.asarray(x, float))
        out = [f(xj, self._thf, D, R) for f, D, R in zip(self._j_vg, self._D, self._R)]
        self.counts["vg"] += 1
        self.counts["grad"] += 1
        self.counts["vjp"] += 1
        dx = (np.asarray(x, float) - self.x0) * self.pw
        c = float(sum(float(o[0]) for o in out)) + float(dx @ dx)
        g = np.asarray(sum(o[1] for o in out)) + 2.0 * dx * self.pw
        return c, g

    def jac_data(self, x):
        """(nbin, n) whitened data Jacobian.  Binning happens inside the jit, so only nbin*n floats
        cross the host/device boundary -- never a (n, n_events) per-event derivative array."""
        xj = jnp.asarray(np.asarray(x, float))
        out = np.empty((self.nbin, self.n))
        for i, (f, D, R) in enumerate(zip(self._j_J, self._D, self._R)):
            blocks = []
            for T, lo, m in self._T:
                blocks.append(np.asarray(f(xj, self._thf, D, R, T))[:m])
            out[self.row0_s[i]:self.row0_s[i + 1]] = np.vstack(blocks).T
        self.counts["jac"] += 1
        self.counts["primal"] += self.nblk
        self.counts["tangent"] += self.nblk * self.B
        self.counts["tangent_eff"] += self.n
        return out

    def jac(self, x):
        """(nbin + n, n) -- the data Jacobian stacked with the prior block diag(1/prior)."""
        return np.vstack([self.jac_data(x), np.diag(self.pw)])

    def hessian(self, x, batch=None):
        """(n, n) exact Hessian of chi2, not the Gauss-Newton approximation.  Computed via ~n
        forward-over-reverse HVPs.  The prior block is exactly 2*diag(1/prior^2) and needs no
        derivative work."""
        B = self.n if not batch else min(int(batch), self.n)
        xj = jnp.asarray(np.asarray(x, float))
        E = np.eye(self.n)
        H = np.zeros((self.n, self.n))
        nblk = 0
        for lo in range(0, self.n, B):
            m_ = min(B, self.n - lo)
            blk = np.zeros((B, self.n)); blk[:m_] = E[lo:lo + m_]
            Vt = jnp.asarray(blk)
            acc = sum(np.asarray(f(xj, self._thf, D, R, Vt))
                      for f, D, R in zip(self._j_H, self._D, self._R))
            H[lo:lo + m_] = np.asarray(acc)[:m_]
            nblk += 1
        self.counts["hess"] += 1
        self.counts["vjp"] += nblk
        self.counts["tangent"] += nblk * B
        H = 0.5 * (H + H.T)
        return H + 2.0 * np.diag(self.pw ** 2)

    def covariance_gn(self, J):
        """(J^T J)^-1 from a whitened jacobian -- the Gauss-Newton covariance, free once J exists."""
        return gn_covariance(J)

    def covariance_exact(self, x, batch=None):
        """(H/2)^-1 from the exact Hessian.  Since H = 2(J^T J + sum_b r_b d2r_b) for chi2 = ||r||^2,
        this reduces to the Gauss-Newton covariance (J^T J)^-1 when the residual term vanishes."""
        return np.linalg.pinv(0.5 * self.hessian(x, batch=batch), rcond=1e-12)

    def warmup(self, x=None, which=("model", "residuals", "chi2", "grad", "jac")):
        """Compile the jitted objects on the exact shapes the timed loop will use, then reset the
        counters.  Nothing after this call may trigger an XLA compile.  `which` lets a caller compile
        only the objects it needs.  Elapsed compile time is kept on `.compile_s`, separate from later
        timing.
        """
        import time as _t
        x = self.x0 if x is None else np.asarray(x, float)
        t0 = _t.perf_counter()
        for w in which:
            {"model": self.model, "residuals": self.residuals, "chi2": self.chi2,
             "grad": self.grad, "jac": self.jac_data, "hess": self.hessian,
             "chi2_and_grad": self.chi2_and_grad}[w](x)
        self.compile_s = _t.perf_counter() - t0
        return self.reset_counts()

    def bounds(self):
        """(lo, hi) per fitted dial from the PHYS_BOUND registry -- one feasible set for both arms."""
        lo = np.array([-np.inf if _K.phys_lo(p) is None else _K.phys_lo(p) for p in self.pnames], float)
        hi = np.array([np.inf if _K.phys_hi(p) is None else _K.phys_hi(p) for p in self.pnames], float)
        return lo, hi

    def describe(self):
        return (f"FitKernel: {self.n} dials, {self.nbin} bins ({self.n_live} live), "
                f"{len(self.eng.samples)} samples, jac batch {self.B} x {self.nblk} dispatch"
                f"{'es' if self.nblk > 1 else ''}"
                + (f" ({self.nblk * self.B - self.n} padded tangent column"
                   f"{'s' if self.nblk * self.B - self.n != 1 else ''})" if self.nblk * self.B != self.n
                   else ""))


_PION_SLOT = ("bc", "sa", "ss_el", "ss", "si", "pi_hh", "pi_a", "sa_c", "ss_el_c", "ss_c", "si_c")
_NUC_SLOT = ("hh", "a", "iso", "finel", "inel", "swap")


def event_windows(n_events, chunk):
    """[(start, C)] covering range(n_events) with a common width C, last window shifted back to n-C.

    Every window shares one width so all windows hit the same compiled program.  Shifting the last
    window back makes it overlap its predecessor; `bank_chunk_plan` assigns each event to exactly
    one window despite the overlap.
    """
    C = int(min(chunk, n_events))
    nch = int(np.ceil(n_events / C))
    return [(int(min(i * C, n_events - C)), C) for i in range(nch)], C, nch


def bank_chunk_plan(JB, n_events, chunk):
    """Per-window slot tables for a bank's ragged FSI records.

    Returns (windows, C, nch, plan) where plan[w] = {"p": idx into the pion slots, "n": idx into the
    nucleon slots, "p_eidx": rebased, "n_eidx": rebased} with every array padded to a common length and
    padding pointing at the trash event C.
    """
    wins, C, nch = event_windows(n_events, chunk)
    pe = np.asarray(JB["f_p_eidx"]); ne = np.asarray(JB["f_n_eidx"])
    own = np.minimum(np.arange(n_events) // C, nch - 1)
    p_own, n_own = own[pe], own[ne]
    Lp = int(np.bincount(p_own, minlength=nch).max())
    Ln = int(np.bincount(n_own, minlength=nch).max())
    plan = []
    for w, (s, _) in enumerate(wins):
        ip = np.where(p_own == w)[0]
        inn = np.where(n_own == w)[0]
        pp = np.full(Lp, 0, np.int64); pe_l = np.full(Lp, C, np.int64)
        nn = np.full(Ln, 0, np.int64); ne_l = np.full(Ln, C, np.int64)
        pp[:len(ip)] = ip; pe_l[:len(ip)] = pe[ip] - s
        nn[:len(inn)] = inn; ne_l[:len(inn)] = ne[inn] - s
        plan.append(dict(p=pp, p_eidx=pe_l, n=nn, n_eidx=ne_l))
    return wins, C, nch, plan, Lp, Ln


def bank_windows(JB, n_events, chunk, drop_source=False):
    """Materialise the per-window sub-banks once.  Returns (nch, C, [sub_bank...], owns).

    Each per-event array is padded with one trash row at index C; the ragged FSI reduction runs over
    C+1 events and `bank_weight_window` truncates back to C.  `owns[w] = (lo, hi)` is the half-open
    range of global event indices window w is responsible for -- windows overlap (see
    `event_windows`), so `owns`, not the window bounds, is what attributes an event to one window.
    `drop_source=True` clears the source bank in place once its arrays have been copied into windows.
    """
    wins, C, nch, plan, Lp, Ln = bank_chunk_plan(JB, n_events, chunk)
    pion = tuple("f_" + f for f in _PION_SLOT)
    nuc = tuple("f_" + f for f in _NUC_SLOT)
    per_event = tuple(k for k in JB if k.startswith("hv_") or k in ("p_struck", "channel", "w0"))
    subs, owns = [], []
    for w, (st, _) in enumerate(wins):
        pl = plan[w]
        B = {}
        for k in per_event:
            v = JB[k][st:st + C]
            B[k] = jnp.concatenate([v, jnp.zeros((1,) + v.shape[1:], v.dtype)], axis=0)
        for k in pion:
            B[k] = jnp.take(JB[k], jnp.asarray(pl["p"]), axis=0)
        for k in nuc:
            B[k] = jnp.take(JB[k], jnp.asarray(pl["n"]), axis=0)
        B["f_p_eidx"] = jnp.asarray(pl["p_eidx"])
        B["f_n_eidx"] = jnp.asarray(pl["n_eidx"])
        subs.append(B)
    owns = [(int(w * C), int(min((w + 1) * C, n_events))) for w in range(nch)]
    if drop_source:
        JB.clear()
    return nch, C, subs, owns


def bank_weight_window(BR, Bw, knobs, grids, C):
    """bank_weight on a pre-built window sub-bank: reduces over C+1 events and drops the trash event
    (see bank_windows).  Equivalent to the unchunked call on a smaller bank.
    """
    return BR.bank_weight(Bw, knobs, grids)[:C]
