"""ONE device-resident objective, consumed by BOTH minimisers.

WHY THIS EXISTS.  Until now Gauss-Newton and MIGRAD reached the same physics through different code:

  * GN   ->  eng.model / eng.jac.  Per-event weights (and, for the Jacobian, per-event DERIVATIVES --
             `(B, n_events)`, ~77 MB per sample per iteration at B=16, N=600k) are pulled back to the
             host and binned there with np.bincount.
  * MIGRAD -> eng.chi2_fn / eng.chi2_grad_fn.  Fully device-resident; only a scalar crosses the bus.

Measured consequence at 60k/sample: 106 ms vs 14.9 ms for the SAME model evaluation.  A timing
comparison built on that is measuring the two implementations, not the two algorithms -- and it moved by
3.6x when a single environment variable (S4_JAX_BIN) was flipped, which is the proof that it was never
measuring an algorithm at all.  The existing device-binning path is no fix: it bins each dial's
derivative for each dataset in a PYTHON LOOP (~170 tiny dispatches, each with its own host sync), and
measured 944 ms against the host path's 260 ms.

The defect is that there was no fused `theta -> bins` primitive with the binning INSIDE the jit.  This
module is that primitive, and every derivative object is built from the one function:

    f_s(x)  =  concat(sample s's binned model blocks)      x = the FITTED dials only

  residuals(x)   (nbin + n,)      whitened data residuals stacked with the prior block
  chi2(x)        scalar           sum of squares of the above
  grad(x)        (n,)             ONE reverse-mode VJP per sample
  jac(x)         (nbin + n, n)    vmapped forward JVP, BINNED ON DEVICE -> nbin*n floats come back,
                                  never per-event arrays

Both minimisers then differ in exactly one thing: which of these they ask for.

DESIGN NOTES, each of them a bug that was actually paid for:

  PER-SAMPLE JITS, SUMMED ON THE HOST.  Jitting the concatenation of all ten samples fuses them into one
  XLA program whose live set is every sample's intermediates at once, and that OOMs an 11 GB turing card.
  Splitting the sum over bins changes no arithmetic and bounds the peak at the largest single sample.

  DATA AND WEIGHTS ARE ARGUMENTS, not captured constants, so a toy ensemble retargets with a
  host-to-device copy of a few hundred floats instead of a fresh XLA compile per toy.

  ONE COMPILED TANGENT SHAPE.  The Jacobian's dial batches are ALL of width B, zero-padded on the last
  block.  The old `S4_JAC_BATCH=16` against 17 dials gave two dispatches of unequal width -- two compiled
  programs, and one extra full primal pass that landed specifically on the n=17 point of a dial scan.
  Padding costs a tangent column; an extra dispatch costs a whole primal.  Both are counted (see
  `counts`), so nothing about the batching is hidden from the report.

  BIN CACHES ARE WARMED EAGERLY.  BinSpec._device() memoises its gather indices on first call; if that
  call happens inside a trace the cached arrays belong to THAT trace and the next transformation of the
  same spec dies with `UnexpectedTracerError`.  One eager pass per sample, in the constructor, before
  anything is jitted.

  NOTHING IS JITTED LAZILY.  Every jitted object is built in the constructor and compiled by `warmup()`.
  `eng.chi2_fn()` returned a fresh `jax.jit` on every call, which is how 65 of 72 measured seconds turned
  out to be XLA compiling inside the timed region.
"""
from __future__ import annotations

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from adonis.analysis import knobs as _K


# ---- event-pass accounting ------------------------------------------------------------------------ #
# The portable unit.  Wall-clock is one machine's realisation of these counts; the counts themselves are
# implementation- and hardware-independent, so they are what a reader can carry to their own setup.
# A "pass" is one traversal of the resident event set:
#   primal model evaluation   1 pass
#   one JVP tangent column    1 pass   (the primal it shares is counted once per DISPATCH, not per column)
#   one VJP                   2 passes (forward sweep + reverse sweep)
# The VJP factor is a convention, stated so it can be re-weighted: the raw counts are kept separately.
VJP_PASSES = 2.0
# One hessian-vector product is a forward sweep carried through a reverse sweep: the primal, the tangent,
# and the cotangent.  Counted as 3 passes, on the same stated-convention basis as VJP_PASSES.
HVP_PASSES = 3.0


class FitKernel:
    """The fused objective over `subset` dials of `eng`, plus its derivative objects.

    Parameters
    ----------
    eng      a MultiEngine (anything exposing .samples[i].model_blocks_jax, .ds, .th0, .prior, .pnames)
    subset   indices of the fitted dials, in the order the fit uses them
    mask     optional per-bin boolean over the FULL stacked binning.  Bins are dropped by setting their
             whitening weight to zero, exactly as trf_fit does.  A frozen mask can only ever REMOVE bins:
             a bin with sigma = inf stays dead however the mask votes, and `n_live` reports what survived
             so a caller freezing a mask across an event scan can assert it actually froze.
    jac_batch  tangent columns per dispatch (default: all n in one).  Changes wall-clock and peak memory,
             never the result; both the effective and the executed tangent counts are recorded.
    """

    def __init__(self, eng, subset, mask=None, jac_batch=None, th_fixed=None):
        self.eng = eng
        self.subset = [int(k) for k in subset]
        self.idx = np.asarray(self.subset, int)
        self.n = len(self.subset)
        self.pnames = [eng.pnames[k] for k in self.subset]
        # th_fixed holds the dials NOT in `subset` -- a profile node pins the scanned dial here while the
        # rest are re-minimised.  The PRIOR CENTRE stays at eng.th0 regardless: pinning a dial away from
        # nominal must not drag the prior along with it, or the node is minimising a different objective
        # than the fit it belongs to.
        self.th0 = np.asarray(eng.th0 if th_fixed is None else th_fixed, float)
        self.x0 = np.asarray(eng.th0, float)[self.idx].copy()
        self.pw = 1.0 / np.asarray(eng.prior, float)[self.idx]      # prior residual weights
        self.nbin_s = [int(sum(d["nbin"] for d in s.ds)) for s in eng.samples]
        self.nbin = int(sum(self.nbin_s))
        self.row0_s = np.cumsum([0] + self.nbin_s)
        self.mask = np.ones(self.nbin, bool) if mask is None else np.asarray(mask, bool).copy()
        if len(self.mask) != self.nbin:
            raise ValueError(f"mask has {len(self.mask)} bins, engine has {self.nbin}")

        self.B = self.n if not jac_batch else min(int(jac_batch), self.n)
        self.nblk = int(np.ceil(self.n / self.B))

        _idx = jnp.asarray(self.idx)
        # THE FIXED DIALS ARE AN ARGUMENT, NOT A CONSTANT.  A profile scan re-minimises the same free
        # subset at node after node, changing only where the scanned dial is pinned.  If that vector were
        # closed over, every node would need its own FitKernel -- and every kernel loads ~50 CUBIN
        # modules that jax.clear_caches() does not unload, so a 13-node scan would exhaust the CUDA
        # context.  As an argument of fixed shape it changes for the price of a host-to-device copy.
        self._thf = jnp.asarray(self.th0)

        def _mk_f(s):
            def f(x, thf):
                return jnp.concatenate(list(s.model_blocks_jax(thf.at[_idx].set(x))))
            return f
        self._f = [_mk_f(s) for s in eng.samples]

        # EAGER BIN-CACHE WARM, before any jax.jit exists (see module docstring).
        for f in self._f:
            np.asarray(f(jnp.asarray(self.x0), self._thf))

        # ---- the jitted objects, all derived from the one f --------------------------------------- #
        def _mk(f):
            def r(x, thf, D, R):                  # whitened residual block for this sample
                return (f(x, thf) - D) * R

            def c(x, thf, D, R):                  # its chi2 contribution -- literally sum(r*r)
                rr = r(x, thf, D, R)
                return jnp.sum(rr * rr)

            def jc(x, thf, D, R, T):              # (B, nbin_s) tangent block: d r / d x . T
                return jax.vmap(lambda t: jax.jvp(lambda u: r(u, thf, D, R), (x,), (t,))[1])(T)

            def hs(x, thf, D, R, V):              # (B, n) Hessian rows: FORWARD-OVER-REVERSE
                # One HVP is a jvp through the gradient, i.e. ~one extra forward sweep on top of the
                # VJP -- so the EXACT hessian costs O(n) passes, against ~2n^2 objective evaluations for
                # a finite-difference HESSE.  That is the whole reason it is worth having.
                return jax.vmap(lambda v: jax.jvp(
                    lambda u: jax.grad(c)(u, thf, D, R), (x,), (v,))[1])(V)

            return (jax.jit(r), jax.jit(c), jax.jit(jax.grad(c)), jax.jit(jc), jax.jit(f), jax.jit(hs))
        (self._j_r, self._j_c, self._j_g, self._j_J, self._j_m,
         self._j_H) = (list(z) for z in zip(*[_mk(f) for f in self._f]))

        # ONE tangent shape for every block: (B, n), zero-padded on the last one.
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
        """Move the NON-fitted dials (e.g. pin a scanned dial at a profile node).  No recompilation:
        a host-to-device copy of NPAR floats.  The prior centre `x0` is untouched -- pinning a dial must
        not drag the prior along with it."""
        self.th0 = np.asarray(th_fixed, float).copy()
        self._thf = jnp.asarray(self.th0)
        return self

    # ---- data ------------------------------------------------------------------------------------- #
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
        """Host copies of the stacked (data, sigma), for reference checks against the old path."""
        return (np.concatenate([d["data"] for s in self.eng.samples for d in s.ds]),
                np.concatenate([d["sigma"] for s in self.eng.samples for d in s.ds]))

    def whiten(self):
        """The stacked whitening vector 1/sigma (0 on dead bins) -- the same w trf_fit builds."""
        return np.concatenate([np.asarray(r) for r in self._R])

    # ---- counters --------------------------------------------------------------------------------- #
    def reset_counts(self):
        self.counts = dict(chi2=0, resid=0, grad=0, jac=0, model=0,
                           primal=0,        # full forward passes over the events
                           tangent=0,       # JVP tangent columns actually EXECUTED (padding included)
                           tangent_eff=0,   # tangent columns that carried a real dial (padding excluded)
                           vjp=0,           # reverse-mode sweeps
                           hvp=0,           # hessian-vector products (forward-over-reverse)
                           hess=0)
        return self

    def event_passes(self):
        """Passes over the resident event set, per the convention at the top of this module."""
        c = self.counts
        return float(c["primal"] + c["tangent"] + VJP_PASSES * c["vjp"]
                     + HVP_PASSES * c["hvp"])

    # ---- the objective ---------------------------------------------------------------------------- #
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
        """Scalar objective: ||whitened data residual||^2 + ||prior residual||^2.

        Computed as sum(r*r) INSIDE the jit, i.e. the identical expression `residuals` squares -- so the
        value-only and residual paths can never drift apart.  One host sync per sample.
        """
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

    def jac_data(self, x):
        """(nbin, n) whitened data Jacobian.  Binning happens INSIDE the jit: what crosses the bus is
        nbin*n floats, never the (n, n_events) per-event derivative the old path shipped."""
        xj = jnp.asarray(np.asarray(x, float))
        out = np.empty((self.nbin, self.n))
        for i, (f, D, R) in enumerate(zip(self._j_J, self._D, self._R)):
            blocks = []
            for T, lo, m in self._T:
                blocks.append(np.asarray(f(xj, self._thf, D, R, T))[:m])   # (m, nbin_s)
            out[self.row0_s[i]:self.row0_s[i + 1]] = np.vstack(blocks).T
        self.counts["jac"] += 1
        self.counts["primal"] += self.nblk          # one primal per DISPATCH, shared by its B tangents
        self.counts["tangent"] += self.nblk * self.B
        self.counts["tangent_eff"] += self.n
        return out

    def jac(self, x):
        """(nbin + n, n) -- the data Jacobian stacked with the prior block diag(1/prior)."""
        return np.vstack([self.jac_data(x), np.diag(self.pw)])

    def hessian(self, x, batch=None):
        """(n, n) EXACT Hessian of chi2 -- not the Gauss-Newton approximation.

        WHY THIS MATTERS FOR THE LAPLACE/OCCAM FACTOR.  The fit's own covariance is (J^T W J)^-1, which
        drops the term sum_b r_b d2m_b: exact only in the small-residual limit.  The Occam correction is
        a log-det of precisely the matrix being approximated, so "the covariance is free" is a claim
        about the GAUSS-NEWTON matrix, not about the Hessian.  This gives the real one in ~n HVPs, where
        a finite-difference HESSE needs ~2n^2 objective evaluations.

        The prior block is exactly 2 diag(1/prior^2) -- quadratic, so no derivative work is needed.
        """
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
        self.counts["hvp"] += nblk * B
        H = 0.5 * (H + H.T)                      # symmetrise: the two sweeps round differently
        return H + 2.0 * np.diag(self.pw ** 2)

    def covariance_gn(self, J):
        """(J^T J)^-1 from a whitened jacobian -- the Gauss-Newton covariance, free once J exists."""
        return np.linalg.pinv(J.T @ J, rcond=1e-12)

    def covariance_exact(self, x, batch=None):
        """(H/2)^-1 from the exact Hessian.  For chi2 = ||r||^2, H = 2(J^T J + sum r d2r), so the
        covariance is (H/2)^-1 and reduces to (J^T J)^-1 exactly when the residual term vanishes."""
        return np.linalg.pinv(0.5 * self.hessian(x, batch=batch), rcond=1e-12)

    # ---- setup ------------------------------------------------------------------------------------ #
    def warmup(self, x=None, which=("model", "residuals", "chi2", "grad", "jac")):
        """Compile the jitted objects on the exact shapes the timed loop will use, then zero the
        counters.  Nothing after this call may trigger an XLA compile.

        Five programs per sample is a MINUTES-long XLA compile at production statistics; `which` exists
        so a caller that only needs one of them (a batch-invariance check, a value-only scan) does not
        pay for the other four.  The elapsed time is kept on `.compile_s` -- it is setup, it is never
        part of a timing, and it should be visible rather than folded into a first iteration.
        """
        import time as _t
        x = self.x0 if x is None else np.asarray(x, float)
        t0 = _t.perf_counter()
        for w in which:
            {"model": self.model, "residuals": self.residuals, "chi2": self.chi2,
             "grad": self.grad, "jac": self.jac_data, "hess": self.hessian}[w](x)
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


# ---- event chunking -------------------------------------------------------------------------------- #
# The model is a SUM over events, m_b = sum_{e in b} coef_e w_e(theta), so it decomposes exactly over
# event windows: m = sum_w m^(w).  That is what makes chunking the event axis free -- unlike splitting
# the DIAL axis, which costs one extra full primal pass per additional dispatch.  Peak memory becomes
# O(C) instead of O(n_events), so the ceiling that stops this kernel at 125k events/sample goes away.
#
# THE RAGGED PART.  A bank is not one array.  hv_*/p_struck/w0/channel are per-EVENT and slice directly,
# but the FSI records (f_bc, f_sa, ... with f_p_eidx / f_n_eidx) are per-INTERACTION SLOT with an event
# index, and pool_fsi_reweight reduces them over `n_events`.  So a window needs its slots SELECTED and
# its event indices REBASED, and the selection is ragged -- different windows hold different slot counts.
# Pad to a common length with a TRASH event index at C, run the reduction over C+1 events, and drop the
# last: padded slots then contribute to an event nobody reads, whatever their values are.  Same device as
# the trash bin in BinSpec.chunks.
_PION_SLOT = ("bc", "sa", "ss_el", "ss", "si", "pi_hh", "pi_a", "sa_c", "ss_el_c", "ss_c", "si_c")
_NUC_SLOT = ("hh", "a", "iso", "finel", "inel", "swap")


def event_windows(n_events, chunk):
    """[(start, C)] covering range(n_events) with a COMMON width, last window shifted back to n-C.

    Common width because every window must hit the same compiled program: a short final window is a
    second shape and a second XLA compile, which is exactly the cost chunking is meant to avoid.
    Shifting the last window back overlaps its predecessor, so each event is assigned to exactly ONE
    window by the tables built in `bank_chunk_plan` -- nothing is double counted.
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
    # own[e] = the window that owns event e; clamped so the shifted last window does not double count
    own = np.minimum(np.arange(n_events) // C, nch - 1)
    p_own, n_own = own[pe], own[ne]
    Lp = int(np.bincount(p_own, minlength=nch).max())
    Ln = int(np.bincount(n_own, minlength=nch).max())
    plan = []
    for w, (s, _) in enumerate(wins):
        ip = np.where(p_own == w)[0]
        inn = np.where(n_own == w)[0]
        pp = np.full(Lp, 0, np.int64); pe_l = np.full(Lp, C, np.int64)      # C == trash event
        nn = np.full(Ln, 0, np.int64); ne_l = np.full(Ln, C, np.int64)
        pp[:len(ip)] = ip; pe_l[:len(ip)] = pe[ip] - s
        nn[:len(inn)] = inn; ne_l[:len(inn)] = ne[inn] - s
        plan.append(dict(p=pp, p_eidx=pe_l, n=nn, n_eidx=ne_l))
    return wins, C, nch, plan, Lp, Ln


def bank_windows(JB, n_events, chunk, drop_source=False):
    """Materialise the per-window sub-banks ONCE.  Returns (nch, C, [sub_bank...], owns).

    THE POINT, and the mistake it fixes.  The first version did the dynamic_slice, the per-slot
    jnp.take and the pad-row concatenate INSIDE the jitted call, so every model evaluation
    re-materialised its sub-bank: measured 4.0 ms against 0.5 ms unchunked at ONE window, i.e. an 8x
    penalty for chunking that was not chunking anything.  The windows are fixed, so this work belongs at
    construction, once.

    Because the windows PARTITION the bank, the caller can drop the source arrays afterwards
    (`drop_source`) and the decomposition costs only the padding rather than a second copy.  Peak
    TRANSIENT memory during a jvp/vjp is then O(C), which is the entire reason for chunking.

    The pad row is baked in here too: every per-event array is one longer than C, the ragged FSI
    reduction runs over C+1 events, and padded slots point at that trash event.  `bank_weight_window`
    truncates back to C.

    `owns[w] = (lo, hi)` is the half-open range of GLOBAL event indices this window is responsible for.
    The last window is shifted back to n-C so all windows share one compiled shape, so it overlaps its
    predecessor; `owns` is what keeps every event counted exactly once.
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
        lo = w * C if w < nch - 1 else st
        owns.append((lo, min(lo + C, n_events)))
    if drop_source:
        JB.clear()                       # the windows hold everything; free the undivided arrays
    return nch, C, subs, owns


def bank_weight_window(BR, Bw, knobs, grids, C):
    """bank_weight on a PRE-BUILT window sub-bank: reduce over C+1 events, drop the trash event.

    The sub-bank already carries its pad row (see bank_windows), so this does no slicing, no gathering
    and no concatenation -- it is exactly the unchunked call on a smaller bank.
    """
    return BR.bank_weight(Bw, knobs, grids)[:C]
