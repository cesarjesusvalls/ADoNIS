"""Section 4 (closure) over the SAME dials + samples as sections 2/3.

Section 2 (Fisher/Gate-I) and section 3 (per-bin gradients) both read ONE stacked-Jacobian object,
`output/altgen/multisample_carbon.npz` (built by analysis.paper.sec3_gradients.build_multisample):

  * 16 DIALS  -- the knobs with combined marginalized shrinkage < 0.5 (Gate I).
  * 20 SAMPLE blocks -- T2K CC0pi/CC1pi STV+muon (7) + MINERvA CC0pi-Np STV (3) + MINERvA qelike pT/p||
    (2) + (e,e') QE/RES omega (2) + pi+/p/n -> C beams (react + abs/pi-prod, 2 each).

This module makes section 4 use the SAME 16 dials and the SAME 20 samples, but as a genuine NONLINEAR
closure (not a Fisher toy): each sample's REAL event bank is loaded and reweighted exactly; the fake data
is the exact-reweighted binned prediction at an injected truth; the blind fit then has to walk the 16
dials back from nominal through the true nonlinear model, across every sample jointly.

`MultiEngine` deliberately duck-types the single-bank `physical_fit_run.Engine` (same `model`, `jac`,
`data_sigma`, `th0`, `prior`, `pnames`, `npar`, `row0`, `ds`), so the already-tested LM fit
(`physical_fit_run.lm_fit`) and the Gate-II / flag machinery run over it UNCHANGED.

Every sample shares the one 28-knob `physical_fit` basis (`theta_nominal`/`knobs_of`); the reweight is
`bank_reweight.bank_weight` for the nu/electron banks and `cascade.pool_fsi_reweight` for the beams (via
`beams.beam_fisher.beam_model`) -- both pure-JAX and differentiable, so the per-knob Jacobian is exact.

    python -m analysis.paper.physfit.multisample            # closure at the default injection
Env: ADONIS_LABEL (default sec4_closure_multisample), PHYSFIT_INJECT, ALTGEN_NIT (default 12),
     S4_NU_CHUNKS / S4_BEAM_CHUNKS / S4_E_CHUNKS (bank subsample sizes; the full banks are ~40M events,
     far more MC precision than a closure needs).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from adonis.reweight import bank_plot as BP, bank_reweight as BR
from adonis.reweight.reweight_model import nominal_knobs
from adonis.workflow import selection as SG
from analysis.paper import info_content as IC
from analysis.paper import fisher_engine as FE
from adonis.analysis.knobs import NPAR, PNAMES, PRIOR, theta_nominal, knobs_of
from adonis.analysis.sample import AnaSample
from analysis.paper.physical_fit import SYST                      # env-driven error-model syst (kept)
from analysis.paper.beams.beam_fisher import beam_model

MULTISAMPLE_NPZ = os.environ.get("S4_GATE_NPZ", "output/altgen/multisample_carbon.npz")
# Jacobian dial-batch: how many tangents go through ONE vmapped jvp.  Default = all dials in one call.
# Under vmap only the TANGENT arrays gain a batch dimension (th/JB are unbatched, so the primal trace is
# NOT replicated) -- the cost is ~1x primal + Bx tangent, not Bx everything.  Results are identical for any
# B; only dispatch count and peak memory change.  S4_JAC_BATCH=1 is NOT the old loop (still vmap, width 1);
# use jac_blocks_ref for that.  On device OOM the batch HALVES and retries rather than killing the job.
JAC_BATCH = int(os.environ.get("S4_JAC_BATCH", "16"))


def _oom(e):
    s = str(e)
    return "RESOURCE_EXHAUSTED" in s or "out of memory" in s.lower() or "OUT_OF_MEMORY" in s


def _batched_jac(subset, call, bin_cols):
    """Run `call(ks)` over dial batches, halving the batch on device-OOM instead of dying.

    call(ks)     -> (len(ks), n_events) stacked per-event derivatives for those dials
    bin_cols(g)  -> the binned column for one dial's per-event derivative
    """
    bs = max(1, JAC_BATCH)
    while True:
        try:
            cols = []
            for lo in range(0, len(subset), bs):
                G = np.asarray(call(subset[lo:lo + bs]))
                cols += [bin_cols(g) for g in G]
            return cols
        except Exception as e:                       # noqa: BLE001 -- re-raised unless it is a device OOM
            if bs == 1 or not _oom(e):
                raise
            bs = max(1, bs // 2)
            print(f"[jac] device OOM at batch {bs * 2} -> retrying at {bs} "
                  f"(identical result, more dispatches)", flush=True)
T2K_KEEP = ("dpt", "dat", "pmu", "cosmu", "pn", "dptt", "daT")   # the 7 T2K obs sec2/sec3 stack


# ------------------------------------------------------------------------------------------------- #
# sub-engines: each owns ONE bank + its binned datasets and exposes weights/model/jac blocks over the
# shared 28-knob theta.  Both types expose `.ds` (a list of per-observable dicts) in stacking order.
# ------------------------------------------------------------------------------------------------- #
class BankSample:
    """A `bank_weight` sample (T2K / MINERvA / (e,e')).  One bank can feed several dataset builders
    (MINERvA STV + qelike pT/p|| share the nu_MINERvA_C bank); `builders` is a list of (fn, keep)."""
    def __init__(self, name, bankdir, builders, max_chunks, log, signal=None, cap=None):
        self.name = name
        # Cache only the SELECTED events (N_selected, streamed via select_bank).  The fit re-evaluates
        # model(theta) every iteration, so caching the signal instead of N_total is what keeps it cheap on
        # arbitrarily large banks.  `cap` further subsamples the signal to at most that many events
        # (fit-side; unbiased normalization) so the resident set fits in memory -- a closure/fit-test needs
        # far fewer events than the full signal (its MC error << SYST).
        # EVERY sample has a signal.  "No selection" is not a special case -- it is the selection that keeps
        # everything, and it must still be written down (select-all) so it goes through the SAME select_bank
        # path and therefore obeys the SAME event cap.  A None signal used to silently fall back to loading
        # the whole bank uncapped, which is exactly how ~1M uncapped events hid behind "S4_SIG_CAP=50000".
        if signal is None:
            raise ValueError(
                f"sample {name!r}: signal is None.  Every sample must declare a selection -- use an explicit "
                f"select-all condition in its config rather than omitting it (None bypasses the event cap).")
        B = SG.select_bank(bankdir, signal, max_chunks=max_chunks, cap=cap)
        self.n_events = int(len(B["w0"]))
        # ALWAYS report the resident event count: this is what the fit holds on the device every iteration,
        # and it is the only number that explains a memory failure.
        log(f"[events] {name:<14} N={self.n_events:>9,}  (cap={cap:,} , {max_chunks} chunks)"
            if cap else f"[events] {name:<14} N={self.n_events:>9,}  (no cap, {max_chunks} chunks)")
        self.JB = BR.to_jax(B); self.grids = BR.default_grids(); self.nom = nominal_knobs()
        w0 = np.asarray(BR.weight_jit(self.JB, self.nom, self.grids))
        ds = []
        for fn, keep in builders:
            built = fn(B, w0, log)
            if keep is not None:
                bykey = {d["key"]: d for d in built}
                built = [bykey[k] for k in keep]        # order + restrict to the kept observables
            ds += built
        self.ds = ds
        _w = lambda th, JB: BR.bank_weight(JB, knobs_of(th, self.nom), self.grids)
        self._wf = jax.jit(_w)
        self._jvp = jax.jit(lambda th, tang, JB: jax.jvp(lambda t: _w(t, JB), (th,), (tang,))[1])
        # Batched Jacobian: ONE vmapped jvp over a (nsub, NPAR) tangent matrix instead of nsub separate jvp
        # calls.  A fit iteration is 1 model eval + nsub Jacobian columns, so ~16/17 of all full-bank passes
        # happen here -- collapsing 16 dispatches into 1 is the single biggest lever on fit wall-clock.
        self._jvpv = jax.jit(lambda th, T, JB: jax.vmap(
            lambda t: jax.jvp(lambda x: _w(x, JB), (th,), (t,))[1])(T))
        # 2nd (diagonal) + 3rd (mixed) directional derivatives (nested jvp) for the higher-order corner
        self._jvp2 = jax.jit(lambda th, tang, JB: jax.jvp(
            lambda t: jax.jvp(lambda s: _w(s, JB), (t,), (tang,))[1], (th,), (tang,))[1])
        self._jvp3 = jax.jit(lambda th, u, w, x, JB: jax.jvp(
            lambda t3: jax.jvp(lambda t2: jax.jvp(lambda t1: _w(t1, JB), (t2,), (u,))[1], (t3,), (w,))[1],
            (th,), (x,))[1])
        self._wbase = _w; self._ddn = {}                          # generic depth-k nested-jvp cache

    def _ddfn(self, k):
        """jit'd depth-k mixed directional derivative: fn(th, T, JB) = sum d^k w . T[0]...T[k-1]."""
        f = self._ddn.get(k)
        if f is None:
            _w = self._wbase
            def make(kk):
                def fn(th, T, JB):
                    def rec(d, t):
                        return _w(t, JB) if d == 0 else jax.jvp(lambda u: rec(d - 1, u), (t,), (T[d - 1],))[1]
                    return rec(kk, th)
                return jax.jit(fn)
            f = make(k); self._ddn[k] = f
        return f

    def ddn_binned(self, theta, tangents):
        """Binned MIXED k-th directional derivative along the k tangent vectors in `tangents`."""
        g = np.asarray(self._ddfn(len(tangents))(jnp.asarray(theta), jnp.asarray(np.stack(tangents)), self.JB))
        return np.concatenate([IC.bin_w0(d, g) for d in self.ds])

    def weights(self, theta):
        return np.asarray(self._wf(jnp.asarray(theta), self.JB))

    def model_blocks(self, theta):
        w = self.weights(theta)
        return [IC.bin_w(d, w) for d in self.ds]

    def jac_blocks(self, theta, subset):
        """Binned Jacobian columns via ONE vmapped jvp per batch of dials (see JAC_BATCH)."""
        nb = sum(d["nbin"] for d in self.ds)
        if not subset:
            return np.zeros((nb, 0))
        th = jnp.asarray(theta)

        def call(ks):
            T = np.zeros((len(ks), NPAR)); T[np.arange(len(ks)), ks] = 1.0
            return self._jvpv(th, jnp.asarray(T), self.JB)                 # (len(ks), n_events)

        return np.column_stack(_batched_jac(
            subset, call, lambda g: np.concatenate([IC.bin_w0(d, g) for d in self.ds])))

    def jac_blocks_ref(self, theta, subset):
        """Reference: one jvp per dial in a Python loop.  Kept ONLY so the vmapped path above can be
        validated element-wise against it; not used by the fit."""
        th = jnp.asarray(theta); cols = []
        for k in subset:
            g = np.asarray(self._jvp(th, jnp.zeros(NPAR).at[k].set(1.0), self.JB))
            cols.append(np.concatenate([IC.bin_w0(d, g) for d in self.ds]))
        nb = sum(d["nbin"] for d in self.ds)
        return np.column_stack(cols) if cols else np.zeros((nb, 0))

    def dd_binned(self, theta, v, order):
        """Binned directional derivative of the model along tangent v: order 1 = J.v, order 2 = v.d2m.v."""
        th = jnp.asarray(theta); vv = jnp.asarray(v)
        g = np.asarray((self._jvp if order == 1 else self._jvp2)(th, vv, self.JB))
        return np.concatenate([IC.bin_w0(d, g) for d in self.ds])

    def dd3_binned(self, theta, u, w, x):
        """Binned MIXED 3rd directional derivative d3m(u,w,x)."""
        g = np.asarray(self._jvp3(jnp.asarray(theta), jnp.asarray(u), jnp.asarray(w), jnp.asarray(x), self.JB))
        return np.concatenate([IC.bin_w0(d, g) for d in self.ds])


class BeamSample:
    """A tagged-beam sample (pi+/p/n -> C): pure-FSI reweight via beam_model.  Two observables/bank."""
    def __init__(self, beam, nbins, syst, max_chunks, log, cap=None):
        self.m = beam_model(beam, nbins=nbins, syst=syst, log=log, max_chunks=max_chunks, cap=cap)
        self.n_events = int(self.m["n_events"]) if "n_events" in self.m else None
        nb = self.m["nbins"]; c = self.m["central"]; s = self.m["sigma"]; e = self.m["mcerr"]
        self.ds = []
        for i, key in enumerate(self.m["keys"]):
            sl = slice(i * nb, (i + 1) * nb)
            self.ds.append(dict(name=f"{beam} {key}", key=key, nbin=nb, edges=self.m["edges"],
                                data=c[sl].copy(), sigma=s[sl].copy(), mcerr=e[sl].copy()))
        self._ddn = {}                                            # generic depth-k nested-jvp cache

    def _ddfn(self, k):
        f = self._ddn.get(k)
        if f is None:
            wf = self.m["w_of"]
            def make(kk):
                def fn(th, T):
                    def rec(d, t):
                        return wf(t) if d == 0 else jax.jvp(lambda u: rec(d - 1, u), (t,), (T[d - 1],))[1]
                    return rec(kk, th)
                return jax.jit(fn)
            f = make(k); self._ddn[k] = f
        return f

    def ddn_binned(self, theta, tangents):
        g = self._ddfn(len(tangents))(jnp.asarray(theta), jnp.asarray(np.stack(tangents)))
        return self.m["binned"](np.asarray(g))

    def weights(self, theta):
        return np.asarray(self.m["w_of"](jnp.asarray(theta)))

    def model_blocks(self, theta):
        b = self.m["binned"](self.weights(theta)); nb = self.m["nbins"]
        return [b[:nb], b[nb:]]

    def jac_blocks(self, theta, subset):
        """Binned Jacobian columns via ONE vmapped jvp per batch of dials (see JAC_BATCH)."""
        nb = 2 * self.m["nbins"]
        if not subset:
            return np.zeros((nb, 0))
        th = jnp.asarray(theta)

        def call(ks):
            T = np.zeros((len(ks), NPAR)); T[np.arange(len(ks)), ks] = 1.0
            return self.m["jvpv"](th, jnp.asarray(T))

        return np.column_stack(_batched_jac(subset, call, self.m["binned"]))

    def jac_blocks_ref(self, theta, subset):
        """Reference (one jvp per dial) kept for element-wise validation of the vmapped path."""
        th = jnp.asarray(theta); cols = []
        for k in subset:
            g = self.m["jvp"](th, jnp.zeros(NPAR).at[k].set(1.0))
            cols.append(self.m["binned"](np.asarray(g)))
        nb = 2 * self.m["nbins"]
        return np.column_stack(cols) if cols else np.zeros((nb, 0))

    def dd_binned(self, theta, v, order):
        """Binned directional derivative of the model along tangent v: order 1 = J.v, order 2 = v.d2m.v."""
        th = jnp.asarray(theta); vv = jnp.asarray(v)
        g = (self.m["jvp"] if order == 1 else self.m["jvp2"])(th, vv)
        return self.m["binned"](np.asarray(g))

    def dd3_binned(self, theta, u, w, x):
        """Binned MIXED 3rd directional derivative d3m(u,w,x)."""
        g = self.m["jvp3"](jnp.asarray(theta), jnp.asarray(u), jnp.asarray(w), jnp.asarray(x))
        return self.m["binned"](np.asarray(g))


class MultiEngine:
    """Duck-types physical_fit_run.Engine over a list of sub-engines (samples concatenated in order)."""
    def __init__(self, samples):
        self.samples = samples
        self.ds = [d for s in samples for d in s.ds]
        self.row0 = np.cumsum([0] + [d["nbin"] for d in self.ds])
        self.th0 = theta_nominal(nominal_knobs())
        self.prior = PRIOR.copy()
        self.pnames = list(PNAMES)
        self.npar = NPAR

    def model(self, theta):
        return np.concatenate([np.asarray(b) for s in self.samples for b in s.model_blocks(theta)])

    def jac(self, theta, subset):
        return np.vstack([s.jac_blocks(theta, subset) for s in self.samples])

    def dd(self, theta, v, order):
        """Binned directional derivative of the FULL stacked model along tangent v (order 1 or 2)."""
        return np.concatenate([s.dd_binned(theta, v, order) for s in self.samples])

    def dd3(self, theta, u, w, x):
        """Binned MIXED 3rd directional derivative d3m(u,w,x) of the FULL stacked model."""
        return np.concatenate([s.dd3_binned(theta, u, w, x) for s in self.samples])

    def ddn(self, theta, tangents):
        """Binned MIXED k-th directional derivative (k = len(tangents)) of the FULL stacked model."""
        return np.concatenate([s.ddn_binned(theta, tangents) for s in self.samples])

    def data_sigma(self):
        return (np.concatenate([d["data"] for d in self.ds]),
                np.concatenate([d["sigma"] for d in self.ds]))

    def set_closure_data(self, truth):
        """Fake data = exact nonlinear reweight at `truth`, per sample; refresh syst+MC sigma on it."""
        for s in self.samples:
            for d, b in zip(s.ds, s.model_blocks(truth)):
                d["data"] = np.asarray(b)
                d["sigma"] = FE.bin_sigma(d["data"], d["mcerr"], SYST)


def build_multisample_engine(log, nu_chunks=None, beam_chunks=None, e_chunks=None):
    """Assemble the 20-block engine in the multisample_carbon order and assert it matches that npz."""
    nu_chunks = nu_chunks or int(os.environ.get("S4_NU_CHUNKS", "4"))        # ~0.79M/chunk -> ~3M
    beam_chunks = beam_chunks or int(os.environ.get("S4_BEAM_CHUNKS", "5"))  # ~0.6M/chunk  -> ~3M
    e_chunks = e_chunks or int(os.environ.get("S4_E_CHUNKS", "64"))          # ~32k/chunk   -> ~2M
    log(f"loading banks: nu={nu_chunks}ch minerva={nu_chunks}ch e={e_chunks}ch beams={beam_chunks}ch")

    sig_cap = int(os.environ.get("S4_SIG_CAP", "1000000"))   # per-sample event cap for the FIT (0 -> no cap)

    def _bank(cfg_name, max_chunks):
        """A BankSample whose datasets come from the CENTRALIZED sample config (AnaSample.bin_datasets):
        same signal + observables + REAL edges as sec1/sec2/sec3, so sec4 fits the SAME sample.  Loaded via
        select_bank -> only the N_selected signal events are cached (not N_total), further capped to
        S4_SIG_CAP events (unbiased) so the resident fit set fits in memory."""
        s = AnaSample.from_config(f"configs/samples/{cfg_name}.yaml")
        return BankSample(s.name, s.bank, [(lambda B, w, l: s.bin_datasets(B, w), None)], max_chunks, log,
                          signal=s.cfg.signal, cap=(sig_cap or None))

    samples = [
        _bank("t2k_cc0pi",    nu_chunks),
        _bank("t2k_cc1pi_ch", nu_chunks),      # CH keys match multisample_carbon; carbon closure (no free-H offset)
        _bank("minerva_stv",  nu_chunks),
        _bank("minerva_ptpz", nu_chunks),
        _bank("ee_omega",     e_chunks),
        BeamSample("pip", 15, SYST, beam_chunks, log, cap=(sig_cap or None)),
        BeamSample("prot", 15, SYST, beam_chunks, log, cap=(sig_cap or None)),
        BeamSample("neut", 15, SYST, beam_chunks, log, cap=(sig_cap or None)),
    ]
    tot = sum(s.n_events for s in samples if getattr(s, "n_events", None))
    log(f"[events] TOTAL RESIDENT = {tot:,} events across {len(samples)} samples "
        f"(S4_SIG_CAP={sig_cap:,}) -- this is what the fit holds on device EVERY iteration")
    eng = MultiEngine(samples)
    want = [str(x) for x in np.load(MULTISAMPLE_NPZ, allow_pickle=True)["dskeys"]]
    got = [d["key"] for d in eng.ds]
    assert got == want, f"sample composition != multisample_carbon:\n  got  {got}\n  want {want}"
    log(f"engine: {len(eng.samples)} samples, {len(eng.ds)} observables, {eng.row0[-1]} bins")
    return eng


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    LABEL = os.environ.get("ADONIS_LABEL", "sec4_closure_multisample")
    INJECT = os.environ.get("PHYSFIT_INJECT",
                            "M_A_res=0.85,kF_sf=1.10,Eb_shift=3.0,s_NN_elastic[1]=1.25,f_NN_cex=0.40")
    NIT = int(os.environ.get("ALTGEN_NIT", "12"))

    from analysis.paper.physfit.physical_fit_run import lm_fit, gate2_Q, gate2_split, flags, parse_inject

    eng = build_multisample_engine(log)

    # ---- the 16 dials: SAME Gate-I selection sec3 uses (combined marginalized shrinkage < 0.5) ------ #
    g = np.load(MULTISAMPLE_NPZ, allow_pickle=True)
    assert [str(x) for x in g["pnames"]] == list(PNAMES), "multisample_carbon knob order != physical_fit"
    subset = [int(i) for i in np.where(g["shrink"] < 0.5)[0]]
    log(f"fit subset ({len(subset)} dials): " + " ".join(eng.pnames[k] for k in subset))

    # ---- inject truth, build the nonlinear closure data across ALL samples ------------------------- #
    # "random": displace EVERY fitted dial by U(0.5,2)*prior with a random sign (Eb_shift positive-only --
    # its physical floor at 0 makes a negative draw unrecoverable), so the full 16-D space is exercised.
    if INJECT.strip().lower() == "random":
        seed = int(os.environ.get("S4_INJECT_SEED", "0"))
        rng = np.random.default_rng(seed)
        truth = eng.th0.copy()
        for k in subset:
            mag = rng.uniform(0.5, 2.0) * eng.prior[k]
            sign = 1.0 if eng.pnames[k] == "Eb_shift" else float(rng.choice([-1.0, 1.0]))
            truth[k] = eng.th0[k] + sign * mag
        inj_desc = f"random(seed={seed}; 0.5-2 sigma, all {len(subset)} dials)"
    else:
        truth, _inj = parse_inject(INJECT, nominal_knobs())
        inj_desc = INJECT
    eng.set_closure_data(truth)
    log(f"closure data ready (nonlinear exact reweight @ {inj_desc})")
    log("  truth: " + " ".join(f"{eng.pnames[k]}={truth[k]:.3f}" for k in subset))

    # Scale the FIT's prior width AFTER the truth is injected (so the injection always uses the real prior).
    # DEFAULT 0.0 = MLE (data-only): section 4's claim is about what the DATA constrain, so the estimator
    # must not be propped up by a prior.  S4_PRIOR_SCALE=1.0 -> MAP (Gate-I prior) for the prior-pull study.
    PRIOR_SCALE = float(os.environ.get("S4_PRIOR_SCALE", "0.0"))
    if PRIOR_SCALE != 1.0:
        eng.prior = eng.prior * (1e6 if PRIOR_SCALE == 0.0 else PRIOR_SCALE)
    # ALWAYS state the estimator: a figure whose caption says MLE while the run used MAP is exactly the
    # kind of silent mismatch that survives into a paper.
    log("estimator: " + ("MLE (data-only, no prior)" if PRIOR_SCALE == 0.0
                         else f"MAP (prior width x{PRIOR_SCALE})"))

    # ---- blind fit from nominal: ONE Gaussian-NLL (chi2_data + prior penalty) LM/GN fit ------------- #
    TOL = float(os.environ.get("PHYSFIT_TOL", "1e-6"))    # Newton-decrement convergence (predicted gap)
    log(f"==== fit (Gaussian NLL = chi2_data + prior)  step_scale x tol={TOL:g}, nit<={NIT} ====")
    traj = []
    th, V, J, m, chi2, chi2_data = lm_fit(eng, subset, "fit", huber=False, nit=NIT, record=traj, tol=TOL)
    traj_theta = np.array([t[0] for t in traj]); traj_chi2 = np.array([t[1] for t in traj])
    traj_chi2data = np.array([t[2] for t in traj])
    log(f"trajectory: {len(traj)} points recorded")

    # ---- report ------------------------------------------------------------------------------------ #
    print(f"\n==== S4 multisample closure [{LABEL}] ({inj_desc}) ====   chi2_data={chi2_data:.1f}/{eng.row0[-1]}")
    print(f"{'dial':>18} {'truth':>7} {'BFP':>8} {'+/-':>7} {'bias/sig':>8}")
    for c, k in enumerate(subset):
        s = np.sqrt(max(V[c, c], 0.0))
        b = (th[k] - truth[k]) / s if s > 0 else 0.0
        print(f"{eng.pnames[k]:>18} {truth[k]:7.3f} {th[k]:8.3f} {s:7.3f} {b:8.2f}")

    # ---- persist (reusable for the sec4 figure; single 'fit' method) ------------------------------- #
    out = f"output/altgen/{LABEL}.npz"
    np.savez(out, mode="closure_multisample", inj=inj_desc, truth=truth, subset=subset,
             pnames=eng.pnames, prior=eng.prior, row0=eng.row0, dskeys=[d["key"] for d in eng.ds],
             data=np.concatenate([d["data"] for d in eng.ds]),
             sigma=np.concatenate([d["sigma"] for d in eng.ds]),
             mcerr=np.concatenate([d["mcerr"] for d in eng.ds]),
             model_nom=eng.model(eng.th0),
             **{f"{d['key']}_edges": d["edges"] for d in eng.ds},
             fit_th=th, fit_V=V, fit_sub=np.array(subset), fit_chi2data=chi2_data,
             fit_model=eng.model(th),
             traj_theta=traj_theta, traj_chi2=traj_chi2, traj_chi2data=traj_chi2data)
    log(f"[out] {out}")
    log("done")


if __name__ == "__main__":
    main()
