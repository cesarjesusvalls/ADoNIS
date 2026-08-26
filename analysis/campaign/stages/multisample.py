"""Closure fit over the SAME dials + samples as the Fisher/gradient studies.

Both read ONE stacked-Jacobian object, `output/altgen/multisample_carbon.npz` (built by
`python -m analysis.campaign.gate1`): the dials passing Gate I (combined marginalized shrinkage < 0.5),
and the joint sample set (T2K CC0pi/CC1pi STV+muon, MINERvA CC0pi-Np STV, MINERvA qelike pT/p||, (e,e')
QE/RES omega, pi+/p/n -> C beams).

This module runs a genuine NONLINEAR closure (not a Fisher toy) over that same set: each sample's REAL
event bank is loaded and reweighted exactly; the fake data is the exact-reweighted binned prediction at
an injected truth; the blind fit walks the dials back from nominal through the true nonlinear model,
across every sample jointly.

`MultiEngine` duck-types the single-bank `fitters.Engine` (`model`, `jac`, `data_sigma`, `th0`, `prior`,
`pnames`, `npar`, `row0`, `ds`), so `fitters.lm_fit` and the Gate-II / flag machinery run over it
unchanged.  Every sample shares the one 28-knob `physical_fit` basis; the reweight is
`bank_reweight.bank_weight` for the nu/electron banks and `cascade.pool_fsi_reweight` for the beams (via
`beams.beam_model`) -- both pure-JAX and differentiable, so the per-knob Jacobian is exact.

    python -m analysis.campaign.stages.multisample            # closure at the default injection
Label, injection, iteration count and bank chunk caps all come from the FitConfig, not the environment;
S4_GATE_NPZ overrides which multisample_carbon.npz is read.
"""
import os
import sys
import sys
import time
from pathlib import Path

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from adonis.reweight import bank_plot as BP, bank_reweight as BR
from adonis.reweight.reweight_model import nominal_knobs
from adonis.workflow import selection as SG
from adonis.fit import binning as IC
from adonis.stats import fisher as _FI
from adonis.stats.gaussian import bin_sigma as _bin_sigma
from adonis.reweight.knobs import NPAR, PNAMES, PRIOR, theta_nominal, knobs_of
from analysis.campaign.sample import AnaSample
from analysis.campaign.beams import beam_model

MULTISAMPLE_NPZ = os.environ.get("S4_GATE_NPZ", "output/altgen/multisample_carbon.npz")
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
        except Exception as e:
            if bs == 1 or not _oom(e):
                raise
            bs = max(1, bs // 2)
            print(f"[jac] device OOM at batch {bs * 2} -> retrying at {bs} "
                  f"(identical result, more dispatches)", flush=True)
_JAX_BIN = os.environ.get("S4_JAX_BIN", "") == "1"

T2K_KEEP = ("dpt", "dat", "pmu", "cosmu", "pn", "dptt", "daT")


class BankSample:
    """A `bank_weight` sample (T2K / MINERvA / (e,e')).  One bank can feed several dataset builders
    (MINERvA STV + qelike pT/p|| share the nu_MINERvA_C bank); `builders` is a list of (fn, keep)."""
    def __init__(self, name, bankdir, builders, max_chunks, log, signal=None, cap=None):
        self.name = name
        if signal is None:
            raise ValueError(
                f"sample {name!r}: signal is None.  Every sample must declare a selection -- use an explicit "
                f"select-all condition in its config rather than omitting it (None bypasses the event cap).")
        B = SG.select_bank(bankdir, signal, max_chunks=max_chunks, cap=cap)
        self.n_events = int(len(B["w0"]))
        log(f"[events] {name:<14} N={self.n_events:>9,}  (cap={cap:,} , {max_chunks} chunks)"
            if cap else f"[events] {name:<14} N={self.n_events:>9,}  (no cap, {max_chunks} chunks)")
        self.JB = BR.to_jax(B); self.grids = BR.default_grids(); self.nom = nominal_knobs()
        w0 = np.asarray(BR.weight_jit(self.JB, self.nom, self.grids))
        ds = []
        for fn, keep in builders:
            built = fn(B, w0, log)
            if keep is not None:
                bykey = {d["key"]: d for d in built}
                built = [bykey[k] for k in keep]
            ds += built
        self.ds = ds
        _w = lambda th, JB: BR.bank_weight(JB, knobs_of(th, self.nom), self.grids)
        self._wf = jax.jit(_w)
        self._jvp = jax.jit(lambda th, tang, JB: jax.jvp(lambda t: _w(t, JB), (th,), (tang,))[1])
        self._jvpv = jax.jit(lambda th, T, JB: jax.vmap(
            lambda t: jax.jvp(lambda x: _w(x, JB), (th,), (t,))[1])(T))
        self._jvp2 = jax.jit(lambda th, tang, JB: jax.jvp(
            lambda t: jax.jvp(lambda s: _w(s, JB), (t,), (tang,))[1], (th,), (tang,))[1])
        self._jvp3 = jax.jit(lambda th, u, w, x, JB: jax.jvp(
            lambda t3: jax.jvp(lambda t2: jax.jvp(lambda t1: _w(t1, JB), (t2,), (u,))[1], (t3,), (w,))[1],
            (th,), (x,))[1])
        self._wbase = _w; self._ddn = {}

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
        if _JAX_BIN:
            wj = self._wf(jnp.asarray(theta), self.JB)
            return [np.asarray(IC.bin_w_dev(d, wj)) for d in self.ds]
        w = self.weights(theta)
        return [IC.bin_w(d, w) for d in self.ds]

    def model_blocks_jax(self, theta):
        """Binned model as JAX arrays, differentiable end to end (needs the BinSpec device path)."""
        wj = self._wf(theta, self.JB)
        return [IC.bin_w_dev(d, wj) for d in self.ds]

    def jac_blocks(self, theta, subset):
        """Binned Jacobian columns via ONE vmapped jvp per batch of dials (see JAC_BATCH)."""
        nb = sum(d["nbin"] for d in self.ds)
        if not subset:
            return np.zeros((nb, 0))
        th = jnp.asarray(theta)

        def call(ks):
            T = np.zeros((len(ks), NPAR)); T[np.arange(len(ks)), ks] = 1.0
            return self._jvpv(th, jnp.asarray(T), self.JB)

        if _JAX_BIN:
            def call_dev(ks):
                T = np.zeros((len(ks), NPAR)); T[np.arange(len(ks)), ks] = 1.0
                G = self._jvpv(th, jnp.asarray(T), self.JB)
                return np.stack([np.concatenate([np.asarray(IC.bin_w0_dev(d, g)) for d in self.ds])
                                 for g in G])
            cols = []
            for lo in range(0, len(subset), max(1, JAC_BATCH)):
                cols += list(call_dev(subset[lo:lo + max(1, JAC_BATCH)]))
            return np.column_stack(cols)
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
        self._ddn = {}

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
        nb = self.m["nbins"]
        if _JAX_BIN:
            b = np.asarray(self.m["binned_dev"](self.m["w_of"](jnp.asarray(theta))))
        else:
            b = self.m["binned"](self.weights(theta))
        return [b[:nb], b[nb:]]

    def model_blocks_jax(self, theta):
        nb = self.m["nbins"]
        b = self.m["binned_dev"](self.m["w_of"](theta))
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
    """Duck-types fitters.Engine over a list of sub-engines (samples concatenated in order)."""
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

    def model_jax(self, theta):
        """Full stacked model as ONE JAX array -- the differentiable counterpart of .model()."""
        return jnp.concatenate([b for s in self.samples for b in s.model_blocks_jax(theta)])

    def _chi2_parts(self, subset):
        """chi2 as a list of PER-SAMPLE terms, each a JAX function of the SUBSET dials, jitted separately
        (not one function over the stacked model) so the arithmetic matches `.model()` exactly.  Shared
        by chi2_fn / chi2_grad_fn so the value-only and value-and-grad objectives can never drift apart.

        DATA AND WEIGHTS ARE ARGUMENTS, not captured constants -- use chi2_data_dev() to get (D, W) per
        sample.
        """
        idx = jnp.asarray(np.asarray(subset, int))
        th0 = jnp.asarray(self.th0)
        parts = []
        for s in self.samples:
            def term(x, D, W, s=s):
                th = th0.at[idx].set(x)
                r = jnp.concatenate([b for b in s.model_blocks_jax(th)]) - D
                return jnp.sum(W * r * r)

            parts.append(term)

        if not getattr(self, "_bins_warm", False):
            D0, W0 = self.chi2_data_dev()
            xw = jnp.asarray(np.asarray(self.th0)[np.asarray(subset, int)])
            for p, d, w in zip(parts, D0, W0):
                p(xw, d, w)
            self._bins_warm = True
        return parts

    def chi2_data_dev(self):
        """Per-sample (data, 1/sigma^2) as device arrays, read from the CURRENT eng.ds.

        Call again after set_closure_data() to pick up a new toy; the jitted objectives take these as
        arguments, so swapping them is a host-to-device copy, not a recompile.
        """
        D, W = [], []
        for s in self.samples:
            sig = np.concatenate([d["sigma"] for d in s.ds])
            ok = np.isfinite(sig) & (sig > 0)
            W.append(jnp.asarray(np.where(ok, 1.0 / np.where(ok, sig, 1.0) ** 2, 0.0)))
            D.append(jnp.asarray(np.concatenate([d["data"] for d in s.ds])))
        return D, W

    def chi2_fn(self, subset):
        """chi2 ALONE, jitted per sample (no gradient) -- for a derivative-free minimiser.

        The returned callable has `.set_data(D, W)` so an ensemble can retarget it at the next toy
        without recompiling; it starts bound to whatever data the engine holds now.
        """
        fs = [jax.jit(p) for p in self._chi2_parts(subset)]
        st = list(self.chi2_data_dev())

        def chi2(x):
            xj = jnp.asarray(np.asarray(x, float))
            return float(sum(f(xj, d, w) for f, d, w in zip(fs, st[0], st[1])))
        chi2.set_data = lambda D, W: st.__setitem__(slice(0, 2), [D, W])
        return chi2

    def chi2_grad_fn(self, subset):
        """(chi2, grad) over `subset` by REVERSE mode, summed over samples -- cheaper than a forward
        Jacobian when only the scalar gradient is needed (MIGRAD, HMC).

        Returns (float, ndarray) rather than JAX scalars: every consumer is a host-side minimiser.
        Same `.set_data(D, W)` retargeting as chi2_fn.
        """
        vgs = [jax.jit(jax.value_and_grad(p)) for p in self._chi2_parts(subset)]
        st = list(self.chi2_data_dev())

        def value_and_grad(x):
            xj = jnp.asarray(np.asarray(x, float))
            out = [f(xj, d, w) for f, d, w in zip(vgs, st[0], st[1])]
            return float(sum(o[0] for o in out)), np.asarray(sum(o[1] for o in out))
        value_and_grad.set_data = lambda D, W: st.__setitem__(slice(0, 2), [D, W])
        return value_and_grad

    def data_sigma(self):
        return (np.concatenate([d["data"] for d in self.ds]),
                np.concatenate([d["sigma"] for d in self.ds]))

    def set_closure_data(self, truth):
        """Fake data = exact nonlinear reweight at `truth`, per sample; refresh sigma on it.

        cfg.data.sigma.mc_term=False drops the MC term (sigma = SYST*data): in a closure the data IS the
        MC reweighted, so the MC statistical fluctuation is common-mode between data and prediction and
        cancels in the residual.

        cfg.data.sigma.mask_mcfrac then masks bins whose MC error alone would exceed that fraction of the
        central value, so dropping mcerr cannot hand a sparse bin a tiny sigma and hence a dominating
        weight.
        """
        sig = self.cfg.data.sigma
        syst_only = not sig.mc_term
        cut = sig.mask_mcfrac if syst_only else None
        for s in self.samples:
            if syst_only and not hasattr(s, "_nom_blocks"):
                s._nom_blocks = [np.abs(np.asarray(b)) for b in s.model_blocks(self.th0)]
            for i, (d, b) in enumerate(zip(s.ds, s.model_blocks(truth))):
                d["data"] = np.asarray(b)
                if not syst_only:
                    d["sigma"] = _bin_sigma(d["data"], d["mcerr"], sig.syst)
                    continue
                if "_sigma0" not in d:
                    c0 = s._nom_blocks[i]
                    bad = (c0 <= 0) | (d["mcerr"] > cut * np.where(c0 > 0, c0, 1.0))
                    d["_sigma0"] = np.where(bad, np.inf, np.where(c0 > 0, sig.syst * c0, np.inf))
                d["sigma"] = d["_sigma0"]


def build_multisample_engine(log, cfg):
    """Assemble the engine described by `cfg` (adonis.fit.config.FitConfig).  The sample list and the
    chunk caps come from the config.
    """
    nu_chunks, beam_chunks, e_chunks = cfg.banks.nu_chunks, cfg.banks.beam_chunks, cfg.banks.e_chunks
    log(f"loading banks: nu={nu_chunks}ch minerva={nu_chunks}ch e={e_chunks}ch beams={beam_chunks}ch")

    sig_cap = cfg.banks.sig_cap

    def _bank(cfg_name, max_chunks):
        """A BankSample whose datasets come from the CENTRALIZED sample config (AnaSample.bin_datasets),
        so this fits the SAME sample the other studies use.  Loaded via select_bank -> only the
        N_selected signal events are cached (not N_total), further capped to cfg.banks.sig_cap events
        (unbiased) so the resident fit set fits in memory."""
        s = AnaSample.from_config(f"configs/samples/{cfg_name}.yaml")
        return BankSample(s.name, s.bank, [(lambda B, w, l: s.bin_datasets(B, w), None)], max_chunks, log,
                          signal=s.cfg.signal, cap=(sig_cap or None))

    samples = [_bank(nm, e_chunks if AnaSample.from_config(f"configs/samples/{nm}.yaml").is_electron
                     else nu_chunks) for nm in cfg.samples]
    samples += [BeamSample(b, 15, cfg.data.sigma.syst, beam_chunks, log, cap=(sig_cap or None))
                for b in cfg.beams]

    tot = sum(s.n_events for s in samples if getattr(s, "n_events", None))
    log(f"[events] TOTAL RESIDENT = {tot:,} events across {len(samples)} samples "
        f"(cap={sig_cap:,}) -- this is what the fit holds on device EVERY iteration")
    eng = MultiEngine(samples)
    eng.cfg = cfg
    want = [str(x) for x in np.load(MULTISAMPLE_NPZ, allow_pickle=True)["dskeys"]]
    got = [d["key"] for d in eng.ds]
    assert got == want, f"sample composition != multisample_carbon:\n  got  {got}\n  want {want}"
    log(f"engine: {len(eng.samples)} samples, {len(eng.ds)} observables, {eng.row0[-1]} bins")
    return eng


def fit_subset(g, pnames, cfg, log=None):
    """The fitted dials: Gate-I shrink<0.5, plus anything `fit.dials` names explicitly.

    Fixing a dial removes it from the fit and from the toy truth throw alike (multisample_coverage
    iterates the same subset), so it is held at nominal everywhere -- i.e. "assumed known".
    """
    sub = [int(i) for i in np.where(g["shrink"] < 0.5)[0]]
    cut = float(getattr(cfg.fit, "vif_cut", 0) or 0)
    if cut > 0 and "F" in g.files:
        vif, rmul = _FI.vif_from_fisher(np.asarray(g["F"]), np.asarray(g["prior"]))
        drop = [k for k in sub if vif[k] > cut]
        if drop and log:
            for k in drop:
                log(f"  GATE-I VIF: dropping {pnames[k]} (shrink {float(g['shrink'][k]):.3f} passes, but "
                    f"VIF={vif[k]:.0f}, R_multi={rmul[k]:.4f} > cut {cut:g})")
        sub = [k for k in sub if vif[k] <= cut]
    add = [] if cfg.fit.dials == "gate1" else list(cfg.fit.dials)
    if add:
        bad = [a for a in add if a not in list(pnames)]
        if bad:
            raise SystemExit(f"S4_ADD_DIALS: unknown dial(s) {bad}")
        for a in add:
            k = list(pnames).index(a)
            if k not in sub:
                sub = sorted(sub + [k])
                if log:
                    log(f"  ADDED (failed Gate I, forced in): {a} (shrink {float(g['shrink'][k]):.3f})")
    fix = []
    if fix:
        bad = [f for f in fix if f not in list(pnames)]
        if bad:
            raise SystemExit(f"S4_FIX_DIALS: unknown dial(s) {bad}")
        keep = [k for k in sub if pnames[k] not in fix]
        if log:
            log(f"  FIXED (held at nominal, not fitted, not thrown): {fix} -> {len(keep)} dials fitted")
        return keep
    return sub


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    from adonis.fit.config import FitConfig
    cfg = FitConfig.load(sys.argv[1] if len(sys.argv) > 1 else "configs/fits/sec4_P1.yaml")
    LABEL, INJECT = cfg.name, cfg.inject_string()
    NIT = cfg.fit.minimizer.max_nfev
    log(f"config {cfg.path}  digest {cfg.digest()}")

    from adonis.fit.fitters import lm_fit, cochran_q, split_half_consistency, flags, parse_inject

    eng = build_multisample_engine(log, cfg)

    g = np.load(MULTISAMPLE_NPZ, allow_pickle=True)
    assert [str(x) for x in g["pnames"]] == list(PNAMES), "multisample_carbon knob order != physical_fit"
    subset = fit_subset(g, eng.pnames, cfg, log)
    log(f"fit subset ({len(subset)} dials): " + " ".join(eng.pnames[k] for k in subset))

    truth, _ = parse_inject(INJECT, nominal_knobs())
    eng.set_closure_data(truth)
    log(f"closure data ready (nonlinear exact reweight @ {INJECT})")
    log("  truth: " + " ".join(f"{eng.pnames[k]}={truth[k]:.3f}" for k in subset))

    PRIOR_SCALE = cfg.fit.prior_scale
    if PRIOR_SCALE != 1.0:
        eng.prior = eng.prior * (1e6 if PRIOR_SCALE == 0.0 else PRIOR_SCALE)
    log(f"estimator: {cfg.fit.estimator.upper()}"
        + (" (data-only, no prior)" if cfg.fit.estimator == "mle"
           else f" (prior width x{PRIOR_SCALE:g})"))

    TOL = float(cfg.fit.minimizer.newton_tol)
    log(f"==== fit (Gaussian NLL = chi2_data + prior)  step_scale x tol={TOL:g}, nit<={NIT} ====")
    traj = []
    if cfg.fit.minimizer.method == "trf":
        from adonis.fit.fitters import trf_fit
        th, V, J, m, chi2, chi2_data = trf_fit(eng, subset, "fit", nit=NIT, record=traj)
    else:
        th, V, J, m, chi2, chi2_data = lm_fit(eng, subset, "fit", huber=False, nit=NIT, record=traj, tol=TOL)
    traj_theta = np.array([t[0] for t in traj]) if traj else np.zeros((0, len(eng.th0)))
    traj_chi2 = np.array([t[1] for t in traj]); traj_chi2data = np.array([t[2] for t in traj])
    log(f"trajectory: {len(traj)} points recorded")

    print(f"\n==== S4 multisample closure [{LABEL}] ({INJECT}) ====   chi2_data={chi2_data:.1f}/{eng.row0[-1]}")
    print(f"{'dial':>18} {'truth':>7} {'BFP':>8} {'+/-':>7} {'bias/sig':>8}")
    for c, k in enumerate(subset):
        s = np.sqrt(max(V[c, c], 0.0))
        b = (th[k] - truth[k]) / s if s > 0 else 0.0
        print(f"{eng.pnames[k]:>18} {truth[k]:7.3f} {th[k]:8.3f} {s:7.3f} {b:8.2f}")

    out = f"output/altgen/{LABEL}.npz"
    np.savez(out, mode="closure_multisample", inj=INJECT, truth=truth, subset=subset,
             pnames=eng.pnames, prior=eng.prior, row0=eng.row0, dskeys=[d["key"] for d in eng.ds],
             data=np.concatenate([d["data"] for d in eng.ds]),
             sigma=np.concatenate([d["sigma"] for d in eng.ds]),
             mcerr=np.concatenate([d["mcerr"] for d in eng.ds]),
             model_nom=eng.model(eng.th0),
             **{f"{d['key']}_edges": d["edges"] for d in eng.ds},
             mask_mcfrac=float(cfg.data.sigma.mask_mcfrac), syst=float(cfg.data.sigma.syst),
             sig_cap=int(cfg.banks.sig_cap),
             fit_th=th, fit_V=V, fit_sub=np.array(subset), fit_chi2data=chi2_data,
             fit_model=eng.model(th),
             traj_theta=traj_theta, traj_chi2=traj_chi2, traj_chi2data=traj_chi2data,
             fit_nfev=getattr(eng, "last_nfev", -1), fit_njev=getattr(eng, "last_njev", -1),
             fit_status=getattr(eng, "last_status", -1), fit_opt=getattr(eng, "last_opt", np.nan),
             fit_gap=getattr(eng, "last_gap", np.nan))
    log(f"[out] {out}")
    log("done")


if __name__ == "__main__":
    main()
