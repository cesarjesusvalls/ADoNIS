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
from analysis.paper import info_content as IC
from analysis.paper import fisher_engine as FE
from adonis.analysis.knobs import NPAR, PNAMES, PRIOR, theta_nominal, knobs_of
from adonis.analysis.sample import AnaSample
from analysis.paper.physical_fit import SYST                      # env-driven error-model syst (kept)
from analysis.paper.beams.beam_fisher import beam_model

MULTISAMPLE_NPZ = os.environ.get("S4_GATE_NPZ", "output/altgen/multisample_carbon.npz")
T2K_KEEP = ("dpt", "dat", "pmu", "cosmu", "pn", "dptt", "daT")   # the 7 T2K obs sec2/sec3 stack


# ------------------------------------------------------------------------------------------------- #
# sub-engines: each owns ONE bank + its binned datasets and exposes weights/model/jac blocks over the
# shared 28-knob theta.  Both types expose `.ds` (a list of per-observable dicts) in stacking order.
# ------------------------------------------------------------------------------------------------- #
class BankSample:
    """A `bank_weight` sample (T2K / MINERvA / (e,e')).  One bank can feed several dataset builders
    (MINERvA STV + qelike pT/p|| share the nu_MINERvA_C bank); `builders` is a list of (fn, keep)."""
    def __init__(self, name, bankdir, builders, max_chunks, log):
        self.name = name
        B = BP.load_bank(bankdir, max_chunks=max_chunks)
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
    def __init__(self, beam, nbins, syst, max_chunks, log):
        self.m = beam_model(beam, nbins=nbins, syst=syst, log=log, max_chunks=max_chunks)
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

    def _bank(cfg_name, max_chunks):
        """A BankSample whose datasets come from the CENTRALIZED sample config (AnaSample.bin_datasets):
        same signal + observables + REAL edges as sec1/sec2/sec3, so sec4 fits the SAME sample."""
        s = AnaSample.from_config(f"configs/samples/{cfg_name}.yaml")
        return BankSample(s.name, s.bank, [(lambda B, w, l: s.bin_datasets(B, w), None)], max_chunks, log)

    samples = [
        _bank("t2k_cc0pi",    nu_chunks),
        _bank("t2k_cc1pi_ch", nu_chunks),      # CH keys match multisample_carbon; carbon closure (no free-H offset)
        _bank("minerva_stv",  nu_chunks),
        _bank("minerva_ptpz", nu_chunks),
        _bank("ee_omega",     e_chunks),
        BeamSample("pip", 15, SYST, beam_chunks, log),
        BeamSample("prot", 15, SYST, beam_chunks, log),
        BeamSample("neut", 15, SYST, beam_chunks, log),
    ]
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

    # optional: scale the FIT's prior width AFTER the truth is injected (so the injection always uses the
    # real prior).  S4_PRIOR_SCALE=0 -> no prior: the BFP then -> injected truth on the data-constrained
    # dials, isolating "is the fit unbiased" from "is the MAP estimate prior-pulled".  Default 1.0 = Gate-I.
    PRIOR_SCALE = float(os.environ.get("S4_PRIOR_SCALE", "1.0"))
    if PRIOR_SCALE != 1.0:
        eng.prior = eng.prior * (1e6 if PRIOR_SCALE == 0.0 else PRIOR_SCALE)
        log(f"FIT prior width scaled x{PRIOR_SCALE} (demonstration; injection unchanged)")

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
