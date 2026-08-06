"""AnaSample -- the fundamental analysis object.

A `sample` is bank(s) + signal + observables(+edges): one definition, built from an
`adonis.workflow.config.AnalysisConfig` (the very YAML sec1 already loads).  Its verbs are methods:

    s = AnaSample.from_config("configs/samples/t2k_cc0pi.yaml")
    s.plot()            # the sec1 ADoNIS-vs-ACHILLES figure
    s.gate1()           # Gate-I Fisher / shrinkage (the old physfit_*.npz)
    s.dump_cache()      # persist the Jacobian npz
    combined = s1 + s2  # SampleSet -> joint Gate-I (Fisher is additive)

The Jacobian bins on the sample's REAL edges (ObservableSpec.edges -- the NUISANCE bins sec1 plots), not
auto design_edges: one sample definition drives plot and gradient identically.  Gate-I shrinkage is
invariant to a per-bin scale (the unit conversion cancels in J/sigma for a carbon sample with no offset),
so the gradient uses scale = 1/binwidth; real display units are a .plot() concern.

Memory: the Jacobian STREAMS the bank one chunk at a time (peak = one chunk), and the SAME streamed pass
accumulates the nominal central + MC error, so J and sigma are consistent and `max_chunks` gives a fast,
self-consistent smoke run.
"""
import glob
import os
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from adonis.reweight import bank_plot as BP, bank_reweight as BR
from adonis.reweight.reweight_model import nominal_knobs
from adonis.workflow.config import load_analysis_config, EleBeamSignalDef
from adonis.workflow import selection as SG
from adonis.analysis import knobs as K

ALTGEN = Path("output/altgen")


# --------------------------------------------------------------------------- small self-contained helpers
def _binidx(mask, vals, edges):
    """(sel_idx, binidx, nbin) for `vals` under boolean `mask`, overflow folded into the edge bins -- the
    binning convention every driver uses (clip to [edge0+eps, edgeN-eps] then digitize)."""
    edges = np.asarray(edges, float); nb = len(edges) - 1
    eps = (edges[-1] - edges[0]) * 1e-12
    vc = np.clip(np.asarray(vals, float), edges[0] + eps, edges[-1] - eps)
    sel = np.where(np.asarray(mask, bool))[0]
    bidx = np.clip(np.searchsorted(edges, vc[sel], side="right") - 1, 0, nb - 1).astype(np.int64)
    return sel, bidx, nb


def _bin_sigma(central, mcerr, syst):
    """Diagonal error sigma = sqrt((syst*central)^2 + mcerr^2), empty-bin guarded (var=0 -> inf, so
    J/sigma = 0 not NaN)."""
    var = (syst * np.asarray(central)) ** 2 + np.asarray(mcerr) ** 2
    return np.where(var > 0, np.sqrt(var), np.inf)


def gate1_from(J, sigma, prior):
    """Asimov Fisher F=(J/sigma)^T(J/sigma), marginalized V=inv(F+diag(1/prior^2)), shrink=sqrt(diagV)/prior
    (FIT when <0.5), reach=sqrt(diagF).  Fisher is ADDITIVE, so composing samples = stacking their rows."""
    Jw = np.asarray(J) / np.asarray(sigma)[:, None]
    F = Jw.T @ Jw
    V = np.linalg.inv(F + np.diag(1.0 / np.asarray(prior) ** 2))
    sig_post = np.sqrt(np.diag(V))
    return F, V, sig_post, sig_post / np.asarray(prior), np.sqrt(np.maximum(np.diag(F), 0.0))


# --------------------------------------------------------------------------- the sample
class AnaSample:
    def __init__(self, cfg, name, syst=0.05):
        self.cfg = cfg
        self.name = name
        self.syst = float(os.environ.get("ADONIS_SYST", syst))
        self._sel = None

    @classmethod
    def from_config(cls, path, **kw):
        return cls(load_analysis_config(path), name=Path(path).stem, **kw)

    # ---- sample properties ----
    @property
    def bank(self):
        return self.cfg.inputs["adonis_bank"][0]

    @property
    def is_electron(self):
        return isinstance(self.cfg.signal, EleBeamSignalDef)

    @property
    def obs_keys(self):
        return [o.key for o in self.cfg.observables]

    def fit_specs(self):
        """Observables that enter the Fisher/fit (ObservableSpec.fit); plot uses ALL of them."""
        return [o for o in self.cfg.observables if getattr(o, "fit", True)]

    def edges(self):
        return {o.key: o.bin_edges() for o in self.cfg.observables}

    # ---- selection (compacted; the plot side) ----
    def selected(self):
        """Per-event {obs_key: values, 'w', 'chan'} for the passing events, streamed + bounded.  This is
        the SAME reducer sec1 plots (bank_signal / ele_signal)."""
        if self._sel is None:
            self._sel = (SG.ele_signal if self.is_electron else SG.bank_signal)(self.bank, self.cfg.signal)
        return self._sel

    # ---- gradient (streamed; J + central + sigma from ONE pass) ----
    def _gradient(self, max_chunks=None, log=print):
        if self.is_electron:
            raise NotImplementedError("electron (e,e') full-length selector lands in P1 (select_full_ele)")
        specs = self.fit_specs(); keys = [o.key for o in specs]
        edges_by = {o.key: o.bin_edges() for o in specs}
        nbins = [len(edges_by[k]) - 1 for k in keys]
        row0 = np.cumsum([0] + nbins)
        J = np.zeros((int(row0[-1]), K.NPAR))
        sumw = [np.zeros(n) for n in nbins]
        sumw2 = [np.zeros(n) for n in nbins]

        grids = BR.default_grids(); nom = nominal_knobs(); th0 = jnp.asarray(K.theta_nominal(nom))
        jvp = jax.jit(lambda th, tang, JB:
                      jax.jvp(lambda t: BR.bank_weight(JB, K.knobs_of(t, nom), grids), (th,), (tang,))[1])
        tang = [jnp.zeros(K.NPAR).at[k].set(1.0) for k in range(K.NPAR)]

        files = sorted(glob.glob(f"{self.bank}/chunk_*.npz"))
        if max_chunks:
            files = files[:max_chunks]
        nch = BP.bank_nchunks(self.bank)
        for ci, f in enumerate(files):
            Bc = BP.load_bank_chunk(f, nch); JBc = BR.to_jax(Bc)
            selm, obs, _w0, _chan = SG.select_full(Bc, self.cfg.signal)      # full-length mask + observables
            w0c = np.asarray(Bc["w0"])
            binned = []
            for j, k in enumerate(keys):
                bw = np.diff(edges_by[k]); s, bidx, nb = _binidx(selm, obs[k], edges_by[k])
                sc = 1.0 / bw
                binned.append((s, bidx, nb, sc))
                sumw[j] += sc * np.bincount(bidx, weights=w0c[s], minlength=nb)
                sumw2[j] += sc ** 2 * np.bincount(bidx, weights=w0c[s] ** 2, minlength=nb)
            for kk in range(K.NPAR):
                g = np.asarray(jvp(th0, tang[kk], JBc))                      # per-event dw/dtheta_kk (chunk)
                for j, (s, bidx, nb, sc) in enumerate(binned):
                    J[row0[j]:row0[j + 1], kk] += sc * np.bincount(bidx, weights=g[s], minlength=nb)
            del Bc, JBc
            if log:
                log(f"  [{self.name}] chunk {ci + 1}/{len(files)}")
        central = {k: sumw[j] for j, k in enumerate(keys)}
        mcerr = {k: np.sqrt(sumw2[j]) for j, k in enumerate(keys)}
        sigma = np.concatenate([_bin_sigma(central[k], mcerr[k], self.syst) for k in keys])
        return dict(J=J, sigma=sigma, row0=row0, keys=keys, edges=edges_by, central=central)

    # ---- verbs ----
    def jacobian(self, **kw):
        r = self._gradient(**kw)
        return r["J"], r["sigma"], r["row0"], r["keys"]

    def gate1(self, max_chunks=None, log=print):
        r = self._gradient(max_chunks=max_chunks, log=log)
        F, V, sig_post, shrink, reach = gate1_from(r["J"], r["sigma"], K.PRIOR)
        r.update(F=F, V=V, sig_post=sig_post, shrink=shrink, reach=reach, prior=K.PRIOR, pnames=K.PNAMES)
        return r

    def plot(self, **kw):
        """The sec1 ADoNIS-vs-ACHILLES figure for this sample (the plot verb).  Delegates to the sec1
        renderer on this sample's config -- the config IS a sec1 spec."""
        from analysis.paper.sec1_validation import helper
        spec = dict(self.cfg.__dict__); spec["_path"] = getattr(self, "_path", None)
        return helper.render(spec, **kw)

    # ---- cache ----
    def dump_cache(self, label=None, res=None, max_chunks=None, log=print):
        """Persist the Gate-I npz (physfit schema: J, sigma, row0, dskeys, shrink, F, V + per-obs edges/
        central) to output/altgen/{label}.npz -- what SampleSet / the figures load."""
        r = res or self.gate1(max_chunks=max_chunks, log=log)
        ALTGEN.mkdir(parents=True, exist_ok=True)
        out = ALTGEN / f"{label or self.name}.npz"
        np.savez(out, J=r["J"], sigma=r["sigma"], row0=r["row0"], prior=K.PRIOR, pnames=K.PNAMES,
                 dskeys=r["keys"], shrink=r["shrink"], F=r["F"], V=r["V"],
                 **{f"{k}_edges": r["edges"][k] for k in r["keys"]},
                 **{f"{k}_central": r["central"][k] for k in r["keys"]})
        if log:
            log(f"[out] {out}")
        return out

    @staticmethod
    def load_cache(label):
        return np.load(ALTGEN / f"{label}.npz", allow_pickle=True)

    def __add__(self, other):
        return SampleSet([self, other])


# --------------------------------------------------------------------------- composition
class SampleSet:
    """Several samples fitted / gated jointly.  Rows are concatenated across samples; the combined Fisher
    is the sum of the per-sample outer products (additive), so this is exactly the old build_multisample."""
    def __init__(self, samples):
        self.samples = list(samples)

    @classmethod
    def from_configs(cls, paths, **kw):
        return cls([AnaSample.from_config(p, **kw) for p in paths])

    def __add__(self, other):
        return SampleSet(self.samples + ([other] if isinstance(other, AnaSample) else other.samples))

    def gate1(self, max_chunks=None, log=print):
        Js, sigs, keys, row0 = [], [], [], [0]
        edges, central = {}, {}
        for s in self.samples:
            r = s._gradient(max_chunks=max_chunks, log=log)
            Js.append(r["J"]); sigs.append(r["sigma"])
            for k in r["keys"]:
                tag = f"{s.name}:{k}"; keys.append(tag)
                edges[tag] = r["edges"][k]; central[tag] = r["central"][k]
                row0.append(row0[-1] + (len(r["edges"][k]) - 1))
        J = np.vstack(Js); sigma = np.concatenate(sigs)
        F, V, sig_post, shrink, reach = gate1_from(J, sigma, K.PRIOR)
        return dict(J=J, sigma=sigma, row0=np.asarray(row0), keys=keys, edges=edges, central=central,
                    F=F, V=V, sig_post=sig_post, shrink=shrink, reach=reach, prior=K.PRIOR, pnames=K.PNAMES)
