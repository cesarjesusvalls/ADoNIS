"""AnaSample -- the fundamental analysis object.

A `sample` is bank(s) + signal + observables(+edges): one definition, built from an
`adonis.workflow.config.AnalysisConfig`.  Its verbs are methods:

    s = AnaSample.from_config("configs/samples/t2k_cc0pi.yaml")
    s.plot()            # ADoNIS-vs-ACHILLES figure
    s.constrained()           # Fisher / shrinkage of the constrained subset
    s.dump_cache()      # persist the Jacobian npz
    combined = s1 + s2  # SampleSet -> joint constrained subset (Fisher is additive)

The Jacobian bins on the sample's REAL edges (ObservableSpec.edges), not auto design_edges: one sample
definition drives plot and gradient identically.  Shrinkage is invariant to a per-bin scale, so the
gradient uses scale = 1/binwidth; real display units are a .plot() concern.

Memory: the Jacobian STREAMS the bank one chunk at a time (peak = one chunk); the same pass accumulates
the nominal central + MC error, so J and sigma stay consistent and `max_chunks` gives a fast smoke run.
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
from adonis.reweight import knobs as K

from adonis.io import bank_path
from analysis._cli import results_dir
T2K_H_BANK = "nu_T2K_H/merged"


def _cfg_h_bank(cfg):
    """The free-H bank for a CH sample: `inputs.h_bank` if the config names one, else the T2K default.

    The hydrogen bank is FLUX-SPECIFIC (nu_mu p -> mu- p pi+ folded with that beam), so a MINERvA CH
    sample cannot borrow the T2K one -- it needs nu_MINERvA_H (NuMI)."""
    hb = getattr(getattr(cfg, "inputs", None), "h_bank", None)
    if hb is None:
        return str(bank_path(T2K_H_BANK))
    return str(bank_path(hb[0] if isinstance(hb, (list, tuple)) else hb))
FREEH_KEYS = ("pn", "dptt", "dalphat", "tpi", "q2")


def _binidx(mask, vals, edges):
    """(sel_idx, binidx, nbin) for `vals` under boolean `mask`, overflow folded into the edge bins -- the
    binning convention every driver uses (clip to [edge0+eps, edgeN-eps] then digitize)."""
    edges = np.asarray(edges, float); nb = len(edges) - 1
    eps = (edges[-1] - edges[0]) * 1e-12
    vc = np.clip(np.asarray(vals, float), edges[0] + eps, edges[-1] - eps)
    sel = np.where(np.asarray(mask, bool))[0]
    bidx = np.clip(np.searchsorted(edges, vc[sel], side="right") - 1, 0, nb - 1).astype(np.int64)
    return sel, bidx, nb


from adonis.stats.gaussian import bin_sigma as _bin_sigma


def constrained_from(J, sigma, prior):
    """Asimov Fisher F=(J/sigma)^T(J/sigma), marginalized V=inv(F+diag(1/prior^2)), shrink=sqrt(diagV)/prior
    (FIT when <0.5), reach=sqrt(diagF).  Fisher is ADDITIVE, so composing samples = stacking their rows."""
    Jw = np.asarray(J) / np.asarray(sigma)[:, None]
    F = Jw.T @ Jw
    V = np.linalg.inv(F + np.diag(1.0 / np.asarray(prior) ** 2))
    sig_post = np.sqrt(np.diag(V))
    return F, V, sig_post, sig_post / np.asarray(prior), np.sqrt(np.maximum(np.diag(F), 0.0))


class AnaSample:
    def __init__(self, cfg, name, syst=0.05):
        self.cfg = cfg
        self.name = name
        self.syst = float(os.environ.get("ADONIS_SYST", syst))
        self._sel = None
        self._path = None

    @classmethod
    def from_config(cls, path, **kw):
        s = cls(load_analysis_config(path), name=Path(path).stem, **kw)
        s._path = str(path)
        return s

    @property
    def bank(self):
        return str(bank_path(self.cfg.inputs["adonis_bank"][0]))

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

    def selected(self):
        """Per-event {obs_key: values, 'w', 'chan'} for the passing events, streamed + bounded.  Uses the
        SAME reducer the plot path does (bank_signal / ele_signal)."""
        if self._sel is None:
            self._sel = (SG.ele_signal if self.is_electron else SG.bank_signal)(self.bank, self.cfg.signal)
        return self._sel

    def _h_bank(self):
        return _cfg_h_bank(self.cfg)

    def bin_datasets(self, B, w0, free_h=None):
        """IC.bin_w-compatible per-observable dataset dicts for the fit engine (MultiEngine / BankSample):
        FIT observables selected + binned on the config's REAL edges, keys namespaced `{name}:{obs}`,
        scale = 1/binwidth so `model(theta) = IC.bin_w(d, weights(theta))` works.  `w0`: nominal per-event
        weight (numpy).  `free_h`: optional {obs_key: frozen offset} added to matching observables'
        central value (theta-independent -> zero gradient).  Numpy-only."""
        selm, obs, _w0, _chan = SG.select_full(B, self.cfg.signal)
        w0 = np.asarray(w0); ds = []
        for o in self.fit_specs():
            edges = o.bin_edges(); nb = len(edges) - 1; bw = np.diff(edges)
            s, bidx, _ = _binidx(selm, obs[o.key], edges)
            off = np.zeros(nb) if free_h is None else np.asarray(free_h.get(o.key, np.zeros(nb)))
            d = dict(name=f"{self.name}:{o.key}", key=f"{self.name}:{o.key}", nbin=nb, edges=edges,
                     sel_idx=s, binidx=bidx, scale_bin=1.0 / bw, offset=off)
            central = d["scale_bin"] * np.bincount(bidx, weights=w0[s], minlength=nb) + off
            mcerr = d["scale_bin"] * np.sqrt(np.bincount(bidx, weights=w0[s] ** 2, minlength=nb))
            d.update(data=central, sigma=_bin_sigma(central, mcerr, self.syst), mcerr=mcerr)
            ds.append(d)
        return ds

    @staticmethod
    def _jvp_ctx():
        """The reweight jvp closure + nominal theta + per-knob tangents (shared by carbon and free-H passes)."""
        grids = BR.default_grids(); nom = nominal_knobs(); th0 = jnp.asarray(K.theta_nominal(nom))
        jvp = jax.jit(lambda th, tang, JB:
                      jax.jvp(lambda t: BR.bank_weight(JB, K.knobs_of(t, nom), grids), (th,), (tang,))[1])
        return th0, jvp, [jnp.zeros(K.NPAR).at[k].set(1.0) for k in range(K.NPAR)]

    def _stream_grad(self, bank, signal, keys, edges_by, ctx, override=None, max_chunks=None, log=print, tag=""):
        """Stream `bank` one chunk at a time; per obs `keys` accumulate the per-bin Jacobian rows (one jvp
        per knob) AND the nominal central (sum w0) + MC-error (sum w0^2) -- all from the SAME pass, so J and
        sigma are consistent.  `override(obs, n)` may rewrite an observable in place (free-H daT randomization).
        Returns (J[sum nbins, NPAR], row0, sumw[list], sumw2[list])."""
        th0, jvp, tang = ctx
        nbins = [len(edges_by[k]) - 1 for k in keys]; row0 = np.cumsum([0] + nbins)
        J = np.zeros((int(row0[-1]), K.NPAR)); sumw = [np.zeros(n) for n in nbins]; sumw2 = [np.zeros(n) for n in nbins]
        files = sorted(glob.glob(f"{bank}/chunk_*.npz"))
        if max_chunks:
            files = files[:max_chunks]
        nch = BP.bank_nchunks(bank)
        for ci, f in enumerate(files):
            Bc = BP.load_bank_chunk(f, nch); JBc = BR.to_jax(Bc)
            selm, obs, _w0, _chan = SG.select_full(Bc, signal)
            if override is not None:
                override(obs, len(selm))
            w0c = np.asarray(Bc["w0"]); binned = []
            for j, k in enumerate(keys):
                bw = np.diff(edges_by[k]); s, bidx, nb = _binidx(selm, obs[k], edges_by[k]); sc = 1.0 / bw
                binned.append((s, bidx, nb, sc))
                sumw[j] += sc * np.bincount(bidx, weights=w0c[s], minlength=nb)
                sumw2[j] += sc ** 2 * np.bincount(bidx, weights=w0c[s] ** 2, minlength=nb)
            for kk in range(K.NPAR):
                g = np.asarray(jvp(th0, tang[kk], JBc))
                for j, (s, bidx, nb, sc) in enumerate(binned):
                    J[row0[j]:row0[j + 1], kk] += sc * np.bincount(bidx, weights=g[s], minlength=nb)
            del Bc, JBc
            if log:
                log(f"  [{self.name}{tag}] chunk {ci + 1}/{len(files)}")
        return J, row0, sumw, sumw2

    def _gradient(self, max_chunks=None, log=print):
        specs = self.fit_specs(); keys = [o.key for o in specs]
        edges_by = {o.key: o.bin_edges() for o in specs}
        ctx = self._jvp_ctx()
        J, row0, sumw, sumw2 = self._stream_grad(self.bank, self.cfg.signal, keys, edges_by, ctx,
                                                 max_chunks=max_chunks, log=log)
        if getattr(self.cfg.signal, "target", "carbon") == "CH":
            fk = [k for k in keys if k in FREEH_KEYS]
            if fk:
                def _rand_daT(obs, n):
                    if "dalphat" in obs:
                        obs["dalphat"] = np.random.default_rng(0).uniform(0.0, np.pi, len(obs["dalphat"]))
                JH, rH, swH, sw2H = self._stream_grad(self._h_bank(), self.cfg.signal, fk, {k: edges_by[k] for k in fk},
                                                      ctx, override=_rand_daT, max_chunks=max_chunks, log=log,
                                                      tag=":freeH")
                for jf, k in enumerate(fk):
                    jc = keys.index(k)
                    J[row0[jc]:row0[jc + 1]] += JH[rH[jf]:rH[jf + 1]]
                    sumw[jc] += swH[jf]; sumw2[jc] += sw2H[jf]
                if log:
                    log(f"  [{self.name}] free-H (CH) added to {fk}")
        central = {k: sumw[j] for j, k in enumerate(keys)}
        mcerr = {k: np.sqrt(sumw2[j]) for j, k in enumerate(keys)}
        sigma = np.concatenate([_bin_sigma(central[k], mcerr[k], self.syst) for k in keys])
        return dict(J=J, sigma=sigma, row0=row0, keys=keys, edges=edges_by, central=central, mcerr=mcerr)

    def jacobian(self, **kw):
        r = self._gradient(**kw)
        return r["J"], r["sigma"], r["row0"], r["keys"]

    def constrained(self, max_chunks=None, log=print):
        r = self._gradient(max_chunks=max_chunks, log=log)
        F, V, sig_post, shrink, reach = constrained_from(r["J"], r["sigma"], K.PRIOR)
        r.update(F=F, V=V, sig_post=sig_post, shrink=shrink, reach=reach, prior=K.PRIOR, pnames=K.PNAMES)
        return r


    def dump_cache(self, label=None, res=None, max_chunks=None, log=print):
        """Persist the constrained-set npz (J, sigma, row0, dskeys, shrink, F, V + per-obs edges/central) to
        <results>/{label}.npz -- what SampleSet / the figures load."""
        r = res or self.constrained(max_chunks=max_chunks, log=log)
        out_dir = results_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{label or self.name}.npz"
        np.savez(out, J=r["J"], sigma=r["sigma"], row0=r["row0"], prior=K.PRIOR, pnames=K.PNAMES,
                 dskeys=r["keys"], shrink=r["shrink"], F=r["F"], V=r["V"],
                 **{f"{k}_edges": r["edges"][k] for k in r["keys"]},
                 **{f"{k}_central": r["central"][k] for k in r["keys"]},
                 **{f"{k}_mcerr": r["mcerr"][k] for k in r["keys"]})
        if log:
            log(f"[out] {out}")
        return out

    @staticmethod
    def load_cache(label):
        return np.load(results_dir() / f"{label}.npz", allow_pickle=True)

    def __add__(self, other):
        return SampleSet([self, other])


class SampleSet:
    """Several samples fitted / gated jointly.  Rows are concatenated across samples; the combined Fisher
    is the sum of the per-sample outer products (additive)."""
    def __init__(self, samples):
        self.samples = list(samples)

    @classmethod
    def from_configs(cls, paths, **kw):
        return cls([AnaSample.from_config(p, **kw) for p in paths])

    def __add__(self, other):
        return SampleSet(self.samples + ([other] if isinstance(other, AnaSample) else other.samples))

    def constrained(self, max_chunks=None, log=print):
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
        F, V, sig_post, shrink, reach = constrained_from(J, sigma, K.PRIOR)
        return dict(J=J, sigma=sigma, row0=np.asarray(row0), keys=keys, edges=edges, central=central,
                    F=F, V=V, sig_post=sig_post, shrink=shrink, reach=reach, prior=K.PRIOR, pnames=K.PNAMES)
