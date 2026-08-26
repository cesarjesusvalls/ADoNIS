"""Tagged-beam samples (pi+/p/n on carbon): the model behind the beam half of the Gate-I stack.

Consumed by analysis.campaign.stages.multisample (BeamSample) via beam_model.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

_BANKS = "output/paper_banks_p4"
BEAM_DIRS = {"pip": f"{_BANKS}/beam_pip_C/merged", "prot": f"{_BANKS}/beam_prot_C/merged",
             "neut": f"{_BANKS}/beam_neut_C/merged"}
BEAM_LABEL = {"pip": "$\\pi^+$–C", "prot": "p–C", "neut": "n–C"}


def beam_model(beam, nbins=15, syst=0.05, log=print, max_chunks=None, cap=None):
    """Load one beam bank and return the DIFFERENTIABLE model pieces a fit needs (not just the Jacobian):

        w_of(theta)   -> per-event FSI reweight (pure JAX, ==1 at nominal)
        jvp(th, tang) -> jitted weight tangent (for the per-knob Jacobian column)
        binned(w)     -> sigma_X(bin) = pi R^2 * sum_bin X_i w_i / n_tried, concatenated [react, second]
        central, sigma, mcerr, edges, keys, th0, n_tried

    model(theta) = binned(w_of(theta)) for ANY theta, using the SAME 28-knob physical_fit basis every
    other sample uses.  max_chunks caps the loaded statistics."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from adonis.workflow.generate_bank import load_bank as _load_bank
    from adonis.reweight.reweight_model import nominal_knobs
    from adonis.reweight import knobs as PF
    from adonis.stats import fisher as FE
    from adonis.stats.gaussian import bin_sigma as _bin_sigma

    B = _load_bank(BEAM_DIRS[beam], max_chunks=max_chunks)
    man = B["manifest"]
    nom = nominal_knobs()
    p = np.asarray(B["beam_p"], float)
    edges = np.linspace(man["pmin"], man["pmax"], nbins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, nbins - 1)
    n_tried = np.bincount(idx, minlength=nbins).astype(float)
    from adonis.workflow import records as REC
    _fl = REC.derive_flags(B)
    is_pion = man["species"] == "PION"
    react = _fl["reacted"].astype(float)
    second = _fl["absorbed"].astype(float) if is_pion else (np.asarray(B["n_pi_out"]) > 0).astype(float)
    PIR2 = man["pir2_mb"]
    keys = [f"{beam}_react", f"{beam}_abs" if is_pion else f"{beam}_pipro"]

    if cap and int(react.sum()) > cap:
        M = int(np.searchsorted(np.cumsum(react), cap) + 1)
        keep_ev = np.zeros(len(react), bool); keep_ev[:M] = True
        n_tried = np.bincount(idx[:M], minlength=nbins).astype(float)
    else:
        M = len(react); keep_ev = np.ones(len(react), bool)

    rmask = react.astype(bool) & keep_ev; ridx = np.where(rmask)[0]
    log(f"[events] beam {beam:<4} N={len(ridx):>9,} signal=reacted  (of {M:,} tried"
        f"{'' if M == len(react) else ', cap=%s' % f'{cap:,}'}, {max_chunks} chunks)")
    remap = np.full(len(rmask), -1, np.int64); remap[ridx] = np.arange(len(ridx))
    idx = idx[rmask]; second = second[rmask]; react = np.ones(len(idx))
    _PION = ("bc", "sa", "ss_el", "ss", "si", "pi_hh", "pi_a", "sa_c", "ss_el_c", "ss_c", "si_c")
    _NUC = ("hh", "a", "iso", "finel", "inel", "swap")
    mp = remap[np.asarray(B["f_p_eidx"])] >= 0; mn = remap[np.asarray(B["f_n_eidx"])] >= 0
    rec = {f: jnp.asarray(np.asarray(B[f"f_{f}"])[mp]) for f in _PION}
    rec.update({f: jnp.asarray(np.asarray(B[f"f_{f}"])[mn]) for f in _NUC})
    rec["p_eidx"] = jnp.asarray(remap[np.asarray(B["f_p_eidx"])[mp]])
    rec["n_eidx"] = jnp.asarray(remap[np.asarray(B["f_n_eidx"])[mn]])
    rec["n_events"] = len(ridx)
    from adonis.fsi.cascade import pool_fsi_reweight

    def w_of(theta):
        k = PF.knobs_of(theta, nom)
        return pool_fsi_reweight(rec, k.sabs, 1.0, s_piN_elastic=k.s_piN_elastic,
                                 s_piN_cex=k.s_piN_cex, s_conv=k.s_conv,
                                 s_NN_elastic=k.s_NN_elastic, s_NN_inelastic=k.s_NN_inelastic,
                                 f_NN_cex=k.f_NN_cex)

    th0 = jnp.asarray(PF.theta_nominal(nom))
    w0 = np.asarray(w_of(th0))
    assert np.abs(w0 - 1).max() < 1e-9, f"nominal identity broken on the {beam} bank: {np.abs(w0-1).max():.2e}"

    from adonis.fit.binning import BinSpec
    _scale = PIR2 / np.maximum(n_tried, 1)
    spec_r = BinSpec(idx, nbins, _scale, coef=react)
    spec_s = BinSpec(idx, nbins, _scale, coef=second)

    def binned(w):
        return np.concatenate([spec_r.apply(np.asarray(w)), spec_s.apply(np.asarray(w))])

    def binned_dev(w):
        import jax.numpy as jnp
        return jnp.concatenate([spec_r.apply_dev(w), spec_s.apply_dev(w)])

    central = binned(w0)
    nr = np.bincount(idx, weights=react, minlength=nbins)
    ns = np.bincount(idx, weights=second, minlength=nbins)
    mcerr = PIR2 * np.concatenate([np.sqrt(nr), np.sqrt(ns)]) / np.maximum(np.concatenate([n_tried] * 2), 1)
    sig = _bin_sigma(central, mcerr, syst)
    jvp = jax.jit(lambda th, tang: jax.jvp(w_of, (th,), (tang,))[1])
    jvpv = jax.jit(lambda th, T: jax.vmap(lambda t: jax.jvp(w_of, (th,), (t,))[1])(T))
    jvp2 = jax.jit(lambda th, tang: jax.jvp(lambda t: jax.jvp(w_of, (t,), (tang,))[1], (th,), (tang,))[1])
    jvp3 = jax.jit(lambda th, u, w, x: jax.jvp(
        lambda t3: jax.jvp(lambda t2: jax.jvp(lambda t1: w_of(t1), (t2,), (u,))[1], (t3,), (w,))[1],
        (th,), (x,))[1])
    log(f"  [{beam}] {len(p):,} tried | {int(react.sum()):,} reacted | {int(ns.sum()):,} second-obs "
        f"| {2*nbins} bins | med MC err {np.median(mcerr[central > 0] / central[central > 0]):.1%}")
    return dict(w_of=w_of, binned_dev=binned_dev, jvp=jvp, jvpv=jvpv, jvp2=jvp2, jvp3=jvp3, binned=binned, central=central, sigma=sig,
                mcerr=mcerr, edges=edges, keys=keys, th0=th0, nbins=nbins, n_tried=n_tried,
                n_events=len(ridx))


def beam_jacobian(beam, nbins=15, syst=0.05, log=print, max_chunks=None):
    """(J (2*nbins, NPAR), sigma (2*nbins,), central, edges, n_tried) for one beam bank.  Rows: reaction
    bins, then the second observable's bins (absorption for pi+, pion production for p/n).  One jax.jvp
    per knob through the SAME reweight, over beam_model.  max_chunks caps the loaded statistics.

    CACHED on disk (plotcache, keyed on the beam bank files + nbins/syst/max_chunks), written per-beam
    as it completes.  max_chunks is part of the key because a capped result must never be served to a
    full run.  The fingerprint is over INPUT FILES + params, NOT this code -- force a rebuild with
    ADONIS_PLOT_REFRESH=1 if beam_model's physics changes."""
    from adonis import cache as plotcache

    def _compute():
        import jax.numpy as jnp
        from adonis.reweight import knobs as PF
        m = beam_model(beam, nbins=nbins, syst=syst, log=log, max_chunks=max_chunks)
        NPAR = PF.NPAR
        J = np.zeros((2 * nbins, NPAR))
        for k in range(NPAR):
            g = m["jvp"](m["th0"], jnp.zeros(NPAR).at[k].set(1.0))
            J[:, k] = m["binned"](g)
        return dict(J=J, sigma=m["sigma"], central=m["central"], edges=m["edges"], n_tried=m["n_tried"])

    cap = "" if max_chunks is None else f"_c{max_chunks}"
    d = plotcache.cached(f"beam_jac_{beam}_n{nbins}_s{syst:g}{cap}", _compute,
                         deps=[BEAM_DIRS[beam]],
                         params={"beam": beam, "nbins": nbins, "syst": syst, "max_chunks": max_chunks})
    return d["J"], d["sigma"], d["central"], d["edges"], d["n_tried"]



def bank_sigma(beam, nbins, target="C", suffix=""):
    import os
    from adonis.workflow.generate_bank import load_bank as _load_bank
    pattern = os.environ.get("ADONIS_BEAM_PATTERN", "output/beam_{beam}_{target}{suffix}")
    path = pattern.format(beam=beam, target=target, suffix=suffix)
    if suffix and not Path(path).is_dir():
        path = pattern.format(beam=beam, target=target, suffix="")
    B = _load_bank(path)
    man = B["manifest"]
    bank_sigma.last = (path, int(man.get("n_total", 0)))
    p = np.asarray(B["beam_p"], float)
    edges = np.linspace(man["pmin"], man["pmax"], nbins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, nbins - 1)
    ntry = np.bincount(idx, minlength=nbins).astype(float)
    from adonis.workflow import records as REC
    _fl = REC.derive_flags(B)
    react = _fl["reacted"].astype(float)
    second = _fl["absorbed"].astype(float) if man["species"] == "PION" \
        else (np.asarray(B["n_pi_out"]) > 0).astype(float)
    nr = np.bincount(idx, weights=react, minlength=nbins)
    ns = np.bincount(idx, weights=second, minlength=nbins)
    PIR2 = man["pir2_mb"]
    with np.errstate(divide="ignore", invalid="ignore"):
        sr, ss = PIR2 * nr / ntry, PIR2 * ns / ntry
        er = PIR2 * np.sqrt(nr * np.clip(1.0 - nr / ntry, 0.0, 1.0)) / ntry
        es = PIR2 * np.sqrt(ns * np.clip(1.0 - ns / ntry, 0.0, 1.0)) / ntry
    return edges, sr, ss, er, es


def chi2(a, b, ea, eb):
    m = np.isfinite(a) & np.isfinite(b) & ((ea > 0) | (eb > 0)) & (a + b > 0)
    if not m.any():
        return np.nan, 0
    d = (a[m] - b[m]) ** 2 / (ea[m] ** 2 + eb[m] ** 2)
    return float(d.sum()), int(m.sum())

