"""KNOB x SAMPLE Fisher: what does each tagged beam add on top of T2K?

Section 3 of the paper asks "which knobs can this sample measure".  The T2K answer (10/27) is CONDITIONAL
on the T2K observable set, and two of the failure modes are curable by a different SAMPLE, not by more
T2K data:
  * DEGENERATE knobs -- the data sees them, another knob spends the sensitivity.  A tagged hadron beam is
    PURE FSI (no hard vertex, no spectral function), so the FSI knobs face NO competing knobs at all.
  * INVISIBLE knobs -- the channel is kinematically shut at T2K.  The beam ENERGY is a free lever, so the
    channel can simply be opened (pi+ > 680 MeV/c -> piN->etaN'; p/n KE > 290 MeV -> NN->NNpi).

Fisher is ADDITIVE, so this needs no new machinery:  F_total = F_T2K + sum_beams F_beam,
with F_s = J_s^T C_s^-1 J_s.  The T2K J is the persisted Gate-I Jacobian (physfit_gate1.npz);
each beam's J is one jax.jvp per knob through the SAME kind-1 reweight.

Observables per beam (sigma in mb, binned in the TAGGED beam momentum):
    pi+  : sigma_reaction(p), sigma_absorption(p)
    p/n  : sigma_reaction(p), sigma_pion-production(p)     <- the s_NN_inelastic handle
    sigma_X(bin, theta) = pi R^2 * sum_{i in bin} X_i * w_i(theta) / n_tried(bin)
The beam (p, b) distribution is theta-INDEPENDENT, so n_tried(bin) is a constant: only the numerator is
reweighted.  This is exact -- and it is only non-trivial because the pion reweight now carries the
SURVIVAL factor (with the old branch-only reweight, a common pion-sigma rescale left w == 1 identically,
so sigma_reaction could not respond to sabs at all).

Usage:  python -m analysis.paper.beams.beam_fisher [--syst 0.05] [--nbins 15]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root -> `from analysis.paper import ...`

# paper_banks_p4 banks keep their manifest + chunk_*.npz under {dir}/merged/ (load_bank reads that dir).
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

    So model(theta) = binned(w_of(theta)) for ANY theta, using the SAME 28-knob physical_fit basis every
    other sample uses.  beam_jacobian (below) is a thin wrapper over this; the multisample closure engine
    (adonis.fit.stages.multisample) is the other consumer.  max_chunks caps the loaded statistics."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from adonis.workflow.generate_bank import load_bank as _load_bank
    from adonis.reweight.reweight_model import nominal_knobs
    from analysis.paper import physical_fit as PF  # SPEC / knobs_of / theta_nominal: the SAME 28 knobs
    from analysis.paper import fisher_engine as FE

    B = _load_bank(BEAM_DIRS[beam], max_chunks=max_chunks)
    man = B["manifest"]
    nom = nominal_knobs()
    p = np.asarray(B["beam_p"], float)
    edges = np.linspace(man["pmin"], man["pmax"], nbins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, nbins - 1)
    n_tried = np.bincount(idx, minlength=nbins).astype(float)        # theta-INDEPENDENT
    from adonis.workflow import records as REC
    _fl = REC.derive_flags(B)                            # reacted/absorbed derived from prim_fate + nsc_prim
    is_pion = man["species"] == "PION"
    react = _fl["reacted"].astype(float)
    second = _fl["absorbed"].astype(float) if is_pion else (np.asarray(B["n_pi_out"]) > 0).astype(float)
    PIR2 = man["pir2_mb"]
    keys = [f"{beam}_react", f"{beam}_abs" if is_pion else f"{beam}_pipro"]

    # N_selected for the beams: the reaction observables sigma_X = PIR2 * sum(X_i w_i) / n_tried involve only
    # REACTED events (non-reacted carry w==1, contribute 0 to the react/second numerators AND have zero
    # gradient), while n_tried (the denominator, over ALL events) is already computed above and is
    # theta-independent.  So compact to the reacted events before building the reweight record -- only ~8%
    # goes on device.  The beam bank has no w0/hv_/fs_ this path needs, only the f_* FSI slot record (pion
    # family via f_p_eidx, nucleon via f_n_eidx), so compact THAT directly, remapping the event index.
    # The beam sample's SIGNAL is `reacted` -- the same role the CC0pi/CC1pi conditions play for the neutrino
    # samples (sigma_X involves only reacted events; the rest carry w==1 and have zero gradient).  Naming it
    # as the signal is what makes the event cap below apply here exactly as it does everywhere else.
    #
    # Event cap (fit-side).  Cap on EVENTS, exactly: keep the prefix of TRIED events that contains the first
    # `cap` signal (reacted) events and recompute n_tried over exactly that prefix.  The retained set is then
    # a smaller but COMPLETE beam exposure, so sigma_X = PIR2*sum(X_i w_i)/n_tried needs no rescaling.
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
    idx = idx[rmask]; second = second[rmask]; react = np.ones(len(idx))   # reacted subset (react == 1)
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

    # SAME binning primitive every other sample uses (IC.BinSpec): m_b = scale_b * sum_b coef_e * w_e.
    # This used to be a second, independent np.bincount implementation; unifying it is what lets the
    # device path -- and hence reverse-mode differentiation -- cover the WHOLE model instead of the
    # BankSamples only.
    from analysis.paper.info_content import BinSpec
    _scale = PIR2 / np.maximum(n_tried, 1)
    spec_r = BinSpec(idx, nbins, _scale, coef=react)
    spec_s = BinSpec(idx, nbins, _scale, coef=second)

    def binned(w):
        return np.concatenate([spec_r.apply(np.asarray(w)), spec_s.apply(np.asarray(w))])

    def binned_dev(w):
        import jax.numpy as jnp
        return jnp.concatenate([spec_r.apply_dev(w), spec_s.apply_dev(w)])

    central = binned(w0)
    # MC error: sqrt(N) on the counts in each bin (the reweight is 1 at nominal)
    nr = np.bincount(idx, weights=react, minlength=nbins)
    ns = np.bincount(idx, weights=second, minlength=nbins)
    mcerr = PIR2 * np.concatenate([np.sqrt(nr), np.sqrt(ns)]) / np.maximum(np.concatenate([n_tried] * 2), 1)
    # EMPTY bins (pion production is EXACTLY zero below the NN->NNpi threshold -> central=mcerr=0) get
    # sigma=inf, not 0, so 0/0 does not put NaN into the Fisher (bin_sigma's var==0 == this empty test).
    sig = FE.bin_sigma(central, mcerr, syst)
    jvp = jax.jit(lambda th, tang: jax.jvp(w_of, (th,), (tang,))[1])
    # Batched: ONE vmapped jvp over a (nsub, NPAR) tangent matrix -- same saving as the bank samples.
    jvpv = jax.jit(lambda th, T: jax.vmap(lambda t: jax.jvp(w_of, (th,), (t,))[1])(T))
    # 2nd directional derivative d^2/dt^2 w(theta + t*tang) via nested jvp (for the higher-order corner)
    jvp2 = jax.jit(lambda th, tang: jax.jvp(lambda t: jax.jvp(w_of, (t,), (tang,))[1], (th,), (tang,))[1])
    # 3rd MIXED directional derivative d^3 w /(du dw dx) via triple-nested jvp (for the 4th-order corner)
    jvp3 = jax.jit(lambda th, u, w, x: jax.jvp(
        lambda t3: jax.jvp(lambda t2: jax.jvp(lambda t1: w_of(t1), (t2,), (u,))[1], (t3,), (w,))[1],
        (th,), (x,))[1])
    log(f"  [{beam}] {len(p):,} tried | {int(react.sum()):,} reacted | {int(ns.sum()):,} second-obs "
        f"| {2*nbins} bins | med MC err {np.median(mcerr[central > 0] / central[central > 0]):.1%}")
    return dict(w_of=w_of, binned_dev=binned_dev, jvp=jvp, jvpv=jvpv, jvp2=jvp2, jvp3=jvp3, binned=binned, central=central, sigma=sig,
                mcerr=mcerr, edges=edges, keys=keys, th0=th0, nbins=nbins, n_tried=n_tried,
                n_events=len(ridx))


def beam_jacobian(beam, nbins=15, syst=0.05, log=print):
    """(J (2*nbins, NPAR), sigma (2*nbins,), central, edges, n_tried) for one beam bank.  Rows: reaction
    bins, then the second observable's bins (absorption for pi+, pion production for p/n).  One jax.jvp
    per knob through the SAME reweight, over beam_model.

    CACHED on disk (plotcache, keyed on the beam bank files + nbins/syst).  The 28-knob jvp over ~12M
    events is the slow part of every multisample rebuild and only changes when the bank or params change;
    the cache is also written per-beam as it completes, so a preempted rebuild resumes without redoing the
    beams it already finished.  NB the fingerprint is over INPUT FILES + params, NOT this code -- if
    beam_model's physics changes, force a rebuild with ADONIS_PLOT_REFRESH=1."""
    from analysis.paper import plotcache

    def _compute():
        import jax.numpy as jnp
        from analysis.paper import physical_fit as PF
        m = beam_model(beam, nbins=nbins, syst=syst, log=log)
        NPAR = PF.NPAR
        J = np.zeros((2 * nbins, NPAR))
        for k in range(NPAR):
            g = m["jvp"](m["th0"], jnp.zeros(NPAR).at[k].set(1.0))
            J[:, k] = m["binned"](g)                                # d sigma_bin / d theta_k (exact)
        return dict(J=J, sigma=m["sigma"], central=m["central"], edges=m["edges"], n_tried=m["n_tried"])

    d = plotcache.cached(f"beam_jac_{beam}_n{nbins}_s{syst:g}", _compute,
                         deps=[BEAM_DIRS[beam]], params={"beam": beam, "nbins": nbins, "syst": syst})
    return d["J"], d["sigma"], d["central"], d["edges"], d["n_tried"]


def main(syst=0.05, nbins=15):
    from analysis.paper import physical_fit as PF
    from analysis.paper import style
    from analysis.paper import fisher_engine as FE
    PNAMES = PF.PNAMES
    PRIOR = PF.PRIOR
    NPAR = PF.NPAR

    d = np.load(style.ALTGEN / "physfit_gate1.npz", allow_pickle=True)
    F = {"T2K": FE.fisher(d["J"], d["sigma"])}                        # (J/sigma)^T(J/sigma)
    print(f"== knob x sample Fisher (syst {syst:.0%}, {nbins} p-bins per observable) ==")
    for beam in ("pip", "prot", "neut"):
        if not Path(BEAM_DIRS[beam]).exists():
            print(f"  [skip] {beam}: no bank at {BEAM_DIRS[beam]}")
            continue
        Jb, sb, _c, _e, _n = beam_jacobian(beam, nbins=nbins, syst=syst)
        F[beam] = FE.fisher(Jb, sb)

    def shrink(Fm):
        return FE.shrink_from_fisher(Fm, PRIOR)

    cols = [("T2K", F["T2K"])]
    for beam in ("pip", "prot", "neut"):
        if beam in F:
            cols.append((f"T2K+{beam}", F["T2K"] + F[beam]))
    if len(F) > 1:
        cols.append(("ALL", sum(F.values())))
    S = np.column_stack([shrink(Fm) for _, Fm in cols])

    print(f"\n{'knob':>20} " + " ".join(f"{n:>10}" for n, _ in cols))
    for k in np.argsort(S[:, -1]):
        marks = "".join("*" if S[k, c] < 0.5 else " " for c in range(len(cols)))
        print(f"{PNAMES[k]:>20} " + " ".join(f"{S[k,c]:10.2f}" for c in range(len(cols))) + f"   {marks}")
    for c, (n, _) in enumerate(cols):
        gained = [PNAMES[k] for k in range(NPAR) if S[k, c] < 0.5 <= S[k, 0]]
        print(f"  {n:>10}: {int((S[:,c] < 0.5).sum()):2d}/{NPAR} FIT" +
              (f"   GAINED vs T2K: {gained}" if gained else ""))
    np.savez("output/altgen/beam_fisher.npz", S=S, cols=[n for n, _ in cols], pnames=PNAMES,
             prior=PRIOR, syst=syst, nbins=nbins)
    print("\n[out] output/altgen/beam_fisher.npz")
    return S, [n for n, _ in cols]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--syst", type=float, default=0.05)
    ap.add_argument("--nbins", type=int, default=15)
    a = ap.parse_args()
    main(syst=a.syst, nbins=a.nbins)
