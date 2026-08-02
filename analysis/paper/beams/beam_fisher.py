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


def beam_model(beam, nbins=15, syst=0.05, log=print, max_chunks=None):
    """Load one beam bank and return the DIFFERENTIABLE model pieces a fit needs (not just the Jacobian):

        w_of(theta)   -> per-event FSI reweight (pure JAX, ==1 at nominal)
        jvp(th, tang) -> jitted weight tangent (for the per-knob Jacobian column)
        binned(w)     -> sigma_X(bin) = pi R^2 * sum_bin X_i w_i / n_tried, concatenated [react, second]
        central, sigma, mcerr, edges, keys, th0, n_tried

    So model(theta) = binned(w_of(theta)) for ANY theta, using the SAME 28-knob physical_fit basis every
    other sample uses.  beam_jacobian (below) is a thin wrapper over this; the multisample closure engine
    (analysis.paper.physfit.multisample) is the other consumer.  max_chunks caps the loaded statistics."""
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

    rec = {f: jnp.asarray(B[f"f_{f}"]) for f in
           ("bc", "sa", "ss_el", "ss", "si", "pi_hh", "pi_a", "sa_c", "ss_el_c", "ss_c", "si_c", "p_eidx",
            "hh", "a", "iso", "finel", "inel", "swap", "n_eidx")}
    rec["n_events"] = len(p)
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

    def binned(w):
        w = np.asarray(w)
        sr = PIR2 * np.bincount(idx, weights=react * w, minlength=nbins) / np.maximum(n_tried, 1)
        ss = PIR2 * np.bincount(idx, weights=second * w, minlength=nbins) / np.maximum(n_tried, 1)
        return np.concatenate([sr, ss])

    central = binned(w0)
    # MC error: sqrt(N) on the counts in each bin (the reweight is 1 at nominal)
    nr = np.bincount(idx, weights=react, minlength=nbins)
    ns = np.bincount(idx, weights=second, minlength=nbins)
    mcerr = PIR2 * np.concatenate([np.sqrt(nr), np.sqrt(ns)]) / np.maximum(np.concatenate([n_tried] * 2), 1)
    # EMPTY bins (pion production is EXACTLY zero below the NN->NNpi threshold -> central=mcerr=0) get
    # sigma=inf, not 0, so 0/0 does not put NaN into the Fisher (bin_sigma's var==0 == this empty test).
    sig = FE.bin_sigma(central, mcerr, syst)
    jvp = jax.jit(lambda th, tang: jax.jvp(w_of, (th,), (tang,))[1])
    # 2nd directional derivative d^2/dt^2 w(theta + t*tang) via nested jvp (for the higher-order corner)
    jvp2 = jax.jit(lambda th, tang: jax.jvp(lambda t: jax.jvp(w_of, (t,), (tang,))[1], (th,), (tang,))[1])
    log(f"  [{beam}] {len(p):,} tried | {int(react.sum()):,} reacted | {int(ns.sum()):,} second-obs "
        f"| {2*nbins} bins | med MC err {np.median(mcerr[central > 0] / central[central > 0]):.1%}")
    return dict(w_of=w_of, jvp=jvp, jvp2=jvp2, binned=binned, central=central, sigma=sig, mcerr=mcerr,
                edges=edges, keys=keys, th0=th0, nbins=nbins, n_tried=n_tried)


def beam_jacobian(beam, nbins=15, syst=0.05, log=print):
    """(J (2*nbins, NPAR), sigma (2*nbins,), central, edges, n_tried) for one beam bank.  Rows: reaction
    bins, then the second observable's bins (absorption for pi+, pion production for p/n).  Thin wrapper
    over beam_model -- one jax.jvp per knob through the SAME reweight."""
    import jax.numpy as jnp
    from analysis.paper import physical_fit as PF
    m = beam_model(beam, nbins=nbins, syst=syst, log=log)
    NPAR = PF.NPAR
    J = np.zeros((2 * nbins, NPAR))
    for k in range(NPAR):
        g = m["jvp"](m["th0"], jnp.zeros(NPAR).at[k].set(1.0))
        J[:, k] = m["binned"](g)                                    # d sigma_bin / d theta_k (exact)
    return J, m["sigma"], m["central"], m["edges"], m["n_tried"]


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
