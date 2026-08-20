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


# beam_model moved to adonis/analysis/beams.py -- it is sample-layer machinery that the fit
# stages construct BeamSample from; only the Fisher driver below is a figure.
from adonis.analysis.beams import beam_model, BEAM_DIRS, BEAM_LABEL   # noqa: F401


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
