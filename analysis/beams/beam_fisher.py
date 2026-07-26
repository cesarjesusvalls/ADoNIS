"""KNOB x SAMPLE Fisher: what does each tagged beam add on top of T2K?

Section 3 of the paper asks "which knobs can this sample measure".  The T2K answer (10/27) is CONDITIONAL
on the T2K observable set, and two of the failure modes are curable by a different SAMPLE, not by more
T2K data:
  * DEGENERATE knobs -- the data sees them, another knob spends the sensitivity.  A tagged hadron beam is
    PURE FSI (no hard vertex, no spectral function), so the FSI knobs face NO competing knobs at all.
  * INVISIBLE knobs -- the channel is kinematically shut at T2K.  The beam ENERGY is a free lever, so the
    channel can simply be opened (pi+ > 680 MeV/c -> piN->etaN'; p/n KE > 290 MeV -> NN->NNpi).

Fisher is ADDITIVE, so this needs no new machinery:  F_total = F_T2K + sum_beams F_beam,
with F_s = J_s^T C_s^-1 J_s.  The T2K J is the persisted Gate-I Jacobian (physfit_gate1_full_v2.npz);
each beam's J is one jax.jvp per knob through the SAME kind-1 reweight.

Observables per beam (sigma in mb, binned in the TAGGED beam momentum):
    pi+  : sigma_reaction(p), sigma_absorption(p)
    p/n  : sigma_reaction(p), sigma_pion-production(p)     <- the s_NN_inelastic handle
    sigma_X(bin, theta) = pi R^2 * sum_{i in bin} X_i * w_i(theta) / n_tried(bin)
The beam (p, b) distribution is theta-INDEPENDENT, so n_tried(bin) is a constant: only the numerator is
reweighted.  This is exact -- and it is only non-trivial because the pion reweight now carries the
SURVIVAL factor (with the old branch-only reweight, a common pion-sigma rescale left w == 1 identically,
so sigma_reaction could not respond to sabs at all).

Usage:  python -m analysis.beams.beam_fisher [--syst 0.05] [--nbins 15]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "altgen"))

BEAM_DIRS = {"pip": "output/beam_pip_C", "prot": "output/beam_prot_C", "neut": "output/beam_neut_C"}
BEAM_LABEL = {"pip": "$\\pi^+$–C", "prot": "p–C", "neut": "n–C"}


def beam_jacobian(beam, nbins=15, syst=0.05, log=print):
    """(J (2*nbins, NPAR), sigma (2*nbins,), central) for one beam bank.  Rows: reaction bins, then the
    second observable's bins (absorption for pi+, pion production for p/n)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from analysis.beams import beam_bank as BB
    from analysis.t2k.differentiability.full_knobs import nominal_knobs
    import physical_fit as PF                      # SPEC / knobs_of / theta_nominal: the SAME 27 knobs

    B = BB.load(BEAM_DIRS[beam])
    man = B["manifest"]
    nom = nominal_knobs()
    p = np.asarray(B["beam_p"], float)
    edges = np.linspace(man["pmin"], man["pmax"], nbins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, nbins - 1)
    n_tried = np.bincount(idx, minlength=nbins).astype(float)        # theta-INDEPENDENT
    react = np.asarray(B["reacted"], float)
    second = np.asarray(B["absorbed"], float) if man["species"] == "PION" \
        else (np.asarray(B["n_pi_out"]) > 0).astype(float)
    PIR2 = man["pir2_mb"]

    rec = {f: jnp.asarray(B[f"f_{f}"]) for f in
           ("bc", "sa", "ss_el", "ss", "si", "pi_hh", "pi_a", "sa_c", "ss_el_c", "ss_c", "si_c", "p_eidx",
            "hh", "a", "iso", "finel", "inel", "swap", "n_eidx")}
    rec["n_events"] = len(p)
    from adonis.fsi.cascade import pool_fsi_reweight

    def w_of(theta):
        k = PF.knobs_of(theta, nom)
        return pool_fsi_reweight(rec, k["sabs"], 1.0, s_piN_elastic=k["s_piN_elastic"],
                                 s_piN_cex=k["s_piN_cex"], s_conv=k["s_conv"],
                                 s_NN_elastic=k["s_NN_elastic"], s_NN_inelastic=k["s_NN_inelastic"],
                                 f_NN_cex=k["f_NN_cex"])

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
    sig = np.sqrt((syst * central) ** 2 + mcerr ** 2)
    # EMPTY bins (no counts -> central 0 and mcerr 0) carry no information: the pion-production observable
    # is EXACTLY zero below the NN->NNpi threshold, so those bins have sigma 0 and would put inf/NaN into
    # the Fisher.  Give them infinite error (zero weight) instead -- J is 0 there anyway.
    empty = (central == 0) & (mcerr == 0)
    sig = np.where(empty, np.inf, sig)

    jvp = jax.jit(lambda th, tang: jax.jvp(w_of, (th,), (tang,))[1])
    NPAR = PF.NPAR
    J = np.zeros((2 * nbins, NPAR))
    for k in range(NPAR):
        g = jvp(th0, jnp.zeros(NPAR).at[k].set(1.0))
        J[:, k] = binned(g)                                         # d sigma_bin / d theta_k (exact)
    log(f"  [{beam}] {len(p):,} tried | {int(react.sum()):,} reacted | {int(ns.sum()):,} second-obs "
        f"| {2*nbins} bins | med MC err {np.median(mcerr[central > 0] / central[central > 0]):.1%}")
    return J, sig, central, edges, n_tried


def main(syst=0.05, nbins=15):
    import physical_fit as PF
    from analysis.paper import style
    PNAMES = PF.PNAMES
    PRIOR = PF.PRIOR
    NPAR = PF.NPAR

    d = np.load(style.ALTGEN / "physfit_gate1_full_v2.npz", allow_pickle=True)
    J_t2k = d["J"] / d["sigma"][:, None]                              # already error-weighted rows
    F = {"T2K": J_t2k.T @ J_t2k}
    print(f"== knob x sample Fisher (syst {syst:.0%}, {nbins} p-bins per observable) ==")
    for beam in ("pip", "prot", "neut"):
        if not Path(BEAM_DIRS[beam]).exists():
            print(f"  [skip] {beam}: no bank at {BEAM_DIRS[beam]}")
            continue
        Jb, sb, _c, _e, _n = beam_jacobian(beam, nbins=nbins, syst=syst)
        Jw = Jb / sb[:, None]
        F[beam] = Jw.T @ Jw

    def shrink(Fm):
        V = np.linalg.inv(Fm + np.diag(1.0 / PRIOR ** 2))
        return np.sqrt(np.diag(V)) / PRIOR

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
