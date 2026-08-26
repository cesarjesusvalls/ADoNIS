"""Fisher-information utilities: pure numpy, no model coupling.

Moved from analysis/paper/fisher_engine.py.  Every function here takes arrays -- a Jacobian, a sigma, a
prior -- and returns arrays.  Nothing imports a bank, a sample or jax, which is why it belongs in the
package rather than in the application that happened to be its first caller.

NOT moved: bank_jacobian / bank_jacobian_chunked.  Those take a jvp callable and a bank directory, so
they are model- and IO-coupled and do not belong in a model-free module.  They are also superseded --
AnaSample._stream_grad is the only one on the live Gate-I path and does strictly more, accumulating the
Jacobian AND the central/MC-error in the SAME pass so J and sigma cannot disagree.  They die with
analysis/paper/physical_fit.py, their only remaining caller.

bin_sigma also left: it had three textually identical copies and now lives in adonis/stats/gaussian.py.
"""
from __future__ import annotations

import numpy as np

def fisher(J, sigma, rows=None):
    """Asimov Fisher F = (J/sigma)^T (J/sigma).  `rows`: restrict to a subset of bin rows (sec3 subsets);
    Fisher is ADDITIVE, so a subset's F is the sum of its bins' outer products -- exactly this slice."""
    if rows is None:
        Jw = np.asarray(J) / np.asarray(sigma)[:, None]
    else:
        Jw = np.asarray(J)[rows] / np.asarray(sigma)[rows, None]
    return Jw.T @ Jw

def posterior(F, prior):
    """Marginalized posterior covariance V = inv(F + diag(1/prior^2)) (Gaussian prior x Asimov Fisher)."""
    return np.linalg.inv(F + np.diag(1.0 / np.asarray(prior) ** 2))

def shrink_from_fisher(F, prior):
    """Gate-I marginalized shrinkage sqrt(diag(V))/prior for a Fisher matrix already in hand."""
    return np.sqrt(np.diag(posterior(F, prior))) / np.asarray(prior)

def vif_from_fisher(F, prior):
    """Gate-I DEGENERACY measure: variance inflation factor and multiple correlation, per knob.

        sigma_marg = sqrt(diag(inv(F + P)))     knob free, ALL others free   (correlation-aware)
        sigma_cond = 1/sqrt(diag(F) + diag(P))  knob free, all others FIXED  (correlation-blind)
        VIF        = (sigma_marg / sigma_cond)^2
        R_multi    = sqrt(1 - 1/VIF)            multiple correlation of this knob with the rest

    Shrinkage alone cannot express this.  Measured on the sec2/sec3 combined set, delta_strength and
    axial_strength pass shrink<0.5 by a hair (0.474, 0.484) while carrying VIF 190 and 371 -- the data
    pins each to ~3% of its prior on its own, and only ~48% once the other knobs move.  delta_strength is
    exactly the dial whose removal takes every pull in the toy ensemble back to 1.

    Use the MULTIPLE correlation, not a pairwise one: axial_strength is the most degenerate knob in the
    set (R_multi 0.9987) yet its largest pairwise correlation with any single knob is only 0.106 -- its
    degeneracy is spread over many knobs at once and a pairwise cut passes it.

    NOTE this is a LINEAR diagnostic.  It flags a flat direction; it cannot flag the second minimum that
    made delta_strength misbehave (the RES weight is quadratic in it, so the parameter -> prediction map
    is two-to-one).  Necessary, not sufficient.
    """
    P = 1.0 / np.asarray(prior) ** 2
    V = np.linalg.inv(np.asarray(F) + np.diag(P))
    s_marg = np.sqrt(np.maximum(np.diag(V), 0.0))
    s_cond = 1.0 / np.sqrt(np.maximum(np.diag(F) + P, 1e-300))
    vif = (s_marg / s_cond) ** 2
    return vif, np.sqrt(np.maximum(1.0 - 1.0 / np.maximum(vif, 1.0), 0.0))

def subset_degeneracy(F, prior, idx):
    """Degeneracy diagnostics for the knobs in `idx` WITH ONLY THOSE FREE (the rest frozen at nominal).

    This conditioning is the whole point.  Gate I quotes `shrink` from the FULL 28-knob marginal, then
    hands the fit a problem in which the knobs that failed the cut are FROZEN -- and freezing changes the
    correlations of the survivors, because a flat direction the frozen knobs used to absorb is dumped onto
    whoever is left.  Measured on this analysis: corr(M_A_res, delta_strength) is -0.489 with all 28 free
    and -0.983 with the 16 survivors free.  Gate I never evaluated the second number, which is the one the
    fit experiences, so a pair that looked benign was in fact near-degenerate and the toy fits fell into a
    second minimum along the resulting valley.

    Returns dict(sigma, vif, rmulti, corr):
      sigma  = sqrt(diag(inv(F_sub + P_sub)))          all of `idx` free
      vif    = (sigma / sigma_conditional)^2           variance inflation, sigma_cond = knob alone
      rmulti = sqrt(1 - 1/vif)                         multiple correlation against the other survivors
      corr   = the correlation matrix over `idx`

    Use rmulti/vif, not a pairwise maximum: a knob can be degenerate against a COMBINATION of others while
    its largest single-pair correlation stays small (axial_strength: rmulti 0.999, max pairwise 0.106).
    """
    idx = list(idx)
    Fs = np.asarray(F)[np.ix_(idx, idx)]
    P = np.diag(1.0 / np.asarray(prior)[idx] ** 2)
    V = np.linalg.inv(Fs + P)
    s = np.sqrt(np.maximum(np.diag(V), 0.0))
    s_cond = 1.0 / np.sqrt(np.maximum(np.diag(Fs) + np.diag(P), 1e-300))
    vif = (s / s_cond) ** 2
    D = np.diag(1.0 / np.where(s > 0, s, 1.0))
    return dict(sigma=s, vif=vif, rmulti=np.sqrt(np.maximum(1.0 - 1.0 / np.maximum(vif, 1.0), 0.0)),
                corr=D @ V @ D)

def prune_by_vif(F, prior, idx, vif_cut=20.0, names=None, log=None):
    """Drop the most degenerate knob, RECOMPUTE, repeat until every survivor has VIF < vif_cut.

    Iteration is required, not cosmetic: each drop re-freezes a direction and changes every remaining
    knob's VIF, so a single pass over a fixed ranking is not the same answer.  Returns the surviving
    index list.  On this analysis (cut 20) it removes s_NN_elastic[1], delta_strength and src_tail --
    delta_strength being exactly the knob whose removal takes every pull in the toy ensemble back to 1.
    """
    cur = list(idx)
    while len(cur) > 1:
        d = subset_degeneracy(F, prior, cur)
        a = int(np.argmax(d["vif"]))
        if d["vif"][a] < vif_cut:
            break
        if log:
            nm = names[cur[a]] if names is not None else cur[a]
            log(f"  prune_by_vif: drop {nm} (VIF {d['vif'][a]:.0f}, R_multi {d['rmulti'][a]:.4f}) "
                f"-> {len(cur)-1} knobs")
        cur.pop(a)
    return cur

def gate1(J, sigma, prior, rows=None):
    """Fisher + Gate-I from a per-bin Jacobian.  Returns (F, V, sig_post, shrink, reach):
      F      = (J/sigma)^T (J/sigma)           Fisher information (ADDITIVE across samples)
      V      = inv(F + diag(1/prior^2))         marginalized posterior covariance
      sig_post = sqrt(diag(V))                  marginalized posterior sigma
      shrink = sig_post / prior                 Gate I: a knob is FIT when shrink < fit_cut (0.5)
      reach  = sqrt(diag(F))                    raw per-knob reach (other knobs held fixed)
    """
    F = fisher(J, sigma, rows)
    V = posterior(F, prior)
    sig_post = np.sqrt(np.diag(V))
    shrink = sig_post / np.asarray(prior)
    reach = np.sqrt(np.maximum(np.diag(F), 0.0))
    return F, V, sig_post, shrink, reach
