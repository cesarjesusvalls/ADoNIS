"""Fisher-information utilities: pure numpy, no model coupling.

Every function here takes arrays -- a Jacobian, a sigma, a prior -- and returns arrays.  Nothing
imports a bank, a sample or jax.
"""
from __future__ import annotations

import numpy as np

def fisher(J, sigma, rows=None):
    """Asimov Fisher F = (J/sigma)^T (J/sigma).  `rows` restricts to a subset of bin rows: Fisher is
    additive, so a subset's F is the sum of its bins' outer products -- exactly this slice."""
    if rows is None:
        Jw = np.asarray(J) / np.asarray(sigma)[:, None]
    else:
        Jw = np.asarray(J)[rows] / np.asarray(sigma)[rows, None]
    return Jw.T @ Jw

def posterior(F, prior):
    """Marginalized posterior covariance V = inv(F + diag(1/prior^2)) (Gaussian prior x Asimov Fisher)."""
    return np.linalg.inv(F + np.diag(1.0 / np.asarray(prior) ** 2))

def shrink_from_fisher(F, prior):
    """Marginalized shrinkage sqrt(diag(V))/prior for a Fisher matrix already in hand."""
    return np.sqrt(np.diag(posterior(F, prior))) / np.asarray(prior)

def vif_from_fisher(F, prior):
    """Degeneracy measure: variance inflation factor and multiple correlation, per knob.

        sigma_marg = sqrt(diag(inv(F + P)))     knob free, all others free   (correlation-aware)
        sigma_cond = 1/sqrt(diag(F) + diag(P))  knob free, all others fixed  (correlation-blind)
        VIF        = (sigma_marg / sigma_cond)^2
        R_multi    = sqrt(1 - 1/VIF)            multiple correlation of this knob with the rest

    Shrinkage alone cannot express this: a knob can look well-constrained by shrinkage while still
    being almost fully degenerate with the others (high VIF).  Use the multiple correlation rather
    than a pairwise one -- degeneracy can be spread over many knobs at once, each pairwise
    correlation small even when R_multi is close to 1.

    This is a linear diagnostic: it flags a flat direction, not a second minimum from a nonlinear
    parameter dependence.  Necessary, not sufficient.
    """
    P = 1.0 / np.asarray(prior) ** 2
    V = np.linalg.inv(np.asarray(F) + np.diag(P))
    s_marg = np.sqrt(np.maximum(np.diag(V), 0.0))
    s_cond = 1.0 / np.sqrt(np.maximum(np.diag(F) + P, 1e-300))
    vif = (s_marg / s_cond) ** 2
    return vif, np.sqrt(np.maximum(1.0 - 1.0 / np.maximum(vif, 1.0), 0.0))

def subset_degeneracy(F, prior, idx):
    """Degeneracy diagnostics for the knobs in `idx` with only those free (the rest frozen at nominal).

    Freezing a knob removes a flat direction it used to absorb, which can push knobs that looked
    only mildly correlated in the full marginal into near-degeneracy once frozen around.  So this
    must be recomputed per subset rather than read off the full-set result.

    Returns dict(sigma, vif, rmulti, corr):
      sigma  = sqrt(diag(inv(F_sub + P_sub)))          all of `idx` free
      vif    = (sigma / sigma_conditional)^2           variance inflation, sigma_cond = knob alone
      rmulti = sqrt(1 - 1/vif)                         multiple correlation against the other survivors
      corr   = the correlation matrix over `idx`

    Use rmulti/vif rather than a pairwise maximum: degeneracy against a combination of knobs can
    leave every single pairwise correlation small.
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
    """Drop the most degenerate knob, recompute, repeat until every survivor has VIF < vif_cut.

    Iterative because each drop re-freezes a direction and changes every remaining knob's VIF; a
    single pass over a fixed ranking would not give the same answer.  Returns the surviving index
    list.
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

def fisher_shrinkage(J, sigma, prior, rows=None):
    """Fisher information and posterior shrinkage from a per-bin Jacobian.  Returns
    (F, V, sig_post, shrink, reach):
      F      = (J/sigma)^T (J/sigma)           Fisher information (additive across samples)
      V      = inv(F + diag(1/prior^2))        marginalized posterior covariance
      sig_post = sqrt(diag(V))                 marginalized posterior sigma
      shrink = sig_post / prior                < 1 means the data, not the prior, sets the width
      reach  = sqrt(diag(F))                   raw per-knob reach (other knobs held fixed)
    """
    F = fisher(J, sigma, rows)
    V = posterior(F, prior)
    sig_post = np.sqrt(np.diag(V))
    shrink = sig_post / np.asarray(prior)
    reach = np.sqrt(np.maximum(np.diag(F), 0.0))
    return F, V, sig_post, shrink, reach
