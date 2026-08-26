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
    """Gate-I marginalized shrinkage sqrt(diag(V))/prior for a Fisher matrix already in hand."""
    return np.sqrt(np.diag(posterior(F, prior))) / np.asarray(prior)

def vif_from_fisher(F, prior):
    """Gate-I degeneracy measure: variance inflation factor and multiple correlation, per knob.

        sigma_marg = sqrt(diag(inv(F + P)))     knob free, all others free   (correlation-aware)
        sigma_cond = 1/sqrt(diag(F) + diag(P))  knob free, all others fixed  (correlation-blind)
        VIF        = (sigma_marg / sigma_cond)^2
        R_multi    = sqrt(1 - 1/VIF)            multiple correlation of this knob with the rest

    Shrinkage alone cannot express this: a knob can pass the shrink<0.5 cut while still being almost
    fully degenerate with the others (high VIF), because shrinkage only measures how well the data
    constrains the knob when everything else is also let float, not how much of that constraint comes
    from correlation with other knobs.

    Use the multiple correlation, not a pairwise one: a knob's degeneracy can be spread over many other
    knobs at once, in which case every single pairwise correlation stays small even though R_multi is
    close to 1.  A pairwise cut would pass such a knob.

    Note this is a linear diagnostic.  It flags a flat direction; it cannot flag a second minimum caused
    by a nonlinear (e.g. quadratic) dependence of the model on the parameter, which makes the
    parameter -> prediction map many-to-one.  Necessary, not sufficient.
    """
    P = 1.0 / np.asarray(prior) ** 2
    V = np.linalg.inv(np.asarray(F) + np.diag(P))
    s_marg = np.sqrt(np.maximum(np.diag(V), 0.0))
    s_cond = 1.0 / np.sqrt(np.maximum(np.diag(F) + P, 1e-300))
    vif = (s_marg / s_cond) ** 2
    return vif, np.sqrt(np.maximum(1.0 - 1.0 / np.maximum(vif, 1.0), 0.0))

def subset_degeneracy(F, prior, idx):
    """Degeneracy diagnostics for the knobs in `idx` with only those free (the rest frozen at nominal).

    This conditioning matters: a full-set marginal shrinkage does not predict the correlations a fit
    will see once some knobs are frozen, because freezing a knob removes a flat direction the frozen
    knob used to absorb, and that direction gets dumped onto whoever is left free.  A pair that looks
    only mildly correlated in the full marginal can become near-degenerate once other knobs are frozen
    around it, pulling the fit into a spurious second minimum along the resulting valley.

    Returns dict(sigma, vif, rmulti, corr):
      sigma  = sqrt(diag(inv(F_sub + P_sub)))          all of `idx` free
      vif    = (sigma / sigma_conditional)^2           variance inflation, sigma_cond = knob alone
      rmulti = sqrt(1 - 1/vif)                         multiple correlation against the other survivors
      corr   = the correlation matrix over `idx`

    Use rmulti/vif, not a pairwise maximum: a knob can be degenerate against a combination of others
    while its largest single-pair correlation stays small.
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

    Iteration is required, not cosmetic: each drop re-freezes a direction and changes every remaining
    knob's VIF, so a single pass over a fixed ranking is not the same answer.  Returns the surviving
    index list.
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
    """Fisher + Gate-I from a per-bin Jacobian.  Returns (F, V, sig_post, shrink, reach):
      F      = (J/sigma)^T (J/sigma)           Fisher information (ADDITIVE across samples)
      V      = inv(F + diag(1/prior^2))         marginalized posterior covariance
      sig_post = sqrt(diag(V))                  marginalized posterior sigma
      shrink = sig_post / prior                 < 1 means the data, not the prior, sets the width
      reach  = sqrt(diag(F))                    raw per-knob reach (other knobs held fixed)
    """
    F = fisher(J, sigma, rows)
    V = posterior(F, prior)
    sig_post = np.sqrt(np.diag(V))
    shrink = sig_post / np.asarray(prior)
    reach = np.sqrt(np.maximum(np.diag(F), 0.0))
    return F, V, sig_post, shrink, reach
