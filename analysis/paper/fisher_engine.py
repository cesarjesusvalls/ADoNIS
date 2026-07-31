"""Shared Gate-I Fisher engine for the section-2/3 per-bin gradient drivers.

Every per-bin gradient driver (physical_fit, minerva_fit, minerva_ptpz_fit, t2k_pcos_fit,
electron_fit) and every assembler that stacks them (sec3_gradients/build_multisample, beams/beam_fisher,
sec2_fisher) builds the SAME object: a per-bin Jacobian J (bins x knobs) via one jax.jvp per knob through
bank_reweight.bank_weight, a diagonal error-model sigma per bin, then the Asimov Fisher
F = (J/sigma)^T (J/sigma) and Gate-I shrinkage sqrt(diag(inv(F + prior^-2)))/prior.

This module is the SINGLE source of that math, so a fix (e.g. the empty-bin sigma=inf guard) lands once
instead of being copy-pasted into eight files that then drift apart.  The functions are exact extractions
of the code they replace -- no behaviour change.
"""
import numpy as np
import jax.numpy as jnp

from analysis.paper import info_content as IC


def bin_sigma(central, mcerr, syst):
    """Diagonal error model sigma = sqrt((syst*central)^2 + mcerr^2) with the EMPTY-BIN GUARD.

    A bin with no selected events has central=mcerr=0 AND J=0 (the bincount reduction is 0 there), so an
    unguarded sqrt(var)=0 makes J/sigma = 0/0 = NaN, and F = (J/sigma)^T(J/sigma) becomes NaN in EVERY
    entry -> every knob silently reports "freeze".  sigma=inf gives J/sigma=0 instead, which is correct
    (an empty bin carries no information).
    """
    var = (syst * np.asarray(central)) ** 2 + np.asarray(mcerr) ** 2
    return np.where(var > 0, np.sqrt(var), np.inf)


def bank_jacobian(jvp_wf, th0, JB, ds, npar, log=None, names=None):
    """Per-bin Jacobian J (sum(nbin) x npar) via one jax.jvp per knob through the bank weight.

    jvp_wf(theta, tangent, JB) -> the directional derivative of the per-event weight; each dataset's
    info_content reduction (IC.bin_w0) turns that into per-bin gradient rows.  Returns (J, row0) where
    row0 is the per-dataset bin-offset array (np.cumsum of the datasets' nbin).
    """
    row0 = np.cumsum([0] + [d["nbin"] for d in ds])
    J = np.zeros((int(row0[-1]), npar))
    th0 = jnp.asarray(th0)
    for k in range(npar):
        g = np.asarray(jvp_wf(th0, jnp.zeros(npar).at[k].set(1.0), JB))
        for j, d in enumerate(ds):
            J[row0[j]:row0[j + 1], k] = IC.bin_w0(d, g)
        if log is not None:
            log(f"  jvp {k + 1:2d}/{npar}" + (f" {names[k]}" if names is not None else ""))
    return J, row0


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
