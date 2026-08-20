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


# The model-free Fisher math moved to adonis/stats/fisher.py; bin_sigma to adonis/stats/gaussian.py.
# Re-exported here so paper-side callers keep working while the move settles.
from adonis.stats.fisher import (fisher, posterior, shrink_from_fisher, vif_from_fisher,  # noqa: F401
                                 subset_degeneracy, gate1_prune, gate1)
from adonis.stats.gaussian import bin_sigma                                               # noqa: F401

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


def bank_jacobian_chunked(jvp_wf, th0, bank_dir, chunk_ds, nbins_list, npar, log=None, names=None,
                          max_chunks=None):
    """Same Jacobian as bank_jacobian, but STREAMED over the bank's chunk files so peak memory is ONE chunk,
    not the whole bank -- which is what lets the 20M-event banks run on the GPU (the concatenated bank is
    11.8GB > GPU memory).  J[bin,k] = sum_events (dw/dtheta_k) is additive over events, and np.bincount
    accumulates, so this is BIT-IDENTICAL to bank_jacobian over the concatenated bank (chunk + within-chunk
    order preserved).

    chunk_ds(B_chunk) -> the per-chunk dataset list (sel_idx/binidx/nbin/scale_bin on the SAME fixed edges
    as the whole-bank datasets; NO central/offset needed -- only the bin_w0 fields).  nbins_list = the
    per-dataset nbin, in the same order chunk_ds returns, fixing the row layout."""
    import glob
    from adonis.reweight import bank_plot as BP, bank_reweight as BR
    row0 = np.cumsum([0] + list(nbins_list))
    J = np.zeros((int(row0[-1]), npar))
    th0 = jnp.asarray(th0)
    tang = [jnp.zeros(npar).at[k].set(1.0) for k in range(npar)]
    files = sorted(glob.glob(f"{bank_dir}/chunk_*.npz"))
    if max_chunks:
        files = files[:max_chunks]                          # subsampled bank (smoke/validation runs)
    nch = BP.bank_nchunks(bank_dir)
    for ci, f in enumerate(files):
        Bc = BP.load_bank_chunk(f, nch); JBc = BR.to_jax(Bc)     # one chunk on the GPU
        dsc = chunk_ds(Bc)
        for k in range(npar):
            g = np.asarray(jvp_wf(th0, tang[k], JBc))            # per-event dw/dtheta_k for THIS chunk
            for j, d in enumerate(dsc):
                J[row0[j]:row0[j + 1], k] += IC.bin_w0(d, g)     # accumulate (bincount is additive)
        del Bc, JBc
        if log is not None:
            log(f"  chunk {ci + 1}/{len(files)} ({int(row0[-1])} bins x {npar} knobs)")
    return J, row0














